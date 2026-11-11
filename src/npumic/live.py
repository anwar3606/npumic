"""Live processing: physical microphone -> processing chain -> virtual microphone.

PipeWire objects:
  npumic          virtual source that apps select ("NPU Denoised Mic")
  npumic_engine   processing node (JACK client via PipeWire's libjack) with one
                  input linked from the physical microphone and one output
                  linked into npumic

Input and output are handled in the same real-time callback, so they share one
clock and one graph cycle: no drift, no resampling, no buffering between
processes. The virtual source stays up when the physical microphone changes,
so apps never lose it.

While inactive, the physical microphone is not linked and the callback outputs
silence without processing; the service activates the engine only while an
app records from the virtual microphone (see `consumers`).

PipeWire's libjack is vendored (vendor/pipewire-jack, extracted from the
official Ubuntu package without root) and loaded before the `jack` module.
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from npumic.devices import VIRTUAL_MIC_PREFIX, Mic, Output, default_output
from npumic.engine import BLOCK, SAMPLE_RATE, Reblocker, Stage
from npumic.micvolume import CLIP_LEVEL
from npumic.monitor import Ring, Spectrograms
from npumic.runtime import ROOT

VIRTUAL_NAME = VIRTUAL_MIC_PREFIX
VIRTUAL_DESCRIPTION = "NPU Denoised Mic"
ENGINE_NAME = f"{VIRTUAL_MIC_PREFIX}_engine"
BLOCK_MS = 1000 * BLOCK / SAMPLE_RATE

_PIPEWIRE_LIBJACK_PATHS = (
    "/usr/lib/x86_64-linux-gnu/pipewire-0.3/jack/libjack.so.0",
    "/usr/lib64/pipewire-0.3/jack/libjack.so.0",
    "/usr/lib/pipewire-0.3/jack/libjack.so.0",
    str(ROOT / "vendor/pipewire-jack/extracted/usr/lib/x86_64-linux-gnu/pipewire-0.3/jack/libjack.so.0"),
)


def _load_jack():
    # Loading PipeWire's libjack first makes the `jack` module bind to it (same soname)
    # instead of a JACK server library.
    for path in _PIPEWIRE_LIBJACK_PATHS:
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            break
    else:
        raise RuntimeError("PipeWire libjack not found; install pipewire-jack (e.g. sudo apt install pipewire-jack)")
    # Ask the graph for 10 ms cycles, matching the processing block size.
    os.environ.setdefault("PIPEWIRE_LATENCY", f"{BLOCK}/{SAMPLE_RATE}")
    import jack

    return jack


def _pw_value(value) -> str:
    if isinstance(value, list):
        return "[ " + " ".join(value) + " ]"
    return json.dumps(value)


def _pw_props(props: dict) -> str:
    return "{ " + " ".join(f"{k} = {_pw_value(v)}" for k, v in props.items()) + " }"


def _node_ids(name: str) -> list[int]:
    objects = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, check=True).stdout)
    return [
        o["id"] for o in objects
        if o.get("type") == "PipeWire:Interface:Node"
        and ((o.get("info") or {}).get("props") or {}).get("node.name") == name
    ]


def _ports(node_name: str, direction: str) -> list[str]:
    """PipeWire port names ("node:port") of a node; direction is "-o" (outputs) or "-i" (inputs)."""
    listed = subprocess.run(["pw-link", direction], capture_output=True, text=True).stdout.splitlines()
    return [p.strip() for p in listed if p.strip().startswith(f"{node_name}:")]


def _wait_for_port(port: str, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        listed = subprocess.run(["pw-link", "-o"], capture_output=True, text=True).stdout.split()
        listed += subprocess.run(["pw-link", "-i"], capture_output=True, text=True).stdout.split()
        if port in listed:
            return
        time.sleep(0.05)
    raise RuntimeError(f"PipeWire port did not appear: {port}")


def _link(src: str, dst: str) -> None:
    subprocess.run(["pw-link", src, dst], capture_output=True)


def _unlink(src: str, dst: str) -> None:
    subprocess.run(["pw-link", "-d", src, dst], capture_output=True)


def consumers() -> list[str]:
    """Names of the apps (and the "hear myself" output device) recording from the virtual microphone, sorted."""
    listed = subprocess.run(["pw-link", "-l", "-o"], capture_output=True, text=True).stdout.splitlines()
    nodes, ours = set(), False
    for line in listed:
        if not line.startswith(" "):
            ours = line.strip() == f"{VIRTUAL_NAME}:capture_MONO"
        elif ours and "|->" in line:
            nodes.add(line.split("|->", 1)[1].strip().rsplit(":", 1)[0])
    return sorted(nodes)


def ensure_virtual_mic() -> None:
    """Creates the virtual source if it does not exist yet. It persists until removed."""
    if _node_ids(VIRTUAL_NAME):
        return
    props = _pw_props({
        "factory.name": "support.null-audio-sink",
        "node.name": VIRTUAL_NAME,
        "node.description": VIRTUAL_DESCRIPTION,
        "media.class": "Audio/Source/Virtual",
        "audio.rate": SAMPLE_RATE,
        "audio.position": ["MONO"],
        "object.linger": True,
    })
    subprocess.run(["pw-cli", "create-node", "adapter", props], capture_output=True, check=True)
    _wait_for_port(f"{VIRTUAL_NAME}:input_MONO")


def remove_virtual_mic() -> None:
    for node_id in _node_ids(VIRTUAL_NAME):
        subprocess.run(["pw-cli", "destroy", str(node_id)], capture_output=True)


@dataclass
class Stats:
    cycles: int = 0
    xruns: int = 0  # audio cycles the system reported as missed
    late_cycles: int = 0  # processing took longer than the cycle's audio duration
    max_ms: float = 0.0
    avg_ms: float = 0.0
    window_max_ms: float = 0.0  # highest processing time since the last take_window_max()
    cycle_frames: int = 0
    in_peak: float = 0.0  # highest sample level since the last take_peaks()
    out_peak: float = 0.0
    in_clipped: int = 0  # input samples at full scale since the last take_clipping()
    in_samples: int = 0

    def take_window_max(self) -> float:
        value, self.window_max_ms = self.window_max_ms, 0.0
        return value

    def take_clipping(self) -> tuple[int, int]:
        counts = self.in_clipped, self.in_samples
        self.in_clipped = self.in_samples = 0
        return counts

    def take_peaks(self) -> tuple[float, float]:
        peaks = self.in_peak, self.out_peak
        self.in_peak = self.out_peak = 0.0
        return peaks

    def add(self, ms: float, frames: int) -> None:
        self.cycles += 1
        self.cycle_frames = frames
        self.late_cycles += ms > 1000 * frames / SAMPLE_RATE
        self.max_ms = max(self.max_ms, ms)
        self.window_max_ms = max(self.window_max_ms, ms)
        self.avg_ms += 0.02 * (ms - self.avg_ms)


class LiveEngine:
    def __init__(self, stage: Stage):
        self.stage = stage
        self.stats = Stats()
        self.mic: Mic | None = None
        self._reblocker = Reblocker(stage)
        self._lock = threading.Lock()
        self._client = None
        self.shutdown_reason: str | None = None  # set when the audio system ends the connection
        self.active = False
        self._linked_inputs: list[str] = []
        self._monitor_links: list[tuple[str, str]] = []
        self.monitor_output: Output | None = None
        self.input_ring = Ring(4 * SAMPLE_RATE)
        self.output_ring = Ring(4 * SAMPLE_RATE)
        self.spectrograms = Spectrograms(self.input_ring, self.output_ring, lambda: self._reblocker.latency)

    @property
    def latency(self) -> int:
        """Processing latency in samples (excluding the audio system's own buffers)."""
        return self._reblocker.latency

    def start(self, mic: Mic | None, active: bool = True) -> None:
        # The audio callback needs the interpreter lock on every cycle; other threads
        # (web server, monitoring) are made to release it after 1 ms instead of 5 ms.
        sys.setswitchinterval(0.001)
        jack = _load_jack()
        self._warm_up()
        ensure_virtual_mic()
        client = jack.Client(ENGINE_NAME, no_start_server=True)
        if client.samplerate != SAMPLE_RATE:
            client.close()
            raise RuntimeError(f"Audio system runs at {client.samplerate} Hz; {SAMPLE_RATE} Hz required")
        self._in = client.inports.register("in")
        self._out = client.outports.register("out")
        client.set_process_callback(self._process)
        client.set_xrun_callback(self._on_xrun)
        client.set_shutdown_callback(self._on_shutdown)
        client.activate()
        self._client = client
        _wait_for_port(f"{ENGINE_NAME}:out")
        _link(f"{ENGINE_NAME}:out", f"{VIRTUAL_NAME}:input_MONO")
        self.mic = mic
        self.set_active(active)

    def _unlink_mic(self) -> None:
        for src in self._linked_inputs:
            _unlink(src, f"{ENGINE_NAME}:in")
        self._linked_inputs = []

    def _link_mic(self) -> None:
        ports = _ports(self.mic.name, "-o")
        if not ports:
            raise RuntimeError(f"Microphone has no capture ports: {self.mic.name}")
        # Multi-channel microphones use their first channel; summing channels would change the level.
        self._linked_inputs = ports[:1]
        with self._lock:
            self._reblocker.reset()
        _link(self._linked_inputs[0], f"{ENGINE_NAME}:in")

    def switch_mic(self, mic: Mic | None) -> None:
        """Changes the physical microphone; the virtual microphone stays connected to apps."""
        self._unlink_mic()
        self.mic = mic
        if self.active and mic is not None:
            self._link_mic()

    def set_active(self, on: bool) -> None:
        """Starts or stops listening to the physical microphone and processing."""
        if not on:
            self.active = False
            self._unlink_mic()
            return
        if not self.active and self.mic is not None:
            self._link_mic()
        self.active = True

    @property
    def monitor(self) -> bool:
        return bool(self._monitor_links)

    def set_monitor(self, on: bool) -> None:
        """Plays the virtual microphone through the default output device, for listening to the result."""
        for src, dst in self._monitor_links:
            _unlink(src, dst)
        self._monitor_links = []
        self.monitor_output = None
        if not on:
            return
        output = default_output()
        if output is None:
            raise RuntimeError("No default output device")
        playback = [p for p in _ports(output.name, "-i") if ":playback_" in p]
        if not playback:
            raise RuntimeError(f"Output device has no playback ports: {output.name}")
        for dst in playback:
            _link(f"{VIRTUAL_NAME}:capture_MONO", dst)
            self._monitor_links.append((f"{VIRTUAL_NAME}:capture_MONO", dst))
        self.monitor_output = output

    def stop(self) -> None:
        self.set_monitor(False)
        if self._client is not None:
            self._client.deactivate()
            self._client.close()
            self._client = None

    def _warm_up(self) -> None:
        """Runs a few blocks so first-call costs (allocation, compilation) happen before going live."""
        silence = np.zeros(BLOCK, dtype=np.float32)
        for _ in range(5):
            self.stage.process(silence)
        self.stage.reset()

    def _on_shutdown(self, status, reason: str) -> None:
        self.shutdown_reason = reason or "audio system stopped"

    def _on_xrun(self, delay_usecs: float) -> None:
        self.stats.xruns += 1

    def _process(self, frames: int) -> None:
        if not self.active:
            self._out.get_array()[:] = 0
            return
        t0 = time.perf_counter()
        inp = self._in.get_array()
        with self._lock:
            out = self._reblocker.push(inp)
        self._out.get_array()[:] = out
        self.input_ring.write(inp)
        self.output_ring.write(out)
        s = self.stats
        s.add((time.perf_counter() - t0) * 1000, frames)
        level = np.abs(inp)
        s.in_peak = max(s.in_peak, float(np.max(level)))
        s.in_clipped += int(np.count_nonzero(level >= CLIP_LEVEL))
        s.in_samples += frames
        s.out_peak = max(s.out_peak, float(np.max(np.abs(out))))
