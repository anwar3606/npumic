"""The clean virtual microphone as a long-running service.

Owns the processing chain, the choice of physical microphone, the saved
settings and the health report:

- Processing: automatic level around DPDFNet noise removal followed by VoxLock
  voice isolation (off until turned on). Both run on the NPU; a model that
  cannot load there, or fails while running, moves to the CPU (see on_npu).
- Idle until used: the physical microphone is linked and processed only while
  an app records from the virtual microphone. Otherwise the microphone, the
  NPU and Bluetooth headsets (which switch to their low-quality call profile
  while their microphone is open) are left alone.
- Microphone choice: the one picked by the user while it is connected, else
  USB, Bluetooth, built-in (devices.pick_mic). Checked whenever PipeWire ports
  or links change (`pw-link --monitor`) and every few seconds.
- Default input: the virtual microphone is made the system default on start;
  the previous default is restored on stop.
- Settings (processing switches, picked microphone) persist in data/settings.json.
"""

import json
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np

from npumic.devices import Mic, default_source, list_mics, pick_mic, set_default_source
from npumic.engine import BLOCK, SAMPLE_RATE, Chain, Fallback, Passthrough, Selector, Switchable
from npumic.health import Health
from npumic.live import VIRTUAL_NAME, LiveEngine, consumers, remove_virtual_mic
from npumic.micvolume import ClipGuard
from npumic.monitor import ResourceMonitor
from npumic.runtime import DATA_DIR
from npumic.stages.autolevel import AutoLevel
from npumic.stages.denoise import Denoise
from npumic.stages.isolation import VOICES_DIR, VoiceIsolation
from npumic.stages.voxlock import Voice, VoxLock

SETTINGS_PATH = DATA_DIR / "settings.json"
IDLE_AFTER_S = 3.0  # apps often close and reopen the microphone in quick succession
RECHECK_S = 5.0
SETTLE_S = 0.2
GLITCH_WINDOW_S = 60.0
GLITCH_LIMIT = 10
SUMMARY_EVERY_S = 300.0


@dataclass
class Settings:
    noise_removal: bool = True
    auto_level: bool = True
    isolation: bool = False
    mic: str | None = None  # node name of the microphone picked by the user; None chooses automatically
    previous_default: str | None = None  # default input before this app made itself the default

    @classmethod
    def load(cls, path) -> "Settings":
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)


def _brief(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:200]


def on_npu(make, title: str, key: str, health: Health) -> Fallback:
    """`make(device)` on the NPU, moved to the CPU if the NPU is missing or the model fails there."""
    try:
        stage = make("NPU")
    except Exception as exc:
        health.problem(key, f"{title} runs on the CPU: it failed to load on the NPU ({_brief(exc)})")
        stage = make("CPU")
    else:
        if stage.device != "NPU":
            health.problem(key, f"{title} runs on the CPU: no NPU found (is the NPU driver loaded?)")

    def failed(exc):
        health.problem(key, f"{title} failed on the {stage.device} ({_brief(exc)}); moving it to the CPU")

    def replaced(result):
        if isinstance(result, Exception):
            health.problem(key, f"{title} could not start on the CPU ({_brief(result)}); audio passes through unprocessed")
        else:
            health.problem(key, f"{title} runs on the CPU after it failed on the {stage.device}")

    return Fallback(stage, lambda: make("CPU"), failed, replaced)


def _device(stage: Fallback) -> str:
    return "none" if stage.failed else stage.inner.device


class Service:
    def __init__(self, settings_path=SETTINGS_PATH, make_default: bool = True, health: Health | None = None,
                 voices_dir=VOICES_DIR):
        self.health = health or Health()
        self.settings_path = settings_path
        self.settings = Settings.load(settings_path)
        self.make_default = make_default
        self.voice = Voice()
        self.denoise = Switchable(on_npu(Denoise, "Noise removal (DPDFNet)", "device-denoise", self.health),
                                  enabled=self.settings.noise_removal)
        self.voxlock = on_npu(lambda d: VoxLock(self.voice, d), "Voice isolation (VoxLock)", "device-voxlock", self.health)
        self.isolation = VoiceIsolation(Selector({"off": Passthrough(), "voxlock": self.voxlock}, "off", pad=False),
                                        self.voice, {"voxlock"}, voices_dir, on_recorded=self._recorded)
        self.level = AutoLevel(Chain([self.denoise, self.isolation]), enabled=self.settings.auto_level)
        self.engine = LiveEngine(self.level)
        self.guard = ClipGuard(self.engine, lambda: self.level.enabled,
                               lambda message: self.health.note(message, logging.WARNING))
        self.resources = ResourceMonitor(self.engine.stats)
        self.in_use: list[str] = []  # names of the apps recording from the virtual microphone
        self.started = time.time()
        self._resources_started = False
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._idle_since: float | None = None
        self._monitor_proc: subprocess.Popen | None = None
        self._glitches: deque[tuple[float, int]] = deque()
        self._summary_at = time.monotonic() + SUMMARY_EVERY_S
        self._warm(self.voxlock)

    @staticmethod
    def _warm(stage) -> None:
        silence = np.zeros(BLOCK, dtype=np.float32)
        for _ in range(5):
            stage.process(silence)
        stage.reset()

    # ---- lifecycle ---------------------------------------------------------------

    def start(self, mic_query: str | None = None) -> None:
        if mic_query:
            match = next((m for m in list_mics() if mic_query.lower() in (m.label + m.name).lower()), None)
            if match is None:
                raise SystemExit(f"No microphone matches '{mic_query}'.")
            self.settings.mic = match.name
            self._save()
        mics = list_mics()
        mic = pick_mic(mics, self.settings.mic)
        self.engine.start(mic, active=False)
        if mic is not None:
            self.isolation.use_mic(mic.name, mic.label)
            self.health.note(f"Microphone: {mic.label}")
        self._apply_isolation()
        self._report_mic(mics, mic)
        if self.make_default:
            self._take_default()
        self.guard.start()
        threading.Thread(target=self._follow_pipewire, name="npumic-pwmonitor", daemon=True).start()
        threading.Thread(target=self._watch, name="npumic-watch", daemon=True).start()
        self._wake.set()

    def stop(self, keep: bool = False) -> None:
        self._stop.set()
        self._wake.set()
        if self._monitor_proc is not None:
            self._monitor_proc.terminate()
        self.guard.stop()
        self.resources.stop()
        self.engine.stop()
        if self.make_default:
            self._restore_default()
        if not keep:
            remove_virtual_mic()
        self.health.note("Stopped")
        self.health.flush()

    def want_resources(self) -> None:
        """Starts resource sampling the first time the control page asks for it."""
        if not self._resources_started:
            self._resources_started = True
            self.resources.start()

    def tick(self) -> None:
        """Called about once a second from the main thread: glitch tracking and logging."""
        s = self.engine.stats
        now = time.monotonic()
        total = s.late_cycles + s.xruns
        self._glitches.append((now, total))
        while self._glitches[0][0] < now - GLITCH_WINDOW_S:
            self._glitches.popleft()
        recent = total - self._glitches[0][1]
        if recent >= GLITCH_LIMIT:
            self.health.problem("glitches", f"Audio glitches: {GLITCH_LIMIT} or more in the last minute "
                                            "(processing too slow, or the system was busy)")
        elif recent == 0:
            self.health.clear("glitches", "No audio glitches in the last minute")
        if now >= self._summary_at:
            self._summary_at = now + SUMMARY_EVERY_S
            if self.engine.active:
                logging.getLogger("npumic").info(
                    "Processing %.2f ms average, %.2f ms max per %.0f ms cycle; %d late cycles, %d glitches since start",
                    s.avg_ms, s.max_ms, 1000 * s.cycle_frames / SAMPLE_RATE, s.late_cycles, s.xruns)
        self.health.flush()

    # ---- default input -----------------------------------------------------------

    def _take_default(self) -> None:
        try:
            current = default_source()
            if current and current != VIRTUAL_NAME:
                self.settings.previous_default = current
                self._save()
            if set_default_source(VIRTUAL_NAME):
                self.health.note("NPU Denoised Mic is the default microphone")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.health.problem("default", f"Could not make NPU Denoised Mic the default microphone ({_brief(exc)})")

    def _restore_default(self) -> None:
        previous = self.settings.previous_default
        try:
            if previous and default_source() == VIRTUAL_NAME and set_default_source(previous):
                self.health.note(f"Default microphone restored: {previous}")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.health.note(f"Could not restore the default microphone ({_brief(exc)})", logging.WARNING)

    # ---- microphone and activity -------------------------------------------------

    def _follow_pipewire(self) -> None:
        """Wakes the watcher whenever PipeWire output ports or links change."""
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(["pw-link", "--monitor", "--output", "--links"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except OSError as exc:
                self.health.problem("pwmonitor", f"Cannot follow audio device changes ({_brief(exc)})")
                self._stop.wait(RECHECK_S)
                continue
            self._monitor_proc = proc
            for _ in proc.stdout:
                self._wake.set()
            proc.wait()
            self._stop.wait(1.0)

    def _watch(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(RECHECK_S if self._idle_since is None else 0.5)
            if self._stop.is_set():
                break
            time.sleep(SETTLE_S)  # changes come in bursts
            self._wake.clear()
            try:
                self.check()
            except Exception as exc:
                self.health.problem("watch", f"Could not read the audio system state ({_brief(exc)})")
            else:
                self.health.clear("watch")

    def check(self) -> None:
        """Picks the microphone and starts or pauses processing depending on whether an app uses it."""
        with self._lock:
            mics = list_mics()
            mic = pick_mic(mics, self.settings.mic)
            current = self.engine.mic
            if (mic and mic.name) != (current and current.name):
                self._use_mic(mic)
            elif mic is not None:
                self.engine.mic = mic  # keeps label and mute state current
            self._report_mic(mics, mic)

            monitor = self.engine.monitor_output
            apps = ["Hear myself" if monitor and name == monitor.name else name for name in consumers()]
            if apps and apps != self.in_use:
                self.health.note(f"Microphone used by: {', '.join(apps)}")
            self.in_use = apps
            wanted = bool(apps) or self.isolation.record_progress is not None
            if wanted:
                self._idle_since = None
                if not self.engine.active:
                    self.engine.set_active(True)
                    self.health.note("An app is using the microphone: processing started")
            elif self.engine.active:
                now = time.monotonic()
                if self._idle_since is None:
                    self._idle_since = now
                elif now - self._idle_since >= IDLE_AFTER_S:
                    self._idle_since = None
                    self.engine.set_active(False)
                    self.health.note("No app is using the microphone: processing paused")

    def _use_mic(self, mic: Mic | None) -> None:
        self.engine.switch_mic(mic)
        if mic is not None:
            self.isolation.use_mic(mic.name, mic.label)
            self._apply_isolation()
            self.health.note(f"Microphone: {mic.label}")

    def _report_mic(self, mics: list[Mic], mic: Mic | None) -> None:
        if mic is None:
            self.health.problem("mic", "No microphone is connected")
        elif mic.muted:
            self.health.problem("mic", f"{mic.label} is muted in the system sound settings, so apps hear silence")
        else:
            self.health.clear("mic")

    # ---- settings ----------------------------------------------------------------

    def _save(self) -> None:
        try:
            self.settings.save(self.settings_path)
        except OSError as exc:
            self.health.note(f"Could not save settings ({_brief(exc)})", logging.WARNING)

    def _apply_isolation(self) -> None:
        if self.settings.isolation and not self.voice.ready:
            self.isolation.selector.selected = "off"
            self.health.problem("voice", "Removing other voices is on, but no voice sample is recorded yet")
        else:
            self.isolation.selector.selected = "voxlock" if self.settings.isolation else "off"
            self.health.clear("voice")

    def _recorded(self, error: str | None) -> None:
        if error:
            self.health.note(f"Voice sample not recorded: {error}", logging.WARNING)
        else:
            self.health.note(f"Voice sample recorded with {self.isolation.voice_mic}")
            with self._lock:
                self._apply_isolation()
        self._wake.set()

    def apply(self, changes: dict) -> None:
        """Applies settings from the control page or the GNOME extension; ValueError for invalid ones."""
        with self._lock:
            if "noise_removal" in changes:
                self.denoise.enabled = self.settings.noise_removal = bool(changes["noise_removal"])
            if "auto_level" in changes:
                self.level.enabled = self.settings.auto_level = bool(changes["auto_level"])
            if "isolation" in changes:
                on = bool(changes["isolation"])
                if on and not self.voice.ready:
                    raise ValueError("Record your voice sample first")
                self.settings.isolation = on
                self._apply_isolation()
            if changes.get("record_voice"):
                if self.engine.mic is None:
                    raise ValueError("No microphone is connected")
                self.isolation.record()
                self.check()
            if "mic" in changes:
                name = changes["mic"] or None
                if name is not None and not any(m.name == name for m in list_mics()):
                    raise ValueError("That microphone is not connected")
                self.settings.mic = name
                self.check()
            if "monitor" in changes:
                self.engine.set_monitor(bool(changes["monitor"]))
                self.check()
            self._save()

    # ---- status ------------------------------------------------------------------

    def status(self) -> dict:
        e, s, iso = self.engine, self.engine.stats, self.isolation
        return {
            "state": "active" if e.active else "idle",
            "apps": self.in_use,
            "problems": list(self.health.problems.values()),
            "events": list(self.health.events)[-30:],
            "mic": e.mic.name if e.mic else None,
            "mic_label": e.mic.label if e.mic else None,
            "mic_choice": self.settings.mic,
            "noise_removal": self.denoise.enabled,
            "noise_removal_device": _device(self.denoise.inner),
            "auto_level": self.level.enabled,
            "gain_db": round(self.level.gain_db, 1),
            "speech_db": round(self.level.output_speech_db, 1) if self.level.output_speech_db is not None else None,
            "clip_action": self.guard.last_action,
            "isolation": self.settings.isolation,
            "isolation_running": iso.selector.active == "voxlock",
            "isolation_device": _device(self.voxlock),
            "voice": {
                "ready": self.voice.ready,
                "mic": iso.voice_mic,
                "recorded": iso.voice_recorded,
                "for_this_mic": iso.voice_for_this_mic,
                "record_progress": iso.record_progress,
                "record_error": iso.record_error,
            },
            "monitor": e.monitor,
            "monitor_output": e.monitor_output.label if e.monitor_output else None,
            "latency_ms": round(1000 * e.latency / SAMPLE_RATE),
            "cycle_ms": round(1000 * s.cycle_frames / SAMPLE_RATE, 1),
            "avg_ms": round(s.avg_ms, 2),
            "max_ms": round(s.max_ms, 2),
            "late_cycles": s.late_cycles,
            "xruns": s.xruns,
            "started": round(self.started),
        }
