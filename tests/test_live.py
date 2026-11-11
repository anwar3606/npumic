"""End-to-end test through PipeWire: a synthetic voice is played into a temporary
test microphone, processed live, and recorded from the virtual microphone.

Requires a running PipeWire session. Run with: uv run pytest -m live
"""

import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
from scipy import signal

from support.signals import noisy_speech
from npumic.devices import Mic
from npumic.engine import SAMPLE_RATE, Passthrough, Stage, process_offline
from npumic.live import (
    VIRTUAL_NAME, LiveEngine, _link, _load_jack, _node_ids, _pw_props, _wait_for_port, remove_virtual_mic,
)
from npumic.stages.denoise import Denoise

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(shutil.which("pw-cli") is None, reason="PipeWire not available"),
]

TEST_SOURCE = "npumictest_source"
TEST_FEED = "npumictest_feed"
STREAM_ARGS = ["-a", "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "f32"]


@pytest.fixture
def test_source():
    if _node_ids(VIRTUAL_NAME):
        pytest.skip("npumic is already running; stop it before running live tests")
    subprocess.run(["pw-cli", "create-node", "adapter", _pw_props({
        "factory.name": "support.null-audio-sink",
        "node.name": TEST_SOURCE,
        "media.class": "Audio/Source/Virtual",
        "audio.rate": SAMPLE_RATE,
        "audio.position": ["MONO"],
        "object.linger": True,
    })], capture_output=True, check=True)
    yield Mic(name=TEST_SOURCE, label="test source", bluetooth=False, is_default=False)
    for node_id in _node_ids(TEST_SOURCE):
        subprocess.run(["pw-cli", "destroy", str(node_id)], capture_output=True)
    remove_virtual_mic()


def play_into_test_source(audio: np.ndarray) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["pw-play", *STREAM_ARGS, "-P", _pw_props({"node.name": TEST_FEED, "node.autoconnect": False, "state.restore-props": False}), "-"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _wait_for_port(f"{TEST_FEED}:output_MONO")
    subprocess.run(["pw-link", f"{TEST_FEED}:output_MONO", f"{TEST_SOURCE}:input_MONO"], check=True)
    threading.Thread(target=lambda: (proc.stdin.write(audio.tobytes()), proc.stdin.close()), daemon=True).start()
    return proc


class VirtualMicRecorder:
    """Records the virtual microphone in memory from a real-time callback, as an app would."""

    NAME = "npumictest_recorder"

    def __init__(self):
        jack = _load_jack()
        self.blocks: list[np.ndarray] = []
        self.client = jack.Client(self.NAME, no_start_server=True)
        self.port = self.client.inports.register("in")
        self.client.set_process_callback(lambda frames: self.blocks.append(self.port.get_array().copy()))
        self.client.activate()
        _wait_for_port(f"{self.NAME}:in")
        _link(f"{VIRTUAL_NAME}:capture_MONO", f"{self.NAME}:in")

    def stop(self) -> np.ndarray:
        self.client.deactivate()
        self.client.close()
        return np.concatenate(self.blocks)


def run_live(stage: Stage, mic: Mic, audio: np.ndarray) -> tuple[np.ndarray, LiveEngine]:
    """Plays audio into the test microphone and returns what apps receive from the virtual microphone."""
    engine = LiveEngine(stage)
    engine.start(mic)
    recorder = VirtualMicRecorder()
    time.sleep(0.5)
    player = play_into_test_source(audio)
    try:
        player.wait(timeout=20)
        time.sleep(1.0)
    finally:
        got = recorder.stop()
        engine.stop()
    return got, engine


def align(got: np.ndarray, expected: np.ndarray) -> np.ndarray:
    lag = int(np.argmax(signal.correlate(got, expected, mode="valid", method="fft")))
    return got[lag : lag + len(expected)]


def test_live_passthrough_is_exact(test_source):
    """The audio path itself must not change the sound: no volume change, no drops, no resampling."""
    _, voice = noisy_speech(5.0, snr_db=10)
    got, engine = run_live(Passthrough(), test_source, voice)
    np.testing.assert_allclose(align(got, voice), voice, atol=1e-6)
    assert engine.stats.xruns == 0


class Tap(Stage):
    """Wraps a stage and keeps a copy of every block it receives and returns."""

    name = "tap"

    def __init__(self, inner: Stage):
        self.inner, self.inputs, self.outputs = inner, [], []
        self.latency = inner.latency

    def reset(self):
        self.inner.reset()
        self.inputs.clear()
        self.outputs.clear()

    def process_block(self, block):
        out = self.inner.process(block)
        self.inputs.append(block.copy())
        self.outputs.append(out.copy())
        return out


def test_live_processing_matches_offline(test_source):
    """Live noise removal must produce exactly what offline processing of the same input produces,
    and apps must receive exactly that output."""
    _, voice = noisy_speech(8.0, snr_db=10)
    tap = Tap(Denoise("CPU"))
    got, engine = run_live(tap, test_source, voice)
    assert engine.stats.xruns == 0

    received = np.concatenate(tap.inputs)
    produced = np.concatenate(tap.outputs)
    expected = process_offline(Denoise("CPU"), received)
    lat = tap.latency
    np.testing.assert_allclose(produced[lat:], expected[: len(produced) - lat], atol=1e-5)

    middle = produced[len(produced) // 3 : 2 * len(produced) // 3]
    np.testing.assert_allclose(align(got, middle), middle, atol=1e-6)
