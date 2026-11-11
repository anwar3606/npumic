import json
import time

import numpy as np
import pytest

from npumic.engine import BLOCK, SAMPLE_RATE, Passthrough, Selector, process_offline
from npumic.runtime import MODELS_DIR, has_npu
from npumic.stages.isolation import RECORD_SECONDS, VoiceIsolation
from npumic.stages.voxlock import FRAME_HOP, MODEL_DIR, N_FFT, SAMPLE_SECONDS, Voice, VoxLock, loudest

SPEECH_SAMPLE = MODELS_DIR / "voxlock" / "assets" / "speech_raw.wav"


@pytest.fixture(scope="module")
def speech_sample():
    """A noisy speech signal at 16 kHz."""
    import soundfile as sf

    if SPEECH_SAMPLE.exists():
        return sf.read(SPEECH_SAMPLE, dtype="float32")[0]
    return _speechlike(5.0, 16000, seed=1)


def _speechlike(seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """Harmonic tone with a syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    x = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    return (0.1 * x * envelope + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


# ---- VoxLock ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def voice():
    v = Voice()
    v.set(_speechlike(SAMPLE_SECONDS, 16000, seed=2))
    return v


def test_loudest_picks_the_loud_part():
    x = np.zeros(16000 * 8, np.float32)
    x[16000 * 4 : 16000 * 7] = 0.5
    picked = loudest(x)
    assert len(picked) == SAMPLE_SECONDS * 16000 and np.all(picked == 0.5)


def test_voice_rejects_wrong_length():
    with pytest.raises(ValueError):
        Voice().set(np.zeros(16000, np.float32))


def test_voxlock_without_voice_outputs_silence():
    stage = VoxLock(Voice(), "CPU")
    y = process_offline(stage, _speechlike(1.0, SAMPLE_RATE))
    assert np.all(y == 0)


def test_voxlock_stream_matches_direct_model_calls(speech_sample):
    """The stage's buffering, pipelining and overlap-add reproduce direct calls of k frames each.

    VoxLock's output depends on how audio is split into calls (one call over a
    whole file differs from calls of k frames), so the reference uses the same split.
    """
    from npumic.runtime import core
    from npumic.stages.voxlock import CPU_THREADS, STEP_FRAMES

    x = speech_sample[: 16000 * 4]
    voice = Voice()
    voice.set(loudest(x))
    stage = VoxLock(voice, "CPU")
    stage.reset()
    out = np.concatenate([stage.step(x[i : i + stage.hop]) for i in range(0, len(x) - stage.hop + 1, stage.hop)])

    k = STEP_FRAMES["CPU"]
    segment = (k - 1) * FRAME_HOP + N_FFT
    model = core().read_model(str(MODEL_DIR / "chunk.onnx"))
    model.reshape({"wav": [1, segment], "enroll_mag": list(voice.mag.shape), "context": list(voice.context.shape),
                   "h": [6, 32, 256], "c": [6, 32, 256]})
    request = core().compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": CPU_THREADS}).create_infer_request()
    padded = np.concatenate([np.zeros(segment - stage.hop, np.float32), x])
    h = np.zeros((6, 32, 256), np.float32)
    c = h.copy()
    ola = np.zeros(len(padded) + N_FFT, np.float32)
    calls = (len(padded) - segment) // stage.hop + 1
    for i in range(calls):
        r = request.infer({"wav": padded[None, i * stage.hop : i * stage.hop + segment], "enroll_mag": voice.mag,
                           "context": voice.context, "h": h, "c": c})
        h, c = r["h_out"].copy(), r["c_out"].copy()
        for j in range(k):
            start = i * stage.hop + j * FRAME_HOP
            ola[start : start + N_FFT] += r["frames"][0, j]
    reference = ola * (-0.5 / 1.5)
    # One pipelined call precedes the stage's output.
    n = (calls - 1) * stage.hop
    a, b = out[stage.hop : stage.hop + n], reference[:n]
    assert 10 * np.log10(np.sum(b ** 2) / (np.sum((a - b) ** 2) + 1e-20)) > 60


def test_voxlock_latency_formula_matches_stage(voice):
    assert VoxLock(voice, "CPU").latency == VoxLock.latency_on("CPU")


@pytest.mark.skipif(not has_npu(), reason="no NPU")
def test_voxlock_on_npu_has_its_own_latency(voice):
    assert VoxLock(voice, "NPU").latency == VoxLock.latency_on("NPU") == 4632


# ---- voice sample recording -----------------------------------------------------


class _FakeVoice:
    def __init__(self):
        self.samples = []

    @property
    def ready(self):
        return bool(self.samples)

    def set(self, sample):
        assert len(sample) == SAMPLE_SECONDS * 16000
        self.samples.append(sample)


def _isolation(tmp_path):
    return VoiceIsolation(Selector({"off": Passthrough()}, "off", pad=False), _FakeVoice(), {"voxlock"}, tmp_path)


def _wait(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


def test_recording_saves_a_sample_per_microphone(tmp_path):
    iso = _isolation(tmp_path)
    iso.use_mic("mic-a", "Mic A")
    assert not iso.voice.ready
    iso.record()
    audio = _speechlike(RECORD_SECONDS + 1, SAMPLE_RATE)
    for i in range(0, len(audio) - BLOCK, BLOCK):
        iso.process(audio[i : i + BLOCK])
    _wait(lambda: iso.record_progress is None)
    assert iso.record_error is None and iso.voice.ready
    assert (tmp_path / "mic-a.wav").exists()
    assert json.loads((tmp_path / "mic-a.json").read_text())["mic"] == "Mic A"
    assert iso.voice_mic == "Mic A" and iso.voice_for_this_mic

    iso.use_mic("mic-b", "Mic B")  # no own sample: keeps Mic A's
    assert iso.voice.ready and not iso.voice_for_this_mic and iso.voice_mic == "Mic A"


def test_recording_silence_is_rejected(tmp_path):
    iso = _isolation(tmp_path)
    iso.use_mic("mic-a", "Mic A")
    iso.record()
    silence = np.zeros(BLOCK, np.float32)
    for _ in range(RECORD_SECONDS * SAMPLE_RATE // BLOCK + 1):
        iso.process(silence)
    _wait(lambda: iso.record_progress is None)
    assert iso.record_error and not iso.voice.ready


def test_saved_sample_is_loaded_at_start(tmp_path):
    import soundfile as sf

    sf.write(tmp_path / "other.wav", _speechlike(SAMPLE_SECONDS, 16000), 16000)
    iso = _isolation(tmp_path)
    iso.use_mic("mic-a", "Mic A")
    assert iso.voice.ready and not iso.voice_for_this_mic
