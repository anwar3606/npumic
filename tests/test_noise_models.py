import numpy as np
import pytest

from support.checks import run_checks
from support.signals import noisy_speech
from npumic.engine import BLOCK, Reblocker, process_offline
from npumic.runtime import has_npu
from npumic.stages.denoise import HOP, Denoise, vorbis_window

_NO_NPU = pytest.mark.skipif(not has_npu(), reason="no NPU")
MODELS = [
    pytest.param(Denoise, "CPU", id="dpdfnet-CPU"),
    pytest.param(Denoise, "NPU", id="dpdfnet-NPU", marks=_NO_NPU),
]


@pytest.fixture(scope="module")
def audio():
    return noisy_speech(8.0, snr_db=10)[1]


def test_vorbis_window_reconstructs_exactly():
    w = vorbis_window(2 * HOP)
    np.testing.assert_allclose(w[:HOP] ** 2 + w[HOP:] ** 2, 1.0, atol=1e-6)


def test_initial_state_from_metadata_matches_reference():
    """The state built from ONNX metadata equals the reference initial state."""
    from npumic.runtime import MODELS_DIR
    from npumic.stages.denoise import initial_state, load_model

    reference = MODELS_DIR / "dpdfnet" / "initial_state.npy"
    if not reference.exists():
        pytest.skip("reference initial state not downloaded")
    np.testing.assert_array_equal(initial_state(load_model()), np.load(reference))


@pytest.mark.parametrize(("model", "device"), MODELS)
def test_output_is_time_aligned(model, device, audio):
    """The declared latency must match the real delay to the sample."""
    out = process_offline(model(device), audio)
    s = slice(4800, -4800)
    lags = range(-3 * HOP, 3 * HOP + 1)
    best = max(lags, key=lambda lag: np.dot(audio[s], np.roll(out, -lag)[s]))
    assert best == 0


@pytest.mark.parametrize(("model", "device"), MODELS)
def test_passes_artifact_checks(model, device, audio):
    out = process_offline(model(device), audio)
    failed = [r.name for r in run_checks(audio, out) if not r.passed]
    assert failed == []


@pytest.mark.parametrize(("model", "device"), MODELS)
def test_live_chunking_matches_offline(model, device, audio):
    reference = process_offline(model(device), audio)
    rb = Reblocker(model(device))
    rng = np.random.default_rng(0)
    pieces, pos = [], 0
    while pos < len(audio):
        size = int(rng.integers(64, 2 * BLOCK))
        pieces.append(rb.push(audio[pos : pos + size]))
        pos += size
    live = np.concatenate(pieces)
    n = (len(audio) - rb.latency) // BLOCK * BLOCK
    np.testing.assert_allclose(live[rb.latency : rb.latency + n], reference[:n], atol=1e-6)


@pytest.mark.parametrize(("model", "device"), MODELS)
def test_reset_restores_initial_behavior(model, device, audio):
    stage = model(device)
    first = process_offline(stage, audio[: 48000])
    process_offline(stage, audio)
    again = process_offline(stage, audio[: 48000])
    np.testing.assert_allclose(first, again, atol=1e-6)


@pytest.mark.parametrize(("model", "device"), MODELS)
def test_quiet_microphone_keeps_speech(model, device):
    """Speech peaks near -26 dBFS (a quiet headset mic) must not be cut.

    Much quieter input is cut by the model on any device; that range is handled
    by automatic input level adjustment before this step.
    """
    from support.checks import dropout_pct

    clean, noisy = noisy_speech(8.0, snr_db=15)
    quiet = noisy * np.float32(10 ** (-6 / 20))
    out = process_offline(model(device), quiet)
    assert dropout_pct(quiet, out) <= 2.0


@_NO_NPU
def test_dpdfnet_npu_matches_cpu(audio):
    """Features are computed in f32 outside the graph, so f16 on the NPU no longer drops high frequencies."""
    from scipy import signal

    def band_levels(x):
        f, p = signal.welch(x, 48000, nperseg=4096)
        edges = [0, 1000, 3000, 6000, 8000, 12000, 24000]
        return np.array([10 * np.log10(p[(f >= lo) & (f < hi)].sum()) for lo, hi in zip(edges, edges[1:])])

    cpu = process_offline(Denoise("CPU"), audio)
    npu = process_offline(Denoise("NPU"), audio)
    assert np.abs(band_levels(npu) - band_levels(cpu)).max() < 2.0
