import numpy as np
import pytest

from support.checks import run_checks
from support.signals import noisy_speech, pink_noise, speech_like
from npumic.engine import BLOCK, Delay, Passthrough, process_offline
from npumic.stages.autolevel import PEAK_LIMIT_DBFS, TARGET_DBFS, AutoLevel


def scaled(x: np.ndarray, db: float) -> np.ndarray:
    return (x * np.float32(10 ** (db / 20))).astype(np.float32)


def speech_level_db(out: np.ndarray, reference: np.ndarray) -> float:
    """Mean power of the output where the reference has speech (blocks within 30 dB of its loudest)."""
    n = len(reference) // BLOCK * BLOCK
    ref = np.mean(reference[:n].reshape(-1, BLOCK) ** 2, axis=1)
    power = np.mean(out[:n].reshape(-1, BLOCK) ** 2, axis=1)
    speech = ref > ref.max() * 1e-3
    return float(10 * np.log10(power[speech].mean()))


@pytest.mark.parametrize("level_db", [-15.0, 0.0, 12.0])
@pytest.mark.parametrize("inner", [Passthrough, lambda: Delay(2880)], ids=["passthrough", "delayed"])
def test_converges_to_target(level_db, inner):
    """Quiet and loud speech end up at the target level after a few seconds."""
    speech = scaled(speech_like(20.0), level_db)
    stage = AutoLevel(inner())
    out = process_offline(stage, speech)
    tail = slice(-10 * 48000, None)
    assert abs(speech_level_db(out[tail], speech[tail]) - TARGET_DBFS) < 2.5
    assert abs(stage.output_speech_db - TARGET_DBFS) < 1.5


def test_gain_holds_in_pauses():
    """Background noise alone after speech must not be raised."""
    rng = np.random.default_rng(1)
    speech = scaled(speech_like(10.0), -20)
    noise = pink_noise(10 * 48000, rng) * np.float32(10 ** (-75 / 20))
    stage = AutoLevel(Passthrough())
    process_offline(stage, speech)
    gain_after_speech = stage.gain_db
    for i in range(0, len(noise), BLOCK):
        stage.process(noise[i : i + BLOCK])
    assert stage.gain_db <= gain_after_speech + 0.1


def test_peaks_stay_below_limit():
    """Loud bursts reduce the gain at once; after settling, peaks stay under the limit."""
    speech = scaled(speech_like(20.0), -30)
    burst = speech.copy()
    burst[12 * 48000 : 14 * 48000] *= 30
    out = process_offline(AutoLevel(Passthrough()), burst)
    after = out[int(12.3 * 48000) :]
    assert 20 * np.log10(np.abs(after).max()) <= PEAK_LIMIT_DBFS + 1.0


def test_disabled_passes_audio_unchanged():
    speech = scaled(speech_like(5.0), -30)
    out = process_offline(AutoLevel(Passthrough(), enabled=False), speech)
    np.testing.assert_array_equal(out, speech)


def test_passes_artifact_checks():
    _, noisy = noisy_speech(10.0, snr_db=20)
    quiet = scaled(noisy, -20)
    out = process_offline(AutoLevel(Passthrough()), quiet)
    # Compared with the input raised to the same overall level, so only artifacts count.
    reference = (quiet * np.sqrt(np.mean(out**2) / np.mean(quiet**2))).astype(np.float32)
    failed = [r.name for r in run_checks(reference, out) if not r.passed]
    assert failed == []


def test_latency_matches_inner():
    assert AutoLevel(Delay(2880)).latency == 2880


def test_follows_a_change_of_the_inner_latency():
    from npumic.engine import Delay, Passthrough, Selector

    sel = Selector({"off": Passthrough(), "slow": Delay(2880)}, "off", warmup_blocks=1, fade_blocks=1, pad=False)
    stage = AutoLevel(sel)
    assert stage.latency == 0
    sel.selected = "slow"
    speech = noisy_speech(2.0, snr_db=30)[1] * 0.01
    out = process_offline(stage, speech)
    assert stage.latency == 2880 and np.all(np.isfinite(out))
