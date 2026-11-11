"""Synthetic test signals for automated tests (no recordings required)."""

import numpy as np

from npumic.engine import SAMPLE_RATE


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spectrum))
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0
    x = np.fft.irfft(spectrum, n)
    return (x / np.std(x)).astype(np.float32)


def speech_like(seconds: float = 10.0, seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Voiced harmonics with gliding pitch, broken into syllables, with fricative-like noise bursts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t) + 15 * np.sin(2 * np.pi * 2.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 30) if k * 250 < sr / 2)
    syllables = np.clip(np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6)), 0, None) ** 2
    pauses = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(float)
    fricative = rng.standard_normal(n) * (np.sin(2 * np.pi * 1.3 * t) > 0.8) * 0.15
    x = (voiced * syllables + fricative) * pauses
    return (0.1 * x / np.max(np.abs(x))).astype(np.float32)


def noisy_speech(seconds: float = 10.0, snr_db: float = 10.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (clean, noisy) with pink background noise at the given speech-to-noise ratio."""
    rng = np.random.default_rng(seed)
    clean = speech_like(seconds, seed)
    noise = pink_noise(len(clean), rng)
    gain = np.sqrt(np.mean(clean**2) / np.mean(noise**2) / 10 ** (snr_db / 10))
    return clean, (clean + gain * noise).astype(np.float32)
