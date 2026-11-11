"""Automatic artifact checks comparing a step's input and output.

Each check catches one kind of breakage that is audible as noise:
non-finite samples, clipping, unexpected loudness, boosted frequency bands,
artifacts repeating at every block edge, and dropouts.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

from npumic.engine import BLOCK, SAMPLE_RATE


@dataclass
class CheckResult:
    name: str
    value: float
    limit: float
    passed: bool
    description: str

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name:<16} {self.value:9.3f}  (limit {self.limit:g})  {self.description}"


@dataclass
class Limits:
    max_level_gain_db: float = 6.0
    max_band_gain_db: float = 6.0
    max_clip_fraction: float = 1e-4
    max_block_edge_score: float = 5.0
    max_dropout_pct: float = 2.0


def _rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def level_gain_db(inp: np.ndarray, out: np.ndarray) -> float:
    """Overall loudness change in dB."""
    return _rms_db(out) - _rms_db(inp)


def band_gain_db(inp: np.ndarray, out: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Largest loudness increase of any third-octave band between 50 Hz and 20 kHz.

    Bands quieter than 60 dB below the loudest input band are ignored.
    """
    f, p_in = signal.welch(inp, sr, nperseg=4096)
    _, p_out = signal.welch(out, sr, nperseg=4096)
    centers = 1000.0 * 2.0 ** (np.arange(-14, 14) / 3.0)
    centers = centers[(centers >= 50) & (centers <= 20000)]
    e_in, e_out = [], []
    for c in centers:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        if sel.any():
            e_in.append(p_in[sel].sum())
            e_out.append(p_out[sel].sum())
    e_in, e_out = np.array(e_in), np.array(e_out)
    floor = e_in.max() * 1e-6
    keep = e_in > floor
    if not keep.any():
        return 0.0
    return float(np.max(10 * np.log10((e_out[keep] + 1e-20) / (e_in[keep] + 1e-20))))


def _outlier_score(profile: np.ndarray) -> float:
    median = np.median(profile)
    spread = np.median(np.abs(profile - median)) * 1.4826
    return float((profile.max() - median) / spread) if spread > 0 else 0.0


INAUDIBLE_DBFS = -90.0


def _with_audibility_floor(x: np.ndarray) -> np.ndarray:
    """Adds fixed white noise at -90 dBFS so that inaudible residue does not dominate statistics."""
    floor = np.random.default_rng(0).standard_normal(len(x)) * 10 ** (INAUDIBLE_DBFS / 20)
    return x.astype(np.float64) + floor


def block_edge_score(x: np.ndarray, block: int = BLOCK) -> float:
    """How strongly one position within every block stands out (robust z-score).

    Natural audio has no preferred position, scoring about 2-4. Processing that
    misbehaves at block boundaries leaves a sharp change at the same position in
    every block, heard as a buzz at 100 Hz and its multiples.
    """
    d2 = np.abs(np.diff(_with_audibility_floor(x), n=2))
    n = len(d2) // block * block
    if n < block * 50:
        return 0.0
    blocks = d2[:n].reshape(-1, block)
    return max(
        _outlier_score(blocks.mean(axis=0)),
        _outlier_score(np.log(blocks + 1e-12).mean(axis=0)),
    )


def dropout_pct(inp: np.ndarray, out: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Percentage of 5 ms windows with the loudest input (speech peaks) that became silent.

    A window counts when its input peak is within 8 dB of the clip's loud level
    (99th percentile) and its output peak is below -100 dBFS.
    """
    win = sr // 200
    n = min(len(inp), len(out)) // win * win
    in_peak = np.abs(inp[:n]).reshape(-1, win).max(axis=1)
    out_peak = np.abs(out[:n]).reshape(-1, win).max(axis=1)
    loud = in_peak >= np.percentile(in_peak, 99) * 10 ** (-8 / 20)
    return float(100.0 * np.sum(loud & (out_peak < 1e-5)) / max(1, np.sum(loud)))


def run_checks(
    inp: np.ndarray, out: np.ndarray, limits: Limits | None = None, has_speech: bool = True
) -> list[CheckResult]:
    """Runs all checks. Set has_speech=False for noise-only clips, where full silencing is correct."""
    limits = limits or Limits()
    inp = np.asarray(inp, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)

    non_finite = int(np.sum(~np.isfinite(out)))
    results = [
        CheckResult("finite", non_finite, 0, non_finite == 0, "NaN/inf samples (cause loud bursts)"),
    ]
    if non_finite:
        return results

    clip = float(np.mean(np.abs(out) >= 0.999))
    edge_out = block_edge_score(out)
    edge_limit = max(limits.max_block_edge_score, block_edge_score(inp) + 2.0)
    level = level_gain_db(inp, out)
    band = band_gain_db(inp, out)
    drops = dropout_pct(inp, out) if has_speech else 0.0

    results += [
        CheckResult("clipping", clip, limits.max_clip_fraction, clip <= limits.max_clip_fraction,
                    "fraction of samples at full scale (harsh distortion)"),
        CheckResult("level_gain_db", level, limits.max_level_gain_db, level <= limits.max_level_gain_db,
                    "overall loudness increase"),
        CheckResult("band_gain_db", band, limits.max_band_gain_db, band <= limits.max_band_gain_db,
                    "largest boost of any frequency band (hum/hiss)"),
        CheckResult("block_edge", edge_out, edge_limit, edge_out <= edge_limit,
                    "artifacts repeating every 10 ms (buzz/clicks)"),
        CheckResult("dropouts_pct", drops, limits.max_dropout_pct, drops <= limits.max_dropout_pct,
                    "% of speech peaks turned into silence (cut-off words)"),
    ]
    return results


def all_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)
