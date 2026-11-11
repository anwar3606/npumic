"""Automatic microphone level around the noise-removal step.

A single gain is applied before noise removal and kept: the model sees speech
at a level it handles well, and apps receive speech at a steady level whatever
the microphone.

The speech level is measured on the output of the wrapped step, i.e. after
noise removal, so background noise does not count as speech and is never
raised. Each output block is referred back to the input by subtracting the gain
that was applied when that audio entered the step (`latency` earlier), which
keeps the control loop free of the step's delay.

- Speech blocks: input-referred output level at least 12 dB above a tracked
  noise floor (follows dips at once, rises 10 dB/s) and above -70 dBFS. Only speech blocks move the gain toward the target, so the
  gain holds still in pauses.
- Speech level: mean power of speech blocks, smoothed over about 2 s of speech
  (0.5 s during the first 3 s of speech after a reset, to settle quickly).
- Gain moves at most +3 dB/s (+10 dB/s while settling) and -6 dB/s, between
  MIN_GAIN_DB and MAX_GAIN_DB, ramped linearly within each block.
- Peak guard: the gain never exceeds the value that would put the loudest
  input-referred output peak of the last 1.5 s at PEAK_LIMIT_DBFS; this limit
  applies immediately.
"""

from collections import deque

import numpy as np

from npumic.engine import BLOCK, SAMPLE_RATE, Stage

TARGET_DBFS = -24.0
MIN_GAIN_DB = -12.0
MAX_GAIN_DB = 30.0
PEAK_LIMIT_DBFS = -3.0
SPEECH_ABOVE_FLOOR_DB = 12.0
SPEECH_MIN_DBFS = -70.0
BLOCKS_PER_S = SAMPLE_RATE / BLOCK
UP_DB_PER_BLOCK = 3.0 / BLOCKS_PER_S
FAST_UP_DB_PER_BLOCK = 10.0 / BLOCKS_PER_S
DOWN_DB_PER_BLOCK = 6.0 / BLOCKS_PER_S
SETTLE_SPEECH_BLOCKS = int(3 * BLOCKS_PER_S)
LEVEL_SMOOTHING_BLOCKS = 2 * BLOCKS_PER_S
FAST_LEVEL_SMOOTHING_BLOCKS = 0.5 * BLOCKS_PER_S
PEAK_HOLD_BLOCKS = int(1.5 * BLOCKS_PER_S)
FLOOR_RISE_DB_PER_BLOCK = 10.0 / BLOCKS_PER_S
FLOOR_MIN_DBFS = -100.0
MAX_DELAY_BLOCKS = 64


def _db(power: float) -> float:
    return 10 * np.log10(power + 1e-20)


class AutoLevel(Stage):
    name = "autolevel"

    def __init__(self, inner: Stage, enabled: bool = True, target_dbfs: float = TARGET_DBFS):
        self.inner = inner
        self.enabled = enabled
        self.target_dbfs = target_dbfs
        self.reset()

    @property
    def latency(self) -> int:
        return self.inner.latency

    def reset(self) -> None:
        self.inner.reset()
        self.gain_db = 0.0
        self.speech_db: float | None = None  # input-referred speech level
        self._floor_db = FLOOR_MIN_DBFS
        self._speech_blocks = 0
        # Gains of recent blocks, newest last; the wrapped step's latency may change while running.
        self._applied = deque([0.0] * MAX_DELAY_BLOCKS, maxlen=MAX_DELAY_BLOCKS)
        self._peaks = deque(maxlen=PEAK_HOLD_BLOCKS)

    @property
    def output_speech_db(self) -> float | None:
        """Current speech level as apps receive it."""
        return None if self.speech_db is None else self.speech_db + self.gain_db

    def _measure(self, out: np.ndarray, gain_then_db: float) -> bool:
        """Updates level statistics from one output block; returns whether it is speech."""
        level_in = _db(float(np.mean(np.square(out, dtype=np.float64)))) - gain_then_db
        peak = 20 * np.log10(float(np.max(np.abs(out))) + 1e-10)
        self._peaks.append(peak - gain_then_db)
        self._floor_db = max(FLOOR_MIN_DBFS, min(level_in, self._floor_db + FLOOR_RISE_DB_PER_BLOCK))
        if level_in < max(SPEECH_MIN_DBFS, self._floor_db + SPEECH_ABOVE_FLOOR_DB):
            return False
        self._speech_blocks += 1
        settling = self._speech_blocks <= SETTLE_SPEECH_BLOCKS
        if self.speech_db is None:
            self.speech_db = level_in
        else:
            n = FAST_LEVEL_SMOOTHING_BLOCKS if settling else LEVEL_SMOOTHING_BLOCKS
            power = 10 ** (self.speech_db / 10)
            power += (10 ** (level_in / 10) - power) / n
            self.speech_db = _db(power)
        return True

    def _next_gain(self, speech: bool) -> float:
        gain = self.gain_db
        if not self.enabled:
            wanted = 0.0
        elif speech and self.speech_db is not None:
            wanted = float(np.clip(self.target_dbfs - self.speech_db, MIN_GAIN_DB, MAX_GAIN_DB))
        else:
            wanted = gain
        up = FAST_UP_DB_PER_BLOCK if self._speech_blocks <= SETTLE_SPEECH_BLOCKS else UP_DB_PER_BLOCK
        if not self.enabled:
            up = FAST_UP_DB_PER_BLOCK
        gain = min(wanted, gain + up) if wanted > gain else max(wanted, gain - DOWN_DB_PER_BLOCK)
        if self.enabled and self._peaks:
            gain = min(gain, PEAK_LIMIT_DBFS - max(self._peaks))
        return gain

    def process_block(self, block: np.ndarray) -> np.ndarray:
        start = 10 ** (self._applied[-1] / 20)
        end = 10 ** (self.gain_db / 20)
        ramp = start + (end - start) * np.arange(1, len(block) + 1, dtype=np.float32) / len(block)
        self._applied.append(self.gain_db)
        out = self.inner.process((block * ramp).astype(np.float32))
        delay = min(round(self.inner.latency / BLOCK), MAX_DELAY_BLOCKS - 1)
        speech = self._measure(out, self._applied[-1 - delay])
        self.gain_db = self._next_gain(speech)
        return out
