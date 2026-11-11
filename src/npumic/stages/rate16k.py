"""Running 16 kHz models inside the 48 kHz, 10 ms block pipeline.

The signal is resampled 48 kHz -> 16 kHz -> 48 kHz around the model with
streaming linear-phase FIR filters; output bandwidth is limited to about
7.5 kHz. Each 10 ms block yields 160 samples at 16 kHz.

A model consumes fixed hops of `hop` samples. When the block does not divide
the hop, model output is queued behind a fixed prefill, the largest remainder
of 160*b mod hop. That guarantees a full block is available on every call and
keeps the delay constant.

Latency at 48 kHz: 2 x 60 (resampling filters) + 3 x (prefill + model delay).
"""

import numpy as np
from scipy import signal

from npumic.engine import BLOCK, SAMPLE_RATE, Stage

MODEL_RATE = 16_000
RATIO = SAMPLE_RATE // MODEL_RATE
BLOCK_16K = BLOCK // RATIO
FIR_TAPS = 121
assert BLOCK % RATIO == 0


def resampling_filter() -> np.ndarray:
    """Low-pass for 48 <-> 16 kHz: passband to ~6.5 kHz, about 60 dB rejection from 8 kHz."""
    return signal.firwin(FIR_TAPS, 7250, fs=SAMPLE_RATE, window=("kaiser", 5.65)).astype(np.float32)


def prefill_for(hop: int) -> int:
    return max((BLOCK_16K * b) % hop for b in range(hop))


class At16k(Stage):
    """Base for steps that run a 16 kHz model one hop at a time.

    Subclasses call `__init__(hop, delay)`, where `delay` is the number of
    samples at 16 kHz by which the output of a hop trails the end of its
    input, and implement `step` and `reset_model`.
    """

    def __init__(self, hop: int, delay: int):
        self.hop = hop
        self._prefill = prefill_for(hop)
        self._fir = resampling_filter()
        # Two filters of (taps - 1) / 2 each, then prefill and model delay at the model rate.
        self.latency = (FIR_TAPS - 1) + RATIO * (self._prefill + delay)

    def step(self, samples: np.ndarray) -> np.ndarray:
        """Processes one hop of 16 kHz input and returns one hop of output."""
        raise NotImplementedError

    def reset_model(self) -> None:
        pass

    def reset(self) -> None:
        self._down_zi = np.zeros(FIR_TAPS - 1, dtype=np.float32)
        self._up_zi = np.zeros(FIR_TAPS - 1, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._ready = np.zeros(self._prefill, dtype=np.float32)
        self.reset_model()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        filtered, self._down_zi = signal.lfilter(self._fir, 1.0, block, zi=self._down_zi)
        self._pending = np.concatenate([self._pending, filtered[::RATIO].astype(np.float32)])
        hops = []
        while len(self._pending) >= self.hop:
            hops.append(self.step(self._pending[: self.hop]))
            self._pending = self._pending[self.hop :]
        self._ready = np.concatenate([self._ready, *hops])
        out16, self._ready = self._ready[:BLOCK_16K], self._ready[BLOCK_16K:]

        stuffed = np.zeros(BLOCK, dtype=np.float32)
        stuffed[::RATIO] = out16 * RATIO
        out, self._up_zi = signal.lfilter(self._fir, 1.0, stuffed, zi=self._up_zi)
        return out.astype(np.float32)
