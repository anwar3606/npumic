"""Measurements for the control page: recent audio for spectrograms, and resource usage.

The audio callback only copies samples into preallocated ring buffers. Spectrograms
are computed on request in the web server thread, and resource usage is sampled by
a low-rate background thread, so neither adds work to the real-time path.
"""

import base64
import glob
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from npumic.engine import BLOCK, SAMPLE_RATE

SPEC_FFT = 1024
SPEC_HOP = BLOCK  # one column per 10 ms
SPEC_BINS = 96
SPEC_MIN_HZ = 40
SPEC_FLOOR_DB = -100.0
SPEC_RANGE_DB = 90.0


class Ring:
    """Fixed-size history of the most recent samples. `written` counts all samples ever written."""

    def __init__(self, size: int):
        self.buf = np.zeros(size, dtype=np.float32)
        self.written = 0

    def write(self, x: np.ndarray) -> None:
        size, n = len(self.buf), len(x)
        if n >= size:
            self.buf[:] = x[-size:]
            self.written += n
            return
        i = self.written % size
        first = min(n, size - i)
        self.buf[i : i + first] = x[:first]
        self.buf[: n - first] = x[first:]
        self.written += n

    def read(self, start: int, end: int) -> np.ndarray | None:
        """Samples [start, end) by absolute position, or None if no longer (or not yet) held."""
        if start < max(0, self.written - len(self.buf)) or end > self.written:
            return None
        size = len(self.buf)
        idx = np.arange(start, end) % size
        return self.buf[idx]


def _band_ranges() -> tuple[np.ndarray, np.ndarray]:
    """FFT bin range [lo, hi) of each log-spaced display band from SPEC_MIN_HZ to Nyquist.

    Bands narrower than one FFT bin use the nearest bin, so low bands may repeat a bin.
    """
    hz = np.geomspace(SPEC_MIN_HZ, SAMPLE_RATE / 2, SPEC_BINS + 1)
    pos = hz / (SAMPLE_RATE / SPEC_FFT)
    lo = np.minimum(np.round(pos[:-1]).astype(int), SPEC_FFT // 2)
    hi = np.maximum(np.round(pos[1:]).astype(int), lo + 1)
    return lo, np.minimum(hi, SPEC_FFT // 2 + 1)


_LO, _HI = _band_ranges()
_WINDOW = np.hanning(SPEC_FFT).astype(np.float32)
_WINDOW_GAIN = float(np.sum(_WINDOW))


def spectrogram_columns(ring: Ring, first_col: int, count: int, offset: int = 0) -> np.ndarray:
    """Columns [first_col, first_col + count) as uint8 levels (0 = SPEC_FLOOR_DB, 255 = floor + range).

    Column c covers the SPEC_FFT samples ending at (c + 1) * SPEC_HOP + offset.
    """
    frames = []
    for c in range(first_col, first_col + count):
        end = (c + 1) * SPEC_HOP + offset
        x = ring.read(end - SPEC_FFT, end)
        frames.append(np.zeros(SPEC_FFT, dtype=np.float32) if x is None else x)
    power = np.abs(np.fft.rfft(np.stack(frames) * _WINDOW, axis=1) / (_WINDOW_GAIN / 2)) ** 2
    cumulative = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(power, axis=1)], axis=1)
    bands = (cumulative[:, _HI] - cumulative[:, _LO]) / (_HI - _LO)
    db = 10 * np.log10(bands + 1e-20)
    return np.clip((db - SPEC_FLOOR_DB) / SPEC_RANGE_DB * 255, 0, 255).astype(np.uint8)


class Spectrograms:
    """Input and output spectrograms, time-aligned by removing the processing latency from the output."""

    def __init__(self, input_ring: Ring, output_ring: Ring, latency, max_cols: int = 200):
        """`latency` is a function returning the current processing latency in samples."""
        # Rings must hold max_cols columns plus one FFT frame plus the latency; older samples read as silence.
        self.input, self.output, self._latency, self.max_cols = input_ring, output_ring, latency, max_cols

    @property
    def latency(self) -> int:
        return self._latency()

    def since(self, col: int) -> dict:
        latency = self.latency
        latest = min(self.input.written, self.output.written - latency) // SPEC_HOP  # exclusive
        first = max(col, latest - self.max_cols)
        count = max(0, latest - first)
        if count == 0:
            return {"next": latest, "count": 0, "bins": SPEC_BINS}
        cols_in = spectrogram_columns(self.input, first, count)
        cols_out = spectrogram_columns(self.output, first, count, offset=latency)
        return {
            "next": first + count,
            "count": count,
            "bins": SPEC_BINS,
            "min_hz": SPEC_MIN_HZ,
            "floor_db": SPEC_FLOOR_DB,
            "range_db": SPEC_RANGE_DB,
            "in": base64.b64encode(cols_in.tobytes()).decode(),
            "out": base64.b64encode(cols_out.tobytes()).decode(),
        }


def _read_int(path: str | None) -> int | None:
    if not path:
        return None
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


class ResourceMonitor:
    """Samples this process's CPU use, NPU busy time and CPU package power twice a second."""

    INTERVAL = 0.5
    HISTORY = 120  # 60 s

    def __init__(self, stats):
        self.stats = stats
        self.history: deque[dict] = deque(maxlen=self.HISTORY)
        self._npu_busy = next(iter(glob.glob("/sys/class/accel/accel*/device/npu_busy_time_us")), None)
        npu_dir = Path(self._npu_busy).parent if self._npu_busy else None
        self._npu_freq = str(npu_dir / "npu_current_frequency_mhz") if npu_dir else None
        self._energy = "/sys/class/powercap/intel-rapl:0/energy_uj"
        self._energy_range = _read_int("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="npumic-monitor", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> dict:
        return self.history[-1] if self.history else {}

    def _run(self) -> None:
        prev_t, prev_cpu = time.monotonic(), time.process_time()
        prev_npu, prev_energy = _read_int(self._npu_busy), _read_int(self._energy)
        while not self._stop.wait(self.INTERVAL):
            t, cpu = time.monotonic(), time.process_time()
            npu, energy = _read_int(self._npu_busy), _read_int(self._energy)
            dt = t - prev_t
            sample = {
                "t": round(time.time(), 2),
                "cpu_pct": round(100 * (cpu - prev_cpu) / dt, 1),
                "npu_pct": None,
                "npu_mhz": _read_int(self._npu_freq),
                "package_w": None,
                "proc_avg_ms": round(self.stats.avg_ms, 2),
                "proc_max_ms": round(self.stats.take_window_max(), 2),
                "budget_ms": round(1000 * self.stats.cycle_frames / SAMPLE_RATE, 2),
            }
            if npu is not None and prev_npu is not None:
                sample["npu_pct"] = round(min(100.0, 100 * (npu - prev_npu) / 1e6 / dt), 1)
            if energy is not None and prev_energy is not None:
                delta = energy - prev_energy
                if delta < 0 and self._energy_range:
                    delta += self._energy_range
                sample["package_w"] = round(delta / 1e6 / dt, 2)
            self.history.append(sample)
            prev_t, prev_cpu, prev_npu, prev_energy = t, cpu, npu, energy
