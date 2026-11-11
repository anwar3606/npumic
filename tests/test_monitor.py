import base64

import numpy as np

from npumic.engine import SAMPLE_RATE
from npumic.monitor import SPEC_BINS, SPEC_HOP, SPEC_MIN_HZ, Ring, Spectrograms, spectrogram_columns


def test_ring_wraps_and_reads_by_absolute_position():
    ring = Ring(100)
    data = np.arange(250, dtype=np.float32)
    for i in range(0, 250, 37):
        ring.write(data[i : i + 37])
    assert ring.written == 250
    np.testing.assert_array_equal(ring.read(160, 250), data[160:250])
    assert ring.read(100, 200) is None  # overwritten
    assert ring.read(240, 260) is None  # not written yet


def _band_of(hz: float) -> int:
    return int(np.log(hz / SPEC_MIN_HZ) / np.log(SAMPLE_RATE / 2 / SPEC_MIN_HZ) * SPEC_BINS)


def test_spectrogram_puts_a_tone_in_its_band():
    ring = Ring(SAMPLE_RATE)
    t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    ring.write((0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
    col = spectrogram_columns(ring, 20, 1)[0]
    assert abs(int(np.argmax(col)) - _band_of(1000)) <= 1
    assert col.max() > 200 and np.median(col) < 60


def test_output_columns_are_aligned_to_input_by_latency():
    latency = 2880
    inp, out = Ring(SAMPLE_RATE), Ring(SAMPLE_RATE)
    burst = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    burst[12000:13000] = 0.5
    inp.write(burst)
    out.write(np.concatenate([np.zeros(latency, dtype=np.float32), burst])[: len(burst)])
    sp = Spectrograms(inp, out, lambda: latency).since(0)
    cols_in = np.frombuffer(base64.b64decode(sp["in"]), np.uint8).reshape(sp["count"], SPEC_BINS)
    cols_out = np.frombuffer(base64.b64decode(sp["out"]), np.uint8).reshape(sp["count"], SPEC_BINS)
    assert sp["count"] == (len(burst) - latency) // SPEC_HOP
    np.testing.assert_array_equal(cols_in, cols_out)
