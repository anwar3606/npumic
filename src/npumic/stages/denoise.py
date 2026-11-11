"""Neural noise removal with DPDFNet-2 (48 kHz full-band, 2.6M parameters), running on the NPU.

The initial state is built from the model's ONNX metadata: the ERB and
deep-filter normalization tables, followed by zeros.

Streaming STFT matching the model's training setup (from its ONNX metadata):
960-sample Vorbis window, 480-sample hop, unnormalized FFT. The Vorbis window
is power-complementary at 50% overlap, so windowed overlap-add reconstructs
the signal exactly when the model leaves the spectrum unchanged.

The NPU computes in float16. Quiet microphones produce spectral values small
enough to lose precision there, which makes the model cut quiet speech. The
input is therefore raised by `input_gain_db` before the model and lowered by
the same amount afterwards; overall level is unchanged.

The graph computes its input features itself: a log-magnitude spectrum
(10 * log10(|X| / 960 + 1e-10)) and, for the 96 deep-filter bins, the magnitude
used for unit normalization. In f16 the 1e-10 floor is zero and quiet bins
underflow, so on the NPU high frequencies were judged silent and suppressed
(11-16 dB below CPU above 6 kHz). Both features are cut out of the graph and
computed here in f32; their running normalizations stay in the graph.

On the NPU, inference is pipelined (see StreamingInfer), adding one hop
(10 ms) of latency so each inference has a full block period to finish.
"""

import numpy as np

from npumic.engine import BLOCK, Stage
from npumic.runtime import MODELS_DIR, StreamingInfer, compile_realtime, core, cut_to_input, ensure_model

MODEL_PATH = MODELS_DIR / "dpdfnet" / "dpdfnet2_48khz_hr.onnx"
WIN = 960
HOP = 480
assert HOP == BLOCK
WNORM = np.float32(1 / WIN)  # spectrum scaling applied by the model before feature extraction
DF_BINS = 96
LOG_FLOOR = 1e-10


def load_model(path=MODEL_PATH):
    """DPDFNet graph with its log-magnitude ("feat_db") and deep-filter magnitude ("feat_mag") features as inputs."""
    if path == MODEL_PATH and not path.exists():
        ensure_model("dpdfnet/dpdfnet2_48khz_hr.onnx")
    model = core().read_model(str(path))
    cut_to_input(model, "/model/Mul", "feat_db")
    cut_to_input(model, "/model/spec_norm/Sqrt", "feat_mag")
    return model


def initial_state(model) -> np.ndarray:
    meta = model.get_rt_info()["framework"]

    def values(key):
        return np.array(meta[key].astype(str).split(","), dtype=np.float32)

    erb, spec = values("erb_norm_init"), values("spec_norm_init")
    state = np.zeros(int(meta["state_size"].astype(str)), dtype=np.float32)
    state[: len(erb)] = erb
    state[len(erb) : len(erb) + len(spec)] = spec
    return state


def vorbis_window(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.sin(0.5 * np.pi * np.sin(np.pi * (k + 0.5) / n) ** 2).astype(np.float32)


class Denoise(Stage):
    name = "denoise"

    def __init__(self, device: str = "NPU", input_gain_db: float = 10.0):
        self._gain_in = np.float32(10 ** (input_gain_db / 20))
        self._gain_out = np.float32(1 / self._gain_in)
        model = load_model(MODEL_PATH)
        state = initial_state(model)
        # The CPU plugin's subgraph fusion fails on this model's GRU cells.
        request, self.device = compile_realtime(model, device, {"SNIPPETS_MODE": "DISABLE"})
        self._infer = StreamingInfer(request, self.device == "NPU", {"state_in": state}, {"state_out": "state_in"})
        self._window = vorbis_window(WIN)
        self._spec = np.zeros((1, 1, WIN // 2 + 1, 2), dtype=np.float32)
        self.latency = self.latency_on(self.device)
        self.reset()

    @classmethod
    def latency_on(cls, device: str) -> int:
        return HOP + (cls.model_delay_frames + (device == "NPU")) * HOP

    @property
    def last_infer_ms(self) -> float:
        return self._infer.last_ms

    # Frames by which the model's output trails its input (40 ms); verified in tests.
    model_delay_frames = 4

    def reset(self) -> None:
        self._infer.reset()
        self._frame = np.zeros(WIN, dtype=np.float32)
        self._overlap = np.zeros(HOP, dtype=np.float32)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        self._frame[:HOP] = self._frame[HOP:]
        self._frame[HOP:] = block * self._gain_in
        spectrum = np.fft.rfft(self._frame * self._window)
        self._spec[0, 0, :, 0] = spectrum.real
        self._spec[0, 0, :, 1] = spectrum.imag

        magnitude = np.abs(spectrum) * WNORM
        feat_db = 10 * np.log10(magnitude + LOG_FLOOR)
        result = self._infer({
            "spec": self._spec,
            "feat_db": feat_db.astype(np.float32)[None, None],
            "feat_mag": magnitude[:DF_BINS].astype(np.float32)[None, None],
        })
        if result is None:  # first pipelined step: no output yet
            enhanced = np.zeros(WIN // 2 + 1, dtype=np.complex64)
        else:
            enhanced = result["spec_e"][0, 0, :, 0] + 1j * result["spec_e"][0, 0, :, 1]

        frame = np.fft.irfft(enhanced, n=WIN).astype(np.float32)
        frame *= self._window
        out = (self._overlap + frame[:HOP]) * self._gain_out
        self._overlap = frame[HOP:].copy()
        return out.astype(np.float32)

