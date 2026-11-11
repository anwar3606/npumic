"""Keeping only the user's voice with VoxLock (16 kHz, 27M parameters, Apache-2.0).

VoxLock is a causal band-split RNN for target-speaker extraction: given a 3 s
sample of the user's voice, it keeps that voice and removes other talkers. It
runs at 16 kHz inside the 48 kHz pipeline through At16k.

Models (models/voxlock/onnx):
- enroll.onnx: voice sample [1, 48000] -> enroll_mag [1, 257, 376], context [1, 512, 298]
- chunk.onnx: wav [1, (k - 1) * 128 + 512] (k STFT frames of 512 samples at a
  128-sample hop), enroll_mag, context and the LSTM state h, c [6, 32, 256]
  -> frames [1, k, 512], h_out, c_out

Output frames are windowed; they are overlap-added and scaled by
GAIN / WIN2_SUM, as in the reference implementation (the squared Hann window
sums to 1.5 at this hop).

Each call processes k frames (k x 8 ms); a larger k costs less per second of
audio but adds delay. Inference is pipelined (see StreamingInfer), adding one
call of delay. k is 5 on both devices: NPU about 14 ms per 40 ms call (k = 3
took 12 ms per 24 ms, which with DPDFNet on the same NPU left it nearly always
busy and made audio callbacks wait), CPU about 18 ms with four threads.

Latency at 48 kHz: 2 x 60 (resampling filters) + 3 x (prefill + 384 + k x 128):
96.5 ms.
"""

import numpy as np

from npumic.runtime import MODELS_DIR, StreamingInfer, available, compile_realtime, core, ensure_model
from npumic.stages.rate16k import FIR_TAPS, MODEL_RATE, RATIO, At16k, prefill_for

MODEL_DIR = MODELS_DIR / "voxlock" / "onnx"
FRAME_HOP = 128
N_FFT = 512
GAIN = -0.5
WIN2_SUM = 1.5
SAMPLE_SECONDS = 3
ENROLL_FRAMES = SAMPLE_SECONDS * MODEL_RATE // FRAME_HOP + 1
CONTEXT_FRAMES = 298
STATE_SHAPE = (6, 32, 256)
STEP_FRAMES = {"NPU": 5, "CPU": 5}
CPU_THREADS = 4


def loudest(sample: np.ndarray, seconds: float = SAMPLE_SECONDS) -> np.ndarray:
    """The loudest `seconds` of a 16 kHz recording, in 10 ms steps."""
    hop = MODEL_RATE // 100
    n = len(sample) // hop
    energy = np.mean(sample[: n * hop].reshape(n, hop) ** 2, axis=1)
    width = int(seconds * 100)
    start = int(np.convolve(energy, np.ones(width), "valid").argmax()) * hop
    return sample[start : start + int(seconds * MODEL_RATE)]


class Voice:
    """The user's voice as VoxLock's enrollment features, shared by every VoxLock option."""

    def __init__(self):
        self._enroll = None
        self.mag: np.ndarray | None = None
        self.context: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.mag is not None

    def set(self, sample: np.ndarray) -> None:
        """Takes a 3 s, 16 kHz voice sample."""
        if len(sample) != SAMPLE_SECONDS * MODEL_RATE:
            raise ValueError(f"voice sample must be {SAMPLE_SECONDS} s at {MODEL_RATE} Hz")
        if self._enroll is None:
            enroll_path = MODEL_DIR / "enroll.onnx"
            if not enroll_path.exists():
                ensure_model("voxlock/onnx/enroll.onnx")
            self._enroll = core().compile_model(str(enroll_path), "CPU").create_infer_request()
        result = self._enroll.infer({"enroll": sample.astype(np.float32)[None]})
        # Replaced as new objects so running steps see either the old or the new voice, never a mix.
        self.mag, self.context = result["enroll_mag"].copy(), result["context"].copy()


class VoxLock(At16k):
    name = "voxlock"

    def __init__(self, voice: Voice, device: str = "NPU"):
        self.voice = voice
        device = device if available(device) else "CPU"
        frames = STEP_FRAMES[device]
        self._segment = (frames - 1) * FRAME_HOP + N_FFT
        shapes = {"wav": [1, self._segment], "enroll_mag": [1, 257, ENROLL_FRAMES],
                  "context": [1, 512, CONTEXT_FRAMES], "h": list(STATE_SHAPE), "c": list(STATE_SHAPE)}
        states = {"h": np.zeros(STATE_SHAPE, np.float32), "c": np.zeros(STATE_SHAPE, np.float32)}
        state_outputs = {"h_out": "h", "c_out": "c"}
        chunk_path = MODEL_DIR / "chunk.onnx"
        if not chunk_path.exists():
            ensure_model("voxlock/onnx/chunk.onnx")
        model = core().read_model(str(chunk_path))
        model.reshape(shapes)
        request, self.device = compile_realtime(model, device, {"INFERENCE_NUM_THREADS": CPU_THREADS})
        for _ in range(3):  # first inferences allocate and initialize
            request.infer({name: np.zeros(shape, np.float32) for name, shape in shapes.items()})
        self._infer = StreamingInfer(request, True, states, state_outputs)
        self._frames = frames
        super().__init__(frames * FRAME_HOP, N_FFT - FRAME_HOP + frames * FRAME_HOP)
        self.reset()

    @staticmethod
    def latency_on(device: str) -> int:
        hop = STEP_FRAMES[device] * FRAME_HOP
        return (FIR_TAPS - 1) + RATIO * (prefill_for(hop) + N_FFT - FRAME_HOP + hop)

    @property
    def last_infer_ms(self) -> float:
        return self._infer.last_ms

    def reset_model(self) -> None:
        self._infer.reset()
        self._wav = np.zeros(self._segment, dtype=np.float32)
        self._ola = np.zeros(self._segment, dtype=np.float32)

    def step(self, samples: np.ndarray) -> np.ndarray:
        self._wav[: -self.hop] = self._wav[self.hop :]
        self._wav[-self.hop :] = samples
        mag, context = self.voice.mag, self.voice.context
        if mag is None:
            return np.zeros(self.hop, dtype=np.float32)
        result = self._infer({"wav": self._wav[None], "enroll_mag": mag, "context": context})
        if result is None:  # first pipelined step: no output yet
            return np.zeros(self.hop, dtype=np.float32)
        frames = result["frames"][0]
        for j in range(self._frames):
            self._ola[j * FRAME_HOP : j * FRAME_HOP + N_FFT] += frames[j]
        out = self._ola[: self.hop] * np.float32(GAIN / WIN2_SUM)
        self._ola[: -self.hop] = self._ola[self.hop :]
        self._ola[-self.hop :] = 0
        return out.astype(np.float32)
