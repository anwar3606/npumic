"""OpenVINO setup, model downloads and the compiled-model cache.

Import this module before `openvino` anywhere in the package.
"""

import hashlib
import logging
import os
import sys
import time
import urllib.request
from functools import cache
from pathlib import Path

ROOT = Path(os.environ.get("NPUMIC_HOME", Path(__file__).resolve().parents[2]))
MODELS_DIR = ROOT / "models"
CACHE_DIR = MODELS_DIR / "cache"
DATA_DIR = ROOT / "data"

MODELS = {
    "dpdfnet/dpdfnet2_48khz_hr.onnx": (
        "https://huggingface.co/Ceva-IP/DPDFNet/resolve/dd6818d00f50c836fed43a6243ebe49116de5964/onnx/dpdfnet2_48khz_hr.onnx",
        "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b",
    ),
    "voxlock/onnx/enroll.onnx": (
        "https://huggingface.co/arbazsiddiqui/voxlock/resolve/a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03/onnx/enroll.onnx",
        "86645cba23f207def0b31a6e74a615d3b7a813b4fdf957c53b3af2905ba2ffd3",
    ),
    "voxlock/onnx/chunk.onnx": (
        "https://huggingface.co/arbazsiddiqui/voxlock/resolve/a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03/onnx/chunk.onnx",
        "3c31444926cdf3d548935344e2f5f1c71538ded6c2c7d1bd22e2c2141ea0bc2a",
    ),
}


def ensure_model(rel_path: str) -> Path:
    """Returns `MODELS_DIR / rel_path`, downloading and verifying it if missing."""
    path = MODELS_DIR / rel_path
    if path.exists():
        return path
    url, expected_sha = MODELS[rel_path]
    log = logging.getLogger("npumic")
    log.info("Downloading %s", rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    sha = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1024 * 1024):
            out.write(chunk)
            sha.update(chunk)
    if sha.hexdigest() != expected_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {rel_path}: got {sha.hexdigest()}")
    tmp.replace(path)
    return path


def ensure_models() -> None:
    """Downloads any missing ONNX models into `MODELS_DIR`."""
    for rel_path in MODELS:
        ensure_model(rel_path)

# hwloc prints a large warning about this CPU's topology on every start; it is harmless.
os.environ.setdefault("HWLOC_HIDE_ERRORS", "2")

# `import openvino` sends a usage event through openvino_telemetry, which forks a
# sender process; forking a multi-threaded real-time audio process can deadlock it.
# With the module unavailable, OpenVINO falls back to its built-in no-op stub.
sys.modules["openvino_telemetry"] = None

import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402


@cache
def core() -> ov.Core:
    c = ov.Core()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    c.set_property({"CACHE_DIR": str(CACHE_DIR)})
    return c


@cache
def available(device: str) -> bool:
    """Whether `device` can run models, loading only that device's plugin.

    `Core.available_devices` loads every plugin; the GPU plugin then opens all OpenCL
    devices, including the discrete NVIDIA GPU, and keeps it out of runtime D3.
    """
    try:
        return bool(core().get_property(device, "AVAILABLE_DEVICES"))
    except RuntimeError:
        return False


def has_npu() -> bool:
    return available("NPU")


class StreamingInfer:
    """Runs a recurrent streaming model one step at a time, carrying its state between steps.

    Synchronous mode returns each step's outputs immediately. Pipelined mode starts
    step n and returns the outputs of step n-1 (None for the first step): the device
    works between audio callbacks, which gives it a full step period instead of a
    fraction of one callback, at the cost of one step of extra latency. Each step
    still receives the state produced by the step before it.
    """

    def __init__(self, request: ov.InferRequest, pipelined: bool, states: dict[str, np.ndarray],
                 state_outputs: dict[str, str]):
        """`states` holds the initial value of each state input; `state_outputs` maps
        each state output name to the state input it feeds."""
        self.request = request
        self.pipelined = pipelined
        self._initial = {k: v.astype(np.float32) for k, v in states.items()}
        self._state_outputs = state_outputs
        # A tensor can carry several names (graph-internal and declared); which one
        # `any_name` returns differs between fresh and cached compiles, so outputs
        # are published under all of them.
        self._outputs = [(o.any_name, o.get_names()) for o in request.model_outputs]
        self.last_ms = 0.0
        self.reset()

    def reset(self) -> None:
        if getattr(self, "_pending", False):
            self.request.wait()
        self._pending = False
        self._states = {k: v.copy() for k, v in self._initial.items()}

    def _collect(self) -> dict[str, np.ndarray]:
        results = {}
        for tensor_name, names in self._outputs:
            value = self.request.get_tensor(tensor_name).data.copy()
            state = next((n for n in names if n in self._state_outputs), None)
            if state is not None:
                self._states[self._state_outputs[state]] = value
            else:
                results.update(dict.fromkeys(names, value))
        return results

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        t0 = time.perf_counter()
        if not self.pipelined:
            self.request.infer({**inputs, **self._states})
            result = self._collect()
        else:
            result = None
            if self._pending:
                self.request.wait()
                result = self._collect()
            self.request.start_async({**inputs, **self._states})
            self._pending = True
        self.last_ms = (time.perf_counter() - t0) * 1000
        return result


def cut_to_input(model: ov.Model, node_name: str, input_name: str) -> None:
    """Replaces the output of `node_name` with a new f32 model input of the same shape.

    Used to compute numerically sensitive feature extraction (logarithms of small
    magnitudes) on the CPU in f32 instead of in the NPU's f16.
    """
    node = next(op for op in model.get_ops() if op.get_friendly_name() == node_name)
    param = ov.op.Parameter(ov.Type.f32, node.get_output_partial_shape(0))
    param.set_friendly_name(input_name)
    param.output(0).get_tensor().set_names({input_name})
    for target in list(node.output(0).get_target_inputs()):
        target.replace_source_output(param.output(0))
    model.add_parameters([param])
    model.validate_nodes_and_infer_types()


def compile_realtime(model_path: Path | ov.Model, device: str, cpu_config: dict | None = None) -> tuple[ov.InferRequest, str]:
    """Compiles a model for per-block inference inside the real-time audio callback.

    Falls back to CPU when `device` is not available. Returns the request and the device used.
    """
    c = core()
    device = device if available(device) else "CPU"
    config = {"PERFORMANCE_HINT": "LATENCY"}
    if device == "NPU":
        # High priority shortens each NPU inference by about a third (DPDFNet: 6.7 -> 4.7 ms).
        config["MODEL_PRIORITY"] = "HIGH"
    if device == "CPU":
        # Worker threads that the audio callback busy-waits on can be starved by it and
        # freeze the process; a single inference thread avoids that and is also faster
        # for small streaming models.
        config["INFERENCE_NUM_THREADS"] = 1
        config["ENABLE_CPU_PINNING"] = False
        config.update(cpu_config or {})
    model = model_path if isinstance(model_path, ov.Model) else str(model_path)
    return c.compile_model(model, device, config).create_infer_request(), device
