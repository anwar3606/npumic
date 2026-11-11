"""Core audio contract shared by every processing step.

All audio is 48 kHz mono float32. Steps process fixed blocks of BLOCK samples
and must return exactly one block of the same size for every block received.
"""

import threading
from collections.abc import Iterable

import numpy as np

SAMPLE_RATE = 48_000
BLOCK = 480  # 10 ms


class Stage:
    """One processing step. Subclasses implement `process_block`."""

    name = "stage"
    latency = 0  # samples of delay this step adds

    def process_block(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def process(self, block: np.ndarray) -> np.ndarray:
        out = self.process_block(block)
        if out.shape != block.shape or out.dtype != np.float32:
            raise ValueError(
                f"{self.name}: returned {out.dtype}{out.shape}, expected float32{block.shape}"
            )
        return out


class Passthrough(Stage):
    name = "passthrough"

    def process_block(self, block: np.ndarray) -> np.ndarray:
        return block


class Chain(Stage):
    """Runs steps in order."""

    name = "chain"

    def __init__(self, stages: Iterable[Stage]):
        self.stages = list(stages)

    @property
    def latency(self) -> int:
        return sum(s.latency for s in self.stages)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        for stage in self.stages:
            block = stage.process(block)
        return block

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()


class Delay(Stage):
    """Delays the signal by a fixed number of samples."""

    name = "delay"

    def __init__(self, samples: int):
        self.latency = samples
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(self.latency, dtype=np.float32)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        joined = np.concatenate([self._buf, block])
        self._buf = joined[len(block) :]
        return joined[: len(block)]


class Selector(Stage):
    """Chooses one of several steps while audio is running, without interrupting it.

    With `pad`, every option is padded to the largest latency so switching never
    shifts timing. Without it, the latency is that of the active option, and a
    switch skips or repeats the difference during the crossfade; that suits
    options with very different delays that are switched rarely.
    Only the active option runs. Setting `selected` starts the new option from a
    clean state in the background for `warmup_blocks` (its output unused, so
    recurrent models settle), then crossfades to it over `fade_blocks` and stops
    the old one. A new selection made during a crossfade is applied after it.
    """

    name = "selector"

    def __init__(self, options: dict[str, Stage], selected: str, warmup_blocks: int = 50, fade_blocks: int = 5,
                 pad: bool = True):
        if selected not in options:
            raise ValueError(f"unknown option: {selected}")
        self.options = options
        self._padded = max(s.latency for s in options.values()) if pad else None
        self._pads = {name: Delay(self._padded - s.latency if pad else 0) for name, s in options.items()}
        self.selected = selected
        self.active = selected
        self._warmup_blocks = warmup_blocks
        self._fade_blocks = fade_blocks
        self.reset()

    @property
    def latency(self) -> int:
        return self.options[self.active].latency if self._padded is None else self._padded

    @property
    def running(self) -> set[str]:
        """Options processing audio now: the active one and, during a switch, the incoming one."""
        incoming = self._next
        return {self.active} if incoming is None else {self.active, incoming}

    def reset(self) -> None:
        self.active = self.selected
        self._next: str | None = None
        self._count = 0
        self._restart(self.active)

    def _restart(self, name: str) -> None:
        self.options[name].reset()
        self._pads[name].reset()

    def _run(self, name: str, block: np.ndarray) -> np.ndarray:
        return self._pads[name].process(self.options[name].process(block))

    def process_block(self, block: np.ndarray) -> np.ndarray:
        fading = self._next is not None and self._count >= self._warmup_blocks
        wanted = self.selected
        if not fading and wanted != (self._next or self.active):
            self._next = None if wanted == self.active else wanted
            self._count = 0
            if self._next is not None:
                self._restart(self._next)

        out = self._run(self.active, block)
        if self._next is None:
            return out
        incoming = self._run(self._next, block)
        self._count += 1
        step = self._count - self._warmup_blocks
        if step <= 0:
            return out
        start, end = (step - 1) / self._fade_blocks, step / self._fade_blocks
        ramp = start + (end - start) * np.arange(len(block), dtype=np.float32) / len(block)
        mixed = (out + ramp * (incoming - out)).astype(np.float32)
        if step >= self._fade_blocks:
            self.active, self._next = self._next, None
        return mixed


class Fallback(Stage):
    """Runs a step and replaces it with a backup when it raises, e.g. an NPU model by its CPU version.

    On the first error the input passes through unprocessed, delayed by the
    failed step's latency, while `make_backup` builds the replacement in a
    background thread; it takes over at the next block boundary. `on_failure`
    is called with the exception from the audio thread and must not block;
    `on_backup` is called from the background thread with the new step, or with
    the exception if building it failed (the input then keeps passing through).
    """

    def __init__(self, primary: Stage, make_backup, on_failure=lambda exc: None, on_backup=lambda result: None):
        self.inner = primary
        self.name = primary.name
        self._make_backup = make_backup
        self._on_failure = on_failure
        self._on_backup = on_backup
        self._backup: Stage | None = None
        self.failed = False
        self._dry = Delay(0)

    @property
    def latency(self) -> int:
        return self.inner.latency

    def reset(self) -> None:
        if not self.failed:
            self.inner.reset()
        self._dry.reset()

    def _build_backup(self) -> None:
        try:
            backup = self._make_backup()
            silence = np.zeros(BLOCK, dtype=np.float32)
            for _ in range(5):
                backup.process(silence)
            backup.reset()
        except Exception as exc:
            self._on_backup(exc)
            return
        self._backup = backup
        self._on_backup(backup)

    def process_block(self, block: np.ndarray) -> np.ndarray:
        backup = self._backup
        if backup is not None:
            self._backup = None
            self.inner, self.failed = backup, False
        if self.failed:
            return self._dry.process(block)
        try:
            return self.inner.process(block)
        except Exception as exc:
            self.failed = True
            self._dry = Delay(self.inner.latency)
            self._on_failure(exc)
            threading.Thread(target=self._build_backup, name=f"npumic-{self.name}-backup", daemon=True).start()
            return self._dry.process(block)


class Switchable(Stage):
    """Wraps a step so it can be turned on and off while audio is running.

    The step keeps running while off and the unprocessed signal is delayed by the
    step's latency, so switching never changes timing. The change is a linear
    crossfade over `fade_blocks` blocks, which avoids clicks. If the step's
    latency changes (see Fallback), the delay of the unprocessed signal follows.
    """

    def __init__(self, inner: Stage, enabled: bool = True, fade_blocks: int = 5):
        self.inner = inner
        self.name = inner.name
        self.enabled = enabled
        self._fade_step = 1.0 / fade_blocks
        self._mix = 1.0 if enabled else 0.0
        self.reset()

    @property
    def latency(self) -> int:
        return self.inner.latency

    def reset(self) -> None:
        self.inner.reset()
        self._dry = np.zeros(self.latency, dtype=np.float32)
        self._mix = 1.0 if self.enabled else 0.0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        wet = self.inner.process(block)
        if len(self._dry) != self.latency:
            self._dry = np.zeros(self.latency, dtype=np.float32)
        delayed = np.concatenate([self._dry, block])
        dry, self._dry = delayed[: len(block)], delayed[len(block) :]
        target = 1.0 if self.enabled else 0.0
        start = self._mix
        end = min(start + self._fade_step, target) if target > start else max(start - self._fade_step, target)
        self._mix = end
        if start == end:  # settled at the target: fully on or fully off
            return wet if end else dry
        ramp = start + (end - start) * np.arange(len(block), dtype=np.float32) / len(block)
        return (dry + ramp * (wet - dry)).astype(np.float32)


class Reblocker:
    """Feeds arbitrary-size chunks (as delivered by audio devices) to a Stage in fixed blocks.

    Output length always equals input length. Adds a constant delay of one block.
    """

    def __init__(self, stage: Stage, block: int = BLOCK):
        self.stage = stage
        self.block = block
        self.reset()

    @property
    def latency(self) -> int:
        return self.block + self.stage.latency

    def reset(self) -> None:
        self.stage.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._ready = np.zeros(self.block, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32, copy=False)])
        n_blocks = len(self._pending) // self.block
        if n_blocks:
            done = [
                self.stage.process(self._pending[i * self.block : (i + 1) * self.block])
                for i in range(n_blocks)
            ]
            self._pending = self._pending[n_blocks * self.block :]
            self._ready = np.concatenate([self._ready, *done])
        out, self._ready = self._ready[: len(chunk)], self._ready[len(chunk) :]
        return out


def process_offline(stage: Stage, audio: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Processes a whole recording block by block.

    The step's latency is removed, so output sample i lines up with input sample i.
    """
    audio = np.asarray(audio, dtype=np.float32)
    stage.reset()
    total = len(audio) + stage.latency
    padded = np.zeros(-(-total // block) * block, dtype=np.float32)
    padded[: len(audio)] = audio
    out = np.concatenate(
        [stage.process(padded[i : i + block]) for i in range(0, len(padded), block)]
    )
    return out[stage.latency : stage.latency + len(audio)]
