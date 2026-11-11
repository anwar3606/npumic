import numpy as np
import pytest

from support.signals import noisy_speech
from npumic.engine import BLOCK, Chain, Delay, Fallback, Passthrough, Reblocker, Selector, Stage, Switchable, process_offline


class RunningGain(Stage):
    """Stateful step whose output depends on all previous blocks."""

    name = "running_gain"

    def reset(self):
        self.energy = 0.0

    def __init__(self):
        self.reset()

    def process_block(self, block):
        self.energy = 0.9 * self.energy + 0.1 * float(np.mean(block**2))
        return (block / (1.0 + 10.0 * self.energy)).astype(np.float32)


class Halve(Stage):
    name = "halve"

    def process_block(self, block):
        return block * np.float32(0.5)


class WrongLength(Stage):
    name = "wrong_length"

    def process_block(self, block):
        return block[:-1]


@pytest.fixture
def audio():
    return noisy_speech(3.0, snr_db=10)[1]


def test_passthrough_offline_is_exact(audio):
    np.testing.assert_array_equal(process_offline(Passthrough(), audio), audio)


def test_offline_removes_latency(audio):
    out = process_offline(Chain([Delay(100), Delay(337)]), audio)
    np.testing.assert_array_equal(out, audio)


def test_stage_contract_rejects_wrong_length():
    with pytest.raises(ValueError, match="wrong_length"):
        WrongLength().process(np.zeros(BLOCK, dtype=np.float32))


@pytest.mark.parametrize("seed", range(5))
def test_reblocker_output_independent_of_chunk_sizes(audio, seed):
    """Audio devices deliver irregular chunk sizes; results must not depend on them."""
    reference = process_offline(RunningGain(), audio)

    rng = np.random.default_rng(seed)
    rb = Reblocker(RunningGain())
    pieces, pos = [], 0
    while pos < len(audio):
        size = int(rng.integers(1, 3 * BLOCK))
        chunk = audio[pos : pos + size]
        out = rb.push(chunk)
        assert len(out) == len(chunk)
        pieces.append(out)
        pos += size
    live = np.concatenate(pieces)

    n = len(audio) - rb.latency
    n -= n % BLOCK
    np.testing.assert_array_equal(live[rb.latency : rb.latency + n], reference[:n])


def test_switchable_on_is_exactly_the_step(audio):
    step = Chain([Delay(337), RunningGain()])
    expected = process_offline(Chain([Delay(337), RunningGain()]), audio)
    np.testing.assert_array_equal(process_offline(Switchable(step), audio), expected)


def test_switchable_off_is_exactly_the_input(audio):
    np.testing.assert_array_equal(process_offline(Switchable(Chain([Delay(337), Halve()]), enabled=False), audio), audio)


def test_switching_keeps_timing_and_fades_smoothly():
    """Turning processing off mid-stream must not shift the audio in time or jump in level."""
    switch = Switchable(Chain([Delay(337), Halve()]), fade_blocks=5)
    ones = np.ones(BLOCK, dtype=np.float32)
    blocks = [switch.process(ones) for _ in range(10)]
    switch.enabled = False
    blocks += [switch.process(ones) for _ in range(10)]
    out = np.concatenate(blocks)[337:]  # after the step's delay the input is constant 1.0

    on, fade, off = out[: 10 * BLOCK - 337], out[10 * BLOCK - 337 : 15 * BLOCK - 337], out[15 * BLOCK - 337 :]
    np.testing.assert_array_equal(on, 0.5)
    np.testing.assert_array_equal(off, 1.0)
    assert np.all(np.diff(fade) >= 0)
    assert np.max(np.abs(np.diff(out))) <= 0.5 / (5 * BLOCK) + 1e-6


class Constant(Stage):
    """Outputs a fixed value after its latency, so timing and mixing are easy to read."""

    def __init__(self, value: float, latency: int):
        self.value, self.latency = np.float32(value), latency
        self.name = f"constant{value}"
        self.reset()

    def reset(self):
        self.delay = Delay(self.latency)
        self.blocks = 0

    def process_block(self, block):
        self.blocks += 1
        return self.delay.process(np.full_like(block, self.value))


def test_selector_pads_options_to_common_latency(audio):
    selector = Selector({"short": Delay(100), "long": Delay(900)}, "short")
    assert selector.latency == 900
    np.testing.assert_array_equal(process_offline(selector, audio), audio)


def test_selector_warms_up_then_crossfades_and_stops_old_option():
    a, b = Constant(1.0, latency=0), Constant(3.0, latency=0)
    selector = Selector({"a": a, "b": b}, "a", warmup_blocks=4, fade_blocks=2)
    ones = np.ones(BLOCK, dtype=np.float32)
    out = [selector.process(ones) for _ in range(3)]
    selector.selected = "b"
    out += [selector.process(ones) for _ in range(10)]
    out = np.concatenate(out)

    blocks_a = a.blocks
    np.testing.assert_array_equal(out[: 7 * BLOCK], 1.0)  # 3 before + 4 warm-up blocks: still "a"
    fade = out[7 * BLOCK : 9 * BLOCK]
    assert np.all(np.diff(fade) > 0) and fade[0] == 1.0
    np.testing.assert_array_equal(out[9 * BLOCK :], 3.0)
    assert selector.active == "b"
    selector.process(ones)
    assert a.blocks == blocks_a  # the old option no longer runs


def test_selector_switch_back_during_warmup_is_silent():
    selector = Selector({"a": Constant(1.0, 0), "b": Constant(3.0, 0)}, "a", warmup_blocks=4, fade_blocks=2)
    ones = np.ones(BLOCK, dtype=np.float32)
    selector.selected = "b"
    out = [selector.process(ones) for _ in range(2)]
    selector.selected = "a"
    out += [selector.process(ones) for _ in range(10)]
    np.testing.assert_array_equal(np.concatenate(out), 1.0)
    assert selector.active == "a"


def test_unpadded_selector_latency_follows_active_option():
    sel = Selector({"off": Passthrough(), "slow": Delay(960)}, "off", warmup_blocks=2, fade_blocks=2, pad=False)
    assert sel.latency == 0
    sel.selected = "slow"
    block = np.ones(BLOCK, np.float32)
    for _ in range(3):
        sel.process(block)
        assert sel.running == {"off", "slow"}
    sel.process(block)
    assert sel.active == "slow" and sel.latency == 960 and sel.running == {"slow"}


class FailsAfter(Stage):
    """Delays by `latency` samples, then raises from block `n` on."""

    name = "fails_after"

    def __init__(self, n: int, latency: int = 480):
        self.n, self.latency, self.count = n, latency, 0
        self._delay = Delay(latency)

    def reset(self):
        self._delay.reset()

    def process_block(self, block):
        self.count += 1
        if self.count > self.n:
            raise RuntimeError("device lost")
        return self._delay.process(block)


def _wait_until(predicate, timeout=5.0):
    import time

    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_fallback_replaces_a_failing_step_with_its_backup():
    import threading

    failures, backups, release = [], [], threading.Event()

    def make_backup():
        release.wait(5)
        return Delay(960)

    stage = Fallback(FailsAfter(3), make_backup, failures.append, backups.append)
    block = np.arange(BLOCK, dtype=np.float32)
    for _ in range(3):
        stage.process(block)
    out = stage.process(block)  # the primary fails: input passes through, delayed by its latency
    assert stage.failed and len(failures) == 1 and str(failures[0]) == "device lost"
    assert np.all(out == 0)
    np.testing.assert_array_equal(stage.process(block), block)
    release.set()
    _wait_until(lambda: backups)
    stage.process(block)
    assert not stage.failed and isinstance(stage.inner, Delay) and stage.latency == 960


def test_fallback_keeps_passing_audio_when_the_backup_fails():
    backups = []

    def broken():
        raise RuntimeError("no CPU either")

    stage = Fallback(FailsAfter(0), broken, on_backup=backups.append)
    block = np.ones(BLOCK, np.float32)
    stage.process(block)
    _wait_until(lambda: backups)
    assert isinstance(backups[0], RuntimeError)
    np.testing.assert_array_equal(stage.process(block), block)


def test_switchable_follows_a_change_of_latency():
    stage = Fallback(FailsAfter(2, latency=480), lambda: Delay(960))
    switch = Switchable(stage, enabled=False)
    block = np.ones(BLOCK, np.float32)
    for _ in range(4):
        switch.process(block)
    _wait_until(lambda: stage._backup is not None or not stage.failed)
    switch.process(block)
    assert switch.latency == 960
    out = [switch.process(block) for _ in range(3)]
    np.testing.assert_array_equal(out[-1], block)
