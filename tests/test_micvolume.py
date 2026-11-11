import time

from npumic import micvolume
from npumic.devices import Mic
from npumic.live import Stats


class FakeEngine:
    def __init__(self):
        self.mic = Mic(name="builtin", label="Built-in", bluetooth=False, is_default=True)
        self.stats = Stats()


def test_clip_guard_lowers_volume_only_while_clipping(monkeypatch):
    volume = {"v": 1.0}
    monkeypatch.setattr(micvolume, "node_id", lambda name: 7)
    monkeypatch.setattr(micvolume, "get_volume", lambda node: volume["v"])
    monkeypatch.setattr(micvolume, "set_volume", lambda node, v: volume.update(v=v))
    monkeypatch.setattr(micvolume, "CHECK_INTERVAL_S", 0.01)
    monkeypatch.setattr(micvolume, "COOLDOWN_S", 0.0)
    engine = FakeEngine()
    enabled = {"on": True}
    guard = micvolume.ClipGuard(engine, lambda: enabled["on"])
    guard.start()
    try:
        engine.stats.in_clipped, engine.stats.in_samples = 100, 1000
        time.sleep(0.1)
        assert volume["v"] == 0.8 and "lowered from 100% to 80%" in guard.last_action
        engine.stats.in_clipped, engine.stats.in_samples = 0, 1000
        time.sleep(0.1)
        assert volume["v"] == 0.8
        enabled["on"] = False
        engine.stats.in_clipped, engine.stats.in_samples = 100, 1000
        time.sleep(0.1)
        assert volume["v"] == 0.8
    finally:
        guard.stop()
