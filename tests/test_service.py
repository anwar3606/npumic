import json
from types import SimpleNamespace

import pytest

from npumic import service as service_mod
from npumic.devices import Mic, pick_mic
from npumic.engine import Stage
from npumic.live import Stats

USB = Mic(name="usb", label="USB Headset", bluetooth=False, is_default=False, usb=True)
BUDS = Mic(name="buds", label="Buds", bluetooth=True, is_default=False)
BUILTIN = Mic(name="builtin", label="Built-in", bluetooth=False, is_default=False)


def test_pick_mic_prefers_the_chosen_one_then_usb_bluetooth_builtin():
    assert pick_mic([BUILTIN, BUDS, USB]) is USB
    assert pick_mic([BUILTIN, BUDS]) is BUDS
    assert pick_mic([BUILTIN]) is BUILTIN
    assert pick_mic([BUILTIN, USB], preferred="builtin") is BUILTIN
    assert pick_mic([BUILTIN, USB], preferred="gone") is USB
    muted = Mic(name="usb", label="USB", bluetooth=False, is_default=False, usb=True, muted=True)
    assert pick_mic([muted, BUILTIN]) is BUILTIN
    assert pick_mic([]) is None


class FakeModel(Stage):
    npu_broken = False
    npu_present = True

    def __init__(self, device):
        if device == "NPU" and self.npu_broken:
            raise RuntimeError("Level Zero error\nlong details")
        self.device = device if self.npu_present else "CPU"
        self.latency = 0

    def process_block(self, block):
        return block


class FakeVoxLock(FakeModel):
    name = "voxlock"

    def __init__(self, voice, device):
        super().__init__(device)


class FakeVoice:
    def __init__(self):
        self.ready = False

    def set(self, sample):
        self.ready = True


class FakeEngine:
    def __init__(self, stage):
        self.stage = stage
        self.stats = Stats()
        self.mic = None
        self.active = False
        self.latency = 480
        self.monitor = False
        self.monitor_output = None
        self.shutdown_reason = None

    def switch_mic(self, mic):
        self.mic = mic

    def set_active(self, on):
        self.active = on


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Fake models and audio system; returns a dict of the audio system's state."""
    state = {"mics": [BUILTIN, USB], "consumers": [], "default": "builtin", "defaults_set": []}
    FakeModel.npu_broken, FakeModel.npu_present = False, True
    monkeypatch.setattr(service_mod, "Denoise", FakeModel)
    monkeypatch.setattr(service_mod, "VoxLock", FakeVoxLock)
    monkeypatch.setattr(service_mod, "Voice", FakeVoice)
    monkeypatch.setattr(service_mod, "LiveEngine", FakeEngine)
    monkeypatch.setattr(service_mod, "list_mics", lambda: state["mics"])
    monkeypatch.setattr(service_mod, "consumers", lambda: state["consumers"])
    monkeypatch.setattr(service_mod, "default_source", lambda: state["default"])

    def set_default(name):
        state["defaults_set"].append(name)
        state["default"] = name
        return True

    monkeypatch.setattr(service_mod, "set_default_source", set_default)
    state["settings"] = tmp_path / "settings.json"
    state["voices"] = tmp_path / "voices"
    return state


def make_service(env):
    s = service_mod.Service(settings_path=env["settings"], voices_dir=env["voices"])
    s.health.flush()
    return s


def test_models_run_on_the_npu_without_problems(env):
    s = make_service(env)
    assert s.status()["noise_removal_device"] == "NPU" and s.status()["isolation_device"] == "NPU"
    assert s.health.problems == {}


def test_model_moves_to_the_cpu_when_the_npu_fails_to_load_it(env):
    FakeModel.npu_broken = True
    s = make_service(env)
    assert s.status()["noise_removal_device"] == "CPU"
    assert "runs on the CPU: it failed to load on the NPU (Level Zero error)" in s.health.problems["device-denoise"]


def test_missing_npu_is_reported(env):
    FakeModel.npu_present = False
    s = make_service(env)
    assert "no NPU found" in s.health.problems["device-denoise"]
    assert "no NPU found" in s.health.problems["device-voxlock"]


def test_processing_follows_apps_using_the_microphone(env, monkeypatch):
    monkeypatch.setattr(service_mod, "IDLE_AFTER_S", 0.0)
    s = make_service(env)
    s.check()
    assert s.engine.mic is USB and not s.engine.active
    env["consumers"] = ["Microsoft Edge"]
    s.check()
    assert s.engine.active and s.status()["apps"] == ["Microsoft Edge"]
    env["consumers"] = []
    s.check()  # starts the idle timer
    assert s.engine.active
    s.check()
    assert not s.engine.active
    s.health.flush()
    messages = [e["message"] for e in s.health.events]
    assert "An app is using the microphone: processing started" in messages
    assert "Microphone used by: Microsoft Edge" in messages
    assert "No app is using the microphone: processing paused" in messages


def test_hear_myself_output_is_named_as_such(env):
    s = make_service(env)
    s.engine.monitor_output = SimpleNamespace(name="alsa_output.speakers", label="Speakers")
    env["consumers"] = ["GNOME Settings", "alsa_output.speakers"]
    s.check()
    assert s.status()["apps"] == ["GNOME Settings", "Hear myself"]


def test_unplugged_microphone_is_replaced(env):
    s = make_service(env)
    s.check()
    assert s.engine.mic is USB
    env["mics"] = [BUILTIN]
    s.check()
    assert s.engine.mic is BUILTIN
    env["mics"] = []
    s.check()
    s.health.flush()
    assert s.engine.mic is None and s.health.problems["mic"] == "No microphone is connected"


def test_chosen_microphone_is_used_and_remembered(env):
    s = make_service(env)
    s.apply({"mic": "builtin"})
    assert s.engine.mic is BUILTIN
    assert json.loads(env["settings"].read_text())["mic"] == "builtin"
    with pytest.raises(ValueError):
        s.apply({"mic": "gone"})
    s.apply({"mic": None})
    assert s.engine.mic is USB


def test_isolation_needs_a_voice_sample(env):
    s = make_service(env)
    with pytest.raises(ValueError, match="voice sample"):
        s.apply({"isolation": True})
    s.voice.ready = True
    s.apply({"isolation": True})
    assert s.isolation.selector.selected == "voxlock" and s.status()["isolation"] is True


def test_settings_persist_across_restarts(env):
    s = make_service(env)
    s.apply({"noise_removal": False, "auto_level": False})
    s = make_service(env)
    assert s.denoise.enabled is False and s.level.enabled is False


def test_saved_isolation_without_a_voice_sample_is_reported(env):
    env["settings"].write_text(json.dumps({"isolation": True}))
    s = make_service(env)
    s._apply_isolation()
    s.health.flush()
    assert s.isolation.selector.selected == "off" and "no voice sample" in s.health.problems["voice"]


def test_default_microphone_is_taken_and_restored(env):
    s = make_service(env)
    s._take_default()
    assert env["default"] == service_mod.VIRTUAL_NAME and s.settings.previous_default == "builtin"
    s._take_default()  # restarting must not remember itself as the previous default
    assert s.settings.previous_default == "builtin"
    s._restore_default()
    assert env["default"] == "builtin"


def test_glitches_are_reported_and_cleared(env, monkeypatch):
    s = make_service(env)
    s.tick()
    s.engine.stats.late_cycles = 25
    s.tick()
    message = s.health.problems["glitches"]
    assert "10 or more in the last minute" in message
    events = len(s.health.events)
    s.engine.stats.late_cycles = 40
    s.tick()
    # A rising count keeps the same problem, so the GNOME extension notifies only once.
    assert s.health.problems["glitches"] == message
    assert len(s.health.events) == events
    monkeypatch.setattr(service_mod, "GLITCH_WINDOW_S", 0.0)
    s.tick()
    s.tick()
    assert "glitches" not in s.health.problems


def test_runtime_npu_failure_is_reported(env):
    s = make_service(env)
    fallback = s.denoise.inner
    fallback._on_failure(RuntimeError("device lost"))
    s.health.flush()
    assert "failed on the NPU (device lost); moving it to the CPU" in s.health.problems["device-denoise"]
    assert s.status()["noise_removal_device"] == "NPU"  # until the next block passes the input through
