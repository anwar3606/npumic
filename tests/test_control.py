import http.client
import json

import pytest

from npumic import control
from npumic.devices import Mic
from npumic.live import Stats

HEADSET = Mic(name="headset", label="Headset", bluetooth=False, is_default=False, usb=True)
BUDS = Mic(name="buds", label="Buds", bluetooth=True, is_default=False)


class FakeResources:
    def __init__(self):
        self.started = False
        self.history = [{"t": 1.0, "cpu_pct": 3.0}]

    def latest(self):
        return self.history[-1]


class FakeEngine:
    def __init__(self):
        self.mic = HEADSET
        self.stats = Stats()
        self.spectrograms = None


class FakeService:
    def __init__(self):
        self.engine = FakeEngine()
        self.resources = FakeResources()
        self.applied = []
        self.resources_wanted = False

    def status(self):
        return {"state": "idle", "noise_removal": True, "problems": []}

    def apply(self, changes):
        if changes.get("mic") == "gone":
            raise ValueError("That microphone is not connected")
        self.applied.append(changes)

    def want_resources(self):
        self.resources_wanted = True


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(control, "list_mics", lambda: [HEADSET, BUDS])
    srv = control.ControlServer(FakeService(), port=0)
    srv.start()
    yield srv
    srv.stop()


def request(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
    hdrs = {"Host": f"127.0.0.1:{srv.port}", **(headers or {})}
    conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=hdrs)
    res = conn.getresponse()
    return res.status, json.loads(res.read() or b"null")


JSON = {"Content-Type": "application/json"}


def test_status_does_not_start_debug_measurements(server):
    status, data = request(server, "GET", "/api/status")
    assert status == 200 and data["state"] == "idle"
    assert not server.service.resources_wanted


def test_debug_adds_levels_and_resources(server):
    server.service.engine.stats.in_peak = 0.5
    status, data = request(server, "GET", "/api/debug")
    assert status == 200 and data["state"] == "idle"
    assert data["in_db"] == pytest.approx(-6.0, abs=0.1) and data["resources"]["cpu_pct"] == 3.0
    assert data["spectra"] is None and server.service.resources_wanted


def test_settings_are_passed_to_the_service(server):
    status, data = request(server, "POST", "/api/settings", {"isolation": True, "mic": "buds"}, JSON)
    assert status == 200 and data["state"] == "idle"
    assert server.service.applied == [{"isolation": True, "mic": "buds"}]


def test_invalid_settings_are_rejected(server):
    status, data = request(server, "POST", "/api/settings", {"mic": "gone"}, JSON)
    assert status == 400 and "not connected" in data["error"]
    status, _ = request(server, "POST", "/api/settings", [1, 2], JSON)
    assert status == 400 and server.service.applied == []


def test_mics_are_listed(server):
    _, data = request(server, "GET", "/api/mics")
    assert [(m["name"], m["current"], m["usb"]) for m in data] == [("headset", True, True), ("buds", False, False)]


def test_other_web_pages_cannot_change_settings(server):
    """A cross-origin form post arrives without a JSON content type; a DNS-rebinding request has a foreign Host."""
    status, _ = request(server, "POST", "/api/settings", {"noise_removal": False}, {"Content-Type": "text/plain"})
    assert status == 403
    status, _ = request(server, "POST", "/api/settings", {"noise_removal": False}, {**JSON, "Host": "evil.example:80"})
    assert status == 403
    status, _ = request(server, "GET", "/api/status", headers={"Host": "evil.example:80"})
    assert status == 403
    assert server.service.applied == []


def test_page_is_served(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request("GET", "/")
    res = conn.getresponse()
    assert res.status == 200 and b"NPU Denoised Mic" in res.read()
