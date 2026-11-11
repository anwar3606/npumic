"""Local control API and debug page of the running service.

Serves on 127.0.0.1 only:
  GET  /                    debug page (levels, spectrograms, resources, events)
  GET  /api/status          state, settings, problems and recent events (used by the GNOME extension)
  GET  /api/debug?since=N   status plus level meters, latest resource sample and
                            spectrogram columns from column N on
  GET  /api/history         resource samples of the last minute
  GET  /api/mics            connected microphones
  POST /api/settings        {"noise_removal": bool, "auto_level": bool, "isolation": bool,
                             "record_voice": true, "mic": node name or null (automatic),
                             "monitor": bool}; any subset

Nothing is measured for the debug page until it is opened: spectrograms are
computed per request and resource sampling starts with the first request for it.

POST requires Content-Type application/json, which browsers cannot send
cross-origin without a CORS preflight that this server never approves, so other
web pages cannot change settings. The Host header is checked against the bound
address to block DNS-rebinding reads.
"""

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from npumic.devices import list_mics

PAGE = Path(__file__).with_name("control.html")
DEFAULT_PORT = 5057


def _db(level: float) -> float:
    return round(20 * math.log10(level), 1) if level > 1e-6 else -120.0


class ControlServer:
    def __init__(self, service, port: int = DEFAULT_PORT):
        """`service` is the running Service."""
        self.service = service
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        threading.Thread(target=self.httpd.serve_forever, name="npumic-control", daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def debug(self, since: int | None) -> dict:
        service = self.service
        service.want_resources()
        engine = service.engine
        in_peak, out_peak = engine.stats.take_peaks()
        return {
            **service.status(),
            "in_db": _db(in_peak),
            "out_db": _db(out_peak),
            "resources": service.resources.latest(),
            "spectra": engine.spectrograms.since(since) if since is not None else None,
        }

    def mics(self) -> list[dict]:
        engine = self.service.engine
        current = engine.mic.name if engine.mic else None
        return [{"name": m.name, "label": m.label, "bluetooth": m.bluetooth, "usb": m.usb, "muted": m.muted,
                 "current": m.name == current} for m in list_mics()]

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: bytes, content_type: str, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, status: int = 200):
                self._send(json.dumps(obj).encode(), "application/json", status)

            def _host_ok(self) -> bool:
                return self.headers.get("Host") in (f"127.0.0.1:{server.port}", f"localhost:{server.port}")

            def do_GET(self):
                if not self._host_ok():
                    return self._json({"error": "forbidden"}, 403)
                url = urlparse(self.path)
                if url.path == "/":
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/status":
                    self._json(server.service.status())
                elif url.path == "/api/debug":
                    since = parse_qs(url.query).get("since", [None])[0]
                    self._json(server.debug(int(since) if since and since.isdigit() else None))
                elif url.path == "/api/history":
                    server.service.want_resources()
                    self._json(list(server.service.resources.history))
                elif url.path == "/api/mics":
                    self._json(server.mics())
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                if not self._host_ok() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self._json({"error": "forbidden"}, 403)
                if self.path != "/api/settings":
                    return self._json({"error": "not found"}, 404)
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("Settings must be a JSON object")
                    server.service.apply(body)
                except (ValueError, RuntimeError) as exc:
                    return self._json({"error": str(exc)}, 400)
                self._json(server.service.status())

        return Handler
