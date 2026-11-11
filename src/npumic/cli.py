"""npumic: the clean virtual microphone."""

import argparse
import json
import logging
import signal
import sys
import time
import urllib.error
import urllib.request

from npumic.control import DEFAULT_PORT

log = logging.getLogger("npumic")


def _setup_logging() -> None:
    # The journal adds its own timestamps when running as a service.
    fmt = "%(asctime)s %(levelname)s %(message)s" if sys.stderr.isatty() else "%(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")


def cmd_run(args) -> int:
    from npumic.control import ControlServer
    from npumic.service import Service

    _setup_logging()
    log.info("Loading models")
    service = Service(make_default=not args.no_default)
    service.health.flush()
    # Binds the port now, so a busy port fails before going live.
    control = ControlServer(service, port=args.port)
    try:
        service.start(args.mic)
    except BaseException:
        service.stop(keep=args.keep)
        control.stop()
        raise
    control.start()
    log.info("Running: select \"NPU Denoised Mic\" in apps; controls and debug page at %s", control.url)

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    code = 0
    try:
        while not stop:
            time.sleep(1)
            service.tick()
            if service.engine.shutdown_reason:
                log.error("Audio system connection ended (%s); exiting", service.engine.shutdown_reason)
                code = 1
                break
    finally:
        control.stop()
        service.stop(keep=args.keep)
    return code


def _status(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def cmd_status(args) -> int:
    from npumic.install import service_state

    s = _status(args.port)
    if s is None:
        print(f"Not running (service: {service_state()}). Logs: journalctl --user -u npumic -n 50")
        return 1
    on = lambda v: "on" if v else "off"
    voice = s["voice"]
    print(f"State          : {s['state']} (used by: {', '.join(s['apps']) or 'no app'})")
    print(f"Microphone     : {s['mic_label'] or 'none'}" + ("" if s["mic_choice"] else " (chosen automatically)"))
    print(f"Noise removal  : {on(s['noise_removal'])}, on {s['noise_removal_device']}")
    print(f"Other voices   : {on(s['isolation'])}, on {s['isolation_device']}"
          + (f"; voice sample from {voice['mic']}, {voice['recorded']}" if voice["ready"] else "; no voice sample"))
    print(f"Automatic level: {on(s['auto_level'])}")
    print(f"Latency        : {s['latency_ms']} ms processing; {s['late_cycles']} late cycles, {s['xruns']} glitches")
    for problem in s["problems"]:
        print(f"Problem        : {problem}")
    if args.events:
        for event in s["events"]:
            print(time.strftime("%H:%M:%S", time.localtime(event["t"])), event["level"], event["message"])
    return 0


def cmd_install(args) -> int:
    from npumic.install import install

    install(extension=not args.no_extension)
    return 0


def cmd_uninstall(args) -> int:
    from npumic.install import uninstall

    uninstall()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="npumic", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the clean virtual microphone in the foreground")
    p.add_argument("--mic", help="part of a microphone's name: use it whenever connected (remembered)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="port of the control API and debug page on 127.0.0.1")
    p.add_argument("--no-default", action="store_true", help="do not make the virtual microphone the system default")
    p.add_argument("--keep", action="store_true", help="keep the (silent) virtual microphone after exit")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show the state of the running microphone")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--events", action="store_true", help="also list recent events")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("install", help="start at login (systemd user service) and add the GNOME quick settings toggle")
    p.add_argument("--no-extension", action="store_true", help="only install the service")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("uninstall", help="remove the service and the GNOME extension")
    p.set_defaults(func=cmd_uninstall)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
