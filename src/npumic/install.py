"""Desktop integration: a systemd user service that starts the microphone at login, and the GNOME extension.

The service runs `npumic run` from this checkout's virtual environment, restarts
it after a failure, and stops with PipeWire (it restarts with it). Its log is in
the journal: journalctl --user -u npumic.

The GNOME extension (quick settings toggle) is copied, not linked, into the
user's extension folder: GNOME Shell deletes an extension folder's contents on
uninstall, following links.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from npumic.runtime import ROOT, ensure_models

UNIT_NAME = "npumic.service"
UNIT_PATH = Path.home() / ".config/systemd/user" / UNIT_NAME
EXTENSION_UUID = "npumic@anwar3606.github.io"
EXTENSION_SOURCE = ROOT / "gnome-extension" / EXTENSION_UUID
EXTENSION_TARGET = Path.home() / ".local/share/gnome-shell/extensions" / EXTENSION_UUID


def unit_text(executable: Path) -> str:
    return f"""[Unit]
Description=NPU Denoised Mic: noise removal on the Intel NPU as a virtual microphone
After=pipewire.service wireplumber.service
Wants=wireplumber.service
PartOf=pipewire.service

[Service]
Type=simple
ExecStart={executable} run
Environment=NPUMIC_HOME={ROOT}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=pipewire.service
"""


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=check)


def service_state() -> str:
    return _systemctl("is-active", UNIT_NAME, check=False).stdout.strip() or "unknown"


def install(extension: bool = True) -> None:
    executable = Path(sys.executable).parent / "npumic"
    if not executable.exists():
        raise SystemExit(f"npumic is not installed in this environment ({executable}); run: uv sync")
    ensure_models()
    UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_PATH.write_text(unit_text(executable))
    _systemctl("daemon-reload")
    _systemctl("enable", UNIT_NAME)
    _systemctl("restart", UNIT_NAME)
    print(f"Service installed and started: {UNIT_PATH}")
    print("  status: npumic status    log: journalctl --user -u npumic -f")
    if extension:
        if EXTENSION_TARGET.is_symlink():
            EXTENSION_TARGET.unlink()
        shutil.copytree(EXTENSION_SOURCE, EXTENSION_TARGET, dirs_exist_ok=True)
        print(f"GNOME extension installed: {EXTENSION_TARGET}")
        if subprocess.run(["gnome-extensions", "enable", EXTENSION_UUID], capture_output=True).returncode != 0:
            # The running shell does not know an extension installed after login; listing it
            # as enabled makes the shell load it at the next login.
            _enable_at_login(EXTENSION_UUID)
        print("  Its quick settings toggle appears after the next login (GNOME Shell loads extension code at login).")


def _enabled_extensions() -> list[str]:
    import ast

    out = subprocess.run(["gsettings", "get", "org.gnome.shell", "enabled-extensions"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return list(ast.literal_eval(out.removeprefix("@as "))) if out else []


def _enable_at_login(uuid: str) -> None:
    enabled = _enabled_extensions()
    if uuid not in enabled:
        subprocess.run(["gsettings", "set", "org.gnome.shell", "enabled-extensions", str(enabled + [uuid])], check=True)


def uninstall() -> None:
    if UNIT_PATH.exists():
        _systemctl("disable", "--now", UNIT_NAME, check=False)
        UNIT_PATH.unlink()
        _systemctl("daemon-reload")
        print(f"Service removed: {UNIT_PATH}")
    if EXTENSION_TARGET.exists() or EXTENSION_TARGET.is_symlink():
        subprocess.run(["gnome-extensions", "disable", EXTENSION_UUID], capture_output=True)
        if EXTENSION_TARGET.is_symlink():
            EXTENSION_TARGET.unlink()
        else:
            shutil.rmtree(EXTENSION_TARGET)
        print(f"GNOME extension removed: {EXTENSION_TARGET}")
