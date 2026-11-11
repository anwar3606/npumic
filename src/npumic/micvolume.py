"""Hardware input volume of the physical microphone.

Clipping happens in the microphone's analog-to-digital stage, before any
processing, and cannot be repaired afterwards. ClipGuard watches the share of
input samples at full scale and lowers the microphone's system input volume
(through WirePlumber's wpctl) until clipping stops. It only ever lowers the
volume; quiet microphones are handled by the digital automatic level.
"""

import re
import subprocess
import threading
import time

from npumic.devices import node_id

CLIP_LEVEL = 0.99
CLIP_FRACTION = 1e-3  # share of samples at full scale in one check interval that counts as clipping
CHECK_INTERVAL_S = 0.5
STEP = 0.8  # volume factor per reduction
MIN_VOLUME = 0.05
COOLDOWN_S = 1.0


def get_volume(node: int) -> float | None:
    result = subprocess.run(["wpctl", "get-volume", str(node)], capture_output=True, text=True)
    match = re.search(r"Volume:\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else None


def set_volume(node: int, volume: float) -> None:
    subprocess.run(["wpctl", "set-volume", str(node), f"{volume:.3f}"], capture_output=True, check=True)


class ClipGuard:
    """Background thread lowering the current microphone's input volume while its signal clips."""

    def __init__(self, engine, is_enabled, on_action=lambda message: None):
        """`engine` is a LiveEngine; `is_enabled` returns whether the guard may act;
        `on_action` receives a description of each volume change."""
        self.engine = engine
        self.is_enabled = is_enabled
        self.on_action = on_action
        self.last_action: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="npumic-clipguard", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        last_change = 0.0
        while not self._stop.wait(CHECK_INTERVAL_S):
            clipped, samples = self.engine.stats.take_clipping()
            mic = self.engine.mic
            if not (mic and samples and self.is_enabled()) or clipped / samples < CLIP_FRACTION:
                continue
            if time.monotonic() - last_change < COOLDOWN_S:
                continue
            try:
                node = node_id(mic.name)
                volume = get_volume(node) if node is not None else None
                if volume is None or volume <= MIN_VOLUME:
                    continue
                new = max(MIN_VOLUME, volume * STEP)
                set_volume(node, new)
            except (OSError, subprocess.SubprocessError, ValueError):
                continue
            last_change = time.monotonic()
            self.last_action = (f"{mic.label} was distorting: input volume lowered from "
                                f"{round(volume * 100)}% to {round(new * 100)}%")
            self.on_action(self.last_action)
