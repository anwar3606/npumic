"""Current problems and recent events of the running microphone.

Problems are conditions that need attention while they last (a model running
on the CPU because the NPU failed, no microphone connected); events are a short
history of what happened. Both are shown on the control page and in the GNOME
extension, and written to the log (the systemd journal when run as a service).

Reports may come from any thread, including the real-time audio callback, so
reporting only appends to a queue; `flush()`, called about once a second from
the main thread, applies the reports and writes the log.
"""

import logging
import time
from collections import deque

log = logging.getLogger("npumic")

MAX_EVENTS = 100


class Health:
    def __init__(self):
        self.problems: dict[str, str] = {}  # replaced as a whole on change, so readers see a consistent dict
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self._pending: deque[tuple] = deque()

    def problem(self, key: str, message: str) -> None:
        """Starts or updates the problem `key`."""
        self._pending.append(("problem", key, message, time.time()))

    def clear(self, key: str, message: str | None = None) -> None:
        """Ends the problem `key`, logging `message` if it was active."""
        self._pending.append(("clear", key, message, time.time()))

    def note(self, message: str, level: int = logging.INFO) -> None:
        self._pending.append(("note", level, message, time.time()))

    def _event(self, t: float, level: int, message: str) -> None:
        log.log(level, message)
        self.events.append({"t": round(t, 1), "level": logging.getLevelName(level).lower(), "message": message})

    def flush(self) -> None:
        while self._pending:
            kind, key, message, t = self._pending.popleft()
            if kind == "note":
                self._event(t, key, message)
            elif kind == "problem":
                if self.problems.get(key) != message:
                    self.problems = {**self.problems, key: message}
                    self._event(t, logging.WARNING, message)
            elif key in self.problems:
                self.problems = {k: v for k, v in self.problems.items() if k != key}
                if message:
                    self._event(t, logging.INFO, message)
