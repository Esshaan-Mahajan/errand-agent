"""Injectable time, so hold expiry and wall-clock limits are testable without sleeping."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

ISO_MINUTE = "%Y-%m-%dT%H:%M"


def fmt(dt: datetime) -> str:
    return dt.strftime(ISO_MINUTE)


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now().replace(microsecond=0)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class FakeClock:
    """Sleeping advances virtual time instantly; shared by the agent and the mock world."""

    def __init__(self, start: str = "2026-11-02T09:00"):
        self._now = parse(start)
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def monotonic(self) -> float:
        with self._lock:
            return self._now.timestamp()

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += timedelta(seconds=seconds)
