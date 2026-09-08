"""Per-search wall-clock budget (hardening PART 14).

A very broad / difficult query must return USEFUL, transparently PARTIAL
results within a bounded time instead of hanging — never by killing
deterministic scoring (which is fast and always runs to completion), only by
skipping OPTIONAL LLM work (further judge batches, the final audit, reason
generation) once the budget is spent. ``SEARCH_MAX_SECONDS <= 0`` disables the
deadline entirely (unlimited), which is what tests use by default.
"""
from __future__ import annotations

import time


class Deadline:
    def __init__(self, seconds: float | None):
        # perf_counter, not monotonic: it is also monotonic but has the highest
        # available resolution on every platform. On Windows time.monotonic()
        # can have a ~15 ms granularity, coarse enough that a sub-second search
        # never observes the deadline as expired (mission STEP 9).
        self._start = time.perf_counter()
        self.seconds = seconds if seconds and seconds > 0 else None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)

    def remaining(self) -> float | None:
        """Seconds left, or ``None`` when unlimited."""
        if self.seconds is None:
            return None
        return self.seconds - (time.perf_counter() - self._start)

    def expired(self) -> bool:
        r = self.remaining()
        return r is not None and r <= 0

    def as_dict(self) -> dict:
        return {
            "seconds": self.seconds,
            "elapsed_ms": self.elapsed_ms(),
            "reached": self.expired(),
        }
