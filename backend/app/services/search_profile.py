"""Lightweight per-search stage profiler (mission STEP 1).

A ``SearchProfile`` accumulates wall-clock time per named stage and integer
counters (``score_candidate_calls``, ``cross_encoder_calls``,
``cross_encoder_pairs``, ``embedding_calls`` …). It is published on a
``contextvars.ContextVar`` for the duration of one ``run_connection_search``
call so leaf modules (``reranker``, ``embeddings``, ``scoring``) can bump a
counter without threading the object through every signature.

Zero overhead when no profile is active (every helper is a cheap ``None``
check), so unit tests that call ``score_candidate`` / ``cross_encode``
directly are unaffected.
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from dataclasses import dataclass, field

_STAGES = (
    "query_interpretation",
    "query_embedding",
    "candidate_fetch",
    "bulk_fact_load",
    "company_classification",
    "hard_gate",
    "semantic_similarity",
    "prescore",
    "cross_encoder",
    "judge",
    "full_verification",
    "verification_validation",
    "verdict_cache_write",
    "judge_validation",
    "rescore",
    "rerank",
    "audit",
    "near_match",
    "reason_generation",
    "model_load",
    "persistence",
    "total",
)


@dataclass
class SearchProfile:
    """Mutated from the main search thread AND, once TASK 3 concurrency is in
    play, from bounded worker threads calling ``incr``/``stage`` concurrently
    (e.g. per-batch LLM call timing). ``_lock`` makes every mutation atomic —
    ``contextvars`` only controls which ``SearchProfile`` a thread sees, not
    thread-safety of the object itself, since a copied context still shares the
    SAME dict/counters by reference (mutable value, not re-created per copy)."""

    timings_ms: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    #: raw per-call duration samples for the concurrent LLM batch loops (TASK 2
    #: point 10/11) — keyed by a caller-chosen label, e.g. "full_verification".
    call_durations_ms: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + dt, 1)

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def record_call(self, label: str, duration_ms: float) -> None:
        """Record ONE LLM call's wall-clock duration under ``label`` (e.g.
        "full_verification") — thread-safe, called from worker threads."""
        with self._lock:
            self.call_durations_ms.setdefault(label, []).append(round(duration_ms, 1))

    def call_stats(self, label: str) -> dict:
        with self._lock:
            samples = list(self.call_durations_ms.get(label, []))
        if not samples:
            return {"count": 0, "total_ms": 0.0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(samples),
            "total_ms": round(sum(samples), 1),
            "avg_ms": round(sum(samples) / len(samples), 1),
            "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1),
        }

    def as_dict(self) -> dict:
        with self._lock:
            timings = {k: self.timings_ms.get(k, 0.0) for k in _STAGES if k in self.timings_ms}
            counters = dict(self.counters)
        return {
            "timings_ms": timings,
            "counters": counters,
            "call_stats": {label: self.call_stats(label) for label in self.call_durations_ms},
        }


_current: contextvars.ContextVar[SearchProfile | None] = contextvars.ContextVar(
    "search_profile", default=None
)


def start() -> SearchProfile:
    """Begin a fresh profile for this context and return it."""
    p = SearchProfile()
    _current.set(p)
    return p


def clear() -> None:
    _current.set(None)


def current() -> SearchProfile | None:
    return _current.get()


def incr(name: str, n: int = 1) -> None:
    """Bump a counter on the active profile, if any (no-op otherwise)."""
    p = _current.get()
    if p is not None:
        p.incr(name, n)


def record_call(label: str, duration_ms: float) -> None:
    """Record one LLM call's duration on the active profile, if any (no-op
    otherwise) — safe to call from a worker thread whose context was copied
    from the search thread (see ``app.services.llm.concurrency``)."""
    p = _current.get()
    if p is not None:
        p.record_call(label, duration_ms)
