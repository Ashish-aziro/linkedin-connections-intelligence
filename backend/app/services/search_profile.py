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
    "judge_validation",
    "rescore",
    "rerank",
    "audit",
    "reason_generation",
    "persistence",
    "total",
)


@dataclass
class SearchProfile:
    timings_ms: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @contextlib.contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + dt, 1)

    def incr(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def as_dict(self) -> dict:
        return {
            "timings_ms": {k: self.timings_ms.get(k, 0.0) for k in _STAGES if k in self.timings_ms},
            "counters": dict(self.counters),
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
