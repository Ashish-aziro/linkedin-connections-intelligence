"""Optional soft per-search LLM call budget (hardening PART 6).

A budget NEVER causes an incorrect result — it only stops SPENDING further
optional LLM calls once the cap is hit. Deterministic results already computed
stand; anything still unresolved stays UNKNOWN (never FALSE, never silently
treated as fully reviewed) and the judge/audit status becomes PARTIAL so the UI
can say verification was incomplete.

Scoped with a ``contextvars.ContextVar`` so it is safe across the sync call
chain of one search request without threading a parameter through every
judge/audit/reason function signature. ``search_service`` starts the budget
AFTER query interpretation (interpretation is foundational, not optional) and
clears it when the search is done.
"""
from __future__ import annotations

import contextvars
import threading

_budget_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar("llm_call_budget", default=None)

#: TASK 3 — worker threads that had the search thread's context copied into
#: them (``app.services.llm.concurrency.run_concurrent_map``) see the SAME
#: budget dict by reference (a context copy shares mutable values, it does not
#: deep-copy them), so concurrent ``try_consume()`` calls need a real lock —
#: the read-then-increment below is not atomic on its own.
_lock = threading.Lock()


def start_budget(max_calls: int) -> None:
    """``max_calls <= 0`` means unlimited (same convention as the other
    ``*_max_candidates`` / ``*_cap`` settings in this codebase)."""
    _budget_var.set(None if max_calls <= 0 else {"max": max_calls, "used": 0})


def clear_budget() -> None:
    _budget_var.set(None)


def try_consume() -> bool:
    """True if a call may proceed (and counts it against the budget); False
    when the budget is exhausted — the caller must NOT attempt the call."""
    b = _budget_var.get()
    if b is None:
        return True
    with _lock:
        if b["used"] >= b["max"]:
            return False
        b["used"] += 1
        return True


def used() -> int:
    b = _budget_var.get()
    return 0 if b is None else b["used"]


def remaining() -> int | None:
    b = _budget_var.get()
    return None if b is None else max(0, b["max"] - b["used"])


def active() -> bool:
    return _budget_var.get() is not None
