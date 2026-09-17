"""TASK 3/8 — concurrency-safety tests for the bounded LLM batch helper.

``run_concurrent_map`` backs both ``full_verification`` and
``near_match_judge``'s batch loops. These tests exercise it directly with
small ``time.sleep`` calls — long enough to force genuine thread overlap
(the fast, instant-mock full-search tests never overlap in practice, so they
would not have caught the ``contextvars`` "context already entered" bug this
module was built to fix), short enough to keep the suite fast.
"""
from __future__ import annotations

import threading
import time

from app.services import search_profile
from app.services.llm import budget
from app.services.llm.concurrency import run_concurrent_map


def test_sequential_fallback_at_concurrency_1_is_strictly_in_order():
    calls: list[int] = []

    def work(x: int) -> int:
        calls.append(x)
        return x * 2

    out = run_concurrent_map([1, 2, 3], work, max_workers=1)
    assert out == [2, 4, 6]
    assert calls == [1, 2, 3]  # no threads involved at all


def test_single_item_never_uses_a_thread_pool():
    calls: list[int] = []
    out = run_concurrent_map([7], lambda x: calls.append(x) or x, max_workers=4)
    assert out == [7]
    assert calls == [7]


def test_concurrent_results_preserve_input_order_not_completion_order():
    def work(x: int) -> int:
        time.sleep(0.03 * (3 - x))  # item 0 finishes LAST, item 2 finishes FIRST
        return x

    out = run_concurrent_map([0, 1, 2], work, max_workers=3)
    assert out == [0, 1, 2]  # merged in INPUT order regardless


def test_batches_actually_overlap_without_crashing():
    """Regression: a shared contextvars.Context object cannot be .run() by two
    threads at once ("cannot enter context: already entered") — only
    reproduces when two tasks are genuinely in flight simultaneously."""
    active: list[int] = []
    max_concurrent = [0]
    lock = threading.Lock()

    def work(x: int) -> int:
        with lock:
            active.append(x)
            max_concurrent[0] = max(max_concurrent[0], len(active))
        time.sleep(0.05)
        with lock:
            active.remove(x)
        return x

    out = run_concurrent_map(list(range(6)), work, max_workers=3)
    assert sorted(out) == list(range(6))
    assert max_concurrent[0] > 1, "workers never actually overlapped — test is not exercising concurrency"


def test_llm_budget_enforced_across_worker_threads():
    """The budget is a contextvars-scoped dict; TASK 3 propagates the calling
    thread's context into every worker AND locks the read-modify-write, so the
    cap holds even when many threads race ``try_consume()`` at once."""
    budget.start_budget(2)
    try:
        results = run_concurrent_map(list(range(20)), lambda _x: budget.try_consume(), max_workers=8)
    finally:
        budget.clear_budget()
    assert sum(results) == 2


def test_search_profile_counters_thread_safe_under_concurrency():
    prof = search_profile.start()
    try:
        def work(_x: int) -> None:
            search_profile.incr("concurrency_test_counter")
            search_profile.record_call("concurrency_test_call", 1.0)

        run_concurrent_map(list(range(50)), work, max_workers=8)
        d = prof.as_dict()
    finally:
        search_profile.clear()
    assert d["counters"]["concurrency_test_counter"] == 50
    assert d["call_stats"]["concurrency_test_call"]["count"] == 50


def test_worker_exception_propagates_like_a_sequential_loop():
    def boom(x: int):
        if x == 1:
            raise ValueError("boom")
        return x

    try:
        run_concurrent_map([0, 1, 2], boom, max_workers=3)
    except ValueError as e:
        assert str(e) == "boom"
    else:
        raise AssertionError("expected the worker's exception to propagate")
