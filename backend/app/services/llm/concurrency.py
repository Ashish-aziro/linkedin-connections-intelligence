"""Bounded concurrency for independent LLM batch calls (TASK 3).

``full_verification`` and ``near_match_judge`` each drive a list of
independent batched Claude requests — independent because every batch carries
its own candidates and nothing downstream reads shared mutable state until
the caller merges results back in the main thread. That makes them safe to
run concurrently: a plain bounded ``ThreadPoolExecutor`` overlaps the network
wait of one HTTP call with another's.

Two things are NOT safe by default and need explicit handling here:

  * ``contextvars`` (``app.services.llm.budget`` / ``app.services.search_profile``)
    are NOT propagated into a ``ThreadPoolExecutor`` worker by Python — a
    fresh worker thread sees the *default* context, not the search thread's.
    Copying the calling context with ``contextvars.copy_context()`` and
    running the worker through it restores the same *bindings* the search
    thread has; the ``dict`` objects those vars point to are still shared by
    reference, so the budget/profile counters still need their own locks
    (see ``budget.py`` / ``search_profile.py``) — copying the context makes
    the worker *see* the right object, it does not make concurrent writes to
    it atomic.
  * a SQLAlchemy ``Session`` must never be touched from more than one thread.
    Neither caller passes a ``db`` handle into the per-batch call — both
    build every packet from already-bulk-loaded facts before entering this
    helper — so that risk does not apply here; do not add a ``db`` parameter
    to ``worker_fn`` without re-auditing that invariant.
"""
from __future__ import annotations

import contextvars
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

log = logging.getLogger("app.llm.concurrency")

T = TypeVar("T")
R = TypeVar("R")


def run_concurrent_map(
    items: list[T],
    worker_fn: Callable[[T], R],
    *,
    max_workers: int,
    profile_label: str | None = None,
) -> list[R]:
    """Run ``worker_fn(item)`` for every item in ``items``, honouring a
    concurrency cap, and return results in the SAME order as ``items``
    (deterministic merge — independent of completion order).

    ``max_workers <= 1`` (or a single item) runs fully sequentially in the
    calling thread — this is the exact pre-TASK-3 behaviour, kept as the
    literal code path (not just an equivalent one) so ``SEMANTIC_JUDGE_
    CONCURRENCY=1`` is a true, trustworthy baseline for benchmarking.

    An exception from ``worker_fn`` propagates to the caller exactly as it
    would from a plain sequential loop — a batch is never silently dropped
    into a "failed" outcome by this helper; that classification stays the
    caller's ``call_fn``'s job (matches the existing single-threaded design).
    """
    if not items:
        return []
    if max_workers <= 1 or len(items) <= 1:
        return [worker_fn(it) for it in items]

    from app.services import search_profile as _sp

    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {}
        for idx, item in enumerate(items):
            t_submit = time.perf_counter()

            def _run(it=item, _t_submit=t_submit):
                wait_ms = (time.perf_counter() - _t_submit) * 1000.0
                if profile_label:
                    _sp.record_call(f"{profile_label}_wait", wait_ms)
                t0 = time.perf_counter()
                try:
                    return worker_fn(it)
                finally:
                    if profile_label:
                        _sp.record_call(profile_label, (time.perf_counter() - t0) * 1000.0)

            # a fresh copy PER submission — a single contextvars.Context object
            # cannot be .run() by more than one thread at a time ("cannot enter
            # context: already entered"). Each copy is taken from the SAME
            # unmodified calling (main) thread context, so every worker still
            # sees identical bindings (the search's SearchProfile / LLM budget).
            ctx = contextvars.copy_context()
            future_to_idx[ex.submit(ctx.run, _run)] = idx
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()
    return results  # type: ignore[return-value]
