"""Warm the local ML models OUTSIDE the user's search request (V4 PART 6 B5).

The PART 6 benchmark measured a ~13 s SentenceTransformer load happening inside
the *first* search. This preloads MiniLM + the cross-encoder once, in a daemon
thread at app startup, so the models are process-cached before the first query.

Failure is logged and swallowed — the app still starts, and a search that races
the warmup simply pays the (unavoidable, one-time) load cost itself, exactly as
before. ``WARM_MODELS_ON_STARTUP=false`` disables it entirely (the test default —
tests must never download real model weights).
"""
from __future__ import annotations

import logging
import threading
import time

from app.config import settings

log = logging.getLogger("app.warmup")

_started = False
_lock = threading.Lock()
_thread: threading.Thread | None = None


def _warm() -> None:
    from app.services import embeddings, reranker

    for name, fn in (("embedding", embeddings.warm), ("reranker", reranker.warm)):
        try:
            t0 = time.perf_counter()
            fn()
            log.info("warmup: %s model ready (%.0f ms)", name, (time.perf_counter() - t0) * 1000.0)
        except Exception:  # noqa: BLE001
            log.warning("warmup: %s model preload failed — first search will load it", name)


def start(*, block: bool = False) -> None:
    """Begin warmup once per process. ``block=True`` runs it synchronously
    (tests). Safe to call more than once — subsequent calls are no-ops."""
    global _started, _thread
    with _lock:
        if _started:
            return
        _started = True
    if not settings.warm_models_on_startup:
        log.info("warmup: disabled (WARM_MODELS_ON_STARTUP=false)")
        return
    if block:
        _warm()
        return
    _thread = threading.Thread(target=_warm, name="model-warmup", daemon=True)
    _thread.start()


def ready() -> dict:
    from app.services import embeddings, reranker

    return {
        "embedding_model_ready": embeddings.model_ready(),
        "reranker_model_ready": reranker.model_ready(),
    }


def _reset_for_tests() -> None:
    global _started, _thread
    _started = False
    _thread = None
