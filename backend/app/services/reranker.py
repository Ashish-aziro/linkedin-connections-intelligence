"""Rerankers over the deterministic candidate list.

* ``cross_encode`` — local `sentence-transformers` CrossEncoder (no new dep, no
  network). Disabled or unavailable → returns zeros (a no-op that leaves ordering
  to the deterministic scorer), mirroring ``embeddings._hash_vector``.
* ``llm_rerank`` — opt-in single structured LLM call that reorders the final top 20
  and can drop clearly-wrong entries.
"""
from __future__ import annotations

import logging
import threading

from pydantic import BaseModel, Field

from app.config import settings

log = logging.getLogger("app.reranker")

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import time as _time

                from sentence_transformers import CrossEncoder

                log.info("loading cross-encoder %s", settings.reranker_model)
                t0 = _time.perf_counter()
                _model = CrossEncoder(settings.reranker_model)
                log.info("cross-encoder loaded in %.1fs", _time.perf_counter() - t0)
    return _model


def preload() -> None:
    """TASK 5 — load the cross-encoder now (app startup). See
    ``embeddings.preload`` — same no-op-once-loaded / never-raises contract."""
    if not settings.reranker_enabled:
        return
    try:
        _get_model()
    except Exception:  # noqa: BLE001
        log.exception("cross-encoder preload failed — will retry lazily on first use")


def cross_encode(query: str, texts: list[str]) -> list[float]:
    """Relevance score in [0,1] per text (min-max normalised across this batch)."""
    if not texts:
        return []
    if not settings.reranker_enabled:
        return [0.0] * len(texts)
    from app.services import search_profile

    search_profile.incr("cross_encoder_calls")
    search_profile.incr("cross_encoder_pairs", len(texts))
    try:
        raw = _get_model().predict(
            [(query, t or "") for t in texts], batch_size=32, show_progress_bar=False
        )
    except Exception:  # noqa: BLE001
        log.exception("cross-encoder unavailable — skipping rerank")
        return [0.0] * len(texts)

    lo, hi = min(raw), max(raw)
    if hi - lo < 1e-6:
        return [0.5] * len(texts)
    return [float((s - lo) / (hi - lo)) for s in raw]


# ─────────────────────────── LLM reranker (opt-in) ───────────────────────────


class _RerankItem(BaseModel):
    person_id: str
    keep: bool = True


class RerankResult(BaseModel):
    order: list[_RerankItem] = Field(default_factory=list)


_SYSTEM = (
    "You are re-ranking a shortlist of professional connections for a search query. "
    "Given the query and a numbered list of candidates (id + one-line profile), return "
    "the ids in the best order for the query, and set keep=false for any candidate that "
    "clearly does not belong. Do not invent ids. Output JSON only."
)


def llm_rerank(query: str, candidates: list[dict]) -> dict[str, object] | None:
    """``candidates``: [{person_id, line}]. Returns {"order": [ids...], "drop": {ids}}
    or ``None`` if the LLM is unavailable."""
    if not settings.llm_rerank_enabled or not candidates:
        return None
    from app.services.llm.router import generate_structured

    listing = "\n".join(f"{i+1}. [{c['person_id']}] {c['line']}" for i, c in enumerate(candidates))
    result = generate_structured(
        _SYSTEM,
        f"Query: {query!r}\n\nCandidates:\n{listing}\n\nReturn the reordered ids.",
        RerankResult,
        max_tokens=700,
        operation="llm_rerank",
    )
    if result is None:
        return None
    parsed = result[0]
    valid = {c["person_id"] for c in candidates}
    order = [it.person_id for it in parsed.order if it.person_id in valid]
    drop = {it.person_id for it in parsed.order if it.person_id in valid and not it.keep}
    if not order:
        return None
    return {"order": order, "drop": drop}
