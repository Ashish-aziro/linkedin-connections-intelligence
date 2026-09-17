"""TASK 5 — reusable concept-vs-career cross-encoder similarity cache.

See ``models.SemanticSimilarityCache`` for the reuse contract. This cache is
consulted for RANKING ONLY (``scoring._score_semantic_concept`` never treats
similarity as proof of a criterion, per V4 §13) — a stale or missing entry
never changes what a search decides, only how fast it decides it.
"""
from __future__ import annotations

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from sqlalchemy.orm import Session

from app.config import settings
from app.models import SemanticSimilarityCache

log = logging.getLogger("app.similarity_cache")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def concept_key(concept: str) -> str:
    return _hash(concept.strip().lower())


def evidence_fingerprint(career_snippet: str) -> str:
    return _hash(career_snippet)


def lookup(
    db: Session, concept: str, snippets_by_person: dict[str, str], *, model: str,
) -> dict[str, float]:
    """Bulk lookup for ONE concept across every candidate, main thread only,
    called before ``CrossEncoder.predict()``. Returns ``{person_id: score}``
    for every hit — a miss is simply absent."""
    if not settings.semantic_similarity_cache_enabled or not snippets_by_person:
        return {}
    ck = concept_key(concept)
    fp_by_person = {pid: evidence_fingerprint(s) for pid, s in snippets_by_person.items()}
    rows = db.scalars(
        select(SemanticSimilarityCache).where(
            SemanticSimilarityCache.person_id.in_(fp_by_person),
            SemanticSimilarityCache.concept_key == ck,
            SemanticSimilarityCache.model == model,
        )
    ).all()
    return {
        r.person_id: r.score for r in rows
        if r.evidence_fingerprint == fp_by_person.get(r.person_id)
    }


def store(
    db: Session, concept: str, scores_by_person: dict[str, float],
    snippets_by_person: dict[str, str], *, model: str,
) -> int:
    """Bulk upsert, main thread only, called right after a ``predict()`` call
    for this concept. Returns rows written; never raises — the cache is an
    optimization, a write failure must not fail the search."""
    if not settings.semantic_similarity_cache_enabled or not scores_by_person:
        return 0
    ck = concept_key(concept)
    from app.models_base import gen_id

    rows = [
        {
            "id": gen_id("simcache"),
            "person_id": pid,
            "concept_key": ck,
            "evidence_fingerprint": evidence_fingerprint(snippets_by_person[pid]),
            "model": model,
            "score": float(score),
        }
        for pid, score in scores_by_person.items()
        if pid in snippets_by_person
    ]
    if not rows:
        return 0
    stmt = sqlite_upsert(SemanticSimilarityCache).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["person_id", "concept_key", "evidence_fingerprint", "model"],
        set_={"score": stmt.excluded.score},
    )
    try:
        db.execute(stmt)
    except Exception:  # noqa: BLE001 — never fail a search over a cache write
        log.exception("similarity cache write failed — continuing without caching this concept")
        return 0
    return len(rows)
