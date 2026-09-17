"""Batched concept-vs-career cross-encoder pass (mission STEP 3/4/13).

``score_candidate`` is PURE — it must never load an ML model. Any
concept-vs-career similarity signal it consults is precomputed here, ONCE per
search, in BATCHES: one ``CrossEncoder.predict`` call per distinct semantic
concept, covering every viable candidate at once (``batch_size=32`` inside the
model), instead of the old one-pair-per-candidate call inside the scoring
loop.

Results land in ``ctx.semantic_similarity[(person_id, concept)]`` — a value in
``[0, 1]`` (min-max normalised across the candidate set for that concept). It
is only ever an UNKNOWN-tier RANKING signal: a similarity hit never by itself
proves a criterion TRUE (STEP 13). Similarity is skipped entirely for a
concept once the deadline is spent — ranking then degrades gracefully rather
than the search hanging (STEP 9/10).
"""
from __future__ import annotations

import logging

from app.config import settings
from app.constants import CriterionType
from app.schemas import ParsedSearchQuery
from app.services import search_profile
from app.services.scoring import ProfileFacts, ScoringContext, _career_snippet

log = logging.getLogger("app.semantic_similarity")

#: semantic criterion types whose scorer consults a concept-vs-career similarity
#: signal (mirrors ``scoring._score_semantic_concept``).
_SIMILARITY_TYPES = {
    CriterionType.SEMANTIC_CONCEPT,
    CriterionType.PROFESSIONAL_CONCEPT,
    CriterionType.INDUSTRY_EXPERIENCE,
    CriterionType.ROLE_FUNCTION,
}


def semantic_concepts(parsed: ParsedSearchQuery) -> list[str]:
    """Distinct, non-empty concept strings the scorer would feed a cross-encoder
    — the parent concept plus every ANY_OF/ALL_OF value (each child keeps the
    parent's type in ``scoring._score_semantic_multi``)."""
    seen: dict[str, None] = {}
    for c in parsed.criteria:
        if c.type not in _SIMILARITY_TYPES:
            continue
        candidates = [c.concept, c.value, *(c.values or [])]
        for raw in candidates:
            v = (raw or "").strip()
            if v:
                seen.setdefault(v, None)
    return list(seen)


def compute_semantic_similarity(
    facts_list: list[ProfileFacts],
    parsed: ParsedSearchQuery,
    ctx: ScoringContext,
    *,
    deadline=None,
    db=None,
) -> None:
    """Populate ``ctx.semantic_similarity`` for every (viable candidate, semantic
    concept) pair, batched by concept. No-op when the reranker is disabled or
    there are no semantic concepts.

    TASK 5 — this is the single biggest measured NON-LLM search cost (a
    ``CrossEncoder.predict()`` call over every viable candidate, per concept —
    see the TASK 2 real-data baseline). A candidate's score for a given
    concept only depends on THEIR OWN career snippet + the concept text, so
    it is safely reusable across searches (``db`` given, TASK 5) — a cache hit
    only changes WHEN the score is computed, never what it is or how it is
    used (still a ranking signal only)."""
    if not settings.reranker_enabled or not facts_list:
        return
    concepts = semantic_concepts(parsed)
    if not concepts:
        return

    snippets: dict[str, str] = {}
    for f in facts_list:
        s = _career_snippet(f)
        if s:
            snippets[f.person.id] = s
    if not snippets:
        return

    from app.services.reranker import cross_encode

    prof = search_profile.current()
    cache = None
    if db is not None:
        from app.services import semantic_similarity_cache as cache
    for concept in concepts:
        if deadline is not None and deadline.expired():
            log.warning(
                "semantic similarity: deadline reached — %d/%d concepts skipped",
                len(concepts) - _done(concepts, concept), len(concepts),
            )
            return

        cached = cache.lookup(db, concept, snippets, model=settings.reranker_model) if cache else {}
        for pid, sc in cached.items():
            ctx.semantic_similarity[(pid, concept)] = sc
        miss_ids = [pid for pid in snippets if pid not in cached]
        if prof is not None:
            prof.incr("semantic_similarity_cache_hits", len(cached))

        if miss_ids:
            miss_texts = [snippets[pid] for pid in miss_ids]
            scores = cross_encode(concept, miss_texts)  # ONE batched predict for the miss set
            fresh = dict(zip(miss_ids, scores))
            for pid, sc in fresh.items():
                ctx.semantic_similarity[(pid, concept)] = sc
            if cache:
                cache.store(db, concept, fresh, snippets, model=settings.reranker_model)
        if prof is not None:
            prof.incr("semantic_similarity_concepts")


def _done(concepts: list[str], current: str) -> int:
    return concepts.index(current)
