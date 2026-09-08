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
) -> None:
    """Populate ``ctx.semantic_similarity`` for every (viable candidate, semantic
    concept) pair, batched by concept. No-op when the reranker is disabled or
    there are no semantic concepts."""
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

    ids = list(snippets)
    texts = [snippets[pid] for pid in ids]
    prof = search_profile.current()
    for concept in concepts:
        if deadline is not None and deadline.expired():
            log.warning(
                "semantic similarity: deadline reached — %d/%d concepts skipped",
                len(concepts) - _done(concepts, concept), len(concepts),
            )
            return
        scores = cross_encode(concept, texts)  # ONE batched predict for the whole pool
        for pid, sc in zip(ids, scores):
            ctx.semantic_similarity[(pid, concept)] = sc
        if prof is not None:
            prof.incr("semantic_similarity_concepts")


def _done(concepts: list[str], current: str) -> int:
    return concepts.index(current)
