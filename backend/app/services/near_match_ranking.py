"""Near Match ranking (near-match design PART 10).

Deliberately NOT ``near_pool.sort(key=match_score)`` — a candidate who strongly
satisfies the query's PRIMARY intent should generally outrank one who merely
matches a secondary constraint exactly (an unrelated person who happens to
live in the requested city must not out-ride a strong professional match who
lives nearby). The weighting below encodes that priority order; it does not
special-case any query, location, or profession.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.constants import CriterionType, GeoRelation
from app.schemas import ParsedSearchQuery
from app.services.near_match_pool import NearCandidate, clamp01

#: GeoRelation -> how "close" that counts as, for ranking a geographic near
#: match against everything else (PART 10 item 3 — closeness of the failed
#: criterion). Generic, not query-specific.
_GEO_CLOSENESS = {
    GeoRelation.SAME_METRO: 1.0,
    GeoRelation.NEARBY_CITY: 0.8,
    GeoRelation.SAME_REGION: 0.55,
    GeoRelation.SAME_STATE_NOT_NEAR: 0.3,
    GeoRelation.FAR: 0.1,
    GeoRelation.UNKNOWN: 0.15,
}

_WEIGHTS = {
    "primary_intent": 0.35,
    "required_satisfied": 0.15,
    "closeness": 0.15,
    "llm_confidence": 0.15,
    "relevance": 0.10,
    "match_score": 0.10,
}


@dataclass
class RankedNearMatch:
    candidate: NearCandidate
    #: the candidate's validated near-match-judge verdict. Always a real dict
    #: for anything that ends up in ``rank_near_matches``'s return value —
    #: there is no "ranked but unverified" entry (near-match design PART 1).
    #: Typed Optional only because ``_closeness`` accepts the same shape
    #: defensively; never actually None on an item this module returns.
    verdict: dict | None
    rank_score: float


def _closeness(nc: NearCandidate, verdict: dict, parsed: ParsedSearchQuery) -> float:
    """Reads the geo relation the VALIDATOR already computed (``verdict
    ["geo_relation"]``) rather than recomputing it — ranking and validation
    can never disagree about how close a geographic near match actually is."""
    relaxed_id = verdict.get("relaxed_criterion_id") or (
        nc.failed_criterion_ids[0] if nc.failed_criterion_ids else ""
    )
    crit = next((c for c in parsed.criteria if c.id == relaxed_id), None)
    if crit is not None and crit.type == CriterionType.LOCATION:
        geo_relation = verdict.get("geo_relation")
        if geo_relation == GeoRelation.UNKNOWN and verdict.get("relation_source") == "llm_inference":
            # the LLM recognised a real-world adjacency structured data can't
            # see (e.g. a well-known suburb) — trust it at NEARBY_CITY
            # strength, no higher, and only because the validator already
            # accepted it (never trusted here for the first time).
            return _GEO_CLOSENESS[GeoRelation.NEARBY_CITY]
        return _GEO_CLOSENESS.get(geo_relation, 0.15)
    if crit is not None:
        comp = next((c for c in nc.scored.components if c.criterion_id == crit.id), None)
        if comp is not None:
            return comp.match_strength
    return 0.3  # unknown gap type — modest default, never zero (it still reached the pool)


def _required_satisfied_ratio(nc: NearCandidate, parsed: ParsedSearchQuery) -> float:
    total_required = sum(1 for c in parsed.criteria if c.required)
    if total_required == 0:
        return 1.0
    return max(0.0, (total_required - len(nc.failed_criterion_ids)) / total_required)


def rank_near_matches(
    pool: list[NearCandidate],
    validated_verdicts: dict[str, dict],
    parsed: ParsedSearchQuery,
) -> list[RankedNearMatch]:
    """Only candidates the near-match LLM judge validated as a useful near
    match are ranked and kept (near-match design PART 1, v2): a candidate with
    no validated verdict is skipped, full stop — this function never falls
    back to a heuristic/code-only recommendation. Every weighted term is
    ``clamp01``-ed before it is combined, so the final ``rank_score`` is
    always in [0, 1] and no single term can dominate purely from a scale
    mismatch (e.g. a raw 0-100 match_score outweighing a 0-1 confidence)."""
    ranked: list[RankedNearMatch] = []
    for nc in pool:
        verdict = validated_verdicts.get(nc.person.id)
        if verdict is None:
            continue
        score = (
            _WEIGHTS["primary_intent"] * clamp01(nc.primary_intent_strength)
            + _WEIGHTS["required_satisfied"] * clamp01(_required_satisfied_ratio(nc, parsed))
            + _WEIGHTS["closeness"] * clamp01(_closeness(nc, verdict, parsed))
            + _WEIGHTS["llm_confidence"] * clamp01(verdict.get("confidence"))
            + _WEIGHTS["relevance"] * clamp01(nc.local_relevance)
            + _WEIGHTS["match_score"] * clamp01(nc.scored.match_score / 100.0)
        )
        ranked.append(RankedNearMatch(candidate=nc, verdict=verdict, rank_score=round(clamp01(score), 4)))
    ranked.sort(key=lambda r: r.rank_score, reverse=True)
    return ranked
