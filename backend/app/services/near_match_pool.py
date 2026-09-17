"""Near Match candidate pool — the LOCAL, LLM-free retrieval + bounding stage
(near-match design PART 4/15; redesign PART 6/7 — multi-channel retrieval).

A Near Match is not "whatever was left over" — it is deliberately assembled
from three UPSTREAM sources (search_service.py), using signals the strict
pipeline already computed (no new embeddings, no extra DB round-trips):

  1. candidates the hard-fact gate rejected for exactly ONE verified
     contradiction (never semantically judged, per the product problem this
     module exists to fix — a location mismatch used to end the story before
     any professional-relevance reasoning happened)
  2. viable candidates whose deterministic rescore is NOT_MATCH (a required
     criterion is FALSE / unmet, but they passed the hard gate)
  3. candidates that qualified (EXACT/POSSIBLE) but were dropped by
     ``MIN_MATCH_SCORE``, or were downgraded by the final audit for
     insufficient evidence — borderline confidence, not a failed requirement

Redesign PART 6/7 — those upstream sources are then split into RETRIEVAL
CHANNELS (geographic / professional-adjacent / partial-requirement /
audit-downgrade / semantic-general) and allocated a bounded share of
``settings.near_match_candidate_pool`` each, so a single large generic
channel (typically "professional NOT_MATCH") cannot crowd out a small but
meaningful channel (e.g. a handful of geographically-nearby, professionally
strong candidates). Unused quota from an empty/small channel is redistributed
to the next-best candidates from OTHER channels — no channel is ever forced
to contribute, and total pool size never exceeds the configured cap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import settings
from app.constants import CriterionType
from app.schemas import ParsedSearchQuery
from app.services.scoring import ProfileFacts, ScoredCandidate, ScoringContext

log = logging.getLogger("app.near_match")

#: a candidate missing more than this many required criteria is not a
#: meaningful Near Match even at maximum relaxation (PART 4).
_MAX_RELAXABLE = 2
#: the primary-intent strength a 2-miss candidate needs to still be considered
#: (PART 4: "allow two missing requirements only when overall semantic
#: relevance is very strong and the primary intent remains clearly satisfied").
_TWO_MISS_INTENT_MIN = 0.6
#: below this, a candidate has essentially no connection to what the query is
#: actually about — not worth spending a judge call on regardless of source.
_MIN_ANY_SIGNAL = 0.05

#: redesign PART 6 — semantic/professional criterion types (mirrors
#: ``scoring._SEMANTIC_TYPES`` conceptually; imported lazily below to avoid a
#: circular import) used to route a candidate into the "professional_adjacent"
#: channel when that is the kind of criterion they failed.
_PROFESSIONAL_TYPES = {
    CriterionType.SEMANTIC_CONCEPT, CriterionType.PROFESSIONAL_CONCEPT,
    CriterionType.INDUSTRY_EXPERIENCE, CriterionType.ROLE_FUNCTION,
    CriterionType.COMPANY_CATEGORY,
}

#: redesign PART 6 — retrieval channels, in priority order for observability
#: (allocation itself is by configured share, not this order). Generic
#: categories, never query-specific.
CHANNELS = ("geographic", "professional_adjacent", "partial_requirement", "audit_downgrade", "semantic")

#: redesign PART 6/15 — configurable per-channel share of the total pool cap.
#: A channel with fewer eligible candidates than its share simply uses less;
#: the rest is redistributed (see ``_allocate_channels``), so this is a
#: FLOOR/target, never a hard quota that pads a channel with weak candidates.
CHANNEL_QUOTA_SHARE: dict[str, float] = {
    "geographic": 0.25,
    "professional_adjacent": 0.25,
    "partial_requirement": 0.20,
    "audit_downgrade": 0.15,
    "semantic": 0.15,
}


def clamp01(value: float | int | None) -> float:
    """Clamp any near-match ranking input to [0.0, 1.0] — shared so every
    signal that feeds ``near_match_ranking`` (which mixes a naturally 0-1
    scale like LLM confidence with things like a 0-100 match_score) is
    guaranteed comparable before weights are applied. Never raises: a bad
    value (None, NaN, a string) becomes 0.0 rather than corrupting a rank."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass
class NearCandidate:
    person: object
    facts: ProfileFacts
    scored: ScoredCandidate
    #: "hard_gate_reject" | "not_match" | "below_threshold" | "audit_downgrade" | "audit_downgrade_possible"
    source: str
    #: criterion ids this candidate actually failed / is uncertain on — the
    #: near-match judge's ``relaxed_criterion_id`` is validated against this
    #: set, never taken on faith (PART 9).
    failed_criterion_ids: list[str] = field(default_factory=list)
    primary_intent_strength: float = 0.0
    local_relevance: float = 0.0
    #: redesign PART 6 — which retrieval channel this candidate was routed
    #: through; observability + allocation only, never changes eligibility.
    channel: str = "semantic"


def _anchor_strength(scored: ScoredCandidate, anchor_ids: list[str]) -> float:
    if not anchor_ids:
        return 0.0
    by_id = {c.criterion_id: c.match_strength for c in scored.components}
    vals = [clamp01(by_id.get(cid, 0.0)) for cid in anchor_ids]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _local_relevance(person_id: str, scored: ScoredCandidate, ctx: ScoringContext) -> float:
    rel = clamp01(ctx.relevance_by_person.get(person_id, 0.0))
    return round(max(rel, clamp01(scored.match_score / 100.0)), 3)


def _is_unrelaxable_location_miss(nc: NearCandidate, crits_by_id: dict) -> bool:
    """True when the candidate's ONLY failure is a LOCATION criterion the
    query marked ``geo_strict`` — an explicit "strictly/only in X" that
    forbids a nearby-city substitute. Such a candidate must never be shown as
    a Near Match on the strength of geographic proximity (review finding F3
    / task step 4: "do not allow geographic expansion to satisfy an
    explicitly mandatory exact-location requirement")."""
    if len(nc.failed_criterion_ids) != 1:
        return False
    crit = crits_by_id.get(nc.failed_criterion_ids[0])
    return bool(crit and crit.type == CriterionType.LOCATION and crit.geo_strict)


def _eligible(nc: NearCandidate, crits_by_id: dict) -> bool:
    if _is_unrelaxable_location_miss(nc, crits_by_id):
        return False
    n_missing = len(nc.failed_criterion_ids)
    if n_missing == 0:
        return nc.primary_intent_strength >= _MIN_ANY_SIGNAL or nc.local_relevance >= _MIN_ANY_SIGNAL
    if n_missing == 1:
        return True
    if n_missing == 2:
        return nc.primary_intent_strength >= _TWO_MISS_INTENT_MIN
    return False


def _classify_channel(nc: NearCandidate, crits_by_id: dict, anchor_ids: set[str]) -> str:
    """Redesign PART 6 — route a candidate into ONE retrieval channel, purely
    from generic signals already computed (criterion TYPES of what they
    failed, and whether the primary intent still holds) — never a
    query-specific rule, never a person/city/profession name."""
    if nc.source in ("audit_downgrade", "audit_downgrade_possible"):
        return "audit_downgrade"
    failed_crits = [crits_by_id[cid] for cid in nc.failed_criterion_ids if cid in crits_by_id]
    if len(failed_crits) == 1 and failed_crits[0].type == CriterionType.LOCATION:
        return "geographic"
    if any(c.type in _PROFESSIONAL_TYPES for c in failed_crits):
        return "professional_adjacent"
    # missed only a SECONDARY (non-primary-intent-anchor) criterion while the
    # primary intent itself is still satisfied — a genuine partial match.
    if failed_crits and not (set(nc.failed_criterion_ids) & anchor_ids) and nc.primary_intent_strength >= _MIN_ANY_SIGNAL:
        return "partial_requirement"
    return "semantic"


def _select_trim_channel(quotas: dict[str, int], cap: int) -> str:
    """Deterministic, fair choice of which channel absorbs one unit of the
    rounding overflow — never a fixed positional bias (review finding F1: the
    previous ``max(quotas, key=quotas.get)`` tie-break returns the FIRST key
    in iteration order on a tie, and ``quotas`` is built by iterating
    ``CHANNELS`` in its literal declared order — "geographic" is always
    first, so it was always the channel silently zeroed out whenever the
    per-channel floors summed above ``cap``, even though it has an equal 25%
    floor and real eligible candidates).

    Channels still above the shared floor of 1 are preferred for trimming (a
    channel is only eliminated entirely as a last resort, once every other
    channel is already at its floor). Among the remaining candidates, the
    channel currently furthest ABOVE its own proportional ideal share
    (``cap * CHANNEL_QUOTA_SHARE``) gives up the slot — i.e. whichever
    channel rounding helped most generously, not whichever is listed first.
    A true tie (identical share, identical current quota) is broken
    alphabetically by channel name so the outcome never depends on iteration
    order."""
    with_slack = [ch for ch, q in quotas.items() if q > 1]
    pool = with_slack or list(quotas)
    return max(pool, key=lambda ch: (quotas[ch] - cap * CHANNEL_QUOTA_SHARE[ch], ch))


def _allocate_channels(
    by_channel: dict[str, list[NearCandidate]], cap: int,
) -> tuple[list[NearCandidate], dict[str, int]]:
    """Redesign PART 6/15 — bounded, diversity-aware selection. Each channel's
    candidates arrive PRE-SORTED (primary_intent_strength, local_relevance)
    descending. Every non-empty channel gets its configured share (rounded,
    at least 1 when it has ANY eligible candidate); unused capacity — from an
    empty channel, or a channel with fewer candidates than its share — is
    redistributed to the next-best remaining candidates from ANY channel, so
    the total never exceeds ``cap`` and a small channel is never padded with
    weak candidates just to hit a quota."""
    if cap <= 0:
        total = sum(len(v) for v in by_channel.values())
        selected = [nc for ch in CHANNELS for nc in by_channel.get(ch, [])]
        return selected, {ch: len(by_channel.get(ch, [])) for ch in CHANNELS} if total else {}

    quotas = {
        ch: min(len(by_channel.get(ch, [])), max(1, round(cap * CHANNEL_QUOTA_SHARE[ch])))
        for ch in CHANNELS if by_channel.get(ch)
    }
    # rounding can push the sum slightly over cap — trim one unit at a time
    # from whichever channel is currently furthest above its own fair share
    # (never the other way — never exceed cap; see _select_trim_channel for
    # why this must not be a fixed positional bias).
    while sum(quotas.values()) > cap:
        victim = _select_trim_channel(quotas, cap)
        quotas[victim] -= 1
        if quotas[victim] == 0:
            del quotas[victim]

    selected: list[NearCandidate] = []
    leftover: list[NearCandidate] = []
    selected_by_channel: dict[str, int] = {}
    for ch in CHANNELS:
        pool = by_channel.get(ch, [])
        q = quotas.get(ch, 0)
        selected.extend(pool[:q])
        selected_by_channel[ch] = q
        leftover.extend(pool[q:])

    remaining = cap - len(selected)
    if remaining > 0 and leftover:
        leftover.sort(key=lambda nc: (nc.primary_intent_strength, nc.local_relevance), reverse=True)
        extra = leftover[:remaining]
        selected.extend(extra)
        for nc in extra:
            selected_by_channel[nc.channel] = selected_by_channel.get(nc.channel, 0) + 1

    return selected, selected_by_channel


def build_near_pool(
    candidates: list[tuple],   # [(person, ProfileFacts, ScoredCandidate, source)]
    parsed: ParsedSearchQuery,
    ctx: ScoringContext,
) -> list[NearCandidate]:
    """``source`` is a retrieval provenance tag — purely for observability
    /channel routing, the eligibility logic treats every candidate the same
    way, driven entirely by ``scored.unmet_required_ids`` (populated
    identically by ``score_candidate`` regardless of which stage produced the
    candidate). Local retrieval keeps FULL recall (every eligible candidate
    from every channel is considered) — only the final, bounded selection
    (``_allocate_channels``) is capped, and that cap only bounds how many
    reach the expensive LLM stage, never which ones exist to consider."""
    anchor_ids = set(parsed.intent_anchor_criterion_ids)
    crits_by_id = {c.id: c for c in parsed.criteria}
    pool: list[NearCandidate] = []
    seen: set[str] = set()
    candidates_before_dedup = len(candidates)
    duplicates_dropped = 0

    for person, facts, scored, source in candidates:
        if person.id in seen:
            duplicates_dropped += 1
            continue  # a candidate audit-downgraded AND already hard-rejected — keep the first
        seen.add(person.id)
        failed = [] if source == "below_threshold" else list(scored.unmet_required_ids)
        nc = NearCandidate(person=person, facts=facts, scored=scored,
                           source=source, failed_criterion_ids=failed)
        nc.primary_intent_strength = _anchor_strength(scored, anchor_ids)
        nc.local_relevance = _local_relevance(person.id, scored, ctx)
        nc.channel = _classify_channel(nc, crits_by_id, anchor_ids)
        pool.append(nc)

    eligible = [nc for nc in pool if _eligible(nc, crits_by_id)]

    by_channel: dict[str, list[NearCandidate]] = {ch: [] for ch in CHANNELS}
    for nc in eligible:
        by_channel[nc.channel].append(nc)
    for ch in CHANNELS:
        by_channel[ch].sort(key=lambda nc: (nc.primary_intent_strength, nc.local_relevance), reverse=True)

    cap = max(0, settings.near_match_candidate_pool)
    bounded, selected_by_channel = _allocate_channels(by_channel, cap)
    bounded.sort(key=lambda nc: (nc.primary_intent_strength, nc.local_relevance), reverse=True)

    by_source: dict[str, int] = {}
    for nc in pool:
        by_source[nc.source] = by_source.get(nc.source, 0) + 1

    eligible_ids = {nc.person.id for nc in eligible}
    bounded_ids = {nc.person.id for nc in bounded}
    eligible_by_source: dict[str, int] = {}
    ineligible_by_source: dict[str, int] = {}
    pooled_by_source: dict[str, int] = {}
    for nc in pool:
        if nc.person.id in eligible_ids:
            eligible_by_source[nc.source] = eligible_by_source.get(nc.source, 0) + 1
        else:
            ineligible_by_source[nc.source] = ineligible_by_source.get(nc.source, 0) + 1
        if nc.person.id in bounded_ids:
            pooled_by_source[nc.source] = pooled_by_source.get(nc.source, 0) + 1

    candidates_per_channel_before_merge = {ch: len(by_channel[ch]) for ch in CHANNELS}

    # redesign PART 16 — safe aggregate diagnostics: counts only, never a
    # name, profile field, evidence string, or API key.
    log.info(
        "near_match_pool: candidates_before_deduplication=%d candidates_after_deduplication=%d "
        "duplicates_dropped=%d sources=%s eligible_by_source=%s ineligible_by_source=%s "
        "candidates_after_eligibility=%d candidates_after_pool_cap=%d cap=%d pooled_by_source=%s "
        "candidates_per_channel_before_merge=%s candidates_per_channel_selected=%s",
        candidates_before_dedup, len(pool), duplicates_dropped, by_source,
        eligible_by_source, ineligible_by_source, len(eligible), len(bounded), cap, pooled_by_source,
        candidates_per_channel_before_merge, selected_by_channel,
    )
    # legacy line (kept for any existing log-scraping/dashboards) — same
    # candidates_in/eligible/pooled/cap meaning as before this hardening pass.
    log.info(
        "near_match_pool: candidates_in=%d sources=%s eligible=%d pooled=%d cap=%d",
        len(pool), by_source, len(eligible), len(bounded), cap,
    )
    return bounded
