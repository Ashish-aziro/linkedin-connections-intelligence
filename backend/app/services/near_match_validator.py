"""Evidence validation for the Near Match judge (near-match design PART 9).

Same principle as ``judge_validator`` elsewhere in this codebase: the LLM
understands meaning, Python verifies facts. Nothing from ``near_match_judge``
reaches the frontend until it passes here.

  * every evidence ref is checked against that candidate's own packet — an
    invented ref is dropped, never trusted
  * a ``useful_near_match=True`` with zero surviving evidence refs is rejected
  * ``relaxed_criterion_id`` must be a criterion this candidate is actually on
    record as failing/uncertain on — never an id the model merely echoed
    incorrectly or invented
  * ``geographic_adjacent`` requires ALL of: the relaxed criterion is
    genuinely LOCATION, the candidate actually has a stored location, and
    deterministic geo data (``geo.classify_relation_deterministic``) does not
    CONTRADICT the claim — a confident deterministic ``FAR`` beats an LLM
    'nearby_city' every time (near-match design PART 5: deterministic facts
    outrank LLM inference, never the reverse). A mislabeled-but-otherwise-
    grounded verdict is downgraded to ``other_relevant`` rather than discarded
    outright. The surviving relation's provenance is tracked as
    ``relation_source``: "deterministic" when geo data itself established it,
    "llm_inference" when only the model's own world knowledge did — an
    inferred adjacency is never presented as a verified geographic fact.
  * a verdict below ``NEAR_MATCH_MIN_CONFIDENCE`` is rejected
  * ``not_meaningful`` is always rejected regardless of the boolean flag
"""
from __future__ import annotations

import logging

from app.config import settings
from app.constants import CriterionType, GeoRelation, NearMatchRelation
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.geo import classify_relation_with_evidence
from app.services.judge_packet import packet_refs
from app.services.near_match_pool import NearCandidate

log = logging.getLogger("app.near_match")

#: deterministic relations compatible with an LLM's "geographic_adjacent"
#: claim — SAME_METRO/SAME_STATE_NOT_NEAR corroborate it (source=deterministic);
#: UNKNOWN means structured data simply couldn't resolve it, so the LLM's own
#: inference is kept but labeled as such (source=llm_inference). FAR actively
#: CONTRADICTS the claim and always wins.
#: redesign PART 4/8 — NEARBY_CITY (a real, distance-confirmed "verified
#: nearby city") gets the SAME "deterministic" confidence as SAME_METRO now
#: that it is backed by real haversine distance, not a guess. SAME_REGION and
#: UNKNOWN are deliberately NOT here (near-match design PART 5 / bug report
#: PART 8: "same state" / "bordering states" must never be silently equated
#: with "nearby") — they fall through to the llm_inference branch below,
#: exactly like before.
_GEO_COMPATIBLE = {GeoRelation.SAME_METRO, GeoRelation.NEARBY_CITY, GeoRelation.SAME_STATE_NOT_NEAR}


#: PART 9 diagnostics — every distinct reason a candidate does not become a
#: shown Near Match, PLUS the bookkeeping case (no pool/packet entry) handled
#: in ``validate_near_verdicts`` itself. Aggregate counts ONLY (never a name,
#: profile field, prompt, or evidence string) — safe to log and to return in
#: the API response's near-match metadata.
REJECTION_REASONS = (
    "llm_rejected",             # useful_near_match=false, or relation_type=not_meaningful
    "low_confidence",           # below settings.near_match_min_confidence
    "invalid_evidence",         # zero evidence_refs survived packet_refs validation
    "invalid_relaxed_criterion",  # relaxed_criterion_id isn't a known gap for this candidate
    "other",                    # missing pool/packet entry, or any future/unclassified reason
)
#: NOT a rejection reason — ``geographic_conflict`` counts how often
#: deterministic geo data contradicted an LLM "geographic_adjacent" claim.
#: The candidate is still ACCEPTED (downgraded to other_relevant, see
#: ``_validate_geo``), so this is tracked separately from ``REJECTION_REASONS``
#: rather than folded into the accept/reject tally.
_GEOGRAPHIC_CONFLICT_KEY = "geographic_conflict"


def validate_near_verdicts(
    raw_verdicts: dict[str, dict],
    packets_by_id: dict[str, dict],
    pool_by_id: dict[str, NearCandidate],
    parsed: ParsedSearchQuery,
) -> tuple[dict[str, dict], dict[str, int]]:
    """Returns ``({person_id: validated_verdict}, diagnostic_counts)``.
    Only candidates that remain a grounded, useful near match after
    validation appear in the first dict — rejected / ungrounded candidates
    are simply absent (never returned with a false ``useful_near_match``).
    ``diagnostic_counts`` (PART 9) is a safe aggregate — every key is one of
    ``REJECTION_REASONS`` or ``geographic_conflict``, values are counts only."""
    location_crit_by_id = {c.id: c for c in parsed.criteria if c.type == CriterionType.LOCATION}
    out: dict[str, dict] = {}
    reasons: dict[str, int] = {r: 0 for r in REJECTION_REASONS}
    reasons[_GEOGRAPHIC_CONFLICT_KEY] = 0
    accepted = 0
    for pid, raw in raw_verdicts.items():
        nc = pool_by_id.get(pid)
        packet = packets_by_id.get(pid)
        if nc is None or packet is None:
            reasons["other"] += 1
            continue
        v, reason, geo_conflict = _validate_one(raw, nc, packet, location_crit_by_id)
        if geo_conflict:
            reasons[_GEOGRAPHIC_CONFLICT_KEY] += 1
        if v is not None:
            out[pid] = v
            accepted += 1
        else:
            reasons[reason or "other"] = reasons.get(reason or "other", 0) + 1
    log.info("near_match_validator: accepted=%d rejected=%d reasons=%s",
             accepted, sum(v for k, v in reasons.items() if k != _GEOGRAPHIC_CONFLICT_KEY), reasons)
    return out, reasons


def _validate_one(
    raw: dict, nc: NearCandidate, packet: dict, location_crit_by_id: dict[str, SearchCriterion],
) -> tuple[dict | None, str | None, bool]:
    """Returns ``(verdict_or_none, rejection_reason_or_none, geographic_conflict)``.
    ``rejection_reason`` is one of ``REJECTION_REASONS`` when the verdict is
    ``None``, else ``None``. ``geographic_conflict`` can be True even when a
    verdict IS returned (a downgrade, not a rejection — see module docstring)."""
    notes: list[str] = []
    if not raw.get("useful_near_match"):
        return None, "llm_rejected", False
    relation_type = raw.get("relation_type") or NearMatchRelation.OTHER_RELEVANT
    if relation_type == NearMatchRelation.NOT_MEANINGFUL:
        return None, "llm_rejected", False

    confidence = float(raw.get("confidence") or 0.0)
    if confidence < settings.near_match_min_confidence:
        return None, "low_confidence", False

    valid_refs = packet_refs(packet)
    raw_refs = list(raw.get("evidence_refs") or [])
    refs = [r for r in raw_refs if r in valid_refs]
    if len(refs) < len(raw_refs):
        notes.append(f"dropped {len(raw_refs) - len(refs)} invalid evidence ref(s)")
    if not refs:
        return None, "invalid_evidence", False  # PART 9 — no grounded evidence -> rejected

    relaxed_id = raw.get("relaxed_criterion_id") or ""
    if relaxed_id not in nc.failed_criterion_ids:
        if len(nc.failed_criterion_ids) == 1:
            notes.append(f"relaxed_criterion_id corrected to the candidate's actual gap ({nc.failed_criterion_ids[0]!r})")
            relaxed_id = nc.failed_criterion_ids[0]
        else:
            notes.append("relaxed_criterion_id does not match a known gap for this candidate — rejected")
            return None, "invalid_relaxed_criterion", False

    relation_source: str | None = None
    geo_relation: str | None = None
    geo_conflict = False
    short_reason = (raw.get("short_reason") or "")[:240]
    if relation_type == NearMatchRelation.GEOGRAPHIC_ADJACENT:
        relation_type, relation_source, geo_relation, geo_notes = _validate_geo(
            nc, relaxed_id, location_crit_by_id,
        )
        notes.extend(geo_notes)
        # a deterministic FAR contradiction doesn't reject the recommendation
        # outright (the candidate may still be professionally relevant, just
        # not geographically) — it downgrades relation_type to other_relevant
        # (see ``_validate_geo``) AND is tracked as its own PART 9 signal so a
        # search-level report can see how often the LLM's geo claim conflicted
        # with structured data, without inspecting any prompt or profile.
        geo_conflict = geo_relation == GeoRelation.FAR
        if geo_conflict:
            # review finding F2 — the LLM's own narrative text ("a short
            # commute from X", "nearby suburb of Y") was written on the
            # belief the candidate was close; passing it through unchanged
            # would still show that claim next to the correct "Missing:
            # located in Y" bullet, actively contradicting it. Replace it
            # with a neutral, evidence-grounded statement instead of the
            # now-disproven LLM claim.
            short_reason = (
                "Does not satisfy the location requirement — deterministic "
                "location data places this candidate outside the requested area."
            )

    return {
        "useful_near_match": True,
        "confidence": round(confidence, 3),
        "satisfied_intent": (raw.get("satisfied_intent") or "")[:300],
        "relaxed_criterion_id": relaxed_id,
        "relation_type": relation_type,
        # provenance (near-match design PART 4/10): "deterministic" or
        # "llm_inference" when relation_type is geographic_adjacent, else None.
        # Ranking reads the exact underlying GeoRelation via ``geo_relation``
        # instead of recomputing it, so the two stages can never disagree.
        "relation_source": relation_source,
        "geo_relation": geo_relation,
        "evidence_refs": refs,
        "short_reason": short_reason,
        "source": nc.source,
        "validation_notes": notes,
    }, None, geo_conflict


def _validate_geo(
    nc: NearCandidate, relaxed_id: str, location_crit_by_id: dict[str, SearchCriterion],
) -> tuple[str, str | None, str | None, list[str]]:
    """Enforce PART 5's full geographic checklist. Returns
    ``(relation_type, relation_source, geo_relation, notes)`` —
    ``relation_type`` is downgraded to ``other_relevant`` (never rejects the
    whole recommendation outright — the candidate may still be professionally
    relevant) whenever the claim can't be substantiated or is actively
    contradicted."""
    crit = location_crit_by_id.get(relaxed_id)
    if crit is None:
        return NearMatchRelation.OTHER_RELEVANT, None, None, [
            "relation_type 'geographic_adjacent' claimed for a non-location gap — downgraded",
        ]

    person = nc.person
    cand_fields = {
        "city": getattr(person, "city", None),
        "state": getattr(person, "state", None),
        "location_text": getattr(person, "location_text", None),
    }
    if not any(cand_fields.values()):
        return NearMatchRelation.OTHER_RELEVANT, None, None, [
            "candidate has no stored location — 'geographic_adjacent' cannot be grounded, downgraded",
        ]

    wanted_values = crit.values or ([crit.value] if crit.value else [])
    if not wanted_values:
        return NearMatchRelation.OTHER_RELEVANT, None, None, [
            "query's location criterion has no value to compare against — downgraded",
        ]

    geo_evidence = classify_relation_with_evidence(cand_fields, wanted_values)
    relation = geo_evidence["relation"]
    if relation == GeoRelation.FAR:
        far_note = "deterministic geo data shows this candidate is FAR from the requested location"
        if geo_evidence.get("distance_miles") is not None:
            far_note += f" (~{geo_evidence['distance_miles']:.0f} miles, real distance)"
        return NearMatchRelation.OTHER_RELEVANT, None, relation, [
            far_note + " — LLM's 'geographic_adjacent' claim rejected, deterministic evidence wins",
        ]
    if relation in _GEO_COMPATIBLE:
        note = f"geographic_adjacent confirmed by structured location data ({relation})"
        if geo_evidence.get("distance_miles") is not None:
            note += f", ~{geo_evidence['distance_miles']:.0f} real miles"
        return NearMatchRelation.GEOGRAPHIC_ADJACENT, "deterministic", relation, [note]
    # UNKNOWN — structured data couldn't resolve it either way; the LLM's own
    # world knowledge is kept, but explicitly labeled as inferred, not verified.
    return NearMatchRelation.GEOGRAPHIC_ADJACENT, "llm_inference", relation, [
        "geographic_adjacent accepted from LLM world knowledge — no structured geo data available "
        "to confirm (labeled llm_inference, never treated as a verified fact)",
    ]
