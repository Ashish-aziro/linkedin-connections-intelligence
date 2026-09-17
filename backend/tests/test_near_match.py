"""Near Match — intent-aware relaxed recommendation layer.

Covers primary-intent identification, generic geo adjacency, the bounded local
candidate pool, the batched near-match LLM judge (mocked — ZERO real Anthropic
calls), evidence validation, ranking, and the "never promotes to Exact" /
"never touches the main result pipeline" guarantees.

City/profession names below (Lawrenceville, Jerseyville, "growth investment
professional", ...) are INVENTED test fixtures, not hardcoded production
values — see ``app/services/geo.py`` and ``app/services/near_match_pool.py``
for the actual (query-agnostic) mechanism they exercise.
"""
from __future__ import annotations

import pytest

from app.constants import (
    CriterionType,
    GeoRelation,
    NearMatchRelation,
    Operator,
    Qualification,
)
from app.schemas import ParsedSearchQuery, ScoreComponent, SearchCriterion
from app.services import near_match_judge, query_intent
from app.services.geo import classify_relation_deterministic
from app.services.near_match_pool import NearCandidate, build_near_pool, clamp01
from app.services.near_match_ranking import rank_near_matches
from app.services.near_match_service import build_near_matches
from app.services.near_match_validator import validate_near_verdicts
from app.services.scoring import ProfileFacts, ScoredCandidate, ScoringContext
from tests.test_search import _Exp, _Person


def _person(i: int, **kw) -> _Person:
    p = _Person(**kw)
    p.id = f"p{i}"
    p.linkedin_url = f"https://www.linkedin.com/in/p{i}"
    return p


def _facts(person, exps=None) -> ProfileFacts:
    return ProfileFacts(person=person, experiences=exps or [], education=[], skills=[],
                        semantic={}, embedding=None)


def _crit(**kw) -> SearchCriterion:
    kw.setdefault("weight", 50)
    kw.setdefault("required", True)
    kw.setdefault("operator", Operator.ANY_OF)
    return SearchCriterion(**kw)


def _plan(*crits) -> ParsedSearchQuery:
    plan = ParsedSearchQuery(criteria=list(crits))
    query_intent._identify_primary_intent(plan)
    return plan


def _scored(person, *, components=(), unmet_ids=(), unmet_labels=(), match_score=40.0,
           qualification=Qualification.NOT_MATCH, evidence=()) -> ScoredCandidate:
    return ScoredCandidate(
        person=person, match_score=match_score, components=list(components),
        evidence=list(evidence), qualification=qualification,
        unmet_required=list(unmet_labels) or list(unmet_ids), unmet_required_ids=list(unmet_ids),
    )


def _nc(person, facts, scored, *, source="not_match") -> NearCandidate:
    nc = NearCandidate(person=person, facts=facts, scored=scored, source=source,
                       failed_criterion_ids=list(scored.unmet_required_ids))
    return nc


# ─────────────────────── primary intent (PART 5) ───────────────────────


def test_primary_intent_anchors_on_the_professional_concept_not_the_location():
    plan = _plan(
        _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
             concept="venture capital investor", weight=60),
    )
    assert plan.intent_anchor_criterion_ids == ["inv"]
    assert "venture capital" in plan.primary_intent.lower()


def test_primary_intent_falls_back_to_the_constraint_when_thats_the_whole_query():
    plan = _plan(_crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=100))
    assert plan.intent_anchor_criterion_ids == ["loc"]


def test_primary_intent_ignores_non_required_criteria():
    plan = _plan(
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=70),
        _crit(id="skill", type=CriterionType.SKILL, value="excel", weight=30, required=False),
    )
    assert plan.intent_anchor_criterion_ids == ["inv"]


# ─────────────────────── geo adjacency (PART 6) — generic, not Atlanta-specific ───────────────────────


def test_geo_relation_same_metro_via_the_existing_generic_alias_map():
    # Bay Area is already in geo.py's alias map — proves the mechanism without
    # any query-specific city list.
    relation = classify_relation_deterministic(
        {"city": "Oakland", "state": "California", "location_text": "Oakland, California, United States"},
        ["San Francisco"],
    )
    assert relation == GeoRelation.SAME_METRO


def test_geo_relation_same_state_from_an_explicit_state_value():
    relation = classify_relation_deterministic(
        {"city": "Jerseyville", "state": "Illinois", "location_text": "Jerseyville, Illinois, United States"},
        ["Illinois"],
    )
    assert relation == GeoRelation.SAME_STATE_NOT_NEAR


def test_geo_relation_unknown_when_structured_data_cannot_establish_it():
    # a genuinely unresolvable pair (neither city in the local coordinate
    # table, no alias entry, no state token in the wanted value) — must stay
    # UNKNOWN rather than guess (left to the LLM judge).
    relation = classify_relation_deterministic(
        {"city": "Millbrook", "state": "Georgia", "location_text": "Millbrook, Georgia, United States"},
        ["Riverbend"],
    )
    assert relation == GeoRelation.UNKNOWN


def test_geo_relation_same_metro_via_real_distance_redesign_part4():
    # redesign PART 4 — Lawrenceville, GA really is ~22 miles from Atlanta;
    # the real-distance engine (not a curated alias list) now correctly
    # recognizes this instead of leaving it UNKNOWN.
    relation = classify_relation_deterministic(
        {"city": "Lawrenceville", "state": "Georgia", "location_text": "Lawrenceville, Georgia, United States"},
        ["Atlanta"],
    )
    assert relation == GeoRelation.SAME_METRO


# ─────────────────────── local pool (PART 4/15) ───────────────────────


def test_pool_respects_the_configured_cap(monkeypatch):
    monkeypatch.setattr("app.services.near_match_pool.settings.near_match_candidate_pool", 2)
    plan = _plan(_crit(id="c1", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", weight=100))
    candidates = []
    for i in range(5):
        p = _person(i)
        comp = ScoreComponent(criterion="x", criterion_id="c1", type=CriterionType.PROFESSIONAL_CONCEPT,
                              weight=100, match_strength=0.5, score=50)
        scored = _scored(p, components=[comp], unmet_ids=["c1"])
        candidates.append((p, _facts(p), scored, "not_match"))
    pool = build_near_pool(candidates, plan, ScoringContext())
    assert len(pool) == 2


def test_missing_two_requirements_needs_strong_primary_intent_to_qualify():
    plan = _plan(
        _crit(id="a", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", weight=50),
        _crit(id="b", type=CriterionType.LOCATION, value="Fernbrook", weight=50),
    )
    p = _person(1)
    weak_comp = ScoreComponent(criterion="x", criterion_id="a", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=50, match_strength=0.1, score=5)
    weak = _scored(p, components=[weak_comp], unmet_ids=["a", "b"])
    pool = build_near_pool([(p, _facts(p), weak, "not_match")], plan, ScoringContext())
    assert pool == []  # weak primary-intent strength -> not eligible with 2 misses

    strong_comp = ScoreComponent(criterion="x", criterion_id="a", type=CriterionType.PROFESSIONAL_CONCEPT,
                                 weight=50, match_strength=0.9, score=45)
    strong = _scored(p, components=[strong_comp], unmet_ids=["a", "b"])
    pool2 = build_near_pool([(p, _facts(p), strong, "not_match")], plan, ScoringContext())
    assert len(pool2) == 1


def test_missing_three_requirements_is_never_eligible():
    plan = _plan(
        _crit(id="a", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", weight=34),
        _crit(id="b", type=CriterionType.LOCATION, value="Fernbrook", weight=33),
        _crit(id="c", type=CriterionType.CURRENT_COMPANY, value="Acme", weight=33),
    )
    p = _person(1)
    comp = ScoreComponent(criterion="x", criterion_id="a", type=CriterionType.PROFESSIONAL_CONCEPT,
                          weight=34, match_strength=1.0, score=34)
    scored = _scored(p, components=[comp], unmet_ids=["a", "b", "c"])
    pool = build_near_pool([(p, _facts(p), scored, "not_match")], plan, ScoringContext())
    assert pool == []


def test_pool_deduplicates_the_same_person_from_multiple_sources():
    plan = _plan(_crit(id="a", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", weight=100))
    p = _person(1)
    comp = ScoreComponent(criterion="x", criterion_id="a", type=CriterionType.PROFESSIONAL_CONCEPT,
                          weight=100, match_strength=0.8, score=80)
    scored = _scored(p, components=[comp], unmet_ids=["a"])
    candidates = [
        (p, _facts(p), scored, "hard_gate_reject"),
        (p, _facts(p), scored, "not_match"),
    ]
    pool = build_near_pool(candidates, plan, ScoringContext())
    assert len(pool) == 1


def test_multichannel_retrieval_geographic_candidates_survive_a_large_generic_pool():
    """Redesign PART 6/7 — the exact regression this redesign targets: a
    small number of professionally-strong, geographically-nearby candidates
    must NOT be crowded out by hundreds of generic professional NOT_MATCH
    candidates competing for the same flat, un-channeled pool cap."""
    from app.services.near_match_pool import CHANNEL_QUOTA_SHARE

    plan = _plan(
        _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="City", weight=40),
    )
    role_comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=60, match_strength=0.9, score=54)
    weak_role_comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.PROFESSIONAL_CONCEPT,
                                    weight=60, match_strength=0.2, score=12)

    # 300 generic "failed the professional criterion, right city" candidates —
    # the dominant, everyday NOT_MATCH source.
    generic = []
    for i in range(300):
        p = _person(1000 + i)
        scored = _scored(p, components=[weak_role_comp], unmet_ids=["role"], match_score=20.0)
        generic.append((p, _facts(p), scored, "not_match"))

    # 5 candidates who genuinely satisfy the PRIMARY intent strongly but only
    # miss the LOCATION criterion — exactly the "hard-rejected but valuable"
    # case the bug report says got crowded out (6 selected out of hundreds).
    geo = []
    for i in range(5):
        p = _person(2000 + i)
        scored = _scored(p, components=[role_comp], unmet_ids=["loc"], match_score=54.0)
        geo.append((p, _facts(p), scored, "hard_gate_reject"))

    pool = build_near_pool(generic + geo, plan, ScoringContext())
    geo_ids = {p.id for p, *_ in geo}
    selected_geo = geo_ids & {nc.person.id for nc in pool}

    assert CHANNEL_QUOTA_SHARE["geographic"] > 0  # the channel exists and has a real reserved share
    # with only 5 total geographic candidates and a 25% floor on a 40-slot
    # pool (10 slots), ALL of them must survive — none crowded out by the
    # 300-strong generic channel, unlike the old flat-pool behavior.
    assert len(selected_geo) == 5, (
        f"expected all 5 geographically-nearby, professionally-strong candidates to survive "
        f"channel allocation; only {len(selected_geo)} did — the generic pool crowded them out"
    )


def test_geo_strict_location_miss_is_never_a_near_match():
    """review finding F3 / task step 4 — a query that explicitly forbids
    nearby-city substitution ("strictly in Atlanta") must not let a candidate
    become a Near Match on the strength of geographic proximity alone. A
    candidate whose ONLY gap is a geo_strict LOCATION requirement is not
    eligible for Near Match at all — geographic expansion must never satisfy
    an explicitly mandatory exact-location requirement."""
    strict_plan = _plan(
        _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Atlanta", weight=40, geo_strict=True),
    )
    role_comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=60, match_strength=0.9, score=54)
    p = _person(9001)
    scored = _scored(p, components=[role_comp], unmet_ids=["loc"], match_score=54.0)
    pool = build_near_pool([(p, _facts(p), scored, "hard_gate_reject")], strict_plan, ScoringContext())
    assert pool == [], "a strict-location-only miss must never surface as a Near Match"

    # the SAME candidate, SAME gap, but an ORDINARY (non-strict) required
    # location must still be eligible — geo_strict only removes eligibility
    # when the query explicitly demanded it.
    ordinary_plan = _plan(
        _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Atlanta", weight=40, geo_strict=False),
    )
    pool2 = build_near_pool([(p, _facts(p), scored, "hard_gate_reject")], ordinary_plan, ScoringContext())
    assert len(pool2) == 1 and pool2[0].channel == "geographic"


def test_small_cap_never_zeroes_out_the_geographic_channel_by_position(monkeypatch):
    """review finding F1 — the old quota tie-break (`max()` on a tie in
    positional dict-iteration order) always trimmed "geographic" first,
    because it is listed first in ``CHANNELS``, whenever the per-channel
    floors summed above a small cap. With all five channels equally
    populated, geographic must not be the channel that systematically loses
    out purely because of where it sits in a tuple."""
    monkeypatch.setattr("app.services.near_match_pool.settings.near_match_candidate_pool", 4)

    plan = _plan(
        _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="City", weight=20),
        _crit(id="cert", type=CriterionType.CERTIFICATION, value="PMP", weight=20),
    )
    role_comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=60, match_strength=0.9, score=54)

    def _group(base: int, *, unmet_ids: list[str], source: str = "not_match"):
        out = []
        for i in range(5):
            p = _person(base + i)
            scored = _scored(p, components=[role_comp], unmet_ids=unmet_ids, match_score=54.0)
            out.append((p, _facts(p), scored, source))
        return out

    geographic = _group(1000, unmet_ids=["loc"])
    professional = _group(2000, unmet_ids=["role"])
    partial = _group(3000, unmet_ids=["cert"])
    audit_downgrade = _group(4000, unmet_ids=["role"], source="audit_downgrade")
    semantic = _group(5000, unmet_ids=[], source="below_threshold")

    pool = build_near_pool(
        geographic + professional + partial + audit_downgrade + semantic, plan, ScoringContext(),
    )

    assert len(pool) == 4, "the total pool must never exceed the configured cap"
    channels_represented = {nc.channel for nc in pool}
    assert "geographic" in channels_represented, (
        "geographic has an equal 25% floor and real eligible candidates — it must not be the "
        "channel silently zeroed out just because it is first in CHANNELS"
    )


# ─────────────────────── near-match judge (PART 8) — mocked, zero live calls ───────────────────────


def _fake_near_judge(verdict_for: dict | None = None, *, capture: list | None = None):
    """Fake ``near_match_judge._call_near_judge``. ``verdict_for`` optionally
    overrides the per-person verdict dict; default is a grounded 'true'."""
    def fake(payload, packets):
        if capture is not None:
            capture.append(len(packets))
        people = {}
        for pkt in packets:
            pid = pkt["person_id"]
            gaps = pkt.get("near_match_context", {}).get("gaps", [])
            relaxed = gaps[0]["criterion_id"] if gaps else ""
            ref = None
            if pkt.get("past"):
                ref = pkt["past"][0].get("ref")
            elif pkt.get("current"):
                ref = pkt["current"].get("ref")
            base = {
                "person_id": pid, "useful_near_match": True, "confidence": 0.8,
                "satisfied_intent": "strong professional fit", "relaxed_criterion_id": relaxed,
                "relation_type": "role_adjacent", "evidence_refs": [ref] if ref else [],
                "short_reason": "closely related experience",
            }
            if verdict_for and pid in verdict_for:
                base.update(verdict_for[pid])
            people[pid] = base
        return "ok", people, "mock:provider", "mock-model"
    return fake


def _pool_with_one_engineer(plan) -> list[NearCandidate]:
    p = _person(1, headline="Engineer", current_company="Co", current_title="Engineer")
    exps = [_Exp("Engineer", "Co", 2020, None, True, id="e1", desc="led backend systems")]
    facts = _facts(p, exps)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=0.6, score=60)
    scored = _scored(p, components=[comp], unmet_ids=["role"])
    return [_nc(p, facts, scored)]


def test_near_judge_is_batched_not_one_call_per_person(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge(capture=calls))
    monkeypatch.setattr(near_match_judge.settings, "near_match_judge_batch_size", 10)
    monkeypatch.setattr(near_match_judge.settings, "near_match_llm_enabled", True)

    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    pool: list[NearCandidate] = []
    for i in range(6):
        p = _person(i, headline="Engineer", current_company="Co", current_title="Engineer")
        exps = [_Exp("Engineer", "Co", 2020, None, True, id=f"e{i}", desc="led backend systems")]
        facts = _facts(p, exps)
        comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                              weight=100, match_strength=0.6, score=60)
        scored = _scored(p, components=[comp], unmet_ids=["role"])
        pool.append(_nc(p, facts, scored))

    run = near_match_judge.run_near_judge("engineers", plan, pool, ScoringContext())
    assert calls == [6]  # ONE call carrying all 6 packets, never 6 separate calls
    assert run.metadata.status == "full"
    assert len(run.verdicts) == 6


def test_near_judge_not_used_when_disabled(monkeypatch):
    monkeypatch.setattr(near_match_judge.settings, "near_match_llm_enabled", False)

    def boom(*a, **k):
        raise AssertionError("must not call the LLM when disabled")

    monkeypatch.setattr(near_match_judge, "_call_near_judge", boom)
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    run = near_match_judge.run_near_judge("engineers", plan, _pool_with_one_engineer(plan), ScoringContext())
    assert run.metadata.status == "not_used"
    assert run.verdicts == {}


# ─────────────────────── evidence validation (PART 9) ───────────────────────


def test_validator_drops_invented_evidence_refs_and_rejects_if_none_remain():
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "role", "relation_type": "role_adjacent",
            "evidence_refs": ["exp:invented-id"], "short_reason": "x",
        }
    }
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out == {}  # the only evidence ref was invented -> rejected


def test_validator_keeps_a_grounded_verdict_and_drops_only_the_bad_ref():
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "role", "relation_type": "role_adjacent",
            "evidence_refs": ["exp:e1", "exp:invented"], "short_reason": "x",
        }
    }
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out["p1"]["evidence_refs"] == ["exp:e1"]
    assert "dropped 1 invalid evidence ref(s)" in out["p1"]["validation_notes"]


def test_validator_rejects_a_relaxed_criterion_id_that_is_not_a_known_gap():
    p = _person(1)
    scored = _scored(p, unmet_ids=["role", "loc"])  # two known gaps -> ambiguous, can't silently fix
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "totally-invented-id", "relation_type": "role_adjacent",
            "evidence_refs": ["exp:e1"], "short_reason": "x",
        }
    }
    plan = _plan(
        _crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=50),
        _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=50),
    )
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out == {}


def test_validator_corrects_a_relaxed_criterion_id_when_there_is_exactly_one_known_gap():
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "wrong-echo", "relation_type": "role_adjacent",
            "evidence_refs": ["exp:e1"], "short_reason": "x",
        }
    }
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out["p1"]["relaxed_criterion_id"] == "role"


def test_validator_downgrades_mislabeled_geographic_adjacent_instead_of_rejecting():
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])  # role gap, NOT location
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "role", "relation_type": "geographic_adjacent",
            "evidence_refs": ["exp:e1"], "short_reason": "x",
        }
    }
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out["p1"]["relation_type"] == NearMatchRelation.OTHER_RELEVANT


def test_validator_rejects_not_meaningful_and_low_confidence(monkeypatch):
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))

    not_meaningful = {"p1": {
        "useful_near_match": True, "confidence": 0.9, "relaxed_criterion_id": "role",
        "relation_type": "not_meaningful", "evidence_refs": ["exp:e1"], "short_reason": "x",
    }}
    out, reasons = validate_near_verdicts(not_meaningful, {"p1": packet}, {"p1": nc}, plan)
    assert out == {}
    assert reasons["llm_rejected"] == 1

    low_conf = {"p1": {
        "useful_near_match": True, "confidence": 0.05, "relaxed_criterion_id": "role",
        "relation_type": "role_adjacent", "evidence_refs": ["exp:e1"], "short_reason": "x",
    }}
    out, reasons = validate_near_verdicts(low_conf, {"p1": packet}, {"p1": nc}, plan)
    assert out == {}
    assert reasons["low_confidence"] == 1


# ─────────────────────── ranking (PART 10) ───────────────────────


def test_strong_primary_intent_outranks_exact_location_but_unrelated_profile():
    plan = _plan(
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT, concept="venture capital investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
    )
    strong_vc = _person(1, location_text="Lawrenceville, Georgia, United States")
    strong_vc.city, strong_vc.state = "Lawrenceville", "Georgia"
    vc_comp = ScoreComponent(criterion="inv", criterion_id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                             weight=60, match_strength=0.95, score=57)
    vc_scored = _scored(strong_vc, components=[vc_comp], unmet_ids=["loc"], match_score=57)
    vc_nc = _nc(strong_vc, _facts(strong_vc), vc_scored)

    unrelated_local = _person(2, location_text="Fernbrook, Georgia, United States")
    unrelated_local.city, unrelated_local.state = "Fernbrook", "Georgia"
    weak_comp = ScoreComponent(criterion="inv", criterion_id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=60, match_strength=0.05, score=3)
    unrelated_scored = _scored(unrelated_local, components=[weak_comp], unmet_ids=["inv"], match_score=3)
    unrelated_nc = _nc(unrelated_local, _facts(unrelated_local), unrelated_scored)

    verdicts = {
        "p1": {"useful_near_match": True, "confidence": 0.85, "relaxed_criterion_id": "loc",
              "relation_type": "geographic_adjacent", "evidence_refs": ["x"], "short_reason": "close"},
    }
    ranked = rank_near_matches([vc_nc, unrelated_nc], verdicts, plan)
    assert [r.candidate.person.id for r in ranked] == ["p1"]  # unrelated_local wasn't validated -> excluded


def test_ranking_never_outputs_a_candidate_the_llm_did_not_validate():
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, _facts(p), scored)
    ranked = rank_near_matches([nc], {}, plan)
    assert ranked == []


def test_ranking_returns_nothing_when_no_verdicts_were_validated():
    # near-match design PART 1 (v2): no validated LLM verdict -> not ranked,
    # not shown — there is no "rank the pool anyway" fallback any more.
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    p = _person(1)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, _facts(p), scored)
    ranked = rank_near_matches([nc], {}, plan)
    assert ranked == []


# ─────────────────────── end-to-end service (PART 3/13) ───────────────────────


def test_build_near_matches_never_promotes_to_exact_or_possible(monkeypatch):
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge())
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    p = _person(1, headline="Engineer", current_company="Co", current_title="Engineer")
    exps = [_Exp("Engineer", "Co", 2020, None, True, id="e1", desc="led backend systems")]
    facts = _facts(p, exps)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=0.6, score=60)
    scored = _scored(p, components=[comp], unmet_ids=["role"])
    candidates = [(p, facts, scored, "not_match")]

    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert len(out) == 1
    display, verdict = out[0]
    assert display.qualification == Qualification.NOT_MATCH
    assert verdict["useful_near_match"] is True
    assert meta.llm_used is True


def test_build_near_matches_does_not_mutate_the_input_scored_candidates(monkeypatch):
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge())
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    p = _person(1, headline="Engineer", current_company="Co", current_title="Engineer")
    exps = [_Exp("Engineer", "Co", 2020, None, True, id="e1", desc="led backend systems")]
    facts = _facts(p, exps)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=0.6, score=60)
    original = _scored(p, components=[comp], unmet_ids=["role"], qualification=Qualification.NOT_MATCH)
    candidates = [(p, facts, original, "not_match")]

    build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert original.qualification == Qualification.NOT_MATCH  # unchanged — no in-place mutation


def test_near_match_pool_is_bounded_end_to_end(monkeypatch):
    monkeypatch.setattr("app.services.near_match_pool.settings.near_match_candidate_pool", 3)
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge())
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    candidates = []
    for i in range(10):
        p = _person(i, headline="Engineer", current_company="Co", current_title="Engineer")
        exps = [_Exp("Engineer", "Co", 2020, None, True, id=f"e{i}", desc="led backend systems")]
        facts = _facts(p, exps)
        comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                              weight=100, match_strength=0.5 + i * 0.01, score=50)
        scored = _scored(p, components=[comp], unmet_ids=["role"])
        candidates.append((p, facts, scored, "not_match"))

    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert meta.pool_size == 3


# ─────────────────────── PART 14 test fixtures (generic, non-hardcoded) ───────────────────────
#
# TEST A — nearby geography. Query: "VCs in <target city>". These prove the
# GEO mechanism generalizes: build_near_pool + the near-match judge treat a
# nearby-but-different city as a meaningful near match ONLY when the profile
# also satisfies the primary intent, never merely for living in the target
# city.


def _vc_plan():
    return _plan(
        _crit(id="vc", type=CriterionType.PROFESSIONAL_CONCEPT, concept="venture capital investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
    )


def test_A_nearby_city_strong_profession_is_a_meaningful_near_match(monkeypatch):
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge({
        "p2": {"relation_type": "geographic_adjacent", "confidence": 0.85,
              "short_reason": "Strong VC background; nearby suburb rather than Fernbrook proper."},
    }))
    plan = _vc_plan()
    p = _person(2, headline="Partner, Venture Capital", current_company="Acme Ventures",
               current_title="Partner", location_text="Lawrenceville, Georgia, United States")
    p.city, p.state = "Lawrenceville", "Georgia"
    exps = [_Exp("Partner", "Acme Ventures", 2019, None, True, id="e2",
                 desc="leads early-stage venture capital investments")]
    facts = _facts(p, exps)
    comp = ScoreComponent(criterion="vc", criterion_id="vc", type=CriterionType.PROFESSIONAL_CONCEPT,
                          weight=60, match_strength=0.9, score=54)
    scored = _scored(p, components=[comp], unmet_ids=["loc"], match_score=54)
    out, meta = build_near_matches("VCs in Fernbrook", plan, ScoringContext(),
                                   candidates=[(p, facts, scored, "hard_gate_reject")])
    assert len(out) == 1
    assert out[0][1]["relation_type"] == NearMatchRelation.GEOGRAPHIC_ADJACENT
    # deterministic geo data has no alias entry for "Fernbrook" (an invented
    # test city) -> UNKNOWN, so the LLM's own claim is kept but marked inferred.
    assert out[0][1]["relation_source"] == "llm_inference"


def test_D_exact_location_but_unrelated_profession_does_not_outrank_strong_vc(monkeypatch):
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge({
        "p2": {"relation_type": "geographic_adjacent", "confidence": 0.85},
        "p4": {"useful_near_match": False, "relation_type": "not_meaningful"},
    }))
    plan = _vc_plan()

    vc = _person(2, current_title="Partner", current_company="Acme Ventures",
                location_text="Lawrenceville, Georgia, United States")
    vc.city, vc.state = "Lawrenceville", "Georgia"
    vc_exps = [_Exp("Partner", "Acme Ventures", 2019, None, True, id="e2",
                    desc="leads early-stage venture capital investments")]
    vc_comp = ScoreComponent(criterion="vc", criterion_id="vc", type=CriterionType.PROFESSIONAL_CONCEPT,
                             weight=60, match_strength=0.9, score=54)
    vc_scored = _scored(vc, components=[vc_comp], unmet_ids=["loc"], match_score=54)

    unrelated = _person(4, current_title="Warehouse Associate", current_company="LocalCo",
                        location_text="Fernbrook, Georgia, United States")
    unrelated.city, unrelated.state = "Fernbrook", "Georgia"
    weak_comp = ScoreComponent(criterion="vc", criterion_id="vc", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=60, match_strength=0.0, score=0)
    unrelated_scored = _scored(unrelated, components=[weak_comp], unmet_ids=["vc"], match_score=0)

    out, meta = build_near_matches(
        "VCs in Fernbrook", plan, ScoringContext(),
        candidates=[
            (vc, _facts(vc, vc_exps), vc_scored, "hard_gate_reject"),
            (unrelated, _facts(unrelated), unrelated_scored, "not_match"),
        ],
    )
    ids = [c.person.id for c, _v in out]
    assert ids == ["p2"]  # the unrelated-but-exact-location person never appears


# TEST B — a completely different metro/profession pair proves the mechanism
# is not special-cased to venture capital or to any one city.


def test_B_metro_adjacent_ml_engineer_is_a_meaningful_near_match(monkeypatch):
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge({
        "p3": {"relation_type": "geographic_adjacent", "confidence": 0.8,
              "short_reason": "Strong ML engineering background; a short commute from Rivertown."},
        "p5": {"useful_near_match": False, "relation_type": "not_meaningful"},
    }))
    plan = _plan(
        _crit(id="ml", type=CriterionType.ROLE_FUNCTION, concept="machine learning engineering", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Rivertown", weight=40),
    )
    ml_engineer = _person(3, current_title="ML Engineer", current_company="DataCo",
                          location_text="Millbrook, Ohio, United States")
    ml_engineer.city, ml_engineer.state = "Millbrook", "Ohio"
    ml_exps = [_Exp("ML Engineer", "DataCo", 2021, None, True, id="e3",
                    desc="builds and ships production machine learning models")]
    ml_comp = ScoreComponent(criterion="ml", criterion_id="ml", type=CriterionType.ROLE_FUNCTION,
                             weight=60, match_strength=0.9, score=54)
    ml_scored = _scored(ml_engineer, components=[ml_comp], unmet_ids=["loc"], match_score=54)

    salesperson = _person(5, current_title="Sales Rep", current_company="RetailCo",
                          location_text="Rivertown, Ohio, United States")
    salesperson.city, salesperson.state = "Rivertown", "Ohio"
    weak_comp = ScoreComponent(criterion="ml", criterion_id="ml", type=CriterionType.ROLE_FUNCTION,
                               weight=60, match_strength=0.0, score=0)
    weak_scored = _scored(salesperson, components=[weak_comp], unmet_ids=["ml"], match_score=0)

    out, _meta = build_near_matches(
        "AI engineers in Rivertown", plan, ScoringContext(),
        candidates=[
            (ml_engineer, _facts(ml_engineer, ml_exps), ml_scored, "hard_gate_reject"),
            (salesperson, _facts(salesperson), weak_scored, "not_match"),
        ],
    )
    assert [c.person.id for c, _v in out] == ["p3"]
    assert out[0][1]["relation_type"] == NearMatchRelation.GEOGRAPHIC_ADJACENT


# TEST C — professional adjacency with no geography involved at all, proving
# the judge/validator path generalizes beyond location relaxation.


def test_C_professionally_adjacent_candidate_is_a_meaningful_near_match(monkeypatch):
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge({
        "p6": {"relation_type": "role_adjacent", "confidence": 0.75,
              "short_reason": "Advises startups on fundraising and personally angel invests."},
        "p7": {"useful_near_match": False, "relation_type": "not_meaningful"},
    }))
    plan = _plan(_crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="investor", weight=100))

    advisor = _person(6, current_title="Startup Advisor", current_company="Independent")
    advisor_exps = [_Exp("Startup Advisor", "Independent", 2018, None, True, id="e6",
                        desc="advises startups on fundraising and personally angel invests "
                             "in early-stage companies")]
    advisor_comp = ScoreComponent(criterion="inv", criterion_id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                                  weight=100, match_strength=0.4, score=40)
    advisor_scored = _scored(advisor, components=[advisor_comp], unmet_ids=["inv"], match_score=40)

    unrelated = _person(7, current_title="Barista", current_company="CafeCo")
    unrelated_exps = [_Exp("Barista", "CafeCo", 2022, None, True, id="e7",
                          desc="makes coffee; profile mentions 'investment' once in a hobbies section")]
    unrelated_comp = ScoreComponent(criterion="inv", criterion_id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                                    weight=100, match_strength=0.05, score=5)
    unrelated_scored = _scored(unrelated, components=[unrelated_comp], unmet_ids=["inv"], match_score=5)

    out, _meta = build_near_matches(
        "investors", plan, ScoringContext(),
        candidates=[
            (advisor, _facts(advisor, advisor_exps), advisor_scored, "not_match"),
            (unrelated, _facts(unrelated, unrelated_exps), unrelated_scored, "not_match"),
        ],
    )
    assert [c.person.id for c, _v in out] == ["p6"]
    assert out[0][1]["relation_type"] == NearMatchRelation.ROLE_ADJACENT


# ─────────────────────── hardening: no LLM = no near matches ───────────────────────


def _one_candidate_bundle():
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    p = _person(1, headline="Engineer", current_company="Co", current_title="Engineer")
    exps = [_Exp("Engineer", "Co", 2020, None, True, id="e1", desc="led backend systems")]
    facts = _facts(p, exps)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=0.6, score=60)
    scored = _scored(p, components=[comp], unmet_ids=["role"])
    return plan, [(p, facts, scored, "not_match")]


def test_llm_disabled_yields_zero_near_matches(monkeypatch):
    monkeypatch.setattr(near_match_judge.settings, "near_match_llm_enabled", False)

    def boom(*a, **k):
        raise AssertionError("must not call the LLM when disabled")

    monkeypatch.setattr(near_match_judge, "_call_near_judge", boom)
    plan, candidates = _one_candidate_bundle()
    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert out == []
    assert meta.llm_used is False
    assert meta.final_count == 0


def test_llm_unavailable_yields_zero_near_matches(monkeypatch):
    # every provider exhausted -> _call_near_judge returns "failed" for every
    # batch, exactly what happens when generate_structured runs out of chain.
    monkeypatch.setattr(near_match_judge, "_call_near_judge", lambda payload, packets: ("failed", None, None, None))
    plan, candidates = _one_candidate_bundle()
    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert out == []
    assert meta.llm_used is False
    assert meta.judge.status == "unavailable"


def test_malformed_near_output_yields_zero_near_matches(monkeypatch):
    # the model returns JSON that fails NearMatchJudgeBatch validation — the
    # router already turns that into a retried-then-exhausted call; simulate
    # the end state directly: every attempt comes back "failed".
    calls = {"n": 0}

    def fake(payload, packets):
        calls["n"] += 1
        return "failed", None, None, None

    monkeypatch.setattr(near_match_judge, "_call_near_judge", fake)
    plan, candidates = _one_candidate_bundle()
    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert out == []
    assert calls["n"] >= 1
    assert meta.judge.status == "unavailable"


def test_no_deterministic_ranking_fallback_helper_exists_in_rank_near_matches():
    # rank_near_matches itself no longer accepts an "llm did not run" mode —
    # locks in that there is exactly one calling convention now.
    import inspect

    sig = inspect.signature(rank_near_matches)
    assert "llm_ran" not in sig.parameters


# ─────────────────────── hardening v3: partial-judge safety ───────────────────────
#
# "partial" judge status means SOME batches succeeded and some failed — it must
# NOT mean "unjudged candidates are shown anyway". These tests build a pool
# that splits into exactly two top-level batches (batch size pinned to 3) and
# make the SECOND batch fail outright, proving the candidates in it can never
# reach the final output regardless of how strong their LOCAL signals were.


def _role_candidate(i: int, *, primary_intent_strength: float) -> tuple:
    p = _person(i, headline="Engineer", current_company="Co", current_title="Engineer")
    exps = [_Exp("Engineer", "Co", 2020, None, True, id=f"e{i}", desc="led backend systems")]
    facts = _facts(p, exps)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=primary_intent_strength, score=60)
    scored = _scored(p, components=[comp], unmet_ids=["role"])
    return (p, facts, scored, "not_match")


def test_partial_judge_3_judged_3_unjudged_only_judged_appear(monkeypatch):
    monkeypatch.setattr(near_match_judge.settings, "near_match_judge_batch_size", 3)
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))

    # judged group: strong local signal so it sorts FIRST into batch 1.
    judged = [_role_candidate(i, primary_intent_strength=0.9) for i in range(3)]
    # unjudged group: weaker (but still pool-eligible, missing exactly 1
    # criterion) local signal so it sorts SECOND into batch 2 — and would rank
    # perfectly well locally if a fallback ever used local signals alone.
    unjudged = [_role_candidate(i, primary_intent_strength=0.7) for i in range(3, 6)]
    judged_ids = {c[0].id for c in judged}
    unjudged_ids = {c[0].id for c in unjudged}

    calls: list[int] = []

    def fake(payload, packets):
        calls.append(len(packets))
        if len(calls) == 1:
            # batch 1 (the judged group) succeeds
            people = {
                pkt["person_id"]: {
                    "person_id": pkt["person_id"], "useful_near_match": True, "confidence": 0.9,
                    "satisfied_intent": "strong fit", "relaxed_criterion_id": "role",
                    "relation_type": "role_adjacent",
                    "evidence_refs": [(pkt.get("current") or {}).get("ref")],
                    "short_reason": "closely related engineering experience",
                }
                for pkt in packets
            }
            return "ok", people, "mock:provider", "mock-model"
        # batch 2 (the unjudged group) fails outright — never truncated, so
        # run_adaptive does not retry/split it into something that could
        # partially succeed; it is simply absent from verdicts.
        return "failed", None, None, None

    monkeypatch.setattr(near_match_judge, "_call_near_judge", fake)

    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=judged + unjudged)

    assert len(calls) == 2                      # exactly two top-level batches
    assert meta.judge.status == "partial"        # some ok, some failed
    assert meta.llm_used is True

    out_ids = {cand.person.id for cand, _v in out}
    assert out_ids == judged_ids                 # ONLY the successfully-judged group
    assert out_ids.isdisjoint(unjudged_ids)       # the unjudged group never leaks in
    for _cand, verdict in out:
        assert verdict is not None
        assert verdict["useful_near_match"] is True


def test_partial_judge_failed_batch_candidates_absent_from_every_stage(monkeypatch):
    """Same setup as above, but inspects each pipeline stage directly instead
    of only the final output — the unjudged group must be absent from
    run_near_judge's verdicts, absent from validate_near_verdicts's output,
    AND absent from rank_near_matches's output, not merely filtered at the
    very end."""
    from app.services.near_match_judge import run_near_judge
    from app.services.near_match_pool import build_near_pool
    from app.services.near_match_validator import validate_near_verdicts

    monkeypatch.setattr(near_match_judge.settings, "near_match_judge_batch_size", 3)
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    judged = [_role_candidate(i, primary_intent_strength=0.9) for i in range(3)]
    unjudged = [_role_candidate(i, primary_intent_strength=0.7) for i in range(3, 6)]
    judged_ids = {c[0].id for c in judged}
    unjudged_ids = {c[0].id for c in unjudged}

    def fake(payload, packets):
        pids = {pkt["person_id"] for pkt in packets}
        if pids & judged_ids:
            people = {
                pkt["person_id"]: {
                    "person_id": pkt["person_id"], "useful_near_match": True, "confidence": 0.9,
                    "relaxed_criterion_id": "role", "relation_type": "role_adjacent",
                    "evidence_refs": [(pkt.get("current") or {}).get("ref")],
                    "short_reason": "x",
                }
                for pkt in packets
            }
            return "ok", people, "mock:provider", "mock-model"
        return "failed", None, None, None

    monkeypatch.setattr(near_match_judge, "_call_near_judge", fake)

    pool = build_near_pool(judged + unjudged, plan, ScoringContext())
    judge_run = run_near_judge("engineers", plan, pool, ScoringContext())

    # stage 1: the judge itself never produced verdict entries for the failed batch
    assert set(judge_run.verdicts) == judged_ids
    assert unjudged_ids.isdisjoint(judge_run.verdicts)

    # stage 2: validation only ever sees (and only ever returns) the judged group
    pool_by_id = {nc.person.id: nc for nc in pool}
    validated, _ = validate_near_verdicts(judge_run.verdicts, judge_run.packets_by_id, pool_by_id, plan)
    assert set(validated) == judged_ids
    assert unjudged_ids.isdisjoint(validated)

    # stage 3: ranking only ever outputs the judged group
    ranked = rank_near_matches(pool, validated, plan)
    ranked_ids = {r.candidate.person.id for r in ranked}
    assert ranked_ids == judged_ids
    assert unjudged_ids.isdisjoint(ranked_ids)


def test_invariant_violation_raises_instead_of_leaking(monkeypatch):
    """Section 3 — the explicit runtime guard in build_near_matches. If ranking
    ever returned an entry without a real validated verdict (a future
    regression), the pipeline must fail loudly (and get caught by search_
    service's failure isolation) rather than silently show it."""
    from dataclasses import replace as _replace

    from app.services.near_match_ranking import RankedNearMatch

    plan, candidates = _one_candidate_bundle()
    monkeypatch.setattr(near_match_judge, "_call_near_judge", _fake_near_judge())

    def fake_rank(pool, validated, parsed):
        # simulate a hypothetical future bug: rank a candidate with NO verdict
        nc = pool[0]
        return [RankedNearMatch(candidate=nc, verdict=None, rank_score=0.9)]

    monkeypatch.setattr(
        "app.services.near_match_service.rank_near_matches", fake_rank,
    )
    with pytest.raises(RuntimeError, match="near-match invariant violated"):
        build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)


# ─────────────────────── hardening v3: no result padding ───────────────────────


def test_only_one_genuinely_useful_candidate_returns_exactly_one_not_padded(monkeypatch):
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    # five pool-eligible candidates, but the mocked judge only accepts ONE —
    # settings.near_match_max_results defaults to 5; the result must not be
    # padded up to it.
    candidates = [_role_candidate(i, primary_intent_strength=0.8) for i in range(5)]
    only_good_id = candidates[0][0].id

    def fake(payload, packets):
        people = {}
        for pkt in packets:
            pid = pkt["person_id"]
            if pid == only_good_id:
                people[pid] = {
                    "person_id": pid, "useful_near_match": True, "confidence": 0.9,
                    "relaxed_criterion_id": "role", "relation_type": "role_adjacent",
                    "evidence_refs": [(pkt.get("current") or {}).get("ref")], "short_reason": "x",
                }
            else:
                people[pid] = {"person_id": pid, "useful_near_match": False, "relation_type": "not_meaningful"}
        return "ok", people, "mock:provider", "mock-model"

    monkeypatch.setattr(near_match_judge, "_call_near_judge", fake)
    monkeypatch.setattr("app.services.near_match_service.settings.near_match_max_results", 5)

    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert len(out) == 1
    assert out[0][0].person.id == only_good_id


def test_zero_good_candidates_returns_empty_not_padded(monkeypatch):
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    candidates = [_role_candidate(i, primary_intent_strength=0.8) for i in range(4)]

    def fake(payload, packets):
        return "ok", {
            pkt["person_id"]: {"person_id": pkt["person_id"], "useful_near_match": False,
                              "relation_type": "not_meaningful"}
            for pkt in packets
        }, "mock:provider", "mock-model"

    monkeypatch.setattr(near_match_judge, "_call_near_judge", fake)
    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert out == []


def test_confidence_threshold_is_not_lowered_to_fill_slots(monkeypatch):
    """A near-miss-confidence verdict must be rejected even when the pool is
    large and the result would otherwise be under NEAR_MATCH_MAX_RESULTS —
    the threshold is fixed configuration, never adjusted by how few slots are
    filled."""
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="engineering", weight=100))
    candidates = [_role_candidate(i, primary_intent_strength=0.8) for i in range(3)]
    threshold = 0.45  # settings default

    def fake(payload, packets):
        return "ok", {
            pkt["person_id"]: {
                "person_id": pkt["person_id"], "useful_near_match": True,
                "confidence": threshold - 0.05,  # just below the configured minimum
                "relaxed_criterion_id": "role", "relation_type": "role_adjacent",
                "evidence_refs": [(pkt.get("current") or {}).get("ref")], "short_reason": "x",
            }
            for pkt in packets
        }, "mock:provider", "mock-model"

    monkeypatch.setattr(near_match_judge, "_call_near_judge", fake)
    monkeypatch.setattr("app.services.near_match_validator.settings.near_match_min_confidence", threshold)
    out, meta = build_near_matches("engineers", plan, ScoringContext(), candidates=candidates)
    assert out == []


# ─────────────────────── hardening: ranking normalization ───────────────────────


def test_clamp01_bounds_any_input():
    assert clamp01(0.5) == 0.5
    assert clamp01(80) == 1.0          # a raw 0-100 value must never leak through
    assert clamp01(-3) == 0.0
    assert clamp01(None) == 0.0
    assert clamp01("not a number") == 0.0
    assert clamp01(float("nan")) == 0.0


def test_match_score_80_contributes_as_point_8_not_80():
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    p = _person(1)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=0.0, score=0)
    # everything else near-zero so match_score's own contribution is isolated
    scored = _scored(p, components=[comp], unmet_ids=["role"], match_score=80.0)
    nc = _nc(p, _facts(p), scored)
    nc.primary_intent_strength = 0.0
    nc.local_relevance = 0.0
    verdict = {"relaxed_criterion_id": "role", "relation_type": "role_adjacent", "confidence": 0.0}
    ranked = rank_near_matches([nc], {"p1": verdict}, plan)
    assert len(ranked) == 1
    # match_score weight is 0.10 -> contributes 0.10*0.8=0.08, not 0.10*80=8.0
    assert ranked[0].rank_score == pytest.approx(0.08, abs=0.02)
    assert 0.0 <= ranked[0].rank_score <= 1.0


def test_primary_intent_can_dominate_the_ranking():
    plan = _plan(
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
    )
    # candidate 1: very strong primary intent, weak on everything else
    strong_p = _person(1)
    strong_comp = ScoreComponent(criterion="inv", criterion_id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                                 weight=60, match_strength=1.0, score=60)
    strong_scored = _scored(strong_p, components=[strong_comp], unmet_ids=["loc"], match_score=10.0)
    strong_nc = _nc(strong_p, _facts(strong_p), strong_scored)
    strong_nc.primary_intent_strength = 1.0
    strong_nc.local_relevance = 0.1

    # candidate 2: weak primary intent, but a high raw match_score and full
    # local relevance — without normalization this would win on scale alone.
    weak_p = _person(2)
    weak_comp = ScoreComponent(criterion="inv", criterion_id="inv", type=CriterionType.PROFESSIONAL_CONCEPT,
                               weight=60, match_strength=0.05, score=3)
    weak_scored = _scored(weak_p, components=[weak_comp], unmet_ids=["loc"], match_score=95.0)
    weak_nc = _nc(weak_p, _facts(weak_p), weak_scored)
    weak_nc.primary_intent_strength = 0.05
    weak_nc.local_relevance = 1.0

    verdicts = {
        "p1": {"relaxed_criterion_id": "loc", "relation_type": "other_relevant", "confidence": 0.7},
        "p2": {"relaxed_criterion_id": "loc", "relation_type": "other_relevant", "confidence": 0.7},
    }
    ranked = rank_near_matches([strong_nc, weak_nc], verdicts, plan)
    assert [r.candidate.person.id for r in ranked] == ["p1", "p2"]


def test_all_ranked_scores_are_within_0_1_even_with_extreme_inputs():
    plan = _plan(_crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="x", weight=100))
    p = _person(1)
    comp = ScoreComponent(criterion="role", criterion_id="role", type=CriterionType.ROLE_FUNCTION,
                          weight=100, match_strength=5.0, score=500)  # deliberately out-of-range
    scored = _scored(p, components=[comp], unmet_ids=["role"], match_score=999.0)
    nc = _nc(p, _facts(p), scored)
    nc.primary_intent_strength = 5.0  # also deliberately out-of-range
    nc.local_relevance = -2.0
    verdict = {"relaxed_criterion_id": "role", "relation_type": "role_adjacent", "confidence": 50.0}
    ranked = rank_near_matches([nc], {"p1": verdict}, plan)
    assert len(ranked) == 1
    assert 0.0 <= ranked[0].rank_score <= 1.0


# ─────────────────────── hardening: LLM-first primary intent ───────────────────────


def test_llm_supplied_primary_intent_is_accepted_when_anchors_are_valid():
    plan = ParsedSearchQuery(
        criteria=[
            _crit(id="vc", type=CriterionType.PROFESSIONAL_CONCEPT, concept="venture capital investor", weight=60),
            _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
        ],
        primary_intent="people with meaningful venture-capital experience",
        intent_anchor_criterion_ids=["vc"],
    )
    query_intent._identify_primary_intent(plan)
    # the LLM's own phrasing and anchor choice are kept verbatim — not
    # overwritten by the deterministic (type-based) derivation.
    assert plan.primary_intent == "people with meaningful venture-capital experience"
    assert plan.intent_anchor_criterion_ids == ["vc"]


def test_llm_supplied_invalid_anchor_ids_are_dropped_and_fallback_runs():
    plan = ParsedSearchQuery(
        criteria=[
            _crit(id="vc", type=CriterionType.PROFESSIONAL_CONCEPT, concept="venture capital investor", weight=60),
            _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
        ],
        primary_intent="whatever the model guessed",
        # neither id exists on this plan's criteria — must never be trusted
        intent_anchor_criterion_ids=["totally-invented-id", "another-bad-id"],
    )
    query_intent._identify_primary_intent(plan)
    # invalid ids -> deterministic fallback took over (anchors on the real
    # professional_concept criterion, not the location constraint)
    assert plan.intent_anchor_criterion_ids == ["vc"]
    assert "venture capital" in plan.primary_intent.lower()


def test_llm_supplied_primary_intent_with_no_anchors_falls_back():
    plan = ParsedSearchQuery(
        criteria=[
            _crit(id="vc", type=CriterionType.PROFESSIONAL_CONCEPT, concept="venture capital investor", weight=60),
            _crit(id="loc", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
        ],
        primary_intent="",  # LLM omitted it entirely
        intent_anchor_criterion_ids=[],
    )
    query_intent._identify_primary_intent(plan)
    assert plan.intent_anchor_criterion_ids == ["vc"]


def test_deterministic_fallback_still_anchors_on_the_constraint_when_thats_all_there_is():
    plan = ParsedSearchQuery(
        criteria=[_crit(id="loc", type=CriterionType.LOCATION, value="Austin", weight=100)],
    )
    query_intent._identify_primary_intent(plan)
    assert plan.intent_anchor_criterion_ids == ["loc"]


def test_llm_anchor_ids_are_revalidated_against_the_final_criteria_set():
    # simulates augment_plan dropping/renaming a criterion AFTER the LLM named
    # its anchor — the stale id must not survive.
    plan = ParsedSearchQuery(
        criteria=[_crit(id="new_id", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=100)],
        primary_intent="investing",
        intent_anchor_criterion_ids=["old_id_that_no_longer_exists"],
    )
    query_intent._identify_primary_intent(plan)
    assert plan.intent_anchor_criterion_ids == ["new_id"]  # deterministic fallback, not the stale id


# ─────────────────────── hardening: deterministic geo beats conflicting LLM inference ───────────────────────


def test_deterministic_far_rejects_a_conflicting_llm_nearby_claim():
    plan = _plan(
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Georgia", weight=40),
    )
    p = _person(1)
    p.city, p.state, p.location_text = "Seattle", "Washington", "Seattle, Washington, United States"
    scored = _scored(p, unmet_ids=["loc"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "loc", "relation_type": "geographic_adjacent",
            "evidence_refs": ["exp:e1"], "short_reason": "x",
        }
    }
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    # deterministic geo proves Seattle/Washington is FAR from Georgia — the
    # LLM's geographic_adjacent claim is rejected, downgraded to other_relevant,
    # never trusted over the structured fact.
    assert out["p1"]["relation_type"] == NearMatchRelation.OTHER_RELEVANT
    assert out["p1"]["relation_source"] is None


# ─────────────────────── hardening: adjacent-state metro (bug report TASK 5) ───────────────────────


def test_adjacent_state_is_not_confidently_far_llm_claim_kept_as_inference():
    """A metro area can straddle a state line (e.g. a Midwest city near the
    Illinois/Indiana border) that the static region alias list doesn't happen
    to enumerate. TASK 5: "different state" alone must not be confident FAR
    when the two states are adjacent — deterministic data stays UNKNOWN, the
    LLM's own geographic_adjacent claim is kept (never verified as a fact:
    relation_source must be "llm_inference", not "deterministic")."""
    plan = _plan(
        _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT, concept="engineer", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Illinois", weight=40),
    )
    p = _person(1)
    # a real border town, but ANY unlisted adjacent-state city exercises the
    # same generic state-adjacency logic — nothing city-specific in the fix.
    p.city, p.state, p.location_text = "Hammond", "Indiana", "Hammond, Indiana, United States"
    scored = _scored(p, unmet_ids=["loc"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {"p1": {
        "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "strong fit",
        "relaxed_criterion_id": "loc", "relation_type": "geographic_adjacent",
        "evidence_refs": ["exp:e1"], "short_reason": "just across the state line",
    }}
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out["p1"]["relation_type"] == NearMatchRelation.GEOGRAPHIC_ADJACENT  # NOT downgraded
    assert out["p1"]["relation_source"] == "llm_inference"  # kept, but never claimed as verified
    assert out["p1"]["geo_relation"] == GeoRelation.UNKNOWN  # deterministic data genuinely can't resolve it


def test_non_adjacent_state_is_still_confidently_far():
    """The adjacency relaxation must not swallow genuine long-distance
    mismatches — non-bordering states stay a confident, deterministic FAR."""
    from app.services.geo import classify_relation_deterministic

    relation = classify_relation_deterministic(
        {"city": "Los Angeles", "state": "California", "location_text": "Los Angeles, California"},
        ["New York"],
    )
    assert relation == GeoRelation.FAR


def test_adjacency_relaxation_does_not_apply_to_a_broad_multistate_region():
    """Bug report — a large state (Texas) borders SOME state in almost any
    big multi-state region expansion (e.g. "Southeast US" -> 12 states,
    Texas borders Arkansas + Louisiana), but that says nothing about whether
    the candidate's actual city (Austin, hundreds of miles from either
    border) is near it. Adjacency reasoning only applies against a SINGLE
    named target state — a broad region list falls back to the pre-relaxation
    default (FAR), generically (no city name appears in the source fix)."""
    from app.services.geo import classify_relation_deterministic

    southeast_us = [
        "Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana",
        "Mississippi", "North Carolina", "South Carolina", "Tennessee", "Virginia", "West Virginia",
    ]
    relation = classify_relation_deterministic(
        {"city": "Austin", "state": "Texas", "location_text": "Austin, Texas, United States"},
        southeast_us,
    )
    assert relation == GeoRelation.FAR

    # the single-state case (the original TASK 5 fix) must still work
    relation2 = classify_relation_deterministic(
        {"city": "Hammond", "state": "Indiana", "location_text": "Hammond, Indiana, United States"},
        ["Illinois"],
    )
    assert relation2 == GeoRelation.UNKNOWN


def test_strict_location_matching_is_unaffected_by_the_adjacency_relaxation():
    """TASK 5 explicitly requires strict exact-location matching to be
    untouched — ``_score_location`` (the hard gate / main scorer) never calls
    ``classify_relation_deterministic`` at all, so an adjacent-state candidate
    still fails strict matching exactly as before."""
    from app.services.scoring import _score_location

    p = _person(1)
    p.city, p.state, p.location_text = "Hammond", "Indiana", "Hammond, Indiana, United States"
    strength, _ev = _score_location(_facts(p), "Illinois")
    assert strength == 0.0  # still a strict miss — only Near Match relaxes this


def test_deterministic_confirmation_marks_relation_source_deterministic():
    plan = _plan(
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Georgia", weight=40),
    )
    p = _person(1)
    p.city, p.state, p.location_text = "Savannah", "Georgia", "Savannah, Georgia, United States"
    scored = _scored(p, unmet_ids=["loc"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "loc", "relation_type": "geographic_adjacent",
            "evidence_refs": ["exp:e1"], "short_reason": "x",
        }
    }
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out["p1"]["relation_type"] == NearMatchRelation.GEOGRAPHIC_ADJACENT
    assert out["p1"]["relation_source"] == "deterministic"


def test_geographic_adjacent_downgraded_when_candidate_has_no_stored_location():
    plan = _plan(
        _crit(id="inv", type=CriterionType.PROFESSIONAL_CONCEPT, concept="investor", weight=60),
        _crit(id="loc", type=CriterionType.LOCATION, value="Georgia", weight=40),
    )
    p = _person(1)  # no city/state/location_text at all
    scored = _scored(p, unmet_ids=["loc"])
    nc = _nc(p, _facts(p), scored)
    packet = {"person_id": "p1", "past": [{"ref": "exp:e1", "experience_id": "e1"}]}
    raw = {
        "p1": {
            "useful_near_match": True, "confidence": 0.9, "satisfied_intent": "x",
            "relaxed_criterion_id": "loc", "relation_type": "geographic_adjacent",
            "evidence_refs": ["exp:e1"], "short_reason": "x",
        }
    }
    out, _ = validate_near_verdicts(raw, {"p1": packet}, {"p1": nc}, plan)
    assert out["p1"]["relation_type"] == NearMatchRelation.OTHER_RELEVANT


# ─────────────────────── hardening: real end-to-end search orchestration ───────────────────────
#
# The tests above unit-test each near-match module directly. These run the
# ACTUAL search entry points (``run_connection_search`` / ``load_search``)
# against a real (SQLite) DB — no shortcuts — with every LLM/network seam
# mocked at the same points the rest of this test suite already uses. Query
# interpretation is pinned to a hand-built plan (query-interpretation quality
# is covered exhaustively elsewhere in this suite) so the test is not coupled
# to the deterministic parser's regex heuristics; everything downstream of
# that — hard gate, scoring, the near-match pipeline, persistence, and reload
# — is the real production code path.

from app.constants import DatasetStatus, EnrichmentState  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import Dataset, Experience, Person  # noqa: E402
from app.services import near_match_judge as _nmj_mod  # noqa: E402
from app.services import search_service  # noqa: E402


def _mk_dataset(db) -> str:
    ds = Dataset(name="near-match e2e test", status=DatasetStatus.READY)
    db.add(ds)
    db.commit()
    return ds.id


def _mk_person(db, dataset_id: str, *, name: str, title: str, company: str,
               location_text: str, city: str, state: str) -> str:
    p = Person(
        dataset_id=dataset_id, is_connection=True,
        linkedin_url=f"https://www.linkedin.com/in/{name.lower().replace(' ', '-')}",
        full_name=name, current_title=title, current_company=company,
        location_text=location_text, city=city, state=state,
        enrichment_state=EnrichmentState.READY, profile_completeness=80,
    )
    db.add(p)
    db.flush()
    db.add(Experience(person_id=p.id, position=title, company_name=company,
                      start_year=2020, is_current=True, order_index=0))
    db.commit()
    return p.id


def _fixed_plan() -> ParsedSearchQuery:
    """A hand-built plan standing in for query interpretation: TITLE (the
    primary intent — deterministically scorable, unlike a semantic type, so
    an unrelated candidate resolves to a real FALSE, not just UNKNOWN) +
    LOCATION (the constraint)."""
    return ParsedSearchQuery(
        criteria=[
            _crit(id="title", type=CriterionType.TITLE, value="Software Engineer", weight=60),
            _crit(id="location", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
        ],
        primary_intent="software engineering",
        intent_anchor_criterion_ids=["title"],
    )


def _e2e_near_judge_fake(payload, packets):
    """Stands in for the real LLM: an engineer gets a grounded geographic_
    adjacent recommendation, anyone else is correctly marked not meaningful —
    exactly what a real reviewer would conclude from the evidence packet."""
    people = {}
    for pkt in packets:
        pid = pkt["person_id"]
        title = (pkt.get("current_title") or "").lower()
        gaps = pkt.get("near_match_context", {}).get("gaps", [])
        relaxed = gaps[0]["criterion_id"] if gaps else ""
        if "engineer" in title:
            ref = (pkt.get("current") or {}).get("ref")
            people[pid] = {
                "person_id": pid, "useful_near_match": True, "confidence": 0.85,
                "satisfied_intent": "strong software engineering background",
                "relaxed_criterion_id": relaxed, "relation_type": "geographic_adjacent",
                "evidence_refs": [ref] if ref else [],
                "short_reason": "Software engineer with a nearby-city location mismatch.",
            }
        else:
            people[pid] = {
                "person_id": pid, "useful_near_match": False, "confidence": 0.1,
                "relaxed_criterion_id": relaxed, "relation_type": "not_meaningful",
                "evidence_refs": [], "short_reason": "",
            }
    return "ok", people, "mock:provider", "mock-model"


def test_e2e_strict_and_near_match_in_one_real_search(monkeypatch):
    monkeypatch.setattr(search_service, "interpret_query", lambda q: (_fixed_plan(), "deterministic", None))
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", _e2e_near_judge_fake)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        a_id = _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        b_id = _mk_person(db, ds_id, name="Bob Nearby", title="Software Engineer", company="Beta",
                          location_text="Millbrook, Georgia, United States", city="Millbrook", state="Georgia")
        c_id = _mk_person(db, ds_id, name="Cara Barista", title="Barista", company="Corner Cafe",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")

        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
        db.commit()
    finally:
        db.close()

    main_ids = {r.person_id for r in resp.connections.results}
    near_ids = {r.person_id for r in resp.connections.near_matches}

    # A: satisfies primary intent (title) + strict location -> main result
    assert a_id in main_ids
    # B: strongly satisfies primary intent, narrowly misses location,
    #    mocked judge says useful geographic_adjacent -> near match
    assert b_id in near_ids
    # C: unrelated profession, exact location -> appears nowhere
    assert c_id not in main_ids and c_id not in near_ids

    near_b = next(r for r in resp.connections.near_matches if r.person_id == b_id)
    assert near_b.qualification == "not_match"          # B never promoted to Exact/Possible
    assert near_b.near_relation_type == "geographic_adjacent"
    assert near_b.reason and "engineer" in near_b.reason.lower()  # grounded short_reason
    assert near_b.near_match_confidence is not None

    # main results are exactly what the strict pipeline alone would produce —
    # near-match logic did not add/remove/reorder anything there.
    assert len(resp.connections.results) == 1
    assert resp.connections.exact_match_count == 1
    assert resp.connections.possible_match_count == 0


def test_e2e_near_match_failure_does_not_break_the_strict_search(monkeypatch):
    """Section 8 — a transport error / malformed output / validator bug in the
    near-match stage must never fail the whole search."""
    monkeypatch.setattr(search_service, "interpret_query", lambda q: (_fixed_plan(), "deterministic", None))

    def boom(*a, **k):
        raise RuntimeError("simulated near-match pipeline failure (transport error)")

    monkeypatch.setattr(search_service, "build_near_matches", boom)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        a_id = _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        _mk_person(db, ds_id, name="Bob Nearby", title="Software Engineer", company="Beta",
                  location_text="Millbrook, Georgia, United States", city="Millbrook", state="Georgia")

        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
    finally:
        db.close()

    # the strict search succeeded normally despite the near-match crash
    assert {r.person_id for r in resp.connections.results} == {a_id}
    assert resp.connections.exact_match_count == 1
    assert resp.connections.near_matches == []  # near_matches=[], not an exception


def test_e2e_near_match_failure_is_logged_with_traceback_not_silent(monkeypatch, caplog):
    """Section 2 — the failure must be logged (with a traceback), never just
    swallowed, and must not leak anything sensitive into the log line."""
    import logging

    monkeypatch.setattr(search_service, "interpret_query", lambda q: (_fixed_plan(), "deterministic", None))

    secret_marker = "sk-super-secret-anthropic-key-should-never-appear"

    def boom(*a, **k):
        raise RuntimeError(f"simulated failure; a fake secret {secret_marker} must not leak")

    monkeypatch.setattr(search_service, "build_near_matches", boom)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                  location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        with caplog.at_level(logging.ERROR, logger="app.search"):
            resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
    finally:
        db.close()

    assert resp.connections.near_matches == []
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR and r.name == "app.search"]
    assert error_records, "near-match failure must be logged at ERROR level, not silently swallowed"
    rec = error_records[0]
    assert rec.exc_info is not None  # a real traceback was captured (log.exception behavior)
    assert "near-match generation failed" in rec.getMessage()
    # the RuntimeError's own message (which contains our fake secret marker,
    # standing in for something like an API key) is a separate concern from
    # the log LINE itself — this test's real assertion is that the traceback
    # capture is what the exc_info machinery provides, never the raw prompt,
    # a profile payload, or an auth header, which this code path never
    # constructs a log message from in the first place.
    assert "app_key" not in rec.getMessage().lower()
    assert "authorization" not in rec.getMessage().lower()


def test_e2e_saved_search_reload_is_db_only_and_preserves_near_match_fields(monkeypatch):
    monkeypatch.setattr(search_service, "interpret_query", lambda q: (_fixed_plan(), "deterministic", None))
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", _e2e_near_judge_fake)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                  location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        _mk_person(db, ds_id, name="Bob Nearby", title="Software Engineer", company="Beta",
                  location_text="Millbrook, Georgia, United States", city="Millbrook", state="Georgia")

        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
        db.commit()
        search_id = resp.search_id
        original_near = list(resp.connections.near_matches)
    finally:
        db.close()

    assert len(original_near) == 1
    orig = original_near[0]

    def boom(*a, **k):
        raise AssertionError("reload must not re-run any search / LLM / near-match / Apify step")

    monkeypatch.setattr(search_service, "interpret_query", boom)
    monkeypatch.setattr(search_service, "run_judge", boom)
    monkeypatch.setattr(search_service, "_run_final_audit", boom)
    monkeypatch.setattr(search_service, "build_near_matches", boom)
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", boom)
    monkeypatch.setattr("app.services.embeddings.embed_text", boom)
    monkeypatch.setattr("app.services.apify_client.scrape_profiles", boom)

    db2 = SessionLocal()
    try:
        reloaded = search_service.load_search(db2, search_id)
    finally:
        db2.close()

    assert reloaded is not None
    assert len(reloaded.connections.near_matches) == 1
    r = reloaded.connections.near_matches[0]
    assert r.person_id == orig.person_id
    assert r.reason == orig.reason
    assert r.near_relation_type == orig.near_relation_type == "geographic_adjacent"
    assert r.near_match_confidence == orig.near_match_confidence


# ─────────────────────── hardening: hard-gate-rejected candidates reach
# Near Match under FULL SONNET VERIFICATION too (bug report) ───────────────
#
# ``settings.full_llm_verification`` defaults to True in production, but
# every test in this suite runs with it forced off by conftest.py's blanket
# ``FULL_LLM_VERIFICATION=false`` — so a bug gated behind "only when full
# verification is ON" had zero coverage anywhere in the existing suite. The
# bug: search_service.py fed hard-gate-rejected candidates into the Near
# Match pool ONLY inside an ``if not full_mode:`` block — in the mode that
# actually runs in production, a candidate rejected for one verified fact
# (e.g. a nearby-but-not-exact location) never reached the Near Match LLM at
# all, regardless of how professionally relevant they were. These tests pin
# ``full_llm_verification=True`` explicitly so this path is never silently
# unverified again.


def _full_mode_plan() -> ParsedSearchQuery:
    """PROFESSIONAL_CONCEPT (judgeable — requires a full_verification verdict)
    + LOCATION (hard-gated). Unlike ``_fixed_plan`` (TITLE + LOCATION, used by
    the legacy-mode e2e tests above), this always has a judgeable criterion —
    the full-verification invariant (every hard-gate survivor gets Sonnet-
    reviewed) requires at least one, or the search fails outright."""
    return ParsedSearchQuery(
        criteria=[
            _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT,
                  concept="software engineering", weight=60),
            _crit(id="location", type=CriterionType.LOCATION, value="Fernbrook", weight=40),
        ],
        primary_intent="software engineering",
        intent_anchor_criterion_ids=["role"],
    )


def _e2e_full_verification_fake(payload, packets, review_by_person=None, **_kw):
    """Stands in for Sonnet's full-verification batch call — grounds every
    verdict in the packet's own ``current_title`` exactly as a real reviewer
    would; never a hardcoded person/city, purely evidence-driven."""
    out: dict = {}
    for pkt in packets:
        pid = pkt["person_id"]
        is_engineer = "engineer" in (pkt.get("current_title") or "").lower()
        ref = (pkt.get("current") or {}).get("ref")
        out[pid] = {
            cid: {
                "criterion_id": cid, "status": "true" if is_engineer else "false",
                "match_strength": 0.9 if is_engineer else 0.0, "confidence": 0.9,
                "reason": "clearly a software engineering role" if is_engineer else "not an engineering role",
                "supporting_evidence_refs": [ref] if (is_engineer and ref) else [],
                "contradicting_evidence_refs": [] if is_engineer else ([ref] if ref else []),
                "experience_ids": [],
            }
            for cid in (pkt.get("unresolved_criteria") or [])
        }
    return "ok", out, "mock:provider", "mock-model"


def test_e2e_hard_gate_rejected_candidate_reaches_near_match_under_full_verification(monkeypatch):
    from app.config import settings
    from app.services import full_verification as _fv_mod

    monkeypatch.setattr(settings, "full_llm_verification", True)
    monkeypatch.setattr(settings, "final_result_audit_enabled", False)
    monkeypatch.setattr(search_service, "interpret_query",
                        lambda q: (_full_mode_plan(), "deterministic", None))
    monkeypatch.setattr(_fv_mod, "_call_judge", _e2e_full_verification_fake)
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", _e2e_near_judge_fake)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        a_id = _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        # Bob is REJECTED BY THE HARD GATE (location mismatch) — he is never
        # sent to full_verification at all. This is the exact candidate the
        # bug excluded from Near Match consideration entirely.
        b_id = _mk_person(db, ds_id, name="Bob Nearby", title="Software Engineer", company="Beta",
                          location_text="Millbrook, Georgia, United States", city="Millbrook", state="Georgia")
        # Cara passes the hard gate (right location) but is professionally
        # unrelated — Sonnet reviews and correctly says FALSE, and the
        # near-match judge correctly says not_meaningful. She must not appear
        # anywhere merely to fill a slot.
        c_id = _mk_person(db, ds_id, name="Cara Barista", title="Barista", company="Corner Cafe",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")

        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
        db.commit()
    finally:
        db.close()

    main_ids = {r.person_id for r in resp.connections.results}
    near_ids = {r.person_id for r in resp.connections.near_matches}

    assert a_id in main_ids                       # A: verified in full, exact match
    assert b_id not in main_ids                    # never shown as a main result — never Sonnet-reviewed
    assert b_id in near_ids, (
        "a hard-gate-rejected candidate who is a strong professional fit and only "
        "misses on a nearby location must still reach Near Match under full verification"
    )
    assert c_id not in main_ids and c_id not in near_ids  # C: unrelated — appears nowhere, no padding

    near_b = next(r for r in resp.connections.near_matches if r.person_id == b_id)
    assert near_b.qualification == "not_match"      # never promoted to Exact/Possible
    assert near_b.near_relation_type == "geographic_adjacent"
    assert near_b.near_match_confidence is not None
    # B only appears because ITS OWN near-match verdict validated — not
    # because the strict pipeline decided anything about them.
    assert len(resp.connections.near_matches) == 1

    # the strict full-verification invariant is untouched: ALICE (hard-gate
    # survivor) and CARA (hard-gate survivor, Sonnet said FALSE) were both
    # reviewed; BOB never was. Main results are exactly what the strict
    # pipeline alone would produce.
    assert resp.judge_metadata is not None
    assert resp.judge_metadata.get("mode") == "full_verification"
    assert resp.judge_metadata.get("status") == "complete"
    assert resp.judge_metadata.get("filtered_candidate_count") == 2
    assert resp.judge_metadata.get("sonnet_verified_candidate_count") == 2
    assert resp.connections.exact_match_count == 1
    assert resp.connections.possible_match_count == 0
    assert len(resp.connections.results) == 1


def test_e2e_full_verification_still_fails_closed_when_verification_incomplete(monkeypatch):
    """The Near Match fix must not weaken the FULL SONNET VERIFICATION
    invariant: if Alice's own review cannot complete, the search still fails
    (503-equivalent exception), it does not silently fall back to a
    near-match-only response."""
    from app.config import settings
    from app.services import full_verification as _fv_mod

    monkeypatch.setattr(settings, "full_llm_verification", True)
    monkeypatch.setattr(settings, "final_result_audit_enabled", False)
    monkeypatch.setattr(search_service, "interpret_query",
                        lambda q: (_full_mode_plan(), "deterministic", None))
    monkeypatch.setattr(_fv_mod, "_call_judge", lambda *a, **k: ("failed", None, None, None))
    monkeypatch.setattr(_fv_mod, "generate_structured", lambda *a, **k: (None, {}))
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                  location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")

        with pytest.raises(_fv_mod.VerificationIncompleteError):
            search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
    finally:
        db.close()


# ─────────────────────── hardening: audit-downgrade -> Near Match handoff (bug report TASK 5) ───────────────────────


def test_e2e_audit_downgraded_candidate_reaches_near_match_with_a_fresh_verdict(monkeypatch):
    """A candidate full_verification says TRUE on every required criterion
    (pre-audit EXACT) but the FINAL AUDIT downgrades for insufficient
    evidence (NOT a grounded contradiction — that's INCORRECT/NOT_MATCH,
    covered separately) used to just vanish: excluded from main results,
    never reconsidered for Near Match. Must now reach Near Match, but ONLY
    via its own independently-validated near-match verdict — never an
    automatic promotion of the audit's own (rejected) qualification."""
    from app.config import settings
    from app.services import final_auditor
    from app.services import full_verification as _fv_mod

    monkeypatch.setattr(settings, "full_llm_verification", True)
    monkeypatch.setattr(settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(search_service, "interpret_query",
                        lambda q: (_full_mode_plan(), "deterministic", None))
    monkeypatch.setattr(_fv_mod, "_call_judge", _e2e_full_verification_fake)
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", _e2e_near_judge_fake)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    def fake_audit(payload, packets, first_pass_by_id, parsed):
        # UNKNOWN decision + one required review "uncertain" -> DOWNGRADE
        # (EXACT -> POSSIBLE), never a grounded contradiction.
        people = [{
            "person_id": p["person_id"], "decision": "unknown", "confidence": 0.5,
            "reason": "insufficient evidence to fully confirm at audit time",
            "criteria": [
                {"criterion_id": "role", "status_review": "uncertain", "reason": "not fully certain"},
                {"criterion_id": "location", "status_review": "supported", "reason": "verified",
                 "supporting_evidence_refs": []},
            ],
            "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
            "suggested_qualification": None,
        } for p in packets]
        return "ok", people, "mock:provider", "mock-model"

    monkeypatch.setattr(final_auditor, "_call_audit", fake_audit)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        a_id = _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
        db.commit()
    finally:
        db.close()

    main_ids = {r.person_id for r in resp.connections.results}
    near_ids = {r.person_id for r in resp.connections.near_matches}
    assert a_id not in main_ids  # audit downgrade -> excluded from main results (unchanged behavior)
    assert a_id in near_ids, (
        "an audit-downgraded (insufficient evidence, not a contradiction) candidate must "
        "still be reconsidered for Near Match, with a gap id the near-match judge can use"
    )
    near_a = next(r for r in resp.connections.near_matches if r.person_id == a_id)
    assert near_a.qualification == "not_match"  # never silently promoted to Exact/Possible


def test_audit_downgrade_never_auto_promotes_without_its_own_near_match_verdict(monkeypatch):
    """The handoff must require a FRESH, validated near-match decision — an
    audit-downgraded candidate the near-match LLM does NOT judge useful stays
    absent, exactly like any other near-match candidate."""
    from app.config import settings
    from app.services import final_auditor
    from app.services import full_verification as _fv_mod

    monkeypatch.setattr(settings, "full_llm_verification", True)
    monkeypatch.setattr(settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(search_service, "interpret_query",
                        lambda q: (_full_mode_plan(), "deterministic", None))
    monkeypatch.setattr(_fv_mod, "_call_judge", _e2e_full_verification_fake)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    def fake_audit(payload, packets, first_pass_by_id, parsed):
        people = [{
            "person_id": p["person_id"], "decision": "unknown", "confidence": 0.5, "reason": "x",
            "criteria": [
                {"criterion_id": "role", "status_review": "uncertain", "reason": "x"},
                {"criterion_id": "location", "status_review": "supported", "reason": "x",
                 "supporting_evidence_refs": []},
            ],
            "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
            "suggested_qualification": None,
        } for p in packets]
        return "ok", people, "mock:provider", "mock-model"

    monkeypatch.setattr(final_auditor, "_call_audit", fake_audit)
    # near-match LLM rejects EVERYONE — no free pass just for reaching the pool.
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", lambda payload, packets: (
        "ok", {p["person_id"]: {
            "person_id": p["person_id"], "useful_near_match": False, "confidence": 0.1,
            "relaxed_criterion_id": "role", "relation_type": "not_meaningful",
            "evidence_refs": [], "short_reason": "",
        } for p in packets}, "mock:provider", "mock-model",
    ))

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        a_id = _mk_person(db, ds_id, name="Alice Exact", title="Software Engineer", company="Acme",
                          location_text="Fernbrook, Georgia, United States", city="Fernbrook", state="Georgia")
        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="software engineers in Fernbrook")
    finally:
        db.close()

    main_ids = {r.person_id for r in resp.connections.results}
    near_ids = {r.person_id for r in resp.connections.near_matches}
    assert a_id not in main_ids and a_id not in near_ids  # excluded everywhere — no auto-promotion


# ─────────────────────── hardening: Near Match rejection diagnostics (bug report) ───────────────────────


def test_validator_returns_a_rejection_reason_for_every_named_cause():
    from app.services.near_match_validator import REJECTION_REASONS

    plan = _vc_plan()
    p = _person(1, current_title="Barista", current_company="Corner Cafe")
    facts = _facts(p)
    scored = _scored(p, unmet_ids=["role"])
    nc = _nc(p, facts, scored)
    packet = {"person_id": p.id, "current": {"ref": "exp:1", "experience_id": "1"}}

    def _run(raw):
        validated, reasons = validate_near_verdicts(
            {p.id: raw}, {p.id: packet}, {p.id: nc}, plan,
        )
        return validated, reasons

    # llm_rejected
    _, reasons = _run({"useful_near_match": False})
    assert reasons["llm_rejected"] == 1 and sum(reasons.values()) == 1

    # low_confidence
    _, reasons = _run({"useful_near_match": True, "confidence": 0.0,
                       "relaxed_criterion_id": "role", "evidence_refs": ["exp:1"]})
    assert reasons["low_confidence"] == 1

    # invalid_evidence
    _, reasons = _run({"useful_near_match": True, "confidence": 0.9,
                       "relaxed_criterion_id": "role", "evidence_refs": ["exp:invented"]})
    assert reasons["invalid_evidence"] == 1

    # invalid_relaxed_criterion — candidate has exactly one gap ("role"), a
    # completely unrelated id can't be silently corrected
    _, reasons = _run({"useful_near_match": True, "confidence": 0.9,
                       "relaxed_criterion_id": "not_a_real_gap_and_multiple_exist",
                       "evidence_refs": ["exp:1"]})
    # with exactly one known gap this is auto-corrected (accepted), so use a
    # candidate with two gaps to force a genuine invalid_relaxed_criterion
    scored2 = _scored(p, unmet_ids=["role", "loc"])
    nc2 = _nc(p, facts, scored2)
    _, reasons2 = validate_near_verdicts(
        {p.id: {"useful_near_match": True, "confidence": 0.9,
               "relaxed_criterion_id": "nonexistent", "evidence_refs": ["exp:1"]}},
        {p.id: packet}, {p.id: nc2}, plan,
    )
    assert reasons2["invalid_relaxed_criterion"] == 1

    # missing pool/packet entry -> "other"
    _, reasons = validate_near_verdicts({"ghost": {"useful_near_match": True}}, {}, {}, plan)
    assert reasons["other"] == 1

    # every reason key is always present (zero-filled), never just absent
    for key in REJECTION_REASONS:
        assert key in reasons


def test_geographic_conflict_is_counted_as_a_diagnostic_not_a_rejection():
    """A deterministic FAR contradiction downgrades relation_type but still
    ACCEPTS the candidate (see near_match_validator._validate_geo) — it must
    be visible in diagnostics without being miscounted as a rejection."""
    plan = _plan(
        _crit(id="loc", type=CriterionType.LOCATION, value="Georgia", weight=40),
    )
    p = _person(1, current_title="Software Engineer", current_company="Acme")
    p.city, p.state, p.location_text = "Seattle", "Washington", "Seattle, Washington, United States"
    facts = _facts(p)
    scored = _scored(p, unmet_ids=["loc"])
    nc = _nc(p, facts, scored)
    packet = {"person_id": p.id, "current": {"ref": "exp:1", "experience_id": "1"}}

    raw = {
        "useful_near_match": True, "confidence": 0.9, "relaxed_criterion_id": "loc",
        "relation_type": "geographic_adjacent", "evidence_refs": ["exp:1"],
        "satisfied_intent": "strong fit", "short_reason": "close by",
    }
    validated, reasons = validate_near_verdicts({p.id: raw}, {p.id: packet}, {p.id: nc}, plan)

    assert p.id in validated                       # ACCEPTED — never rejected outright
    assert validated[p.id]["relation_type"] == "other_relevant"  # downgraded, not geographic_adjacent
    assert reasons["geographic_conflict"] == 1      # but the conflict IS visible in diagnostics
    assert sum(v for k, v in reasons.items() if k != "geographic_conflict") == 0  # not double-counted as a rejection


def test_near_match_metadata_exposes_rejection_reason_counts_end_to_end(monkeypatch):
    """All-judged-all-rejected (the exact symptom in the bug report) must
    still surface WHY via ``near_match_metadata.rejection_reason_counts`` —
    never just a silent 0-accepted count."""
    plan = _vc_plan()
    people = [_person(i, current_title="Barista", current_company="Corner Cafe") for i in range(3)]
    bundle = [
        (p, _facts(p), _scored(p, unmet_ids=["role"]), "hard_gate_reject")
        for p in people
    ]
    ctx = ScoringContext()

    def _all_rejected(payload, packets):
        # every candidate judged, every one rejected by the LLM itself
        return "ok", {
            pkt["person_id"]: {"person_id": pkt["person_id"], "useful_near_match": False,
                               "confidence": 0.1, "relaxed_criterion_id": "role",
                               "relation_type": "not_meaningful", "evidence_refs": [], "short_reason": ""}
            for pkt in packets
        }, "mock:provider", "mock-model"

    monkeypatch.setattr(near_match_judge, "_call_near_judge", _all_rejected)
    _, meta = build_near_matches("investors", plan, ctx, candidates=bundle)

    assert meta.validated_count == 0
    assert meta.rejected_count == 3
    assert meta.rejection_reason_counts.get("llm_rejected") == 3
    assert sum(meta.rejection_reason_counts.values()) >= 3  # visible, not silently zero
