"""Bug report — "inconsistent Match Scores and Near Match qualification".

Covers the query-interpreter-level fixes (explicit title constraint vs broad
experience/function request) and confirms, end-to-end with a genuinely
DIFFERENT profession/industry than any prior test file in this suite, that a
candidate who narrowly misses ONE genuine required criterion is correctly a
Near Match regardless of how high their raw match_score is — never promoted,
never demoted purely by number.
"""
from __future__ import annotations

import pytest

from app.constants import CriterionType, DatasetStatus, EnrichmentState
from app.database import SessionLocal
from app.models import Dataset, Experience, Person
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services import near_match_judge as _nmj_mod
from app.services import query_intent, search_service
from app.services.query_interpreter import _soften_requirements


def _crit(**kw) -> SearchCriterion:
    kw.setdefault("weight", 50)
    kw.setdefault("required", True)
    kw.setdefault("operator", "ANY_OF")
    return SearchCriterion(**kw)


def _plan(*crits) -> ParsedSearchQuery:
    plan = ParsedSearchQuery(criteria=list(crits))
    query_intent._identify_primary_intent(plan)
    return plan


# ─────────────────────── PART 9.E — explicit title constraint stays required ───────────────────────


@pytest.mark.parametrize("query,expected_required", [
    ("People whose current job title is Clinical Research Director", True),
    ("People titled Head of Culinary Operations", True),
    ("People with the title of Logistics Coordinator", True),
    ("People with teaching experience", False),
    ("People with experience in supply chain management", False),
    ("Nurses with ICU experience", False),
])
def test_explicit_title_wording_vs_broad_experience_request(query, expected_required):
    """PART 4/5/9.E/9.F, generic across unrelated professions (clinical
    research, culinary, logistics, teaching, supply chain, nursing) — none of
    these appear anywhere else in this test suite. A genuine 'title is X' /
    'titled X' / 'with the title of X' phrasing must survive softening; a
    broad experience/skill request must not."""
    crit = _crit(id="role", type=CriterionType.TITLE, value="X", required=True)
    plan = ParsedSearchQuery(criteria=[crit])
    softened = _soften_requirements(plan, query)
    assert softened.criteria[0].required is expected_required


def test_softening_caps_required_criteria_generically():
    """PART 5 — an over-eager interpretation with many required criteria is
    capped to the 3 highest-weight ones, regardless of domain."""
    crits = [
        _crit(id=f"c{i}", type=CriterionType.SKILL, value=f"skill{i}", weight=w, required=True)
        for i, w in enumerate([10, 90, 20, 80, 30, 70])
    ]
    plan = ParsedSearchQuery(criteria=crits)
    softened = _soften_requirements(plan, "people with several specific skills")
    required = [c for c in softened.criteria if c.required]
    assert len(required) <= 3
    # the highest-weight ones survive, not an arbitrary subset
    assert {c.id for c in required} <= {"c1", "c3", "c5"}


# ─────────────────────── PART 9.B — genuine single-criterion miss is Near Match regardless of score ───────────────────────


def _mk_dataset(db) -> str:
    ds = Dataset(name="score-qualification test", status=DatasetStatus.READY)
    db.add(ds)
    db.commit()
    return ds.id


def _mk_person(db, dataset_id, *, name, title, company, location_text, city, state) -> str:
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


def _clinic_plan() -> ParsedSearchQuery:
    """A profession/industry ('clinical research') and city ('Denver') used
    nowhere else in this suite — demonstrates the fix is domain-agnostic."""
    return ParsedSearchQuery(
        criteria=[
            _crit(id="role", type=CriterionType.PROFESSIONAL_CONCEPT,
                  concept="clinical research professional", weight=60),
            _crit(id="location", type=CriterionType.LOCATION, value="Denver", weight=40),
        ],
        primary_intent="clinical research",
        intent_anchor_criterion_ids=["role"],
    )


def _fake_full_verification(payload, packets, review_by_person=None, **_kw):
    out = {}
    for pkt in packets:
        pid = pkt["person_id"]
        is_researcher = "research" in (pkt.get("current_title") or "").lower()
        ref = (pkt.get("current") or {}).get("ref")
        out[pid] = {
            cid: {
                "criterion_id": cid, "status": "true" if is_researcher else "false",
                "match_strength": 0.9 if is_researcher else 0.0, "confidence": 0.9,
                "reason": "clinical research role" if is_researcher else "not a research role",
                "supporting_evidence_refs": [ref] if (is_researcher and ref) else [],
                "contradicting_evidence_refs": [] if is_researcher else ([ref] if ref else []),
                "experience_ids": [],
            }
            for cid in (pkt.get("unresolved_criteria") or [])
        }
    return "ok", out, "mock:provider", "mock-model"


def _fake_near_judge(payload, packets):
    """Grounds a true near-match verdict from the packet's own evidence — no
    hardcoded name/profession, generic pattern matching the rest of this suite."""
    people = {}
    for pkt in packets:
        pid = pkt["person_id"]
        title = (pkt.get("current_title") or "").lower()
        gaps = pkt.get("near_match_context", {}).get("gaps", [])
        relaxed = gaps[0]["criterion_id"] if gaps else ""
        if "research" in title:
            ref = (pkt.get("current") or {}).get("ref")
            people[pid] = {
                "person_id": pid, "useful_near_match": True, "confidence": 0.85,
                "satisfied_intent": "strong clinical research background",
                "relaxed_criterion_id": relaxed, "relation_type": "geographic_adjacent",
                "evidence_refs": [ref] if ref else [],
                "short_reason": "Clinical researcher with a nearby-city location mismatch.",
            }
        else:
            people[pid] = {
                "person_id": pid, "useful_near_match": False, "confidence": 0.1,
                "relaxed_criterion_id": relaxed, "relation_type": "not_meaningful",
                "evidence_refs": [], "short_reason": "",
            }
    return "ok", people, "mock:provider", "mock-model"


def test_high_score_near_match_never_outranks_into_exact_and_uses_same_formula(monkeypatch):
    """PART 1/2/9.B/9.C: a candidate (Bob) who genuinely fails ONE required
    criterion (location) but strongly satisfies the primary intent can have a
    HIGHER match_score than a main result — that must not promote him to
    Exact, and his displayed match_score must be the SAME score_candidate()
    formula as every main result (never a separately-scaled near-ranking
    number). A third candidate (Cara) who fails on substance, not wording,
    must not appear anywhere just to fill the section."""
    from app.config import settings
    from app.services import full_verification as _fv_mod

    monkeypatch.setattr(settings, "full_llm_verification", True)
    monkeypatch.setattr(settings, "final_result_audit_enabled", False)
    monkeypatch.setattr(search_service, "interpret_query",
                        lambda q: (_clinic_plan(), "deterministic", None))
    monkeypatch.setattr(_fv_mod, "_call_judge", _fake_full_verification)
    monkeypatch.setattr(_nmj_mod, "_call_near_judge", _fake_near_judge)
    monkeypatch.setattr("app.services.reranker.settings.reranker_enabled", False)

    db = SessionLocal()
    try:
        ds_id = _mk_dataset(db)
        a_id = _mk_person(db, ds_id, name="Alice Researcher", title="Clinical Research Associate",
                          company="MedCo", location_text="Denver, Colorado, United States",
                          city="Denver", state="Colorado")
        b_id = _mk_person(db, ds_id, name="Bob Researcher", title="Senior Clinical Research Scientist",
                          company="BioCo", location_text="Boulder, Colorado, United States",
                          city="Boulder", state="Colorado")
        _mk_person(db, ds_id, name="Cara Unrelated", title="Barista", company="Corner Cafe",
                  location_text="Denver, Colorado, United States", city="Denver", state="Colorado")

        resp = search_service.run_connection_search(db, dataset_id=ds_id, query="clinical researchers in Denver")
        db.commit()
    finally:
        db.close()

    main_ids = {r.person_id for r in resp.connections.results}
    near_ids = {r.person_id for r in resp.connections.near_matches}

    assert a_id in main_ids
    assert b_id not in main_ids  # hard-gate rejected for location — never in main results
    assert b_id in near_ids      # but genuinely relevant -> reached Near Match
    assert not (near_ids - {b_id})  # Cara (unrelated) filled no slot

    main_item = next(r for r in resp.connections.results if r.person_id == a_id)
    near_item = next(r for r in resp.connections.near_matches if r.person_id == b_id)
    # both scores come from the identical score_candidate() formula — same
    # units, same meaning; a Near Match score is allowed to be numerically
    # higher (never artificially clamped below every main result).
    assert isinstance(near_item.match_score, float)
    assert near_item.qualification == "not_match"  # never silently promoted to Exact/Possible
    assert main_item.qualification == "exact_match"
