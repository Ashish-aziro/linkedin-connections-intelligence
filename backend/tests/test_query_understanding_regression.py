"""Real search-pipeline regressions for general query understanding.

Mocked Anthropic query interpretation (structural transport plan only) + the
fixture / synthesized profiles; every stage after interpretation (fact
safety-net, candidate gate, deterministic scoring, qualification, ranking,
evidence) runs for real. Judge / audit are simply unavailable (no keys), which
in this pre-PART-6 build means deterministic results still stand — that is the
desired behavior and is asserted, not worked around.

No literal example word (Atlanta / Chicago / nonprofit) drives any production
code; these are regression fixtures only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import repositories as repo
from app.config import settings
from app.constants import CriterionType, Qualification
from app.database import SessionLocal
from app.schemas import LenientSearchPlan

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


@pytest.fixture(autouse=True)
def _llm_on(monkeypatch):
    monkeypatch.setattr(settings, "llm_query_interpretation", True)


def _mock_plan(monkeypatch, plan: dict):
    def fake(system, user, schema, **kw):  # noqa: ARG001
        return schema.model_validate(plan), "anthropic:paid", "claude-haiku-4-5-20251001"
    monkeypatch.setattr("app.services.query_interpreter.generate_structured", fake)


def _enriched(client) -> str:
    ds = client.post("/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}
                     ).json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds}/enrich")
    return ds


def _seed(db, ds_id, *, name, location, title, company, exp_desc):
    slug = name.lower().replace(" ", "-")
    p = repo.add_person(
        db, dataset_id=ds_id, is_connection=True,
        linkedin_url=f"https://www.linkedin.com/in/{slug}", full_name=name,
        first_name=name.split()[0], last_name=name.split()[-1],
        current_title=title, current_company=company,
        location_text=location, city=location.split(",")[0].strip(),
        enrichment_state="READY", profile_completeness=80,
    )
    repo.replace_experiences(db, p.id, [{
        "position": title, "company_name": company, "is_current": True,
        "start_year": 2019, "description": exp_desc,
    }])
    from app.services.search_text import build_search_text
    from app.services.embeddings import embed_text
    txt = build_search_text(db, p)
    repo.upsert_embedding(db, p.id, model=settings.embedding_model,
                          dim=settings.embedding_dim, vector=embed_text(txt), search_text=txt)
    return p.id


# ─────────────────────── §20 — role + explicit location must gate ───────────────────────


def test_required_location_rejects_contradictory_candidates(client, monkeypatch):
    """The reported bug: role + explicit location ended up with EVERY profile
    viable. Generalized — a required LOCATION criterion must be present and a
    candidate whose known location clearly conflicts must NOT be an exact match.
    """
    ds = _enriched(client)
    db = SessionLocal()
    try:
        # two synthetic engineers: one in the target metro, one far away
        here = _seed(db, ds, name="Local Engineer", location="Seattle, Washington, United States",
                     title="Senior Software Engineer", company="Acme",
                     exp_desc="Backend distributed systems, senior engineer.")
        far = _seed(db, ds, name="Remote Engineer", location="Bangalore, Karnataka, India",
                    title="Senior Software Engineer", company="Acme",
                    exp_desc="Backend distributed systems, senior engineer.")
        db.commit()
    finally:
        db.close()

    _mock_plan(monkeypatch, {
        "intent": "find_people",
        "criteria": [
            {"id": "role", "type": "role_function", "concept": "software engineering",
             "required": True, "weight": 40, "operator": None},
            {"id": "sen", "type": "seniority", "value": "senior", "required": True, "weight": 25},
            {"id": "loc", "type": "location", "value": "Seattle", "required": True, "weight": 35,
             "operator": None},
        ],
    })
    body = client.post("/search", json={"dataset_id": ds, "query": "senior software engineers in seattle"}).json()

    iq = body["interpreted_query"]
    loc_crit = [c for c in iq["criteria"] if c["type"] == "location"]
    assert loc_crit and loc_crit[0]["required"] is True          # location present + required

    results = {r["name"]: r for r in body["connections"]["results"]}
    near = {r["name"] for r in body["connections"]["near_matches"]}

    # the far-away engineer is NOT an exact/possible result (gated or near-match)
    assert "Remote Engineer" not in results or \
        results["Remote Engineer"]["qualification"] == Qualification.NOT_MATCH
    # the local engineer survives as a real candidate
    assert "Local Engineer" in results
    assert results["Local Engineer"]["qualification"] in (
        Qualification.EXACT_MATCH, Qualification.POSSIBLE_MATCH)
    # NOT every profile viable
    assert body["connections"]["total_candidates"] < 900


# ─────────────────────── §21 — nonprofit + Chicago regression fixture ───────────────────────


def test_location_plus_industry_concept_regression(client, monkeypatch):
    ds = _enriched(client)
    db = SessionLocal()
    try:
        match = _seed(
            db, ds, name="Cara Local", location="Chicago, Illinois, United States",
            title="Program Director", company="Chicago Community Trust",
            exp_desc="Program director at a nonprofit foundation; grantmaking, community programs.",
        )
        wrong_city = _seed(
            db, ds, name="Nora Elsewhere", location="Denver, Colorado, United States",
            title="Program Director", company="Mountain Nonprofit Alliance",
            exp_desc="Program director at a nonprofit; grantmaking and community programs.",
        )
        db.commit()
    finally:
        db.close()

    _mock_plan(monkeypatch, {
        "intent": "professional_recommendation",
        "criteria": [
            {"id": "loc", "type": "location", "value": "Chicago", "required": True, "weight": 40,
             "operator": None},
            {"id": "np", "type": "company_category", "concept": "nonprofit",
             "scope": "any_experience", "required": True, "weight": 35, "operator": None,
             "value": None},
            {"id": "ind", "type": "industry_experience",
             "concept": "professional experience in the nonprofit sector",
             "required": True, "weight": 25, "operator": None},
        ],
    })
    body = client.post("/search", json={
        "dataset_id": ds, "query": "people with nonprofit experience in Chicago"}).json()

    iq = body["interpreted_query"]
    assert [c for c in iq["criteria"] if c["type"] == "location" and c["required"]]
    assert [c for c in iq["criteria"]
            if c["type"] in ("company_category", "industry_experience")
            and "nonprofit" in (c.get("concept") or c.get("value") or "").lower()]

    names = {r["name"]: r for r in body["connections"]["results"]}
    near = {r["name"] for r in body["connections"]["near_matches"]}

    # wrong-city candidate fails the required location (not an exact/possible result)
    assert "Nora Elsewhere" not in names or names["Nora Elsewhere"]["qualification"] == Qualification.NOT_MATCH
    # the Chicago nonprofit person is surfaced (exact or possible — conservative when
    # the semantic side is unverified) with evidence from their real profile
    if "Cara Local" in names:
        r = names["Cara Local"]
        assert r["qualification"] in (Qualification.EXACT_MATCH, Qualification.POSSIBLE_MATCH)
        ev_text = " ".join(e["text"].lower() for e in r["evidence"]) + " " + r["reason"].lower()
        assert "chicago" in ev_text or any(
            "chicago" in (x.get("company_name") or "").lower()
            or "chicago" in (x.get("location") or "").lower()
            for x in r["relevant_experience"]
        ) or "nonprofit" in ev_text
    else:
        assert "Cara Local" in near


# ─────────────────────── partial verification still shows conservative results (§0/§12/§22) ───────────────────────


def test_partial_ai_verification_does_not_erase_results(client, monkeypatch):
    ds = _enriched(client)
    _mock_plan(monkeypatch, {
        "criteria": [
            {"id": "co", "type": "past_company", "value": "Amazon", "scope": "any_experience",
             "required": True, "weight": 60, "operator": None},
            {"id": "sk", "type": "skill", "value": "aws", "required": False, "weight": 40},
        ],
    })
    # judge / audit unavailable (no keys) — pre-PART-6 behavior: results stand
    body = client.post("/search", json={"dataset_id": ds, "query": "people who worked at Amazon and know AWS"}).json()

    assert body.get("search_status") in (None, "success", "success_with_fallback")  # NOT verification_incomplete
    assert body["connections"]["results"], body                     # results NOT erased
    top = body["connections"]["results"][0]
    assert top["match_score"] > 0
    scores = [r["match_score"] for r in body["connections"]["results"]]
    assert scores == sorted(scores, reverse=True)                    # ranked by deterministic score


def test_no_part6_no_results_policy(monkeypatch):
    """The mandatory-LLM / mandatory-audit / results=[] policy must NOT be back."""
    assert not hasattr(settings, "require_llm_for_results") or \
        getattr(settings, "require_llm_for_results", False) is False
    assert not hasattr(settings, "search_require_final_audit") or \
        getattr(settings, "search_require_final_audit", False) is False
