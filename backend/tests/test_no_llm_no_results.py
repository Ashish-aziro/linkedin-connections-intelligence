"""V4 PART 6 B2 / B12 / B13 / B32 — NO LLM ⇒ NO user-visible results.

Deterministic code still filters / scores / validates internally, but a search
returns candidate cards only when an LLM successfully participated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.constants import SearchStatus, VerificationStatus
from app.schemas import LenientSearchPlan

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


@pytest.fixture
def ds_id(client):
    d = client.post("/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}
                    ).json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{d}/enrich")
    return d


@pytest.fixture(autouse=True)
def _require_llm(monkeypatch):
    monkeypatch.setattr(settings, "require_llm_for_results", True)
    monkeypatch.setattr(settings, "llm_query_interpretation", True)


def _plan(*crits, **kw):
    return LenientSearchPlan.model_validate({"criteria": list(crits), **kw})


def _mock_interpretation(monkeypatch, plan, provider="anthropic:paid", model="claude-sonnet-5"):
    def fake(system, user, schema, **kw):  # noqa: ARG001
        return (schema.model_validate(plan.model_dump()), provider, model)

    monkeypatch.setattr("app.services.query_interpreter.generate_structured", fake)


def _mock_interpretation_fails(monkeypatch):
    monkeypatch.setattr("app.services.query_interpreter.generate_structured",
                        lambda *a, **k: None)  # -> deterministic parser


# ── 1 / 2 — no LLM plan ⇒ AI_UNAVAILABLE, zero results ──────────────────────


def test_deterministic_interpretation_fallback_returns_zero_results(client, ds_id, monkeypatch):
    _mock_interpretation_fails(monkeypatch)
    r = client.post("/search", json={"dataset_id": ds_id, "query": "people who worked at Amazon"}).json()
    assert r["search_status"] == SearchStatus.AI_UNAVAILABLE
    assert r["connections"]["results"] == []
    assert r["connections"]["near_matches"] == []
    assert r["llm_verified"] is False


def test_llm_disabled_entirely_returns_zero_results(client, ds_id, monkeypatch):
    monkeypatch.setattr(settings, "llm_query_interpretation", False)
    r = client.post("/search", json={"dataset_id": ds_id, "query": "engineers at Google"}).json()
    assert r["search_status"] == SearchStatus.AI_UNAVAILABLE
    assert r["connections"]["results"] == []


# ── 3 — required semantic judge produced nothing ⇒ VERIFICATION_INCOMPLETE ───


def test_semantic_judge_unavailable_returns_zero_results(client, ds_id, monkeypatch):
    _mock_interpretation(monkeypatch, _plan(
        {"id": "np", "type": "professional_concept", "concept": "nonprofit experience",
         "required": True, "weight": 100},
    ))
    # judge path: every provider exhausted
    monkeypatch.setattr("app.services.semantic_judge.generate_structured",
                        lambda *a, **k: (None, {"attempts": [{"provider": None, "status": "offline"}]})
                        if k.get("return_meta") else None)
    r = client.post("/search", json={"dataset_id": ds_id, "query": "people with nonprofit experience"}).json()
    assert r["search_status"] == SearchStatus.VERIFICATION_INCOMPLETE
    assert r["connections"]["results"] == []
    assert r["connections"]["near_matches"] == []
    assert r["verification_status"] == VerificationStatus.INCOMPLETE


# ── 4 — deadline expires before verification ⇒ zero results ─────────────────


def test_deadline_before_verification_returns_zero_results(client, ds_id, monkeypatch):
    _mock_interpretation(monkeypatch, _plan(
        {"id": "np", "type": "professional_concept", "concept": "nonprofit experience",
         "required": True, "weight": 100},
    ))
    monkeypatch.setattr(settings, "search_max_seconds", 0.0001)
    import time as _t
    _t.sleep(0.001)
    monkeypatch.setattr("app.services.semantic_judge.generate_structured",
                        lambda *a, **k: (None, {"attempts": []}) if k.get("return_meta") else None)
    r = client.post("/search", json={"dataset_id": ds_id, "query": "nonprofit people"}).json()
    assert r["search_status"] in (SearchStatus.VERIFICATION_INCOMPLETE, SearchStatus.AI_UNAVAILABLE)
    assert r["connections"]["results"] == []


# ── 5 — Anthropic verifies the search ⇒ normal results ─────────────────────


def test_anthropic_success_shows_results(client, ds_id, monkeypatch):
    # a plan with NO required semantic criterion — deterministic facts are
    # authoritative, judge not required, Anthropic interpreted it.
    _mock_interpretation(monkeypatch, _plan(
        {"id": "co", "type": "past_company", "value": "Amazon", "required": True, "weight": 100},
    ))
    r = client.post("/search", json={"dataset_id": ds_id, "query": "former Amazon people"}).json()
    assert r["search_status"] == SearchStatus.SUCCESS
    assert r["ai_provider"] == "anthropic"
    assert r["anthropic_attempted"] is True
    assert r["anthropic_succeeded"] is True
    assert r["fallback_used"] is False


# ── 8 — SEARCH_REQUIRE_ANTHROPIC ──────────────────────────────────────────


def test_require_anthropic_true_blocks_fallback_provider(client, ds_id, monkeypatch):
    monkeypatch.setattr(settings, "search_require_anthropic", True)
    _mock_interpretation(monkeypatch, _plan(
        {"id": "co", "type": "past_company", "value": "Amazon", "required": True, "weight": 100},
    ), provider="groq:primary", model="gpt-oss-120b")
    r = client.post("/search", json={"dataset_id": ds_id, "query": "amazon people"}).json()
    assert r["search_status"] == SearchStatus.AI_UNAVAILABLE
    assert r["connections"]["results"] == []


def test_require_anthropic_false_allows_fallback(client, ds_id, monkeypatch):
    monkeypatch.setattr(settings, "search_require_anthropic", False)
    _mock_interpretation(monkeypatch, _plan(
        {"id": "co", "type": "past_company", "value": "Amazon", "required": True, "weight": 100},
    ), provider="groq:primary", model="gpt-oss-120b")
    r = client.post("/search", json={"dataset_id": ds_id, "query": "amazon people"}).json()
    assert r["search_status"] == SearchStatus.SUCCESS_WITH_FALLBACK
    assert r["fallback_used"] is True
    assert r["ai_provider"] == "groq_primary"


# ── 9 / 10 — failed search never leaks candidates, even on reload ──────────


def test_failed_search_persists_zero_results_and_reload_stays_empty(client, ds_id, monkeypatch):
    _mock_interpretation_fails(monkeypatch)
    sid = client.post("/search", json={"dataset_id": ds_id, "query": "amazon people"}).json()["search_id"]

    reloaded = client.get(f"/search/{sid}").json()
    assert reloaded["search_status"] == SearchStatus.AI_UNAVAILABLE
    assert reloaded["connections"]["results"] == []
    assert reloaded["connections"]["near_matches"] == []

    from app.models import SearchResult
    from app.database import SessionLocal
    with SessionLocal() as s:
        assert s.query(SearchResult).filter_by(search_id=sid).count() == 0


def test_reload_of_failed_search_runs_no_llm(client, ds_id, monkeypatch):
    _mock_interpretation_fails(monkeypatch)
    sid = client.post("/search", json={"dataset_id": ds_id, "query": "amazon"}).json()["search_id"]

    def boom(*a, **k):  # noqa: ARG001
        raise AssertionError("reload must not call the interpreter")

    monkeypatch.setattr("app.services.search_service.interpret_query", boom)
    reloaded = client.get(f"/search/{sid}").json()
    assert reloaded["connections"]["results"] == []
