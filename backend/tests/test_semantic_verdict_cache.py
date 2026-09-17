"""TASK 4 — semantic verdict cache tests.

Reuses the FULL SONNET VERIFICATION fixtures from ``test_full_verification``
(same mocking seam: ``full_verification._call_judge``, never a real Anthropic
call). Each test counts how many BATCHES were actually sent to verify the
cache genuinely skips already-answered (person, criterion) pairs, not just
that the search still returns a result.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import repositories as repo
from app.config import settings
from app.database import SessionLocal
from app.models import SemanticVerdictCache
from app.services import semantic_verdict_cache as vcache
from tests.test_full_verification import CID, _PLAN, _enriched, _fake_call, _full_mode  # noqa: F401

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


def _batch_call_count(seen: list) -> int:
    return len(seen)


def test_second_identical_search_hits_the_cache(client, monkeypatch):
    ds = _enriched(client)
    seen1: list = []
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call(capture=seen1))
    first = client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()
    assert first["judge_metadata"]["cache_hits"] == 0  # nothing cached yet

    seen2: list = []
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call(capture=seen2))
    second = client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()

    assert second["judge_metadata"]["cache_hits"] > 0
    assert _batch_call_count(seen2) == 0  # every candidate was fully cached — zero NEW batches
    assert second["judge_metadata"]["cache_fully_cached_candidates"] == \
        second["judge_metadata"]["filtered_candidate_count"]
    # same evidence, same criterion, same query context -> same qualification outcome
    n1 = {r["name"]: r["qualification"] for r in first["connections"]["results"]}
    n2 = {r["name"]: r["qualification"] for r in second["connections"]["results"]}
    assert n1 == n2


def test_changing_evidence_invalidates_only_that_person(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())
    client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()

    # mutate ONE person's experience description -> their evidence_fingerprint changes
    db = SessionLocal()
    try:
        people = client.get(f"/datasets/{ds}/people").json()
        victim = people[0]["person_id"]
        repo.replace_experiences(db, victim, [{
            "position": "Totally Different Role", "company_name": "New Co",
            "is_current": True, "start_year": 2024, "description": "Brand new evidence.",
        }])
        db.commit()
    finally:
        db.close()

    seen: list = []
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call(capture=seen))
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()
    sent = {pid for batch in seen for pid in batch}
    assert victim in sent  # the CHANGED person needed fresh review
    assert body["judge_metadata"]["cache_hits"] > 0  # everyone ELSE still hit cache


def test_different_query_context_does_not_share_a_cache_hit(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())
    client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()

    from app.services import search_service

    different_plan = _PLAN.model_copy(deep=True)
    different_plan.intent = "a_totally_different_intent"
    monkeypatch.setattr(
        search_service, "interpret_query",
        lambda q: (different_plan, "anthropic:paid", "claude-sonnet-4-6"),
    )
    seen: list = []
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call(capture=seen))
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()
    assert _batch_call_count(seen) > 0  # different context -> NOT served from the first search's cache
    assert body["judge_metadata"]["cache_hits"] == 0


def test_only_validated_verdicts_are_cached_never_a_failed_batch(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr(
        "app.services.full_verification._call_judge",
        lambda *a, **k: ("failed", None, None, None),
    )
    monkeypatch.setattr("app.services.full_verification.generate_structured", lambda *a, **k: (None, {}))
    r = client.post("/search", json={"dataset_id": ds, "query": "cloud"})
    assert r.status_code == 503  # unrecoverable — matches test_full_verification's existing invariant

    with SessionLocal() as db:
        assert db.query(SemanticVerdictCache).count() == 0  # nothing written for a failed run


def test_cache_disabled_setting_skips_lookup_and_write(client, monkeypatch):
    monkeypatch.setattr(settings, "semantic_verdict_cache_enabled", False)
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())
    client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()

    seen: list = []
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call(capture=seen))
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()
    assert _batch_call_count(seen) > 0  # cache never consulted -> full re-verification every time
    assert body["judge_metadata"]["cache_hits"] == 0
    with SessionLocal() as db:
        assert db.query(SemanticVerdictCache).count() == 0


def test_criterion_key_ignores_the_query_generated_id():
    from app.constants import CriterionType
    from app.schemas import SearchCriterion

    a = SearchCriterion(id="crit-abc", type=CriterionType.SEMANTIC_CONCEPT,
                        concept="cloud infrastructure", required=True, weight=100)
    b = SearchCriterion(id="crit-xyz", type=CriterionType.SEMANTIC_CONCEPT,
                        concept="cloud infrastructure", required=True, weight=100)
    assert vcache.criterion_key(a) == vcache.criterion_key(b)

    c = SearchCriterion(id="crit-xyz", type=CriterionType.SEMANTIC_CONCEPT,
                        concept="a different concept entirely", required=True, weight=100)
    assert vcache.criterion_key(a) != vcache.criterion_key(c)


def test_saved_search_reload_still_makes_zero_llm_or_cache_calls(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())
    sid = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()["search_id"]

    def boom(*a, **k):
        raise AssertionError("reload must not touch the verdict cache or the LLM")

    monkeypatch.setattr(vcache, "lookup", boom)
    monkeypatch.setattr(vcache, "store", boom)
    monkeypatch.setattr("app.services.full_verification._call_judge", boom)
    reloaded = client.get(f"/search/{sid}").json()
    assert reloaded["query"] == "cloud"
