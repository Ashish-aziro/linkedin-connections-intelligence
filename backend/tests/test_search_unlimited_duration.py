"""fix/unlimited-search-duration — SEARCH_MAX_SECONDS=0 means NO application
wall-clock deadline on a search.

A broad semantic query must keep processing every required judge batch and the
final audit until verification finishes, however long that takes. Elapsed wall
time alone must never produce ``verification_incomplete`` / ``ai_unavailable``
when the setting is 0. A POSITIVE value is still honoured for operators who opt
into a safety ceiling.

No live LLM / Apify calls — the four provider seams are the same fakes the E2E
flow test uses; the "slow" variants just sleep between batches.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.config import Settings, settings
from app.constants import JudgeStatus, SearchStatus, VerificationStatus
from app.services import semantic_judge
from app.services.deadline import Deadline
from tests.test_e2e_application_flow import (
    QUERY,
    _fake_call_audit,
    _fake_call_judge,
    _fake_derive_semantics,
    _fake_interpretation,
)

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"

# per-batch pause that dwarfs any tiny positive budget used below, without
# making the suite slow.
_SLOW_BATCH_SECONDS = 0.15


def _slow_judge(pause=_SLOW_BATCH_SECONDS):
    inner = _fake_call_judge()

    def fake(payload, packets, unresolved_by_person=None):
        time.sleep(pause)
        return inner(payload, packets, unresolved_by_person)

    return fake


@pytest.fixture
def enriched(client, monkeypatch):
    """A small enriched dataset with the full LLM stack wired to fakes and every
    verification rule ON. Each test overrides ``search_max_seconds`` / the judge
    seam as needed."""
    monkeypatch.setattr(settings, "semantic_enabled", True)
    monkeypatch.setattr(settings, "llm_query_interpretation", True)
    monkeypatch.setattr(settings, "require_llm_for_results", True)
    monkeypatch.setattr(settings, "semantic_judge_enabled", True)
    monkeypatch.setattr(settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(settings, "semantic_judge_batch_size", 1)  # 1 batch / candidate -> several batches
    monkeypatch.setattr(settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(settings, "search_require_final_audit", True)
    monkeypatch.setattr(settings, "llm_reason_generation", False)

    monkeypatch.setattr("app.services.query_interpreter.generate_structured", _fake_interpretation())
    monkeypatch.setattr("app.services.semantic_llm.derive_semantics", _fake_derive_semantics())
    monkeypatch.setattr("app.services.semantic_judge._call_judge", _fake_call_judge())
    audit_calls: list = []
    monkeypatch.setattr("app.services.final_auditor._call_audit", _fake_call_audit(capture=audit_calls))

    ds_id = client.post(
        "/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}
    ).json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")
    return {"ds_id": ds_id, "audit_calls": audit_calls}


# ─────────────────────────── TEST A ───────────────────────────


def test_A_zero_is_the_default_and_deadline_never_expires():
    # the class default itself is 0 — running with no .env override does not
    # silently restore a deadline.
    assert Settings.model_fields["search_max_seconds"].default == 0.0

    for seconds in (0, 0.0, -5, None):
        d = Deadline(seconds)
        assert d.seconds is None
        assert d.remaining() is None
        assert d.expired() is False
        time.sleep(0.02)
        assert d.expired() is False  # still not expired after real elapsed time
        assert d.as_dict()["reached"] is False


# ─────────────────────────── TEST B ───────────────────────────


def test_B_slow_judge_runs_to_completion_when_unlimited(client, enriched, monkeypatch):
    monkeypatch.setattr(settings, "search_max_seconds", 0.0)
    monkeypatch.setattr("app.services.semantic_judge._call_judge", _slow_judge())

    r = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()

    jm = r["judge_metadata"]
    assert jm["status"] == JudgeStatus.FULL
    assert jm["judge_batch_count"] >= 2           # multiple batches actually ran
    assert jm["judge_failed_batches"] == 0
    assert r["search_status"] == SearchStatus.SUCCESS
    assert r["verification_status"] == VerificationStatus.COMPLETE


# ─────────────────────────── TEST C ───────────────────────────


def test_C_no_batches_skipped_by_wall_time_when_zero_but_a_tiny_budget_does_skip(
    client, enriched, monkeypatch
):
    # 1) tiny positive budget + a slow judge -> later batches ARE skipped and the
    #    search fails on elapsed wall time (proves wall time is the only lever).
    monkeypatch.setattr(settings, "search_max_seconds", 0.05)
    monkeypatch.setattr("app.services.semantic_judge._call_judge", _slow_judge())
    tiny = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()
    assert tiny["judge_metadata"]["deadline_reached"] is True
    assert tiny["search_status"] == SearchStatus.VERIFICATION_INCOMPLETE
    assert tiny["connections"]["results"] == []

    # 2) same dataset, same slow judge, ONLY the budget changes to 0 -> every
    #    batch runs, nothing skipped, results are returned.
    monkeypatch.setattr(settings, "search_max_seconds", 0.0)
    monkeypatch.setattr("app.services.semantic_judge._call_judge", _slow_judge())
    unlimited = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()
    jm = unlimited["judge_metadata"]
    assert jm["deadline_reached"] is False
    assert jm["judge_successful_batches"] == jm["judge_batch_count"]  # none skipped
    assert jm["omitted_people"] == 0
    assert unlimited["search_status"] == SearchStatus.SUCCESS
    assert unlimited["connections"]["returned"] >= 1


# ─────────────────────────── TEST D ───────────────────────────


def test_D_final_audit_still_runs_after_a_long_judge_stage(client, enriched, monkeypatch):
    monkeypatch.setattr(settings, "search_max_seconds", 0.0)
    monkeypatch.setattr("app.services.semantic_judge._call_judge", _slow_judge(0.25))

    r = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()

    am = r["audit_metadata"]
    assert am is not None and am["enabled"] is True
    assert am["status"] in ("full", "partial")
    assert am["successful_batches"] >= 1
    assert am["batch_count"] >= 1
    assert len(enriched["audit_calls"]) >= 1  # the audit LLM seam was actually hit
    assert am.get("deadline_reached") in (False, None)


# ─────────────────────────── TEST E ───────────────────────────


def test_E_full_verification_still_produces_llm_verified_results(client, enriched, monkeypatch):
    monkeypatch.setattr(settings, "search_max_seconds", 0.0)

    r = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()

    assert r["search_status"] == SearchStatus.SUCCESS
    assert r["llm_verified"] is True
    assert r["anthropic_succeeded"] is True
    verified = [x for x in r["connections"]["results"] if x["llm_verified"]]
    assert verified, r
    assert verified[0]["audit_decision"] == "approved"


# ─────────────────────────── TEST F ───────────────────────────


def test_F_real_provider_failure_still_fails_safely_even_when_unlimited(client, enriched, monkeypatch):
    monkeypatch.setattr(settings, "search_max_seconds", 0.0)

    # judge genuinely exhausted (not a timeout) -> verification_incomplete, no results
    monkeypatch.setattr(
        "app.services.semantic_judge._call_judge", lambda *a, **k: ("failed", None, None, None)
    )
    j = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()
    assert j["search_status"] == SearchStatus.VERIFICATION_INCOMPLETE
    assert j["connections"]["results"] == []
    assert j["connections"]["near_matches"] == []

    # interpretation genuinely unavailable -> ai_unavailable, no results
    monkeypatch.setattr("app.services.query_interpreter.generate_structured", lambda *a, **k: None)
    i = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()
    assert i["search_status"] == SearchStatus.AI_UNAVAILABLE
    assert i["connections"]["results"] == []


# ─────────────────────────── TEST G ───────────────────────────


def test_G_positive_budget_is_still_honoured_when_an_operator_configures_one(
    client, enriched, monkeypatch
):
    monkeypatch.setattr(settings, "search_max_seconds", 0.05)
    monkeypatch.setattr("app.services.semantic_judge._call_judge", _slow_judge())

    r = client.post("/search", json={"dataset_id": enriched["ds_id"], "query": QUERY}).json()
    # an explicitly configured ceiling still cuts a runaway search off
    assert r["search_status"] in (
        SearchStatus.VERIFICATION_INCOMPLETE,
        SearchStatus.AI_UNAVAILABLE,
    )
    assert r["connections"]["results"] == []


def test_G_unit_positive_deadline_object_still_expires():
    d = Deadline(0.01)
    assert d.expired() is False
    time.sleep(0.03)
    assert d.expired() is True
