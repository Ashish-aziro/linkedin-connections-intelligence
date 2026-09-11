"""FULL SONNET VERIFICATION EXPERIMENT — targeted tests.

Small, high-value set (10). Only the Sonnet batch call
(``full_verification._call_judge``) and query interpretation are mocked; the
packet builder, ``run_full_verification`` orchestration, the fact-consistency
validator, deterministic scoring, qualification, ranking and the router all run
for real.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import repositories as repo
from app.config import Settings, settings
from app.constants import CriterionType, FullVerificationStatus, Qualification
from app.database import SessionLocal
from app.schemas import ParsedSearchQuery, SearchCriterion

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"

CID = "cloud"
_PLAN = ParsedSearchQuery(criteria=[
    SearchCriterion(id=CID, type=CriterionType.PROFESSIONAL_CONCEPT,
                    concept="hands-on cloud infrastructure experience",
                    required=True, weight=100),
])


@pytest.fixture(autouse=True)
def _full_mode(monkeypatch):
    monkeypatch.setattr(settings, "full_llm_verification", True)
    monkeypatch.setattr(settings, "final_result_audit_enabled", False)
    monkeypatch.setattr(settings, "semantic_judge_enabled", True)
    # every search in this file uses the same required semantic plan
    monkeypatch.setattr(
        "app.services.search_service.interpret_query",
        lambda q: (_PLAN.model_copy(deep=True), "anthropic:paid", "claude-sonnet-4-6"),
    )


def _verdict(cid, status="true", *, exp_ref="exp:x"):
    return {
        "criterion_id": cid, "status": status,
        "match_strength": 0.9 if status == "true" else 0.0,
        "confidence": 0.9, "reason": "role covers building cloud infra",
        "supporting_evidence_refs": [exp_ref] if status == "true" else [],
        "contradicting_evidence_refs": [], "experience_ids": [],
    }


def _fake_call(*, statuses=None, omit=(), fail_first_n=0, truncate_multi=False, capture=None):
    """Build a fake ``_call_judge``. ``statuses``: pid -> verdict status."""
    state = {"n": 0}
    statuses = statuses or {}

    def fake(payload, packets, review_by_person=None):  # noqa: ARG001
        state["n"] += 1
        if capture is not None:
            capture.append([p["person_id"] for p in packets])
        if state["n"] <= fail_first_n:
            return "failed", None, None, None
        if truncate_multi and len(packets) > 1:
            return "truncated", None, None, None
        out = {}
        for pkt in packets:
            pid = pkt["person_id"]
            if pid in omit:
                continue
            cur = (pkt.get("current") or {}).get("experience_id") or "x"
            per = {}
            for cid in pkt.get("unresolved_criteria") or [CID]:
                per[cid] = _verdict(cid, statuses.get(pid, "true"), exp_ref=f"exp:{cur}")
            out[pid] = per
        return "ok", out, "anthropic:paid", "claude-sonnet-4-6"

    return fake


def _enriched(client) -> str:
    ds = client.post("/datasets", files={"file": ("c.csv", FIXTURE_CSV.read_bytes(), "text/csv")}
                     ).json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds}/enrich")
    return ds


def _viable_ids(client, ds):
    """The person_ids the hard-fact gate lets through for _PLAN (no hard facts
    in the plan -> everyone with an experience row is viable)."""
    return {p["person_id"] for p in client.get(f"/datasets/{ds}/people").json()}


# ─────────────────────────── 1 — every hard-gate survivor is sent ───────────────────────────


def test_every_filtered_candidate_is_sent_to_sonnet(client, monkeypatch):
    ds = _enriched(client)
    seen: list = []
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call(capture=seen))
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud infra people"}).json()

    sent = {pid for batch in seen for pid in batch}
    jm = body["judge_metadata"]
    assert jm["mode"] == "full_verification"
    assert jm["status"] == "complete"
    assert jm["filtered_candidate_count"] >= 5
    assert len(sent) == jm["filtered_candidate_count"]        # everyone was sent


# ─────────────────────────── 2 — success invariant ───────────────────────────


def test_success_invariant_filtered_equals_verified(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())
    jm = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()["judge_metadata"]
    assert jm["filtered_candidate_count"] == jm["sonnet_verified_candidate_count"]
    assert jm["sonnet_verified_candidate_count"] == jm["judge_candidate_count"]


# ─────────────────────────── 3 — an omitted candidate is recovered ───────────────────────────


def test_candidate_omitted_by_a_batch_is_recovered(client, monkeypatch):
    ds = _enriched(client)
    victim = sorted(_viable_ids(client, ds))[0]
    good = _fake_call()

    def wrapper(payload, packets, review_by_person=None):
        # the victim is omitted from every MULTI-person batch; the bounded
        # single-person retry (one packet) includes them.
        if len(packets) > 1:
            return _fake_call(omit=(victim,))(payload, packets, review_by_person)
        return good(payload, packets, review_by_person)

    monkeypatch.setattr("app.services.full_verification._call_judge", wrapper)
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()
    jm = body["judge_metadata"]
    assert jm["status"] == "complete"
    assert jm["single_person_retries"] >= 1
    assert jm["filtered_candidate_count"] == jm["sonnet_verified_candidate_count"]


# ─────────────────────────── 4 — truncation -> split, not partial ───────────────────────────


def test_truncation_splits_and_still_completes(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr(settings, "full_verification_batch_size", 50)  # one big batch -> forced to split
    monkeypatch.setattr("app.services.full_verification._call_judge",
                        _fake_call(truncate_multi=True))
    jm = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()["judge_metadata"]
    assert jm["status"] == "complete"
    assert jm["adaptive_splits"] >= 1
    assert jm["truncations"] >= 1
    assert jm["filtered_candidate_count"] == jm["sonnet_verified_candidate_count"]


# ─────────────────────────── 5 — unrecoverable failure fails the SEARCH ───────────────────────────


def test_unrecoverable_verification_returns_503_and_persists_nothing(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge",
                        lambda *a, **k: ("failed", None, None, None))
    monkeypatch.setattr("app.services.full_verification.generate_structured", lambda *a, **k: (None, {}))
    r = client.post("/search", json={"dataset_id": ds, "query": "cloud"})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["error"] == "verification_incomplete"
    assert detail["retryable"] is True

    # nothing persisted for a failed search
    with SessionLocal() as s:
        from app.models import SearchQuery
        assert s.query(SearchQuery).filter_by(dataset_id=ds).count() == 0
    assert client.get(f"/datasets/{ds}/searches").json() == []


# ─────────────────────────── 6 — INSUFFICIENT_EVIDENCE required -> excluded, no warning ───────────────────────────


def test_insufficient_evidence_required_criterion_is_excluded_from_main(client, monkeypatch):
    ds = _enriched(client)
    db = SessionLocal()
    try:
        ok = _seed(db, ds, name="Clear Cloud", desc="Runs AWS and Kubernetes infrastructure daily.")
        vague = _seed(db, ds, name="Vague Person", desc="Works on things.")
        db.commit()
    finally:
        db.close()
    monkeypatch.setattr(
        "app.services.full_verification._call_judge",
        _fake_call(statuses={ok: "true", vague: "unknown"}),
    )
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()

    names = {r["name"]: r for r in body["connections"]["results"]}
    assert "Vague Person" not in names                          # required INSUFFICIENT_EVIDENCE -> excluded
    for r in body["connections"]["results"]:
        assert r["qualification"] == "exact_match"              # only verified-TRUE candidates shown
        assert not r["uncertain_criteria"]                      # never "needs verification"
    assert body["connections"]["possible_match_count"] == 0
    assert body["judge_metadata"]["excluded_insufficient_evidence"] >= 1


# ─────────────────────────── 7 — deterministic score stays backend-owned ───────────────────────────


def test_match_score_is_deterministic_not_from_sonnet(client, monkeypatch):
    ds = _enriched(client)
    db = SessionLocal()
    try:
        _seed(db, ds, name="Strong Cloud",
              desc="Principal cloud architect — AWS, Kubernetes, Terraform, cost optimization at scale.")
        _seed(db, ds, name="Light Cloud", desc="Some AWS exposure on one project.")
        db.commit()
    finally:
        db.close()
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())  # both TRUE, same confidence
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud infrastructure"}).json()
    by_name = {r["name"]: r for r in body["connections"]["results"]}
    if "Strong Cloud" in by_name and "Light Cloud" in by_name:
        # identical Sonnet verdicts, but deterministic relevance/evidence differ
        assert by_name["Strong Cloud"]["match_score"] != by_name["Light Cloud"]["match_score"]
    # score is a real number in range, and no criterion asked Sonnet for a 0-100 score
    for r in body["connections"]["results"]:
        assert 0.0 <= r["match_score"] <= 100.0


# ─────────────────────────── 8 — saved search reload makes zero new LLM calls ───────────────────────────


def test_saved_search_reload_makes_no_llm_calls(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr("app.services.full_verification._call_judge", _fake_call())
    sid = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()["search_id"]

    def boom(*a, **k):
        raise AssertionError("reload must not call any LLM / verification step")

    monkeypatch.setattr("app.services.full_verification._call_judge", boom)
    monkeypatch.setattr("app.services.full_verification.run_full_verification", boom)
    monkeypatch.setattr("app.services.search_service.interpret_query", boom)
    reloaded = client.get(f"/search/{sid}").json()
    assert reloaded["query"] == "cloud"
    assert reloaded["judge_metadata"]["mode"] == "full_verification"


# ─────────────────────────── 9 — model default ───────────────────────────


def test_sonnet_is_the_default_model():
    assert Settings.model_fields["anthropic_model"].default == "claude-sonnet-4-6"
    assert Settings.model_fields["full_llm_verification"].default is True


# ─────────────────────────── 10 — not-used path (legacy) still works ───────────────────────────


def test_full_verification_off_falls_back_to_legacy_judge(client, monkeypatch):
    ds = _enriched(client)
    monkeypatch.setattr(settings, "full_llm_verification", False)

    def boom(*a, **k):
        raise AssertionError("full verification must not run when disabled")

    monkeypatch.setattr("app.services.full_verification._call_judge", boom)
    body = client.post("/search", json={"dataset_id": ds, "query": "cloud"}).json()
    # legacy path: judge unavailable (no key) -> conservative results still returned, no 503
    assert body["judge_metadata"]["mode"] != "full_verification"
    assert "connections" in body


# ─────────────────────────── helper ───────────────────────────


def _seed(db, ds_id, *, name, desc):
    slug = name.lower().replace(" ", "-")
    p = repo.add_person(
        db, dataset_id=ds_id, is_connection=True,
        linkedin_url=f"https://www.linkedin.com/in/{slug}", full_name=name,
        first_name=name.split()[0], last_name=name.split()[-1],
        current_title="Engineer", current_company="Acme",
        location_text="Seattle, Washington, United States", city="Seattle",
        enrichment_state="READY", profile_completeness=80,
    )
    repo.replace_experiences(db, p.id, [{
        "position": "Engineer", "company_name": "Acme", "is_current": True,
        "start_year": 2019, "description": desc,
    }])
    from app.services.embeddings import embed_text
    from app.services.search_text import build_search_text
    txt = build_search_text(db, p)
    repo.upsert_embedding(db, p.id, model=settings.embedding_model,
                          dim=settings.embedding_dim, vector=embed_text(txt), search_text=txt)
    return p.id
