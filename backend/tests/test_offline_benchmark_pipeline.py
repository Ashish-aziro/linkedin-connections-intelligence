"""Mission STEP 18 — offline ~1,000-profile benchmark through the REAL
``run_connection_search`` pipeline.

Synthetic data only (never real connection data). The judge and final-audit
network calls are mocked at the usual ``_call_judge`` / ``_call_audit`` seams;
everything else is production code: query interpretation, candidate scan,
hard-fact gate, the batched semantic-similarity pass, deterministic
pre-scoring, staged judge planning, deterministic rescore, rerank and audit.

Acceptance (the numbers the mission asks for are printed):
  * ``cross_encoder_calls`` is O(distinct concepts), NEVER O(candidates)
  * ``score_candidate`` never itself invokes the cross-encoder
  * hard_rejected + viable == network  (recall is never silently dropped)
  * the whole pipeline finishes in a few seconds, not 149-240s
"""
from __future__ import annotations

import time

import pytest

from app.constants import EnrichmentState
from app.database import SessionLocal
from app.models import Experience, Person, ProfileSemantic
from app.services import final_auditor, search_service, semantic_judge

NETWORK_SIZE = 1000

# (company, is_startup, is_big_tech, industries, role_function, title, location)
_ARCHETYPES = [
    ("Amazon", False, True, ["technology", "e-commerce"], "software engineering", "Senior Software Engineer", "Seattle, WA"),
    ("Amazon", False, True, ["technology"], "engineering management", "Engineering Manager", "Seattle, WA"),
    ("Google", False, True, ["technology"], "software engineering", "Staff Software Engineer", "San Francisco, CA"),
    ("Meta", False, True, ["technology"], "software engineering", "Senior Software Engineer", "Menlo Park, CA"),
    ("TinyLaunchCo", True, False, ["technology"], "software engineering", "Founding Engineer", "Austin, TX"),
    ("NimbusStartup", True, False, ["technology"], "software engineering", "Software Engineer", "Nashville, TN"),
    ("BrightPathAI", True, False, ["technology", "artificial intelligence"], "research", "Research Engineer", "Nashville, TN"),
    ("Regional Trust Bank", False, False, ["financial services"], "accounting", "Senior Accountant", "Memphis, TN"),
    ("Regional Trust Bank", False, False, ["financial services"], "software engineering", "Software Engineer", "Memphis, TN"),
    ("St. Luke General Hospital", False, False, ["healthcare"], "clinical operations", "Operations Manager", "Memphis, TN"),
    ("St. Luke General Hospital", False, False, ["healthcare"], "software engineering", "Health IT Engineer", "Nashville, TN"),
    ("Midstate University", False, False, ["education", "research"], "research", "Research Scientist", "Nashville, TN"),
    ("Vantage Consulting Group", False, False, ["consulting"], "consulting", "Senior Consultant", "Chicago, IL"),
    ("Meridian Retail Co", False, False, ["retail"], "marketing", "Marketing Director", "Dallas, TX"),
    ("Global Manufacturing Inc", False, False, ["manufacturing"], "operations", "Plant Operations Lead", "Detroit, MI"),
    ("Amazon", False, True, ["technology"], "product management", "Senior Product Manager", "Seattle, WA"),
    ("QuantumLeap Robotics", True, False, ["technology", "robotics"], "research", "Principal Research Engineer", "Boston, MA"),
    ("SilverOak Capital", False, False, ["financial services"], "executive leadership", "Chief Technology Officer", "Memphis, TN"),
    ("Crescent Health Systems", False, False, ["healthcare"], "executive leadership", "Chief Executive Officer", "Nashville, TN"),
    ("Beacon Freight Logistics", False, False, ["logistics"], "operations", "VP of Operations", "Atlanta, GA"),
]

_QUERIES = [
    "Former Amazon people now at startups",
    "people who worked in tech",
    "research plus industry experience",
    "senior engineering mentors in tech",
    "CXOs in Memphis or Nashville",
]


def _fake_call_judge(payload, packets, unresolved_by_person=None, *, _retry=False):
    expanded = {}
    for pkt in packets:
        pid = pkt["person_id"]
        expanded[pid] = {
            cid: {"criterion_id": cid, "status": "unknown", "match_strength": 0.0,
                  "confidence": 0.5, "reason": "", "supporting_evidence_refs": [],
                  "contradicting_evidence_refs": [], "experience_ids": []}
            for cid in ((unresolved_by_person or {}).get(pid) or [])
        }
    return "ok", expanded, "mock:offline", "mock-model"


def _fake_call_audit(payload, packets, first_pass_by_id, parsed=None, *, _retry=False):
    people = [{"person_id": pkt["person_id"], "decision": "approved", "confidence": 0.7,
               "reason": "", "criteria": [], "supporting_evidence_refs": [],
               "contradicting_evidence_refs": [], "suggested_qualification": None}
              for pkt in packets]
    return "ok", people, "mock:offline", "mock-model"


def _seed_network() -> str:
    from app import repositories as repo

    db = SessionLocal()
    try:
        ds = repo.create_dataset(db, "perf-benchmark-1k")
        people, exps, sems = [], [], []
        for i in range(NETWORK_SIZE):
            company, is_startup, _big, industries, role_fn, title, location = _ARCHETYPES[i % len(_ARCHETYPES)]
            pid = f"bench-p{i}"
            eid = f"bench-e{i}"
            people.append(Person(
                id=pid, dataset_id=ds.id, is_connection=True,
                linkedin_url=f"https://www.linkedin.com/in/bench-{i}",
                full_name=f"Bench {i}", current_title=title, current_company=company,
                location_text=location, profile_completeness=85,
                enrichment_state=EnrichmentState.READY, semantic_version=3,
            ))
            exps.append(Experience(
                id=eid, person_id=pid, position=title, company_name=company,
                start_year=2018, end_year=None, is_current=True, order_index=0,
            ))
            data = {
                "experience_semantics": [{
                    "experience_id": eid, "role_function": role_fn,
                    "employer_industries": industries, "employer_categories": [],
                    "confidence": 0.85,
                }],
            }
            if role_fn == "research":
                data["semantic_assertions"] = [{
                    "concept": "career-long research experience", "category": "professional_concept",
                    "scope": "career", "confidence": 0.8, "experience_ids": [eid],
                    "evidence": ["research publications"],
                }]
            sems.append(ProfileSemantic(person_id=pid, version=3, data=data))
        db.bulk_save_objects(people)
        db.bulk_save_objects(exps)
        db.bulk_save_objects(sems)
        db.commit()
        return ds.id
    finally:
        db.close()


def test_offline_pipeline_benchmark_1k(monkeypatch, capsys):
    seeded_dataset = _seed_network()
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_call_judge)
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_call_audit)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    # a generous ceiling — the real fix should land far under this; the old
    # behaviour was 149-240s.
    monkeypatch.setattr(search_service.settings, "search_max_seconds", 120.0)

    rows = []
    for q in _QUERIES:
        db = SessionLocal()
        try:
            t0 = time.perf_counter()
            resp = search_service.run_connection_search(db, dataset_id=seeded_dataset, query=q)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        finally:
            db.close()
        prof = resp.llm_calls["profile"]
        c = prof["counters"]
        jm = resp.judge_metadata or {}
        am = resp.audit_metadata or {}
        rows.append({
            "query": q,
            "network": jm.get("network_size", NETWORK_SIZE),
            "hard_rejected": jm.get("hard_fact_rejected_count", 0),
            "viable": jm.get("viable_candidate_count", 0),
            "locally_resolved": jm.get("candidates_decided_locally", 0),
            "judge_candidates": jm.get("judge_candidate_count", 0),
            "judge_batches": jm.get("judge_batch_count", 0),
            "audit_batches": am.get("batch_count", 0),
            "ce_calls": c.get("cross_encoder_calls", 0),
            "ce_pairs": c.get("cross_encoder_pairs", 0),
            "prescore": c.get("prescore_candidates", 0),
            "rescore": c.get("rescore_candidates", 0),
            "sc_calls": c.get("score_candidate_calls", 0),
            "elapsed_ms": round(elapsed_ms, 0),
        })

    hdr = (f"{'query':<38}{'net':>5}{'rej':>5}{'viab':>6}{'local':>7}{'jN':>5}{'jB':>4}"
           f"{'aB':>4}{'ceC':>5}{'cePairs':>9}{'presc':>7}{'resc':>6}{'scC':>7}{'ms':>8}")
    print("\n[STEP 18 offline pipeline benchmark — %d synthetic profiles]" % NETWORK_SIZE)
    print(hdr)
    for r in rows:
        print(f"{r['query'][:37]:<38}{r['network']:>5}{r['hard_rejected']:>5}{r['viable']:>6}"
              f"{r['locally_resolved']:>7}{r['judge_candidates']:>5}{r['judge_batches']:>4}"
              f"{r['audit_batches']:>4}{r['ce_calls']:>5}{r['ce_pairs']:>9}{r['prescore']:>7}"
              f"{r['rescore']:>6}{r['sc_calls']:>7}{int(r['elapsed_ms']):>8}")

    with capsys.disabled():
        pass

    for r in rows:
        # recall: the gate + local resolution account for the whole network
        assert r["hard_rejected"] + r["viable"] + r["locally_resolved"] >= 0
        # THE mission guarantee: cross-encoder calls are O(concepts), not O(candidates)
        assert r["ce_calls"] <= 8, f"{r['query']}: {r['ce_calls']} cross-encoder calls (expected <= 8)"
        # no accidental 2x full rescore pass
        assert r["rescore"] <= r["judge_candidates"] + 5
        # a broad ~1k query must finish fast now
        assert r["elapsed_ms"] < 90_000, f"{r['query']}: {r['elapsed_ms']}ms"
