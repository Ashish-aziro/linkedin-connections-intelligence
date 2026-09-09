"""V4 PART 5 — grounded final result audit.

No live LLM: ``final_auditor._audit_batch`` (or the router) is mocked. These
prove the audit is a correctness BRAKE — it downgrades / removes but never
upgrades, never overturns a verified fact or chronology, runs one batched pass
over TOP_N + BUFFER, and survives partial failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.constants import (
    AuditDecision,
    AuditStatus,
    CriterionType,
    Qualification,
    Scope,
    TriState,
)
from app.schemas import (
    FinalAuditBatch,
    FinalAuditCriterionReview,
    FinalAuditPersonDecision,
    ParsedSearchQuery,
    ScoreComponent,
    SearchCriterion,
)
from app.services import final_auditor
from app.services.company_intel import company_key
from app.services.final_audit_validator import validate_audit
from app.services.scoring import ProfileFacts, ScoringContext
from tests.test_search import _Exp, _Person


# ─────────────────────── fixtures / helpers ───────────────────────


def _pf(person, exps=None, sem=None, edu=None):
    return ProfileFacts(person=person, experiences=exps or [], education=edu or [], skills=[],
                        semantic=sem or {}, embedding=None)


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _plan(*c, **kw):
    return ParsedSearchQuery(criteria=list(c), **kw)


@dataclass
class _Scored:
    """Stand-in for scoring.ScoredCandidate (dataclasses.replace-compatible)."""

    person: object
    match_score: float = 40.0
    components: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    matched_criteria: list = field(default_factory=list)
    excluded_reason: str | None = None
    qualification: str = Qualification.POSSIBLE_MATCH
    unmet_required: list = field(default_factory=list)
    uncertain_required: list = field(default_factory=list)


def _comp(cid, *, required=True, strength=0.9, ctype=CriterionType.PROFESSIONAL_CONCEPT):
    return ScoreComponent(criterion=cid, criterion_id=cid, type=ctype, weight=50,
                          match_strength=strength, score=45, required=required, evidence=[])


def _packet(pid="p0", *, current=None, past=(), education=()):
    return {
        "person_id": pid,
        "current": current or {"experience_id": "cur", "title": "Engineer", "is_current": True},
        "past": list(past),
        "education": list(education),
        "experience_semantics": [], "company_classifications": [],
        "skills": [], "certifications": [], "semantic_assertions": [],
    }


# ─────────────────────── §29 — hard fact wins over an APPROVE ───────────────────────


def test_verified_location_mismatch_cannot_be_approved():
    plan = _plan(_crit(id="loc", type=CriterionType.LOCATION, value="Nashville", required=True))
    facts = _pf(_Person(location_text="Atlanta, Georgia, United States"),
                [_Exp("Engineer", "Co", 2020, None, True, id="cur")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.95,
           "reason": "Atlanta is close enough to Nashville", "criteria": []}
    out = validate_audit(raw, _packet(), plan, facts, ScoringContext(),
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] == Qualification.NOT_MATCH
    assert out["decision"] == AuditDecision.INCORRECT


# ─────────────────────── §34 — Amazon -> startup, company fact locked ───────────────────────


def test_current_startup_false_survives_an_approve():
    ctx = ScoringContext(company_class={
        company_key("99", "Microsoft"): {"is_startup": False, "confidence": 0.96,
                                         "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    plan = _plan(
        _crit(id="past", type=CriterionType.PAST_COMPANY, value="Amazon", required=True,
              scope=Scope.PAST_COMPANY),
        _crit(id="su", type=CriterionType.COMPANY_CATEGORY, concept="startup", required=True,
              scope=Scope.CURRENT_COMPANY),
    )
    facts = _pf(_Person(current_company="Microsoft", about="I advise startups on the side"),
                [_Exp("Principal PM", "Microsoft", 2018, None, True, id="cur", company_id="99"),
                 _Exp("SDE", "Amazon", 2013, 2017, False, id="amz", company_id="1586")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.9,
           "reason": "advises startups so counts as startup", "criteria": []}
    out = validate_audit(raw, _packet(past=[{"experience_id": "amz", "is_current": False}]),
                         plan, facts, ctx, first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] == Qualification.NOT_MATCH


# ─────────────────────── §35 — education is not academic employment ───────────────────────


def test_auditor_cannot_upgrade_academia_transition_from_a_degree():
    plan = _plan(_crit(id="tr", type=CriterionType.CAREER_TRANSITION,
                       concept="from academia to industry", required=True))
    facts = _pf(_Person(), [_Exp("Software Engineer", "Datadog", 2019, None, True, id="cur",
                                 desc="backend at a tech company")],
                edu=[_Exp("BS", "State University", 2015, 2019, False, id="edu1")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.9,
           "reason": "studied at a university then went to industry — that is academia to industry",
           "criteria": [{"criterion_id": "tr", "status_review": "supported", "reason": "degree counts"}]}
    out = validate_audit(raw, _packet(education=[{"education_id": "edu1"}]), plan, facts,
                         ScoringContext(), first_pass_qualification=Qualification.POSSIBLE_MATCH)
    assert out["applied_qualification"] != Qualification.EXACT_MATCH


# ─────────────────────── §30 — mentor false positive ───────────────────────


def _mentor_plan():
    return _plan(_crit(id="mgmt", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="engineering management or people leadership experience",
                       required=True, scope=Scope.CAREER),
                 _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="evidence of mentoring or advising others",
                       required=True, scope=Scope.CAREER),
                 intent="mentor_recommendation")


def test_senior_engineer_with_no_management_is_removed_by_audit():
    # PART 5.5: a career-scoped "unsupported" needs broad coverage of a complete
    # history — here 3 dated IC roles, completeness 80, all cited.
    plan = _mentor_plan()
    facts = _pf(_Person(current_title="Senior Backend Engineer", completeness=80),
                [_Exp("Backend Engineer", "A", 2013, 2016, False, id="e1", desc="built services"),
                 _Exp("Senior Backend Engineer", "B", 2016, 2019, False, id="e2", desc="built services"),
                 _Exp("Senior Backend Engineer", "C", 2019, None, True, id="e3", desc="built services")])
    packet = _packet(current={"experience_id": "e3", "is_current": True},
                     past=[{"experience_id": "e1", "is_current": False},
                           {"experience_id": "e2", "is_current": False}])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {
        "mgmt": {"status": TriState.TRUE, "match_strength": 0.8},
        "mentor": {"status": TriState.TRUE, "match_strength": 0.8},
    }
    raw = {"person_id": "p0", "decision": "incorrect", "confidence": 0.9,
           "reason": "no management or mentoring in every role across the full work history",
           "criteria": [
               {"criterion_id": "mgmt", "status_review": "unsupported",
                "reason": "every role across the full history is an individual contributor",
                "contradicting_evidence_refs": ["exp:e1", "exp:e2", "exp:e3"]},
               {"criterion_id": "mentor", "status_review": "unsupported",
                "reason": "no mentoring in any of the roles",
                "contradicting_evidence_refs": ["exp:e1", "exp:e2", "exp:e3"]},
           ]}
    out = validate_audit(raw, packet, plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["decision"] == AuditDecision.INCORRECT
    assert out["applied_qualification"] == Qualification.NOT_MATCH
    assert out["failed_required"]


def test_ungrounded_incorrect_downgrades_to_unknown_not_removed():
    plan = _mentor_plan()
    facts = _pf(_Person(), [_Exp("Senior Backend Engineer", "Co", 2019, None, True, id="cur")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"mgmt": {"status": TriState.TRUE}, "mentor": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "incorrect", "confidence": 0.6,
           "reason": "feels wrong", "criteria": [
               {"criterion_id": "mentor", "status_review": "unsupported",
                "reason": "no proof", "contradicting_evidence_refs": ["exp:invented999"]}]}
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["decision"] == AuditDecision.UNKNOWN
    assert out["applied_qualification"] == Qualification.POSSIBLE_MATCH


# ─────────────────────── §31 — professor false positive ───────────────────────


def test_ai_engineer_with_a_degree_is_not_a_professor():
    plan = _plan(_crit(id="prof", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="a professor / faculty appointment at a university", required=True))
    facts = _pf(_Person(current_title="AI Engineer"),
                [_Exp("AI Engineer", "Startup", 2021, None, True, id="cur")],
                edu=[_Exp("MS", "Stanford", 2019, 2021, False, id="edu1")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"prof": {"status": TriState.TRUE, "match_strength": 0.7}}
    raw = {"person_id": "p0", "decision": "incorrect", "confidence": 0.9,
           "reason": "current role is AI Engineer at a startup; the Stanford MS is a degree, not a faculty job",
           "criteria": [{"criterion_id": "prof", "status_review": "unsupported",
                         "reason": "no faculty appointment", "contradicting_evidence_refs": ["exp:cur"]}]}
    out = validate_audit(raw, _packet(education=[{"education_id": "edu1"}]), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] == Qualification.NOT_MATCH


# ─────────────────────── §32 — HIPAA modality possible ───────────────────────


def test_possible_modality_candidate_stays_possible_not_verified_expert():
    plan = _plan(SearchCriterion(id="hipaa", type=CriterionType.PROFESSIONAL_CONCEPT,
                                 concept="HIPAA compliance experience", weight=100,
                                 required=False, modality="possible"))
    facts = _pf(_Person(current_title="Software Engineer", current_company="HealthCo"),
                [_Exp("Software Engineer", "HealthCo", 2021, None, True, id="cur")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.5,
           "reason": "healthcare employer — may have some HIPAA exposure but nothing verified",
           "criteria": [{"criterion_id": "hipaa", "status_review": "uncertain",
                         "reason": "no explicit compliance work"}],
           "suggested_qualification": "exact_match"}
    out = validate_audit(raw, _packet(), plan, facts, ScoringContext(),
                         first_pass_qualification=Qualification.POSSIBLE_MATCH)
    assert out["applied_qualification"] == Qualification.POSSIBLE_MATCH


# ─────────────────────── §33 — cross-domain: only one side supported ───────────────────────


def test_cross_domain_one_side_unsupported_cannot_stay_exact():
    plan = _plan(_crit(id="cyber", type=CriterionType.PROFESSIONAL_CONCEPT,
                       concept="cybersecurity experience", required=True, scope=Scope.CAREER),
                 _crit(id="health", type=CriterionType.INDUSTRY_EXPERIENCE,
                       concept="healthcare industry experience", required=True, scope=Scope.CAREER))
    facts = _pf(_Person(current_title="Cloud Security Engineer"),
                [_Exp("Cloud Security Engineer", "SaaSCo", 2020, None, True, id="cur",
                      desc="application and cloud security")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"cyber": {"status": TriState.TRUE}, "health": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "downgrade", "confidence": 0.8,
           "reason": "cybersecurity is solid but there is no healthcare evidence at all",
           "criteria": [
               {"criterion_id": "cyber", "status_review": "supported", "supporting_evidence_refs": ["exp:cur"]},
               {"criterion_id": "health", "status_review": "unsupported",
                "reason": "no healthcare employer or context", "contradicting_evidence_refs": ["exp:cur"]}]}
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["applied_qualification"] != Qualification.EXACT_MATCH


# ─────────────────────── §37 / §38 — downgrade keeps; no upgrade ───────────────────────


def test_downgrade_moves_exact_to_possible_and_keeps_the_person():
    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    facts = _pf(_Person(), [_Exp("Eng", "Co", 2020, None, True, id="cur")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"c": {"status": TriState.TRUE}}
    raw = {"person_id": "p0", "decision": "downgrade", "confidence": 0.7,
           "reason": "the concept match rests on a weak inference", "criteria": []}
    out = validate_audit(raw, _packet(), plan, facts, ctx,
                         first_pass_qualification=Qualification.EXACT_MATCH)
    assert out["decision"] == AuditDecision.DOWNGRADE
    assert out["applied_qualification"] == Qualification.POSSIBLE_MATCH


def test_auditor_cannot_manufacture_exact():
    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    facts = _pf(_Person(), [_Exp("Eng", "Co", 2020, None, True, id="cur")])
    raw = {"person_id": "p0", "decision": "approved", "confidence": 0.99,
           "reason": "this is clearly a perfect match", "suggested_qualification": "exact_match",
           "criteria": []}
    out = validate_audit(raw, _packet(), plan, facts, ScoringContext(),
                         first_pass_qualification=Qualification.POSSIBLE_MATCH)
    assert out["applied_qualification"] == Qualification.POSSIBLE_MATCH


# ─────────────────────── §36 — promotion from the same audited buffer ───────────────────────


def _fake_audit(decisions_by_pid, *, capture=None, none_after=None):
    state = {"n": 0}

    def fake(payload, packets, first_pass_by_id, parsed=None):
        idx = state["n"]
        state["n"] += 1
        if capture is not None:
            capture.append({"payload": payload, "packets": packets, "first_pass": first_pass_by_id})
        if none_after is not None and idx >= none_after:
            return "failed", None, None, None
        people = []
        for pkt in packets:
            pid = pkt["person_id"]
            d = decisions_by_pid.get(pid, "approved")
            cur_id = (pkt.get("current") or {}).get("experience_id") or "x"
            people.append({
                "person_id": pid, "decision": d, "confidence": 0.9, "reason": "ok",
                "criteria": [
                    {"criterion_id": c["id"],
                     "status_review": "unsupported" if d == "incorrect" else "supported", "reason": "r",
                     "supporting_evidence_refs": [],
                     "contradicting_evidence_refs": ([f"exp:{cur_id}"] if d == "incorrect" else [])}
                    for c in payload["criteria"] if c["required"]
                ],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
                "suggested_qualification": None,
            })
        return "ok", people, "mock:prov", "mock-1"

    return fake


def _pool(n):
    pool = []
    for i in range(n):
        p = _Person(current_title="Engineer")
        p.id = f"p{i}"
        pool.append(_Scored(person=p, match_score=100 - i,
                            qualification=Qualification.POSSIBLE_MATCH, components=[_comp("c")]))
    return pool


def test_promotion_uses_only_the_already_audited_buffer(monkeypatch):
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_top_n", 3)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_buffer", 2)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_batch_size", 10)
    calls = []
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_audit({"p1": "incorrect"}, capture=calls))

    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    pool = _pool(5)
    facts_by_id = {sc.person.id: _pf(sc.person, [_Exp("Eng", "Co", 2020, None, True, id="cur")])
                   for sc in pool}
    bundle_by_id = {sc.person.id: (sc.person, facts_by_id[sc.person.id],
                                   {"volunteering": [], "recommendations": []}) for sc in pool}

    run = final_auditor.run_final_audit("q", plan, pool, ScoringContext(), bundle_by_id=bundle_by_id)
    assert len(calls) == 1
    assert run.metadata.requested_candidates == 5

    survivors = []
    for sc in pool:
        v = validate_audit(run.decisions[sc.person.id], run.packets_by_id.get(sc.person.id, {}),
                           plan, facts_by_id[sc.person.id], ScoringContext(),
                           first_pass_qualification=sc.qualification)
        if v["applied_qualification"] != Qualification.NOT_MATCH:
            survivors.append(sc.person.id)
    assert survivors[:3] == ["p0", "p2", "p3"]


# ─────────────────────── §39 — partial audit failure ───────────────────────


def test_partial_audit_failure_does_not_crash(monkeypatch):
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_batch_size", 10)
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_audit({}, none_after=1))

    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    pool = _pool(30)
    bundle_by_id = {sc.person.id: (sc.person, _pf(sc.person, [_Exp("E", "C", 2020, None, True, id="cur")]),
                                   {"volunteering": [], "recommendations": []}) for sc in pool}
    run = final_auditor.run_final_audit("q", plan, pool, ScoringContext(), bundle_by_id=bundle_by_id)

    assert run.metadata.status == AuditStatus.PARTIAL
    assert run.metadata.successful_batches == 1 and run.metadata.failed_batches == 2
    unaudited = [pid for pid, d in run.decisions.items() if d.get("audit_missing")]
    assert len(unaudited) == 20
    assert all(run.decisions[pid]["decision"] == AuditDecision.UNKNOWN for pid in unaudited)


# ─────────────────────── hardening PART 14 — search deadline ───────────────────────


class _FakeDeadline:
    def __init__(self, expire_after: int):
        self.calls = 0
        self.expire_after = expire_after

    def expired(self) -> bool:
        self.calls += 1
        return self.calls > self.expire_after


def test_deadline_reached_stops_further_audit_batches(monkeypatch):
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_batch_size", 10)
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_audit({}))

    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    pool = _pool(30)
    bundle_by_id = {sc.person.id: (sc.person, _pf(sc.person, [_Exp("E", "C", 2020, None, True, id="cur")]),
                                   {"volunteering": [], "recommendations": []}) for sc in pool}
    run = final_auditor.run_final_audit("q", plan, pool, ScoringContext(), bundle_by_id=bundle_by_id,
                                        deadline=_FakeDeadline(1))

    assert run.metadata.deadline_reached is True
    assert run.metadata.status == AuditStatus.PARTIAL
    assert run.metadata.successful_batches == 1
    unaudited = [pid for pid, d in run.decisions.items() if d.get("audit_missing")]
    assert len(unaudited) == 20
    assert all(run.decisions[pid]["decision"] == AuditDecision.UNKNOWN for pid in unaudited)


# ─────────────────────── §40 — oversized audit packet ───────────────────────


def test_oversized_audit_packet_is_left_unaudited(monkeypatch):
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_max_batch_chars", 400)
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_audit({}))
    monkeypatch.setattr(final_auditor, "build_packets", lambda *a, **k: [
        {"person_id": "small", "current": {"experience_id": "e1", "is_current": True}, "past": [],
         "education": [], "experience_semantics": [], "company_classifications": [], "skills": [],
         "certifications": [], "semantic_assertions": []},
        {"person_id": "huge", "current": {"experience_id": "e1", "is_current": True}, "past": [],
         "education": [], "experience_semantics": [], "company_classifications": [], "skills": [],
         "certifications": [], "semantic_assertions": [], "blob": "z" * 5000},
    ])
    plan = _plan(_crit(id="c", type=CriterionType.PROFESSIONAL_CONCEPT, concept="x", required=True))
    pool = _pool(2)
    pool[0].person.id, pool[1].person.id = "small", "huge"
    bundle_by_id = {sc.person.id: (sc.person, _pf(sc.person), {}) for sc in pool}
    run = final_auditor.run_final_audit("q", plan, pool, ScoringContext(), bundle_by_id=bundle_by_id)

    assert run.metadata.oversized_packets == 1
    assert run.decisions["huge"].get("audit_missing") is True
    assert run.metadata.status == AuditStatus.PARTIAL


# ─────────────────────── the SearchPlan + first-pass reach the auditor ───────────────────────


def test_query_plan_and_first_pass_reach_the_auditor(monkeypatch):
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)
    calls = []
    monkeypatch.setattr(final_auditor, "_call_audit", _fake_audit({}, capture=calls))

    plan = _mentor_plan()
    plan.target_person_context = {"field": "backend engineering", "goal": "engineering management"}
    p = _Person()
    p.id = "p0"
    sc = _Scored(person=p, qualification=Qualification.EXACT_MATCH,
                 components=[_comp("mgmt"), _comp("mentor")])
    ctx = ScoringContext()
    ctx.judge_results["p0"] = {"mgmt": {"status": TriState.TRUE, "reason": "led a team",
                                        "supporting_evidence_refs": ["exp:cur"]}}
    bundle_by_id = {"p0": (p, _pf(p, [_Exp("EM", "Co", 2019, None, True, id="cur")]),
                           {"volunteering": [], "recommendations": []})}
    final_auditor.run_final_audit("Who could mentor a backend engineer moving into management?",
                                  plan, [sc], ctx, bundle_by_id=bundle_by_id)

    payload = calls[0]["payload"]
    assert payload["original_query"].startswith("Who could mentor")
    assert payload["intent"] == "mentor_recommendation"
    assert payload["target_person_context"]["goal"] == "engineering management"
    assert {c["id"] for c in payload["criteria"]} == {"mgmt", "mentor"}
    fp = calls[0]["first_pass"]["p0"]
    assert fp["qualification"] == Qualification.EXACT_MATCH
    assert any(c["criterion_id"] == "mgmt" and c["judge"] for c in fp["criteria"])


# ─────────────────────── end-to-end through /search ───────────────────────


def test_search_runs_audit_after_scoring_and_returns_metadata(client, monkeypatch):
    from tests.test_search import _enriched_dataset
    from app.services import semantic_judge

    # no live LLM anywhere
    import app.services.company_intel as ci
    monkeypatch.setattr(ci, "get_or_classify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no classify during search")))
    monkeypatch.setattr(semantic_judge, "_call_judge",
                        lambda *a, **k: ("failed", None, None, None))  # judge unavailable -> deterministic
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", True)

    seen_pids = []

    def fake_audit(payload, packets, first_pass_by_id, parsed=None):
        seen_pids.extend(p["person_id"] for p in packets)
        people = [{
            "person_id": p["person_id"], "decision": "approved", "confidence": 0.88, "reason": "checks out",
            "criteria": [{"criterion_id": c["id"], "status_review": "supported", "reason": "",
                         "supporting_evidence_refs": [], "contradicting_evidence_refs": []}
                        for c in payload["criteria"] if c["required"]],
            "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "suggested_qualification": None,
        } for p in packets]
        return "ok", people, "mock:prov", "mock-1"

    monkeypatch.setattr(final_auditor, "_call_audit", fake_audit)

    ds = _enriched_dataset(client)
    body = client.post("/search", json={
        "dataset_id": ds, "query": "Senior engineers at Microsoft",
    }).json()

    am = body["audit_metadata"]
    assert am and am["enabled"] is True and am["status"] in ("full", "partial")
    assert am["requested_candidates"] >= 1
    assert am["audited_candidates"] == len(set(seen_pids))
    # every shown result carries its audit outcome
    for r in body["connections"]["results"]:
        assert r["audit_decision"] == "approved"
        assert r["llm_verified"] is True
        assert r["reason"]  # reason generated only for the post-audit survivors
