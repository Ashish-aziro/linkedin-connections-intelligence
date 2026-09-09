"""Exhaustive semantic judge + validator (V4 PART 3 §46–§57).

No live LLM — ``semantic_judge._judge_batch`` (or the router) is mocked. These
prove: EVERY viable candidate is judged in all_viable mode, batching is real,
partial failure is survivable, a missing verdict is UNKNOWN, the universal
SearchPlan (intent / relational context / modality / boolean structure) reaches
the judge, and the fact-consistency validator rejects invented / out-of-scope /
locked-fact-contradicting verdicts.
"""
from __future__ import annotations

from app.constants import CriterionType, JudgeStatus, Modality, Operator, Qualification, Scope, TriState
from app.schemas import (
    CompactJudgeBatch,
    ParsedSearchQuery,
    SearchCriterion,
)
from app.services import semantic_judge
from app.services.company_intel import company_key
from app.services.judge_validator import validate_person
from app.services.scoring import ProfileFacts, ScoredCandidate, ScoringContext
from tests.test_search import _Exp, _Person


# ─────────────────────── fixtures / helpers ───────────────────────


def _person(i: int, **kw) -> _Person:
    p = _Person(**kw)
    p.id = f"p{i}"
    return p


def _pf(person, exps=None, sem=None) -> ProfileFacts:
    return ProfileFacts(person=person, experiences=exps or [], education=[], skills=[],
                        semantic=sem or {}, embedding=None)


def _bundle(n: int) -> list[tuple]:
    out = []
    for i in range(n):
        p = _person(i, headline="Engineer", current_company="Co", current_title="Engineer")
        exps = [_Exp("Engineer", "Co", 2020, None, True, id=f"e{i}",
                     desc="led a team and mentored engineers")]
        out.append((p, _pf(p, exps), {"volunteering": [], "recommendations": []}))
    return out


def _mentor_plan(*, extra_crits=()) -> ParsedSearchQuery:
    crits = [
        SearchCriterion(id="mentor_evidence", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="evidence of mentoring, coaching, advising or people leadership",
                        scope=Scope.CAREER, weight=60, required=True),
        *extra_crits,
    ]
    return ParsedSearchQuery(
        intent="mentor_recommendation", criteria=crits,
        target_person_context={"field": "backend engineering", "goal": "engineering management"},
    )


def _fake_judge(*, true_ids=None, none_after=None, omit_people=(), omit_criteria=None, capture=None):
    """Fake ``_call_judge`` (the adaptive-splitter's CallFn seam). ``none_after``
    -> batch index from which the call fails (outcome "failed", never "truncated" —
    these fakes never trigger a split, so one call == one top-level batch, same
    as the pre-adaptive-splitting behaviour)."""
    state = {"batch": 0}

    def fake(payload, packets, unresolved_by_person=None):
        idx = state["batch"]
        state["batch"] += 1
        if capture is not None:
            capture.append({"payload": payload, "packets": packets})
        if none_after is not None and idx >= none_after:
            return "failed", None, None, None
        expanded: dict[str, dict[str, dict]] = {}
        for pkt in packets:
            pid = pkt["person_id"]
            if pid in omit_people:
                continue
            crit_ids = (unresolved_by_person or {}).get(pid) or [c["id"] for c in payload["criteria_to_judge"]]
            want_true = true_ids if true_ids is not None else crit_ids
            per_person: dict[str, dict] = {}
            for cid in crit_ids:
                if omit_criteria and (pid, cid) in omit_criteria:
                    continue
                first_exp = (pkt.get("current") or {}).get("experience_id") \
                    or (pkt["past"][0]["experience_id"] if pkt.get("past") else None)
                is_true = cid in want_true
                per_person[cid] = {
                    "criterion_id": cid,
                    "status": "true" if is_true else "unknown",
                    "match_strength": 0.9 if is_true else 0.0,
                    "confidence": 0.9,
                    "reason": "clear evidence in the role description",
                    "supporting_evidence_refs": [f"exp:{first_exp}"] if (first_exp and is_true) else [],
                    "contradicting_evidence_refs": [],
                    "experience_ids": [first_exp] if (first_exp and is_true) else [],
                }
            expanded[pid] = per_person
        return "ok", expanded, "mock:provider", "mock-model-1"

    return fake


def _raise_judge(*a, **k):
    raise AssertionError("_judge_batch must not be called")


# ─────────────────────── §46 — all viable candidates are judged ───────────────────────


def test_all_viable_candidates_are_judged_in_batches(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 0)
    calls = []
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge(capture=calls))

    run = semantic_judge.run_judge(
        "career mentors", _mentor_plan(), _bundle(100), ScoringContext(),
        network_size=991, pool_size=991, hard_rejected_count=0,
    )
    assert run.metadata.judge_candidate_count == 100
    assert run.metadata.judge_batch_count == 10           # NOT 60, NOT ambiguity-band
    assert len(calls) == 10
    assert sum(len(c["packets"]) for c in calls) == 100
    assert run.metadata.judge_status == JudgeStatus.FULL
    assert len(run.verdicts) == 100
    assert all("mentor_evidence" in v for v in run.verdicts.values())


def test_no_hidden_60_person_cap(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 0)
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge())
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(75), ScoringContext(),
                                   network_size=75, pool_size=75, hard_rejected_count=0)
    assert run.metadata.judge_candidate_count == 75 and not run.metadata.capped


def test_explicit_cap_is_respected_and_reported(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 20)
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge())
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(50), ScoringContext(),
                                   network_size=50, pool_size=50, hard_rejected_count=0)
    assert run.metadata.judge_candidate_count == 20
    assert run.metadata.capped is True and run.metadata.cap_limit == 20


# ─────────────────────── §52 — partial batch failure ───────────────────────


def test_partial_batch_failure_keeps_earlier_verdicts(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge(none_after=6))
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(100), ScoringContext(),
                                   network_size=100, pool_size=100, hard_rejected_count=0)
    judged = [pid for pid, v in run.verdicts.items()
              if not v["mentor_evidence"].get("judge_missing")]
    unjudged = [pid for pid, v in run.verdicts.items()
                if v["mentor_evidence"].get("judge_missing")]
    assert len(judged) == 60 and len(unjudged) == 40
    assert run.metadata.judge_status == JudgeStatus.PARTIAL
    assert run.metadata.judge_successful_batches == 6 and run.metadata.judge_failed_batches == 4
    assert all(run.verdicts[pid]["mentor_evidence"]["status"] == TriState.UNKNOWN for pid in unjudged)


def test_all_batches_fail_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge(none_after=0))
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(30), ScoringContext(),
                                   network_size=30, pool_size=30, hard_rejected_count=0)
    assert run.metadata.judge_status == JudgeStatus.UNAVAILABLE
    assert all(v["mentor_evidence"]["status"] == TriState.UNKNOWN for v in run.verdicts.values())


# ─────────────────────── hardening PART 22 — exact live-failure simulation ───────────────────────


def test_exact_live_failure_recovers_via_split_and_single_person_retry(monkeypatch):
    """Reproduces the live incident: a query with 4 required criteria per
    person ("research plus industry experience" style), a batch that
    truncates at every size down to a single person, and even that single
    person truncating once at the estimated budget. Proves: adaptive splitting
    isolates single-person leaves, the bounded single-person double-budget
    retry (hardening PART 7) recovers them, there is no infinite recursion /
    unbounded call count, and the run still returns full, usable verdicts —
    never hangs, never crashes."""
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 9)  # the size that got stuck live

    four_crits = [
        SearchCriterion(id="research", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="research experience", scope=Scope.CAREER, weight=25, required=True),
        SearchCriterion(id="industry", type=CriterionType.INDUSTRY_EXPERIENCE,
                        concept="industry experience", scope=Scope.CAREER, weight=25, required=True),
        SearchCriterion(id="leadership", type=CriterionType.ROLE_FUNCTION,
                        concept="engineering leadership", scope=Scope.CAREER, weight=25, required=True),
        SearchCriterion(id="mentoring", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="mentoring", scope=Scope.CAREER, weight=25, required=True),
    ]
    plan = ParsedSearchQuery(criteria=four_crits)
    bundle = _bundle(9)

    call_sizes: list[int] = []
    single_person_seen: set[str] = set()

    import re

    def fake_generate_structured(system, user, schema, *, max_tokens, operation, return_meta):
        n = user.count('"person_id":')
        call_sizes.append(n)
        if n == 1:
            pid = re.search(r'"person_id":\s*"(p\d+)"', user).group(1)
            if pid in single_person_seen:
                # the bounded double-budget retry (hardening PART 7) succeeds
                batch = CompactJudgeBatch.model_validate({"people": [{
                    "person_id": pid,
                    "criteria": [{"criterion_id": c.id, "status": "true", "confidence": 0.8,
                                 "supporting_refs": ["exp:e0"], "reason": ""} for c in four_crits],
                }]})
                meta = {"operation": operation, "selected_provider": "mock", "selected_model": "m1",
                       "attempts": [{"provider": "mock", "status": "success"}]}
                return batch, "mock", "m1", meta
            single_person_seen.add(pid)
        meta = {"operation": operation, "selected_provider": None, "selected_model": None,
               "attempts": [{"provider": "mock", "status": "output_truncated"}]}
        return None, meta

    monkeypatch.setattr(semantic_judge, "generate_structured", fake_generate_structured)
    run = semantic_judge.run_judge(
        "research plus industry experience", plan, bundle, ScoringContext(),
        network_size=987, pool_size=987, hard_rejected_count=0,
    )

    assert len(run.verdicts) == 9
    assert all(v["research"]["status"] == TriState.TRUE for v in run.verdicts.values())
    assert all(v["industry"]["status"] == TriState.TRUE for v in run.verdicts.values())
    # every eventual leaf was size 1 (fully split down), and each single-person
    # leaf was called at most twice (initial + one bounded retry) -- no
    # unbounded recursion / repeated identical calls
    assert max(call_sizes) <= 9
    assert call_sizes.count(1) == 18  # 9 people x (1 truncated attempt + 1 successful retry)
    assert len(call_sizes) < 40
    assert run.metadata.judge_status == JudgeStatus.FULL
    # the internal single-person retry (hardening PART 7) fully resolves inside
    # _call_judge and is never surfaced to adaptive_batch as a "truncated"
    # outcome once it succeeds -- adaptive_batch's own counter only reflects
    # the multi-person splits that actually happened above the leaf level.
    assert run.metadata.truncations >= 3
    assert run.metadata.adaptive_splits >= 3


# ─────────────────────── hardening PART 14 — search deadline ───────────────────────


class _FakeDeadline:
    """Expires starting from the (expire_after + 1)-th check — one check happens
    per batch iteration, right before that batch would be attempted."""

    def __init__(self, expire_after: int):
        self.calls = 0
        self.expire_after = expire_after

    def expired(self) -> bool:
        self.calls += 1
        return self.calls > self.expire_after


def test_deadline_reached_stops_further_batches_and_marks_partial(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge())

    run = semantic_judge.run_judge(
        "x", _mentor_plan(), _bundle(100), ScoringContext(),
        network_size=100, pool_size=100, hard_rejected_count=0,
        deadline=_FakeDeadline(3),
    )
    assert run.metadata.deadline_reached is True
    assert run.metadata.judge_status == JudgeStatus.PARTIAL
    assert run.metadata.judge_batch_count == 3
    judged = [pid for pid, v in run.verdicts.items() if not v["mentor_evidence"].get("judge_missing")]
    unjudged = [pid for pid, v in run.verdicts.items() if v["mentor_evidence"].get("judge_missing")]
    assert len(judged) == 30 and len(unjudged) == 70
    assert all(run.verdicts[pid]["mentor_evidence"]["status"] == TriState.UNKNOWN for pid in unjudged)


def test_deadline_prioritises_candidates_nearest_the_qualification_boundary(monkeypatch):
    """hardening PART 15 — when a run is cut short (budget or deadline),
    candidates with fewer remaining unresolved required criteria (one verdict
    from a decided outcome) and a higher local match score must be judged
    BEFORE candidates still ambiguous on every dimension."""
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 5)
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge())

    leadership = SearchCriterion(id="leadership", type=CriterionType.ROLE_FUNCTION,
                                 concept="engineering leadership", scope=Scope.CAREER,
                                 weight=40, required=True)
    plan = _mentor_plan(extra_crits=[leadership])
    bundle = _bundle(20)

    local_scored = {}
    for i, (p, _f, _x) in enumerate(bundle):
        if i < 10:
            local_scored[p.id] = ScoredCandidate(
                person=p, match_score=90.0, components=[], evidence=[],
                qualification=Qualification.POSSIBLE_MATCH,
                status_by_criterion={"mentor_evidence": TriState.TRUE, "leadership": TriState.UNKNOWN},
            )
        else:
            local_scored[p.id] = ScoredCandidate(
                person=p, match_score=50.0, components=[], evidence=[],
                qualification=Qualification.POSSIBLE_MATCH,
                status_by_criterion={"mentor_evidence": TriState.UNKNOWN, "leadership": TriState.UNKNOWN},
            )

    run = semantic_judge.run_judge(
        "x", plan, bundle, ScoringContext(),
        network_size=20, pool_size=20, hard_rejected_count=0, local_scored=local_scored,
        deadline=_FakeDeadline(2),  # only 2 of 4 batches (10 of 20 candidates) run
    )
    judged_ids = {pid for pid, v in run.verdicts.items() if not v["leadership"].get("judge_missing")}
    near_ids = {p.id for i, (p, _f, _x) in enumerate(bundle) if i < 10}
    assert judged_ids == near_ids


def test_no_deadline_never_stops_a_run(monkeypatch):
    """deadline=None (the default) — behaves exactly as if the deadline never existed."""
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge())
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(30), ScoringContext(),
                                   network_size=30, pool_size=30, hard_rejected_count=0)
    assert run.metadata.deadline_reached is False
    assert run.metadata.judge_status == JudgeStatus.FULL


# ─────────────────────── §53 — missing person / criterion ───────────────────────


def test_missing_person_and_criteria_become_unknown(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)
    plan = _mentor_plan(extra_crits=(
        SearchCriterion(id="ai", type=CriterionType.INDUSTRY_EXPERIENCE, concept="AI industry",
                        weight=20, required=True),
        SearchCriterion(id="hc", type=CriterionType.INDUSTRY_EXPERIENCE, concept="healthcare industry",
                        weight=20, required=True),
    ))
    monkeypatch.setattr(semantic_judge, "_call_judge",
                        _fake_judge(omit_people={"p0"}, omit_criteria={("p1", "ai"), ("p1", "hc")}))
    run = semantic_judge.run_judge("x", plan, _bundle(10), ScoringContext(),
                                   network_size=10, pool_size=10, hard_rejected_count=0)
    for cid in ("mentor_evidence", "ai", "hc"):
        assert run.verdicts["p0"][cid].get("judge_missing")
        assert run.verdicts["p0"][cid]["status"] == TriState.UNKNOWN
    assert run.verdicts["p1"]["ai"].get("judge_missing") and run.verdicts["p1"]["hc"].get("judge_missing")
    assert run.verdicts["p1"]["mentor_evidence"]["status"] == TriState.TRUE
    assert run.metadata.omitted_people == 1
    assert run.metadata.omitted_criteria == 2
    assert run.metadata.judge_status == JudgeStatus.PARTIAL


# ─────────────────────── §54 — provider routing (through the router) ───────────────────────


def test_judge_batch_uses_the_router(monkeypatch):
    seen = {}

    def fake_generate(system, user, schema, **kw):
        seen["operation"] = kw.get("operation")
        meta = {"operation": kw.get("operation"), "selected_provider": "anthropic:paid",
               "selected_model": "claude-haiku-4-5", "attempts": [{"provider": "anthropic:paid", "status": "success"}]}
        return CompactJudgeBatch(people=[]), "anthropic:paid", "claude-haiku-4-5", meta

    monkeypatch.setattr(semantic_judge, "generate_structured", fake_generate)
    outcome, payload, provider, model = semantic_judge._call_judge({"criteria_to_judge": []}, [{"person_id": "p0"}])
    assert outcome == "ok" and provider == "anthropic:paid"
    assert seen["operation"] == "semantic_judge"


def test_judge_batch_none_when_router_exhausted(monkeypatch):
    monkeypatch.setattr(semantic_judge, "generate_structured",
                        lambda *a, **k: (None, {"attempts": [{"provider": "anthropic:paid", "status": "bad_output"}]}))
    outcome, payload, provider, model = semantic_judge._call_judge({"criteria_to_judge": []}, [{"person_id": "p0"}])
    assert outcome == "failed"


# ─────────────────────── §55/§56/§57 — the SearchPlan reaches the judge ───────────────────────


def test_query_intent_and_relational_context_reach_the_judge(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    calls = []
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge(capture=calls))
    semantic_judge.run_judge("Who could mentor a backend engineer moving into management?",
                             _mentor_plan(), _bundle(3), ScoringContext(),
                             network_size=3, pool_size=3, hard_rejected_count=0)
    payload = calls[0]["payload"]
    assert payload["intent"] == "mentor_recommendation"
    assert payload["target_person_context"]["field"] == "backend engineering"
    assert payload["target_person_context"]["goal"] == "engineering management"


def test_modality_reaches_the_judge(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    calls = []
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge(capture=calls))
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="hipaa", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="HIPAA compliance experience", weight=100,
                        required=False, modality=Modality.POSSIBLE),
    ])
    semantic_judge.run_judge("might have HIPAA experience", plan, _bundle(2), ScoringContext(),
                             network_size=2, pool_size=2, hard_rejected_count=0)
    crit = calls[0]["payload"]["criteria_to_judge"][0]
    assert crit["modality"] == "possible" and crit["required"] is False


def test_boolean_structure_reaches_the_judge(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    calls = []
    monkeypatch.setattr(semantic_judge, "_call_judge", _fake_judge(capture=calls))
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="cyber", type=CriterionType.PROFESSIONAL_CONCEPT,
                        concept="cybersecurity experience", weight=33, required=True),
        SearchCriterion(id="health", type=CriterionType.INDUSTRY_EXPERIENCE,
                        concept="healthcare industry experience", weight=33, required=True),
        SearchCriterion(id="skills", type=CriterionType.ROLE_FUNCTION,
                        values=["security engineering", "cloud engineering"],
                        operator=Operator.ANY_OF, weight=34, required=False),
    ])
    semantic_judge.run_judge("cybersecurity AND healthcare", plan, _bundle(2), ScoringContext(),
                             network_size=2, pool_size=2, hard_rejected_count=0)
    crits = {c["id"]: c for c in calls[0]["payload"]["criteria_to_judge"]}
    assert crits["cyber"]["required"] and crits["health"]["required"]      # two required dimensions
    assert crits["skills"]["operator"] == "ANY_OF"
    assert crits["skills"]["values"] == ["security engineering", "cloud engineering"]  # not flattened


# ─────────────────────── §50/§51/§18 — fact-consistency validator ───────────────────────


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _packet(pid="p0"):
    return {
        "person_id": pid,
        "current": {"experience_id": "e2", "title": "Product Manager", "company": "Co", "is_current": True},
        "past": [{"experience_id": "e1", "title": "Software Engineer", "company": "Co", "is_current": False,
                  "description": "backend services"}],
        "education": [{"education_id": "ed1", "school": "Stanford", "degree": "MS", "field": "CS"}],
        "experience_semantics": [], "company_classifications": [], "skills": [],
        "certifications": [], "semantic_assertions": [],
    }


def _facts0():
    p = _person(0, current_title="Product Manager", current_company="Co")
    return _pf(p, [_Exp("Software Engineer", "Co", 2018, 2021, False, id="e1"),
                   _Exp("Product Manager", "Co", 2022, None, True, id="e2")])


def test_invalid_evidence_ref_downgrades_unsupported_true():
    raw = {"mentor": {"criterion_id": "mentor", "status": "true", "match_strength": 0.9,
                      "supporting_evidence_refs": ["exp:e999"], "reason": "led people"}}
    crit = _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and people leadership", required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["mentor"]["status"] == TriState.UNKNOWN
    assert any("invalid evidence ref" in n for n in out["mentor"]["validation"]["notes"])


def test_wrong_scope_evidence_downgrades_current_claim():
    raw = {"role": {"criterion_id": "role", "status": "true", "match_strength": 0.9,
                    "supporting_evidence_refs": ["exp:e1"], "experience_ids": ["e1"],
                    "reason": "was a software engineer"}}
    crit = _crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                 scope=Scope.CURRENT, required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["role"]["status"] == TriState.UNKNOWN
    assert any("not current" in n for n in out["role"]["validation"]["notes"])


def test_was_mentored_does_not_prove_is_a_mentor():
    """named correctness scenario (hardening PART 27): 'was mentored' / 'as a
    mentee' describes RECEIVING mentorship, the opposite of a mentor-experience
    criterion. Even a claimed TRUE with a grounded reference must be downgraded
    when the judge's own reason text says the person was on the receiving end."""
    raw = {"mentor": {"criterion_id": "mentor", "status": "true", "match_strength": 0.85,
                      "supporting_evidence_refs": ["exp:e1"],
                      "reason": "was mentored by a senior engineer early in her career"}}
    crit = _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring, coaching or advising others", required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["mentor"]["status"] == TriState.UNKNOWN
    assert any("receiving mentorship" in n for n in out["mentor"]["validation"]["notes"])


def test_actually_mentoring_others_survives_validation():
    """The contrast case: a grounded TRUE genuinely describing GIVING mentorship
    must NOT be caught by the was-mentored guard."""
    raw = {"mentor": {"criterion_id": "mentor", "status": "true", "match_strength": 0.85,
                      "supporting_evidence_refs": ["exp:e1"],
                      "reason": "mentored three junior engineers and led their onboarding"}}
    crit = _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring, coaching or advising others", required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["mentor"]["status"] == TriState.TRUE


def test_professor_from_education_only_is_downgraded():
    raw = {"prof": {"criterion_id": "prof", "status": "true", "match_strength": 0.8,
                    "supporting_evidence_refs": ["edu:ed1"], "reason": "studied at Stanford"}}
    crit = _crit(id="prof", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="a professor / faculty appointment at a university", required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["prof"]["status"] == TriState.UNKNOWN


def test_locked_company_classification_forces_false():
    ctx = ScoringContext(company_class={
        company_key("99", "Microsoft"): {"is_startup": False, "confidence": 0.95,
                                         "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    p = _person(0, current_company="Microsoft")
    facts = _pf(p, [_Exp("Principal PM", "Microsoft", 2018, None, True, id="e1", company_id="99")])
    key = company_key("99", "Microsoft")
    pkt = {"person_id": "p0",
           "current": {"experience_id": "e1", "company": "Microsoft", "is_current": True},
           "past": [], "company_classifications": [
               {"ref": f"company:{key}", "company": "Microsoft", "is_startup": False}],
           "education": [], "experience_semantics": [], "skills": [], "certifications": [],
           "semantic_assertions": []}
    raw = {"su": {"criterion_id": "su", "status": "true", "match_strength": 0.8,
                  "supporting_evidence_refs": [f"company:{key}"], "reason": "feels like a startup"}}
    crit = _crit(id="su", type=CriterionType.COMPANY_CATEGORY, concept="startup",
                 scope=Scope.CURRENT_COMPANY, required=True)
    out = validate_person(raw, pkt, ParsedSearchQuery(criteria=[crit]), facts, ctx)
    assert out["su"]["status"] == TriState.FALSE


def test_well_grounded_true_survives_validation():
    raw = {"mentor": {"criterion_id": "mentor", "status": "true", "match_strength": 0.88,
                      "supporting_evidence_refs": ["exp:e1"], "experience_ids": ["e1"],
                      "reason": "explicitly mentored five engineers per the role description"}}
    crit = _crit(id="mentor", type=CriterionType.PROFESSIONAL_CONCEPT,
                 concept="evidence of mentoring and people leadership", scope=Scope.CAREER, required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert out["mentor"]["status"] == TriState.TRUE
    assert out["mentor"]["supporting_evidence_refs"] == ["exp:e1"]


def test_non_judgeable_criterion_verdict_is_dropped():
    raw = {"tr": {"criterion_id": "tr", "status": "true", "match_strength": 0.9,
                  "supporting_evidence_refs": ["exp:e1"]}}
    crit = _crit(id="tr", type=CriterionType.CAREER_TRANSITION, concept="from consulting to tech",
                 required=True)
    out = validate_person(raw, _packet(), ParsedSearchQuery(criteria=[crit]), _facts0(), ScoringContext())
    assert "tr" not in out  # chronology stays code-authoritative (§16)


# ─────────────────────── mode = off / no judgeable criteria ───────────────────────


def test_mode_off_never_judges(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "off")
    monkeypatch.setattr(semantic_judge, "_call_judge", _raise_judge)
    run = semantic_judge.run_judge("x", _mentor_plan(), _bundle(5), ScoringContext(),
                                   network_size=5, pool_size=5, hard_rejected_count=0)
    assert run.metadata.judge_status == JudgeStatus.NOT_USED and run.verdicts == {}


def test_no_judgeable_criteria_is_not_used(monkeypatch):
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge, "_call_judge", _raise_judge)
    plan = ParsedSearchQuery(criteria=[
        SearchCriterion(id="c", type=CriterionType.CURRENT_COMPANY, value="Google", weight=100, required=True),
    ])
    run = semantic_judge.run_judge("at Google", plan, _bundle(5), ScoringContext(),
                                   network_size=5, pool_size=5, hard_rejected_count=0)
    assert run.metadata.judge_status == JudgeStatus.NOT_USED


# ─────────────────────── §22/§47/§48/§59 — end-to-end through /search ───────────────────────


def _all_true_batch(payload, packets, unresolved_by_person=None):
    expanded: dict[str, dict[str, dict]] = {}
    for pkt in packets:
        pid = pkt["person_id"]
        first_exp = (pkt.get("current") or {}).get("experience_id") \
            or (pkt["past"][0]["experience_id"] if pkt.get("past") else None)
        crit_ids = (unresolved_by_person or {}).get(pid) or [c["id"] for c in payload["criteria_to_judge"]]
        expanded[pid] = {
            cid: {
                "criterion_id": cid, "status": "true", "match_strength": 0.9, "confidence": 0.9,
                "reason": "role shows mentoring and leadership",
                "supporting_evidence_refs": [f"exp:{first_exp}"] if first_exp else [],
                "contradicting_evidence_refs": [], "experience_ids": [first_exp] if first_exp else [],
            }
            for cid in crit_ids
        }
    return "ok", expanded, "mock:provider", "mock-model-1"


def test_search_judges_every_viable_candidate_and_min_score_does_not_prefilter(client, monkeypatch):
    from tests.test_search import _enriched_dataset

    # §59 — normal search never classifies companies
    import app.services.company_intel as ci
    monkeypatch.setattr(ci, "get_or_classify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no classify during search")))

    seen = []
    monkeypatch.setattr(semantic_judge, "_call_judge",
                        lambda payload, packets, unresolved_by_person=None: (
                            seen.append(len(packets)),
                            _all_true_batch(payload, packets, unresolved_by_person))[1])
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_max_candidates", 0)

    ds = _enriched_dataset(client)
    body = client.post("/search", json={
        "dataset_id": ds,
        "query": "engineers who could mentor someone moving into management",
    }).json()

    jm = body["judge_metadata"]
    assert jm is not None and jm["mode"] == "all_viable"
    assert jm["hard_fact_rejected_count"] == 0
    assert jm["judge_candidate_count"] == jm["viable_candidate_count"] >= 3
    assert jm["judge_candidate_count"] == sum(seen)  # every viable candidate was in a packet
    # candidates whose DETERMINISTIC mentor score is ~0 still appear — proof
    # MIN_MATCH_SCORE no longer pre-filters and the judge verdict lifted them
    assert body["connections"]["returned"] >= 2
    assert body["connections"]["exact_match_count"] >= 1


def test_search_survives_judge_unavailable(client, monkeypatch):
    from tests.test_search import _enriched_dataset

    monkeypatch.setattr(semantic_judge, "_call_judge", lambda *a, **k: ("failed", None, None, None))
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    ds = _enriched_dataset(client)
    body = client.post("/search", json={
        "dataset_id": ds, "query": "people with mentoring and leadership experience",
    }).json()
    assert body["judge_metadata"]["status"] in ("unavailable", "not_used")
    # search still returns — deterministic scoring stands
    assert "connections" in body
