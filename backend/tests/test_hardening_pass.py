"""Full-application hardening pass — Anthropic-primary routing, structured-
output truncation handling, adaptive batch splitting, staged local resolution
(call reduction), the soft LLM call budget, and the career-transition
dedup fix. No live LLM calls — every provider boundary is mocked/offline.
"""
from __future__ import annotations

import pytest

from app.constants import CriterionType, Qualification, Scope, TriState
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services import career_chronology
from app.services.llm import adaptive_batch, budget
from app.services.llm.base import LLMBadOutput, LLMOutputTruncated
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from app.services.semantic_judge import candidate_needs_judge, needs_semantic_judge
from tests.test_search import _Exp, _Person


def _facts(person, exps=None, sem=None):
    return ProfileFacts(person=person, experiences=exps or [], education=[], skills=[],
                        semantic=sem or {}, embedding=None)


def _crit(**kw):
    kw.setdefault("weight", 100)
    return SearchCriterion(**kw)


def _plan(*c, **kw):
    return ParsedSearchQuery(criteria=list(c), **kw)


# ═══════════════════ Phase 2 — structured-output truncation ═══════════════════


def test_anthropic_truncation_is_classified_distinctly(monkeypatch):
    import httpx

    from app.services.llm import anthropic_client

    class _Resp:
        status_code = 200

        def json(self):
            return {"stop_reason": "max_tokens",
                    "content": [{"type": "text", "text": '{"people": [{"person_id": "p1"'}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(LLMOutputTruncated):
        anthropic_client.messages_json(api_key="x", model="m", system_prompt="s",
                                       user_prompt="u", max_tokens=50)


def test_anthropic_genuine_malformed_json_stays_bad_output_not_truncated(monkeypatch):
    import httpx

    from app.services.llm import anthropic_client

    class _Resp:
        status_code = 200

        def json(self):
            return {"stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "not json at all {{{"}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(LLMBadOutput) as exc:
        anthropic_client.messages_json(api_key="x", model="m", system_prompt="s",
                                       user_prompt="u", max_tokens=50)
    assert not isinstance(exc.value, LLMOutputTruncated)  # ended normally -> not truncation


def test_openai_compatible_length_finish_reason_is_truncated(monkeypatch):
    import httpx

    from app.services.llm import openai_compatible

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"finish_reason": "length",
                                 "message": {"content": '{"reasons": [{"person_id": "p1"'}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    with pytest.raises(LLMOutputTruncated):
        openai_compatible.chat_json(base_url="https://x", api_key="k", model="m",
                                    system_prompt="s", user_prompt="u", max_tokens=50)


def test_router_does_not_retry_truncation_identically(monkeypatch):
    from app.services.llm import router as llm_router
    from app.services.llm.base import LLMProvider

    calls = {"n": 0}

    class _P(LLMProvider):
        name = "fake:provider"
        model = "fake-model"

        def available(self):
            return True

        def generate_json(self, *a, **k):
            calls["n"] += 1
            raise LLMOutputTruncated("hit max_tokens")

    from pydantic import BaseModel

    class _Schema(BaseModel):
        x: int = 0

    monkeypatch.setattr(llm_router.settings, "llm_max_retries", 2)
    result = llm_router.generate_structured("sys", "usr", _Schema, chain=[_P()],
                                            operation="test", return_meta=True)
    model, meta = result
    assert model is None
    # exactly ONE attempt against the truncating provider — no identical retry
    assert calls["n"] == 1
    assert meta["attempts"][0]["status"] == "output_truncated"


# ═══════════════════ Phase 3 — adaptive batch splitting ═══════════════════


def test_adaptive_split_on_truncation_keeps_successful_siblings():
    packets = [{"person_id": f"p{i}"} for i in range(10)]

    def call_fn(pkts):
        # the whole-10 batch truncates; splits of size <= 5 succeed
        if len(pkts) > 5:
            return "truncated", None, None, None
        return "ok", {"ids": [p["person_id"] for p in pkts]}, "prov", "model-1"

    leaves, stats = adaptive_batch.run_adaptive(packets, call_fn)
    assert stats.adaptive_splits == 1
    assert stats.truncations == 1
    ok_leaves = [leaf for leaf in leaves if leaf.outcome == "ok"]
    assert sum(len(leaf.packets) for leaf in ok_leaves) == 10
    assert stats.failed_batches == 0


def test_single_packet_unresolved_truncation_becomes_failed_leaf_not_infinite_loop():
    packets = [{"person_id": "solo"}]

    def call_fn(pkts):
        return "truncated", None, None, None  # never resolves, even at size 1

    leaves, stats = adaptive_batch.run_adaptive(packets, call_fn)
    assert stats.failed_batches == 1
    assert leaves[0].outcome == "failed"
    assert leaves[0].packets[0]["person_id"] == "solo"
    # bounded — did not recurse past a single packet
    assert stats.batches_attempted == 1


def test_genuine_failure_does_not_trigger_a_split():
    packets = [{"person_id": f"p{i}"} for i in range(6)]
    attempts = []

    def call_fn(pkts):
        attempts.append(len(pkts))
        return "failed", None, None, None

    leaves, stats = adaptive_batch.run_adaptive(packets, call_fn)
    assert attempts == [6]           # one shot, no split — only truncation splits
    assert stats.adaptive_splits == 0
    assert stats.failed_batches == 1


def test_no_duplicate_candidates_across_split_leaves():
    packets = [{"person_id": f"p{i}"} for i in range(8)]
    seen_across_leaves: list[str] = []

    def call_fn(pkts):
        if len(pkts) > 2:
            return "truncated", None, None, None
        return "ok", None, "prov", "m"

    leaves, _stats = adaptive_batch.run_adaptive(packets, call_fn)
    for leaf in leaves:
        seen_across_leaves.extend(p["person_id"] for p in leaf.packets)
    assert sorted(seen_across_leaves) == sorted(p["person_id"] for p in packets)
    assert len(seen_across_leaves) == len(set(seen_across_leaves))  # no duplicates


# ═══════════════════ Phase 6 — soft LLM call budget ═══════════════════


def test_budget_stops_after_max_calls_and_never_over_reports():
    budget.start_budget(2)
    try:
        assert budget.try_consume() is True
        assert budget.try_consume() is True
        assert budget.try_consume() is False   # exhausted — third call refused
        assert budget.used() == 2
    finally:
        budget.clear_budget()


def test_budget_zero_or_negative_means_unlimited():
    budget.start_budget(0)
    try:
        for _ in range(50):
            assert budget.try_consume() is True
    finally:
        budget.clear_budget()


def test_router_skips_the_call_when_budget_exhausted(monkeypatch):
    from app.services.llm import router as llm_router
    from app.services.llm.base import LLMProvider
    from pydantic import BaseModel

    class _Schema(BaseModel):
        x: int = 0

    class _P(LLMProvider):
        name = "fake:provider"
        model = "fake-model"

        def available(self):
            return True

        def generate_json(self, *a, **k):
            raise AssertionError("must not be called once the budget is exhausted")

    budget.start_budget(1)
    try:
        budget.try_consume()  # pre-exhaust
        result = llm_router.generate_structured("sys", "usr", _Schema, chain=[_P()],
                                                operation="test", return_meta=True)
        model, meta = result
        assert model is None
        assert meta["attempts"][0]["status"] == "budget_exhausted"
    finally:
        budget.clear_budget()


# ═══════════════════ Phase 4/5 — staged local resolution (call reduction) ═══════════════════


def _prescored_for(status_by_criterion, qualification):
    from app.services.scoring import ScoredCandidate

    return ScoredCandidate(
        person=_Person(), match_score=50.0, components=[], evidence=[],
        qualification=qualification, status_by_criterion=status_by_criterion,
    )


def test_needs_judge_false_when_deterministic_true_already_resolved():
    crit = _crit(id="c1", type=CriterionType.INDUSTRY_EXPERIENCE, concept="healthcare", required=True)
    prescored = _prescored_for({"c1": TriState.TRUE}, Qualification.EXACT_MATCH)
    assert needs_semantic_judge(prescored, crit) is False


def test_needs_judge_true_when_required_and_unknown():
    crit = _crit(id="c1", type=CriterionType.INDUSTRY_EXPERIENCE, concept="healthcare", required=True)
    prescored = _prescored_for({"c1": TriState.UNKNOWN}, Qualification.POSSIBLE_MATCH)
    assert needs_semantic_judge(prescored, crit) is True


def test_needs_judge_false_when_not_required():
    crit = _crit(id="c1", type=CriterionType.INDUSTRY_EXPERIENCE, concept="healthcare", required=False)
    prescored = _prescored_for({"c1": TriState.UNKNOWN}, Qualification.POSSIBLE_MATCH)
    assert needs_semantic_judge(prescored, crit) is False


def test_needs_judge_false_when_already_sealed_not_match_by_other_criterion():
    """A different required criterion is already FALSE -> this candidate is
    excluded regardless of this criterion's own outcome; no call is worth it."""
    crit = _crit(id="c1", type=CriterionType.INDUSTRY_EXPERIENCE, concept="healthcare", required=True)
    prescored = _prescored_for({"c1": TriState.UNKNOWN, "c2": TriState.FALSE}, Qualification.NOT_MATCH)
    assert needs_semantic_judge(prescored, crit) is False


def test_needs_judge_false_for_code_authoritative_type():
    crit = _crit(id="c1", type=CriterionType.CAREER_TRANSITION, concept="from consulting to tech",
                required=True)
    prescored = _prescored_for({}, Qualification.POSSIBLE_MATCH)
    assert needs_semantic_judge(prescored, crit) is False


def test_candidate_needs_judge_short_circuits_when_all_resolved():
    crits = [
        _crit(id="c1", type=CriterionType.ROLE_FUNCTION, concept="engineering", required=True),
        _crit(id="c2", type=CriterionType.INDUSTRY_EXPERIENCE, concept="fintech", required=True),
    ]
    prescored = _prescored_for({"c1": TriState.TRUE, "c2": TriState.TRUE}, Qualification.EXACT_MATCH)
    assert candidate_needs_judge(prescored, crits) is False


def test_broad_query_does_not_judge_everyone_when_most_are_decided(monkeypatch):
    """The actual call-reduction integration: run_judge on a synthetic pool where
    most candidates are already resolved by stored facts — proves the judge
    bundle shrinks to the genuinely ambiguous subset (V4 hardening PART 4)."""
    from app.services import semantic_judge

    plan = _plan(_crit(id="ind", type=CriterionType.INDUSTRY_EXPERIENCE,
                       concept="technology industry experience", required=True))

    def make(pid, *, resolved: bool):
        p = _Person()
        p.id = pid
        status = TriState.TRUE if resolved else TriState.UNKNOWN
        return p, _prescored_for({"ind": status}, Qualification.EXACT_MATCH if resolved else Qualification.POSSIBLE_MATCH)

    n_total, n_ambiguous = 50, 5
    people = [make(f"p{i}", resolved=(i >= n_ambiguous)) for i in range(n_total)]
    local_scored = {pid: sc for (p, sc) in people for pid in [p.id]}
    bundle = [(p, _facts(p), {"volunteering": [], "recommendations": []}) for p, _sc in people]

    seen_batches = []

    def fake_call_judge(payload, packets, unresolved_by_person=None):
        seen_batches.append(len(packets))
        return "ok", {}, "mock", "m1"

    monkeypatch.setattr(semantic_judge, "_call_judge", fake_call_judge)
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")

    run = semantic_judge.run_judge("technology experience", plan, bundle, ScoringContext(),
                                   network_size=n_total, pool_size=n_total, hard_rejected_count=0,
                                   local_scored=local_scored)
    assert run.metadata.candidates_decided_locally == n_total - n_ambiguous
    assert run.metadata.candidates_needing_llm == n_ambiguous
    assert run.metadata.judge_candidate_count == n_ambiguous
    assert sum(seen_batches) == n_ambiguous   # the judge never even saw the decided ones


# ═══════════════════ Phase 7 — career-transition dedup ═══════════════════


def test_transition_derives_true_from_already_verified_company_and_category():
    """'Former Amazon now at a startup' where BOTH endpoints are independently
    verifiable via the SAME matchers past_company/company_category already use
    -> the transition itself resolves TRUE, generalized (no company special-casing:
    this reuses company_matches + the shared category_field/company_class lookup)."""
    from app.services.company_intel import company_key

    p = _Person(current_company="Foo Startup")
    exps = [
        _Exp("SDE", "Consultia", 2015, 2019, False, id="e1"),
        _Exp("Founding Engineer", "Foo Startup", 2020, None, True, id="e2", company_id="fs1"),
    ]
    facts = _facts(p, exps)
    ctx = ScoringContext(company_class={
        company_key("fs1", "Foo Startup"): {"is_startup": True, "confidence": 0.9,
                                            "provenance": "ai_company_inference",
                                            "industries": [], "categories": []},
    })
    crit = _crit(id="tr", type=CriterionType.CAREER_TRANSITION,
                concept="from Consultia to a startup", required=True)
    strength, _ev, status = career_chronology.score_transition(facts, crit, ctx)
    assert status == TriState.TRUE


def test_transition_without_ctx_falls_back_to_concept_overlap_only():
    """Backward compatible — omitting ctx (existing unit tests / callers that
    don't have one yet) must not raise, just behave as before."""
    p = _Person()
    exps = [_Exp("Consultant", "Big Consulting", 2015, 2019, False, id="e1",
                desc="management consulting"),
           _Exp("Engineer", "Tech Co", 2020, None, True, id="e2", desc="software engineering")]
    facts = _facts(p, exps)
    crit = _crit(id="tr", type=CriterionType.CAREER_TRANSITION,
                concept="from consulting to tech", required=True)
    _strength, _ev, status = career_chronology.score_transition(facts, crit)
    assert status in (TriState.TRUE, TriState.UNKNOWN)  # never raises, never a hard FALSE here


def test_transition_does_not_double_penalize_when_components_already_true(monkeypatch):
    """End-to-end at the scoring layer: a candidate with verified past_company +
    company_category BOTH true, plus a career_transition criterion for the same
    A->B, must not be knocked to POSSIBLE just because the transition criterion
    exists — chronology confirms the transition too, so it resolves TRUE."""
    from app.services.company_intel import company_key

    p = _Person(current_company="Foo Startup")
    exps = [
        _Exp("SDE", "Amazon", 2015, 2019, False, id="e1", company_id="amz"),
        _Exp("Founding Engineer", "Foo Startup", 2020, None, True, id="e2", company_id="fs1"),
    ]
    facts = _facts(p, exps)
    ctx = ScoringContext(company_class={
        company_key("fs1", "Foo Startup"): {"is_startup": True, "confidence": 0.9,
                                            "provenance": "ai_company_inference",
                                            "industries": [], "categories": []},
    })
    plan = _plan(
        _crit(id="past", type=CriterionType.PAST_COMPANY, value="Amazon", required=True,
             scope=Scope.PAST_COMPANY),
        _crit(id="cat", type=CriterionType.COMPANY_CATEGORY, concept="startup", required=True,
             scope=Scope.CURRENT_COMPANY),
        _crit(id="tr", type=CriterionType.CAREER_TRANSITION, concept="from Amazon to a startup",
             required=True),
    )
    result = score_candidate(facts, plan, ctx)
    assert result.qualification == Qualification.EXACT_MATCH
    assert not result.uncertain_required


# ═══════════════════ Phase 8 — role_function / industry_experience / company_category ═══════════════════


def test_technical_role_at_non_tech_employer_vs_tech_employer_non_technical_role():
    from app.services.company_intel import company_key

    ctx = ScoringContext(company_class={
        company_key(None, "Google"): {"is_technology_company": True, "confidence": 0.95,
                                      "provenance": "ai_company_inference", "industries": [], "categories": []},
    })
    # Software Engineer at JPMorgan: technical role_function TRUE, NOT a tech company.
    # role_function/industry_experience are resolved from STORED ProfileSemantic
    # v3 data (what real enrichment produces) — a bare description alone is not
    # enough for a confident deterministic TRUE (only a similarity signal, §Phase8).
    swe_jpm = _facts(
        _Person(current_company="JPMorgan"),
        [_Exp("Software Engineer", "JPMorgan", 2020, None, True, id="e1",
             desc="writes backend services")],
        sem={"experience_semantics": [{
            "experience_id": "e1", "role_function": "software engineering",
            "employer_industries": ["financial services"], "confidence": 0.9,
        }]},
    )
    role_crit = _crit(id="role", type=CriterionType.ROLE_FUNCTION, concept="software engineering",
                      required=True)
    cat_crit = _crit(id="cat", type=CriterionType.COMPANY_CATEGORY, concept="tech company", required=True)

    from app.services.scoring import _score_company_category, _score_semantic_multi

    _s, _e, role_status = _score_semantic_multi(swe_jpm, role_crit, ctx)
    _s2, _e2, cat_status = _score_company_category(swe_jpm, cat_crit, ctx)
    assert role_status == TriState.TRUE
    assert cat_status != TriState.TRUE  # JPMorgan is not classified as a tech company

    # Accountant at Google: tech-company employment TRUE, NOT a technical role
    acct_google = _facts(_Person(current_company="Google"),
                        [_Exp("Accountant", "Google", 2020, None, True, id="e2",
                             desc="manages financial reporting")])
    _s3, _e3, cat_status2 = _score_company_category(acct_google, cat_crit, ctx)
    _s4, _e4, role_status2 = _score_semantic_multi(acct_google, role_crit, ctx)
    assert cat_status2 == TriState.TRUE
    assert role_status2 != TriState.TRUE  # accounting is not software engineering


# ═══════════════════ Phase 9 — event context stays out of criteria (regression) ═══════════════════


def test_networking_event_context_not_a_skill_criterion():
    from app.services.query_facts import strip_context

    stripped, ctx = strip_context("Who should I invite to a CXO networking event in Memphis?")
    assert "purpose" in ctx
    assert "networking" not in stripped.lower() or "event" not in stripped.lower()


# ═══════════════════ Phase 20 — offline ~1000-profile acceptance benchmark ═══════════════════


def test_offline_1000_profile_benchmark_does_not_explode_judge_calls(monkeypatch, capsys):
    """Synthetic ~1000-profile network (NOT real profile data). Most candidates
    are decidable from stored facts/semantics; only a small ambiguous slice
    should ever reach the judge — proving a broad query does not turn into
    ~100 judge requests just because the network is ~1,000 people."""
    from app.services import semantic_judge

    plan = _plan(_crit(id="ind", type=CriterionType.INDUSTRY_EXPERIENCE,
                       concept="technology industry experience", required=True))

    NETWORK_SIZE = 991
    HARD_REJECTED = 40
    VIABLE = NETWORK_SIZE - HARD_REJECTED
    AMBIGUOUS = 47   # the genuinely-unresolved slice

    local_scored = {}
    bundle = []
    for i in range(VIABLE):
        p = _Person()
        p.id = f"p{i}"
        resolved = i >= AMBIGUOUS
        status = TriState.TRUE if resolved else TriState.UNKNOWN
        qual = Qualification.EXACT_MATCH if resolved else Qualification.POSSIBLE_MATCH
        local_scored[p.id] = _prescored_for({"ind": status}, qual)
        bundle.append((p, _facts(p), {"volunteering": [], "recommendations": []}))

    judge_requests = {"n": 0}

    def fake_call_judge(payload, packets, unresolved_by_person=None):
        judge_requests["n"] += 1
        return "ok", {}, "mock", "m1"

    monkeypatch.setattr(semantic_judge, "_call_judge", fake_call_judge)
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "all_viable")
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_batch_size", 10)

    run = semantic_judge.run_judge("technology industry experience", plan, bundle, ScoringContext(),
                                   network_size=NETWORK_SIZE, pool_size=NETWORK_SIZE,
                                   hard_rejected_count=HARD_REJECTED, local_scored=local_scored)

    report = {
        "network_size": NETWORK_SIZE,
        "locally_evaluated": VIABLE,
        "hard_rejected": HARD_REJECTED,
        "locally_resolved": run.metadata.candidates_decided_locally,
        "llm_ambiguous": run.metadata.candidates_needing_llm,
        "judge_candidates": run.metadata.judge_candidate_count,
        "judge_batches": run.metadata.judge_batch_count,
        "estimated_llm_calls": judge_requests["n"],
    }
    print(f"\n[PART 20 offline benchmark] {report}")

    assert run.metadata.candidates_decided_locally == VIABLE - AMBIGUOUS
    assert run.metadata.candidates_needing_llm == AMBIGUOUS
    # the actual assertion the mission cares about: NOT ~100 judge requests for
    # a ~1,000-person network just because the network is large.
    assert judge_requests["n"] <= 5
    assert judge_requests["n"] == run.metadata.judge_batch_count
