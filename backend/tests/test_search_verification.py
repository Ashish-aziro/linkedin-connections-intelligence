"""V4 PART 6 B2 / B9 / B12 / B13 — the AI-verification decision helpers.

Unit tests for ``search_verification`` (no DB, no LLM): who verified the search,
whether the query needed LLM verification and got it, and which typed state the
search ends in.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.constants import JudgeStatus, SearchStatus, VerificationStatus
from app.services import search_verification as sv


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(settings, "require_llm_for_results", True)
    monkeypatch.setattr(settings, "search_require_anthropic", False)
    monkeypatch.setattr(settings, "search_require_final_audit", False)
    monkeypatch.setattr(settings, "final_result_audit_enabled", True)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")


# ── ai_identity ────────────────────────────────────────────────────────────


def test_identity_anthropic_won():
    out = sv.ai_identity("anthropic:paid", "claude-sonnet-5",
                         {"anthropic:paid": 3}, {"anthropic:paid": 2})
    assert out["ai_provider"] == "anthropic"
    assert out["anthropic_succeeded"] is True
    assert out["fallback_used"] is False
    assert out["ai_model"] == "Claude Sonnet 5"


def test_identity_fallback_used():
    out = sv.ai_identity("groq:primary", None, {"groq:primary": 2}, None)
    assert out["ai_provider"] == "groq_primary"
    assert out["anthropic_succeeded"] is False
    assert out["fallback_used"] is True


def test_identity_deterministic_only():
    out = sv.ai_identity("deterministic", None)
    assert out["ai_provider"] is None  # nothing AI verified this search
    assert out["ai_model"] is None
    assert out["anthropic_succeeded"] is False
    assert out["fallback_used"] is False


# ── interpretation_gate ────────────────────────────────────────────────────


def test_gate_blocks_deterministic_when_llm_required():
    s, vstat, reason = sv.interpretation_gate("deterministic")
    assert s == SearchStatus.AI_UNAVAILABLE and reason


def test_gate_allows_deterministic_when_not_required(monkeypatch):
    monkeypatch.setattr(settings, "require_llm_for_results", False)
    assert sv.interpretation_gate("deterministic") is None


def test_gate_blocks_non_anthropic_when_require_anthropic(monkeypatch):
    monkeypatch.setattr(settings, "search_require_anthropic", True)
    s, _, reason = sv.interpretation_gate("groq:primary")
    assert s == SearchStatus.AI_UNAVAILABLE and "Anthropic" in reason


def test_gate_allows_anthropic():
    assert sv.interpretation_gate("anthropic:paid") is None


# ── decide ─────────────────────────────────────────────────────────────────


def _judge(status, ok_batches=1):
    return {"status": status, "judge_successful_batches": ok_batches}


def test_decide_success_no_semantic_criterion():
    s, vstat, reason = sv.decide(
        interp_provider="anthropic:paid", required_semantic=False,
        judge_meta=None, audit_meta=None, audit_ran=False, deadline_reached=False,
    )
    assert s == SearchStatus.SUCCESS
    assert vstat == VerificationStatus.NOT_REQUIRED
    assert reason is None


def test_decide_success_semantic_judge_full():
    s, vstat, _ = sv.decide(
        interp_provider="anthropic:paid", required_semantic=True,
        judge_meta=_judge(JudgeStatus.FULL), audit_meta={"status": "full", "successful_batches": 2},
        audit_ran=True, deadline_reached=False,
    )
    assert s == SearchStatus.SUCCESS
    assert vstat == VerificationStatus.COMPLETE


def test_decide_judge_unavailable_is_verification_incomplete():
    s, vstat, reason = sv.decide(
        interp_provider="anthropic:paid", required_semantic=True,
        judge_meta=_judge(JudgeStatus.UNAVAILABLE, ok_batches=0),
        audit_meta=None, audit_ran=False, deadline_reached=False,
    )
    assert s == SearchStatus.VERIFICATION_INCOMPLETE
    assert vstat == VerificationStatus.INCOMPLETE and reason


def test_decide_required_audit_missing_blocks(monkeypatch):
    monkeypatch.setattr(settings, "search_require_final_audit", True)
    s, _, reason = sv.decide(
        interp_provider="anthropic:paid", required_semantic=True,
        judge_meta=_judge(JudgeStatus.FULL), audit_meta=None,
        audit_ran=False, deadline_reached=True,
    )
    assert s == SearchStatus.VERIFICATION_INCOMPLETE and "verification" in reason.lower()


def test_decide_required_audit_present_passes(monkeypatch):
    monkeypatch.setattr(settings, "search_require_final_audit", True)
    s, vstat, _ = sv.decide(
        interp_provider="anthropic:paid", required_semantic=True,
        judge_meta=_judge(JudgeStatus.FULL),
        audit_meta={"status": "full", "successful_batches": 3},
        audit_ran=True, deadline_reached=False,
    )
    assert s == SearchStatus.SUCCESS


def test_decide_deterministic_interpretation_is_ai_unavailable():
    s, _, _ = sv.decide(
        interp_provider="deterministic", required_semantic=True,
        judge_meta=None, audit_meta=None, audit_ran=False, deadline_reached=False,
    )
    assert s == SearchStatus.AI_UNAVAILABLE
