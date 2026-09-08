"""AI-verification state for a search (V4 PART 6 B2 / B9 / B12 / B22 / B30).

Pure helpers — no DB, no LLM. ``search_service`` uses these to decide whether a
search may return user-visible candidates and to build the safe metadata block
the API + frontend show ("Verified by Claude" / "AI verification could not be
completed").
"""
from __future__ import annotations

from app.config import settings
from app.constants import (
    JudgeStatus,
    LLMProviderName,
    SearchStatus,
    VerificationStatus,
)

#: provider id  ->  (family, human name shown in the UI)
_PROVIDER_DISPLAY = {
    LLMProviderName.ANTHROPIC: ("anthropic", "Claude"),
    LLMProviderName.GROQ_PRIMARY: ("groq_primary", "Groq (primary)"),
    LLMProviderName.GROQ_FALLBACK: ("groq_fallback", "Groq (fallback)"),
    LLMProviderName.OPENROUTER: ("openrouter", "OpenRouter (free)"),
    "deterministic": ("deterministic", None),
}

_MODEL_DISPLAY = {
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-5": "Claude Opus 5",
}


def _is_anthropic(provider: str | None) -> bool:
    return bool(provider) and provider.startswith("anthropic")


def model_display_name(model: str | None) -> str | None:
    if not model:
        return None
    return _MODEL_DISPLAY.get(model, model)


def provider_family(provider: str | None) -> str | None:
    fam, _ = _PROVIDER_DISPLAY.get(provider or "", (provider, None))
    return fam


def ai_identity(interp_provider: str | None, interp_model: str | None,
                *provider_dicts: dict | None) -> dict:
    """Collapse the interpretation provider + every judge/audit ``providers``
    tally into one safe identity block. ``anthropic_succeeded`` is True when
    Anthropic produced validated output for ANY stage."""
    seen: set[str] = set()
    if interp_provider and interp_provider != "deterministic":
        seen.add(interp_provider)
    for d in provider_dicts:
        for k, n in (d or {}).items():
            if n:
                seen.add(k)

    anthropic_succeeded = any(_is_anthropic(p) for p in seen)
    anthropic_attempted = (
        bool(settings.anthropic_api_key) or anthropic_succeeded or _is_anthropic(interp_provider)
    )
    non_anthropic = sorted(p for p in seen if not _is_anthropic(p) and p != "deterministic")
    fallback_used = bool(non_anthropic) and not anthropic_succeeded

    if anthropic_succeeded:
        winner, model = LLMProviderName.ANTHROPIC, (settings.anthropic_model
                                                    if _is_anthropic(interp_provider) else interp_model
                                                    or settings.anthropic_model)
    elif non_anthropic:
        winner, model = non_anthropic[0], None
    else:
        winner, model = None, None

    return {
        "ai_provider": provider_family(winner),
        "ai_model": model_display_name(model),
        "anthropic_attempted": anthropic_attempted,
        "anthropic_succeeded": anthropic_succeeded,
        "fallback_used": fallback_used,
    }


def judge_verified(required_semantic: bool, judge_meta: dict | None) -> bool:
    """Did the semantic judge actually produce validated verdicts for a query
    that needed them?"""
    if not required_semantic:
        return True
    m = judge_meta or {}
    if m.get("status") in (JudgeStatus.FULL, JudgeStatus.PARTIAL):
        return int(m.get("judge_successful_batches") or 0) > 0
    return False


def interpretation_gate(interp_provider: str | None) -> tuple[str, str, str] | None:
    """Step 1 only — run right after ``interpret_query`` so an un-interpretable
    query bails BEFORE the expensive pipeline. Returns a failure triple or
    ``None`` to continue."""
    if interp_provider == "deterministic":
        if settings.require_llm_for_results:
            return (SearchStatus.AI_UNAVAILABLE, VerificationStatus.INCOMPLETE,
                    "AI could not interpret the query — no results were returned to avoid "
                    "showing keyword-only matches.")
    elif settings.search_require_anthropic and not _is_anthropic(interp_provider):
        return (SearchStatus.AI_UNAVAILABLE, VerificationStatus.INCOMPLETE,
                "SEARCH_REQUIRE_ANTHROPIC is set and Anthropic did not handle this search.")
    return None


def decide(
    *,
    interp_provider: str | None,
    required_semantic: bool,
    judge_meta: dict | None,
    audit_meta: dict | None,
    audit_ran: bool,
    deadline_reached: bool,
) -> tuple[str, str, str | None]:
    """Full post-pipeline decision. Return ``(search_status,
    verification_status, reason_or_None)``. ``reason`` is a short human sentence
    for the diagnostic response — never a normal result explanation."""
    require_llm = settings.require_llm_for_results

    gate = interpretation_gate(interp_provider)
    if gate is not None:
        return gate

    # 2 — required semantic verification must have happened
    if not judge_verified(required_semantic, judge_meta):
        if require_llm:
            return (SearchStatus.VERIFICATION_INCOMPLETE, VerificationStatus.INCOMPLETE,
                    "AI semantic verification could not be completed. No results were "
                    "returned to avoid showing unverified matches.")

    # 3 — final audit, when the deployment requires it (B10). Only enforced when
    # the audit is actually enabled, results require an LLM, and the query has a
    # semantic criterion the audit would review.
    if (settings.search_require_final_audit and settings.final_result_audit_enabled
            and require_llm and required_semantic):
        audit_ok = audit_ran and (audit_meta or {}).get("status") in ("full", "partial") \
            and int((audit_meta or {}).get("successful_batches") or 0) > 0
        if not audit_ok:
            return (SearchStatus.VERIFICATION_INCOMPLETE, VerificationStatus.INCOMPLETE,
                    "Final AI verification could not be completed within the time budget. "
                    "No results were returned.")

    # 4 — success. fallback vs anthropic is decided by ai_identity(); the caller
    #     maps SUCCESS -> SUCCESS_WITH_FALLBACK when fallback_used is True.
    if not required_semantic and interp_provider == "deterministic":
        # only possible when require_llm is False — nothing AI actually verified
        return (SearchStatus.SUCCESS, VerificationStatus.NOT_REQUIRED, None)
    vs = VerificationStatus.NOT_REQUIRED if not required_semantic else VerificationStatus.COMPLETE
    if required_semantic and deadline_reached and not (judge_meta or {}).get("status") == JudgeStatus.FULL:
        vs = VerificationStatus.INCOMPLETE
    return (SearchStatus.SUCCESS, vs, None)
