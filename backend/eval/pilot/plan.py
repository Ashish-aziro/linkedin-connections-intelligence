"""Live-pilot cost / call plan (V4 PART 10 §5/§15/§17, PART 10.1 §11).

Pure estimation from configuration + the offline dry-run funnels. No dollar
figures are invented — only call counts and configured routing.
"""
from __future__ import annotations

import math

from app.config import settings
from app.services.llm.providers import default_chain

#: the cheap first slice to run live (V4 PART 10.1 §7)
STAGED_QUERY_IDS = [
    "q03_cxo_event_memphis_nashville",
    "q09_ex_amazon_now_startup",
    "q10_cyber_and_healthcare",
    "q11_academia_to_industry",
    "q12_backend_to_management_mentor",
]


def provider_routing() -> dict:
    chain = default_chain()
    return {
        "chain": [{"name": p.name, "model": getattr(p, "model", None)} for p in chain],
        "anthropic_configured": bool(settings.anthropic_api_key),
        "anthropic_workspace_id_configured": bool(settings.anthropic_workspace_id),
        "anthropic_first": bool(chain) and chain[0].name.startswith("anthropic"),
        "anthropic_model": settings.anthropic_model if settings.anthropic_api_key else None,
        "expected_first_provider": chain[0].name if chain else None,
        "anthropic_would_incur_paid_usage": bool(settings.anthropic_api_key),
    }


def semantic_v3_plan(missing_ids: list[str], present_ids: list[str]) -> dict:
    n = len(missing_ids)
    return {
        "target_semantic_version": settings.semantic_profile_version,
        "pilot_profiles": n + len(present_ids),
        "already_at_target": len(present_ids),
        "needing_enrichment": n,
        "estimated_llm_requests": n,   # derive_semantics == 1 structured call each
        "max_tokens_per_request": 2200,
        "batching": "one profile per request (not batched)",
        "source_data": "normalized DB rows via build_compact_profile — NO Apify, NO re-scrape",
        "provider_routing": provider_routing(),
    }


def _judge_batches(viable: int | None) -> int:
    if not viable:
        return 0
    return math.ceil(viable / settings.semantic_judge_batch_size)


def _audit_batches(returned: int | None, exact: int | None, possible: int | None) -> int:
    pool = min(
        settings.top_connections + settings.final_result_audit_buffer,
        (exact or 0) + (possible or 0) if exact is not None else (returned or 0),
    )
    return math.ceil(pool / settings.final_result_audit_batch_size) if pool > 0 else 0


def _per_query(rec: dict, *, reasons_enabled: bool) -> dict:
    f = rec.get("funnel", {})
    viable = f.get("viable_candidate_count") or f.get("candidate_pool_size")
    jb = _judge_batches(viable)
    ab = _audit_batches(f.get("returned"), f.get("exact"), f.get("possible")) \
        if settings.final_result_audit_enabled else 0
    reason = min(f.get("returned") or 0, settings.llm_reason_top_n) \
        if (settings.llm_reason_generation and reasons_enabled) else 0
    interp = 1 if settings.llm_query_interpretation else 0
    return {
        "query_id": rec["query_id"],
        "interpretation_calls": interp,
        "viable_candidates_offline": viable,
        "expected_judge_batches": jb,
        "expected_audit_batches": ab,
        "expected_reason_calls": reason,
        "expected_total": interp + jb + ab + reason,
    }


def _aggregate(records: list[dict], *, reasons_enabled: bool) -> dict:
    per = [_per_query(r, reasons_enabled=reasons_enabled) for r in records]
    totals = {
        "interpretation": sum(p["interpretation_calls"] for p in per),
        "judge_batches": sum(p["expected_judge_batches"] for p in per),
        "audit_batches": sum(p["expected_audit_batches"] for p in per),
        "reason": sum(p["expected_reason_calls"] for p in per),
    }
    totals["all_llm_requests"] = sum(totals.values())
    return {"per_query": per, "totals": totals}


def search_call_estimate(records: list[dict], *, reasons_enabled: bool = True) -> dict:
    """Single-scenario estimate (kept for back-compat)."""
    est = _aggregate(records, reasons_enabled=reasons_enabled)
    est["reasons_enabled"] = reasons_enabled
    est["note"] = ("offline funnels understate viable counts when the deterministic parser "
                   "produces fewer criteria than the LLM interpreter — treat as a floor")
    return est


def full_estimate(records: list[dict]) -> dict:
    """Both scenarios (production-like vs pilot with reasons off), full set + staged 5."""
    staged = [r for r in records if r["query_id"] in set(STAGED_QUERY_IDS)]
    return {
        "note": "offline funnels are a FLOOR — the LLM interpreter usually produces tighter "
                "criteria, shifting viable counts. No dollar cost is estimated.",
        "full_21_production_like": _aggregate(records, reasons_enabled=True),
        "full_21_reasons_disabled": _aggregate(records, reasons_enabled=False),
        "staged_5_production_like": _aggregate(staged, reasons_enabled=True),
        "staged_5_reasons_disabled": _aggregate(staged, reasons_enabled=False),
        "staged_query_ids": STAGED_QUERY_IDS,
    }


def live_commands(dataset_id: str) -> list[str]:
    joined = ",".join(STAGED_QUERY_IDS)
    return [
        "# 0. config-only preflight (zero network):",
        "python -m eval.pilot.run_pilot preflight",
        "# 1. (optional) semantic v3 backfill — pilot.db ONLY, stored data, NO Apify:",
        "python -m eval.pilot.run_pilot enrich --live --i-understand-costs",
        "# 2. STAGED live eval — 5 queries first:",
        f"python -m eval.pilot.run_pilot run --live --i-understand-costs --only {joined}",
        "# 3. full live eval (optionally --no-reasons to cut ~half the calls):",
        "python -m eval.pilot.run_pilot run --live --i-understand-costs",
        f"#    (all operate ONLY on eval/pilot/pilot.db built from {dataset_id}; production untouched)",
    ]
