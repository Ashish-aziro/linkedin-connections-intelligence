"""FULL SONNET VERIFICATION (experimental).

Every candidate that survives the hard-fact viability gate is reviewed by Claude
Sonnet against the WHOLE search plan before a search may successfully return.
There is no "judge only where locally unresolved" shortcut and no
partial/degraded success: if a candidate's review cannot be completed after
bounded recovery, the whole SEARCH fails with a retryable error.

Reuses the mature machinery already in the tree:
  * ``judge_packet.build_packets`` (with ``full_profile=True``) — a complete,
    evidence-referenced, size-bounded profile packet per candidate
  * ``semantic_judge._call_judge`` — one batched structured request through the
    Anthropic-only router (retries, circuit breaker, model selection)
  * ``llm.adaptive_batch.run_adaptive`` — truncation -> split-in-half retry
  * ``judge_validator.validate_person`` — every verdict / evidence reference is
    validated against that person's exact packet + the locked deterministic
    facts BEFORE it can move a score. FACTS STAY AUTHORITATIVE.

Recovery ladder for a candidate/criterion the model omitted or truncated:
  1. batched pass (adaptive split on truncation)
  2. retry the candidate alone (bounded)
  3. per-required-criterion targeted single call on a compacted packet
  4. still missing a REQUIRED criterion -> raise VerificationIncompleteError
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.constants import FullVerificationStatus, TriState, VerdictState
from app.schemas import CompactJudgeBatch, ParsedSearchQuery
from app.services.judge_packet import build_packets, plan_payload
from app.services.llm.adaptive_batch import run_adaptive
from app.services.llm.router import generate_structured
from app.services.llm.token_estimate import estimate_judge_output_tokens
from app.services.semantic_judge import (
    _call_judge,
    _expand_compact,
    _make_batches,
    judgeable_criteria,
)

log = logging.getLogger("app.full_verification")

_TARGETED_SYSTEM = (
    "You are a professional-network analyst. Decide whether ONE person satisfies ONE "
    "criterion, using ONLY the evidence packet given. FACTS ARE LOCKED — do not "
    "contradict or invent an employer, role, date, degree, skill or reference.\n"
    "MISSING EVIDENCE IS NOT FALSE: return \"unknown\" when the packet does not "
    "establish the claim. Ground a \"true\" with a supporting_refs entry "
    "(\"exp:<id>\", \"edu:<id>\", \"cert:<id>\", \"skill:<name>\", \"assertion:<n>\", "
    "\"company:<key>\", \"pub:<id>\", \"vol:<id>\", \"rec:<id>\"). Return JSON only: "
    '{"people":[{"person_id":"...","criteria":[{"criterion_id":"...","status":"true|false|unknown",'
    '"confidence":0-1,"supporting_refs":[],"contradicting_refs":[],"reason":""}]}]}'
)


class VerificationIncompleteError(RuntimeError):
    """Full Sonnet verification could not be completed — the SEARCH must fail
    (retryable), never return partial/unverified results."""

    def __init__(self, message: str, *, metadata: dict | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


@dataclass
class FullVerificationRun:
    #: person_id -> criterion_id -> RAW verdict dict (``semantic_judge`` shape),
    #: ready for ``judge_validator.validate_person``.
    verdicts: dict[str, dict[str, dict]]
    packets_by_id: dict[str, dict]
    metadata: dict = field(default_factory=dict)


def _unknown_verdict(cid: str, *, missing: bool, reason: str = "") -> dict:
    return {
        "criterion_id": cid, "status": TriState.UNKNOWN,
        "match_strength": 0.0, "confidence": 0.0, "reason": reason,
        "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
        "experience_ids": [], "judge_missing": bool(missing),
    }


def run_full_verification(
    query: str,
    parsed: ParsedSearchQuery,
    bundle: list[tuple],
    ctx,
    *,
    network_size: int,
    pool_size: int,
    hard_rejected_count: int,
    local_scored: dict | None = None,
) -> FullVerificationRun:
    """``bundle``: ``[(person, ProfileFacts, {"volunteering": [...], "recommendations": [...]})]``
    — EVERY candidate that passed the hard-fact gate. No filtering here."""
    jcrits = judgeable_criteria(parsed)
    all_ids = [c.id for c in jcrits]
    required_ids = {c.id for c in jcrits if c.required}
    filtered_count = len(bundle)

    meta: dict = {
        "mode": "full_verification",
        "status": FullVerificationStatus.NOT_USED,
        "model": settings.anthropic_model,
        "network_size": network_size,
        "candidate_pool_size": pool_size,
        "hard_fact_rejected_count": hard_rejected_count,
        "filtered_candidate_count": filtered_count,
        "sonnet_verified_candidate_count": 0,
        "judgeable_criteria": all_ids,
        "required_criteria": sorted(required_ids),
        "criteria_reviewed": 0,
        "batch_count": 0,
        "successful_batches": 0,
        "failed_batches": 0,
        "truncations": 0,
        "adaptive_splits": 0,
        "single_person_retries": 0,
        "chunked_profile_reviews": 0,
        "targeted_criterion_calls": 0,
        "total_llm_calls": 0,
        "providers": {},
        "models": [],
        "excluded_false": 0,
        "excluded_insufficient_evidence": 0,
        # ── frontend-compat keys (mirror JudgeMetadata.as_dict) ──
        "judge_candidate_count": filtered_count,
        "judge_batch_count": 0,
        "judge_successful_batches": 0,
        "judge_failed_batches": 0,
        "omitted_criteria": 0,
        "omitted_people": 0,
    }

    if not settings.full_llm_verification or not jcrits or not bundle:
        # legacy judge path owns non-full-verification searches
        return FullVerificationRun({}, {}, meta)

    review_by_person = {p.id: list(all_ids) for (p, _f, _x) in bundle}
    packets = build_packets(
        bundle, parsed, ctx, query=query, unresolved_by_person=review_by_person,
        max_packet_chars=settings.full_verification_max_packet_chars, full_profile=True,
    )
    packets_by_id = {pkt["person_id"]: pkt for pkt in packets}
    payload = plan_payload(query, parsed, jcrits)

    verdicts: dict[str, dict[str, dict]] = {}

    def _absorb(leaves) -> None:
        for leaf in leaves:
            if leaf.outcome != "ok":
                continue
            for pid, crit_verdicts in leaf.payload.items():
                verdicts.setdefault(pid, {}).update(crit_verdicts)

    def _absorb_stats(stats) -> None:
        meta["batch_count"] += stats.batches_attempted
        meta["successful_batches"] += stats.successful_batches
        meta["failed_batches"] += stats.failed_batches
        meta["truncations"] += stats.truncations
        meta["adaptive_splits"] += stats.adaptive_splits
        meta["total_llm_calls"] += stats.batches_attempted
        for prov, n in stats.providers.items():
            meta["providers"][prov] = meta["providers"].get(prov, 0) + n
        for m in stats.models:
            if m not in meta["models"]:
                meta["models"].append(m)

    def _call(pkts):
        return _call_judge(payload, pkts, review_by_person)

    # ── 1. batched pass ────────────────────────────────────────────────
    normal = [pkt for pkt in packets if not pkt.get("_packet_too_large")]
    oversized_ids = [pkt["person_id"] for pkt in packets if pkt.get("_packet_too_large")]
    batches, over_by_chars = _make_batches(
        normal, size=settings.full_verification_batch_size,
        max_chars=settings.semantic_judge_max_batch_chars,
    )
    oversized_ids += [x for x in over_by_chars if x]
    for batch in batches:
        leaves, stats = run_adaptive(batch, _call)
        _absorb_stats(stats)
        _absorb(leaves)

    def _gaps(pid: str) -> list[str]:
        got = verdicts.get(pid, {})
        return [cid for cid in review_by_person.get(pid, []) if cid not in got]

    # ── 2. single-person bounded retry ────────────────────────────────
    incomplete = [pid for pid in packets_by_id if _gaps(pid) and pid not in oversized_ids]
    for pid in incomplete:
        for _ in range(max(0, settings.full_verification_single_retries)):
            if not _gaps(pid):
                break
            meta["single_person_retries"] += 1
            leaves, stats = run_adaptive([packets_by_id[pid]], _call)
            _absorb_stats(stats)
            _absorb(leaves)

    # ── 3. per-required-criterion targeted calls (also handles oversized) ──
    needs_targeted = {pid for pid in packets_by_id if _gaps(pid)} | set(oversized_ids)
    for pid in needs_targeted:
        missing = [cid for cid in _gaps(pid)] or (list(all_ids) if pid in oversized_ids else [])
        # prioritise required criteria
        missing.sort(key=lambda c: (c not in required_ids))
        did_chunk = False
        for cid in missing:
            if cid in verdicts.get(pid, {}):
                continue
            crit = next((c for c in jcrits if c.id == cid), None)
            if crit is None:
                continue
            v = _targeted_criterion_call(payload, packets_by_id[pid], pid, crit, meta)
            if v is not None:
                verdicts.setdefault(pid, {})[cid] = v
                did_chunk = True
        if did_chunk:
            meta["chunked_profile_reviews"] += 1

    # ── 4. resolve gaps ──────────────────────────────────────────────
    unrecovered: dict[str, list[str]] = {}
    for pid in packets_by_id:
        for cid in _gaps(pid):
            if cid in required_ids:
                unrecovered.setdefault(pid, []).append(cid)
            else:
                # a preferred criterion we could not review — a completed search
                # tolerates this (only soft score), recorded as insufficient.
                verdicts.setdefault(pid, {})[cid] = _unknown_verdict(
                    cid, missing=False, reason="preferred criterion not reviewed")

    meta["criteria_reviewed"] = sum(len(v) for v in verdicts.values())
    meta["sonnet_verified_candidate_count"] = sum(
        1 for pid in packets_by_id if not any(c in required_ids for c in _gaps(pid))
    )
    meta["judge_batch_count"] = meta["batch_count"]
    meta["judge_successful_batches"] = meta["successful_batches"]
    meta["judge_failed_batches"] = meta["failed_batches"]

    if unrecovered:
        meta["status"] = FullVerificationStatus.INCOMPLETE
        meta["unrecovered_candidates"] = len(unrecovered)
        log.error("full verification INCOMPLETE — %d candidate(s) missing a required verdict: %s",
                  len(unrecovered), {k: v for k, v in list(unrecovered.items())[:5]})
        raise VerificationIncompleteError(
            f"Full Sonnet verification could not be completed for {len(unrecovered)} "
            f"of {filtered_count} candidate(s).",
            metadata=meta,
        )

    meta["status"] = FullVerificationStatus.COMPLETE
    log.info(
        "full verification COMPLETE — filtered=%d verified=%d criteria_reviewed=%d "
        "batches=%d/%d ok truncations=%d splits=%d single_retries=%d chunked=%d targeted=%d calls=%d model=%s",
        filtered_count, meta["sonnet_verified_candidate_count"], meta["criteria_reviewed"],
        meta["successful_batches"], meta["batch_count"], meta["truncations"], meta["adaptive_splits"],
        meta["single_person_retries"], meta["chunked_profile_reviews"], meta["targeted_criterion_calls"],
        meta["total_llm_calls"], settings.anthropic_model,
    )
    return FullVerificationRun(verdicts, packets_by_id, meta)


def _targeted_criterion_call(payload: dict, packet: dict, pid: str, crit, meta: dict) -> dict | None:
    """One tiny call: this person, this one criterion, a compacted packet. Output
    is a single verdict so it never truncates. Returns a raw verdict dict, or
    ``None`` when even this could not be completed."""
    meta["targeted_criterion_calls"] += 1
    meta["total_llm_calls"] += 1
    compact = _compact_packet_for(packet, crit)
    user = (
        "SEARCH PLAN:\n" + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n\nPERSON (one evidence packet):\n" + json.dumps(compact, ensure_ascii=False, default=str)
        + f"\n\nDecide ONLY criterion_id={crit.id!r}."
    )
    result = generate_structured(
        _TARGETED_SYSTEM, user, CompactJudgeBatch,
        max_tokens=estimate_judge_output_tokens(1, [1]) * 2,
        operation="full_verification_targeted", return_meta=True,
    )
    if result[0] is None:
        return None
    batch, provider, model_id, _m = result
    if provider:
        meta["providers"][provider] = meta["providers"].get(provider, 0) + 1
    if model_id and model_id not in meta["models"]:
        meta["models"].append(model_id)
    for pv in batch.people:
        if pv.person_id != pid:
            continue
        for cv in pv.criteria:
            if cv.criterion_id == crit.id:
                return _expand_compact(cv, set())
    return None


_HEAVY_KEYS = ("recommendations_received", "publications")


def _compact_packet_for(packet: dict, crit) -> dict:
    """A smaller copy of the packet for a single-criterion call — keep identity +
    every career row + classifications + assertions; drop the bulkiest prose."""
    out = {k: v for k, v in packet.items() if k not in _HEAVY_KEYS and not k.startswith("_")}
    if isinstance(out.get("about"), str):
        out["about"] = out["about"][:400]
    if isinstance(out.get("career_summary"), str):
        out["career_summary"] = out["career_summary"][:400]
    for e in out.get("past", []) or []:
        if isinstance(e, dict) and e.get("description"):
            e["description"] = e["description"][:200]
    out["unresolved_criteria"] = [crit.id]
    return out
