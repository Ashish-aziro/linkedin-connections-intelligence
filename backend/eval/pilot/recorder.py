"""Run the evaluation queries through the REAL search pipeline and capture detail.

``run_query`` calls ``run_connection_search`` unchanged, against a Session bound
to the isolated pilot DB. Judge verdicts, validator downgrades and final-audit
before→after transitions are captured by wrapping (monkeypatching) the seams the
search service uses — production code is NOT modified.

``offline=True`` forces every LLM provider to report "exhausted" so the run is
deterministic and free: query interpretation falls back to the deterministic
parser, the semantic judge and final audit report ``unavailable`` / ``not_used``.

``reasons_enabled=False`` disables LLM display-reason generation for the duration
of the run only (restored afterward) — it does not affect ranking (V4 PART 10.1 §8).
"""
from __future__ import annotations

import contextlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

# criterion types the deterministic UNKNOWN-rate heuristic treats as semantic
JUDGEABLE_TYPES = (
    "professional_concept", "semantic_concept", "role_function", "industry_experience",
    "company_category", "skill", "seniority", "career_transition",
)

_DROPPED_RE = re.compile(r"dropped (\d+) invalid evidence ref")
_SCOPE_NOTE_RE = re.compile(r"cannot prove a .*-scoped claim|supporting experience is (?:not current|not past)")


@dataclass
class JudgeTrace:
    verdict_status_by_type: dict[str, dict[str, int]] = field(default_factory=dict)
    validator_downgrades: int = 0          # TRUE/FALSE -> UNKNOWN (any reason)
    validator_dropped: int = 0             # whole verdicts removed (unknown criterion id etc.)
    grounding_downgrades: int = 0          # TRUE/FALSE -> UNKNOWN specifically for failed grounding
    invalid_evidence_refs: int = 0         # count of invented refs the validator stripped
    wrong_scope_refs: int = 0              # verdicts whose refs were the wrong current/past scope
    people_judged: int = 0
    missing_criterion_verdicts: int = 0    # judgeable criteria the model omitted (per person, summed)
    judgeable_criteria_expected: int = 0   # judgeable criteria * people judged
    # direct judge signal for REQUIRED semantic criteria (V4 PART 10.1 §6)
    required_semantic_total: int = 0
    required_semantic_true: int = 0
    required_semantic_false: int = 0
    required_semantic_unknown: int = 0

    def _bump(self, ctype: str, status: str) -> None:
        d = self.verdict_status_by_type.setdefault(ctype, {"true": 0, "false": 0, "unknown": 0})
        d[status] = d.get(status, 0) + 1

    @property
    def required_semantic_unknown_rate(self) -> float | None:
        if not self.required_semantic_total:
            return None
        return round(self.required_semantic_unknown / self.required_semantic_total, 4)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["required_semantic_unknown_rate"] = self.required_semantic_unknown_rate
        return d


@dataclass
class QueryRecord:
    query_id: str
    query: str
    group: str
    interpretation: dict[str, Any]
    funnel: dict[str, Any]
    results: list[dict]
    near_matches: list[dict]
    judge_metadata: dict | None
    audit_metadata: dict | None
    judge_trace: dict
    audit_transitions: list[dict]          # per person: first_pass -> final, decision, reason, issues
    audit_transition_tally: dict[str, int]
    audit_changes: list[dict]              # non-approved transitions only (kept for back-compat)
    final_uncertainty_rate: float | None   # from final uncertain_criteria (renamed, key kept below)
    unknown_required_rate: float | None    # alias of final_uncertainty_rate for older readers
    reason_generation_enabled: bool
    llm_provider: str | None
    llm_model: str | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


_AUDIT_REMOVED = "not_match"
_TRANSITION_BUCKETS = (
    "exact_to_exact", "exact_to_possible", "exact_to_removed",
    "possible_to_possible", "possible_to_removed", "other",
)


def _transition_bucket(first: str, final: str) -> str:
    f = "exact" if first == "exact_match" else "possible" if first == "possible_match" else "other"
    if final == _AUDIT_REMOVED:
        t = "removed"
    elif final == "exact_match":
        t = "exact"
    elif final == "possible_match":
        t = "possible"
    else:
        t = "other"
    key = f"{f}_to_{t}"
    return key if key in _TRANSITION_BUCKETS else "other"


@contextlib.contextmanager
def _instrument(trace: JudgeTrace, audit_sink: list[dict]):
    """Wrap the judge, judge-validator and final-audit-validator seams."""
    from app.services import final_audit_validator as fav
    from app.services import search_service as ss
    from app.services.semantic_judge import judgeable_criteria

    real_run_judge = ss.run_judge
    real_validate = ss.validate_person
    real_validate_audit = fav.validate_audit

    def wrapped_run_judge(query, parsed, bundle, ctx, **kw):
        run = real_run_judge(query, parsed, bundle, ctx, **kw)
        ctype = {c.id: c.type for c in parsed.criteria}
        expected = {c.id for c in judgeable_criteria(parsed)}
        trace.people_judged = len(run.verdicts)
        trace.judgeable_criteria_expected = len(expected) * len(run.verdicts)
        for _pid, crits in run.verdicts.items():
            for cid, raw in crits.items():
                status = str(raw.get("status", "unknown")).lower()
                if status not in ("true", "false", "unknown"):
                    status = "unknown"
                trace._bump(ctype.get(cid, "unknown_type"), status)
                if raw.get("judge_missing") and cid in expected:
                    trace.missing_criterion_verdicts += 1
        return run

    def wrapped_validate(person_verdicts, packet, parsed, facts, ctx):
        before = {cid: str(v.get("status", "unknown")).lower() for cid, v in person_verdicts.items()}
        out = real_validate(person_verdicts, packet, parsed, facts, ctx)

        required_ids = {c.id for c in parsed.criteria if c.required}
        judgeable_ids = {c.id for c in _judgeable(parsed)}

        for cid, bstatus in before.items():
            if cid not in out:
                trace.validator_dropped += 1
                continue
            v = out[cid]
            astatus = str(v.get("status", "unknown")).lower()
            notes = " ".join((v.get("validation") or {}).get("notes", []))
            m = _DROPPED_RE.search(notes)
            if m:
                trace.invalid_evidence_refs += int(m.group(1))
            if _SCOPE_NOTE_RE.search(notes):
                trace.wrong_scope_refs += 1
            if bstatus in ("true", "false") and astatus == "unknown":
                trace.validator_downgrades += 1
                if "grounded" in notes or "grounding" in notes or "not FALSE" in notes:
                    trace.grounding_downgrades += 1

        for cid, v in out.items():
            if cid in required_ids and cid in judgeable_ids:
                st = str(v.get("status", "unknown")).lower()
                trace.required_semantic_total += 1
                if st == "true":
                    trace.required_semantic_true += 1
                elif st == "false":
                    trace.required_semantic_false += 1
                else:
                    trace.required_semantic_unknown += 1
        return out

    def wrapped_validate_audit(raw, packet, parsed, facts, ctx, *, first_pass_qualification,
                               first_pass_uncertain=None):
        v = real_validate_audit(raw, packet, parsed, facts, ctx,
                                first_pass_qualification=first_pass_qualification,
                                first_pass_uncertain=first_pass_uncertain)
        final_q = v.get("applied_qualification")
        first_q = v.get("first_pass_qualification", first_pass_qualification)
        notes = " ".join((v.get("validation") or {}).get("notes", []))
        m = _DROPPED_RE.search(notes)
        if m:
            trace.invalid_evidence_refs += int(m.group(1))
        audit_sink.append({
            "person_id": v.get("person_id") or raw.get("person_id"),
            "first_pass_qualification": first_q,
            "final_qualification": final_q,
            "audit_decision": v.get("decision"),
            "reason": (v.get("reason") or "")[:300],
            "issues": v.get("audit_issues", []),
            "failed_required": v.get("failed_required", []),
            "missing_required_reviews": v.get("missing_required_reviews", 0),
            "bucket": _transition_bucket(first_q or "", final_q or ""),
        })
        return v

    ss.run_judge = wrapped_run_judge
    ss.validate_person = wrapped_validate
    fav.validate_audit = wrapped_validate_audit
    try:
        yield
    finally:
        ss.run_judge = real_run_judge
        ss.validate_person = real_validate
        fav.validate_audit = real_validate_audit


def _judgeable(parsed):
    from app.services.semantic_judge import judgeable_criteria
    return judgeable_criteria(parsed)


@contextlib.contextmanager
def _offline():
    """Hard-guarantee zero network LLM calls (three independent locks)."""
    from app.config import settings
    from app.services.llm import providers as _providers
    from app.services.llm import router as _router

    def _empty_chain():
        return []

    def _exhausted(*a, **k):
        meta = {"operation": k.get("operation"), "selected_provider": None,
               "selected_model": None, "attempts": [{"provider": None, "status": "offline"}]}
        return (None, meta) if k.get("return_meta") else None

    saved_chain = _providers.default_chain
    saved_router_chain = _router.default_chain
    saved_key = settings.anthropic_api_key
    saved_audit = settings.final_result_audit_enabled
    _providers.default_chain = _empty_chain
    _router.default_chain = _empty_chain
    settings.anthropic_api_key = ""
    settings.final_result_audit_enabled = False

    patched: list[tuple] = []
    for modname in (
        "app.services.llm.router", "app.services.query_interpreter",
        "app.services.semantic_judge", "app.services.semantic_llm",
        "app.services.reason_generator", "app.services.final_auditor",
    ):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, "generate_structured"):
            patched.append((mod, mod.generate_structured))
            mod.generate_structured = _exhausted
    try:
        yield
    finally:
        _providers.default_chain = saved_chain
        _router.default_chain = saved_router_chain
        settings.anthropic_api_key = saved_key
        settings.final_result_audit_enabled = saved_audit
        for mod, fn in patched:
            mod.generate_structured = fn


@contextlib.contextmanager
def _reasons(enabled: bool):
    from app.config import settings

    if enabled:
        yield
        return
    saved = settings.llm_reason_generation
    settings.llm_reason_generation = False
    try:
        yield
    finally:
        settings.llm_reason_generation = saved


def _funnel_from_judge(md: dict | None) -> dict:
    md = md or {}
    return {k: md.get(k) for k in (
        "network_size", "candidate_pool_size", "hard_fact_rejected_count",
        "viable_candidate_count", "judge_candidate_count", "judge_batch_count",
    )}


def _result_row(item: dict) -> dict:
    return {
        "rank": item["rank"],
        "person_id": item["person_id"],
        "name": item.get("name"),
        "qualification": item.get("qualification"),
        "match_score": item.get("match_score"),
        "matched_criteria": item.get("matched_criteria", []),
        "uncertain_criteria": item.get("uncertain_criteria", []),
        "unmet_criteria": item.get("unmet_criteria", []),
        "audit_decision": item.get("audit_decision"),
        "audit_reason": item.get("audit_reason"),
        "audit_issues": item.get("audit_issues", []),
        "llm_verified": item.get("llm_verified", False),
        "evidence_refs": [
            {"type": c["type"], "criterion": c["criterion"], "match_strength": c["match_strength"],
             "evidence": [e.get("text", "")[:160] for e in c.get("evidence", [])]}
            for c in item.get("score_breakdown", [])
        ],
    }


def _final_uncertainty_rate(interp: dict, results: list[dict]) -> float | None:
    req_semantic = [
        c for c in interp.get("criteria", [])
        if c.get("required") and c.get("type") in JUDGEABLE_TYPES
    ]
    if not req_semantic or not results:
        return None
    total = len(req_semantic) * len(results)
    unknown = sum(len(r.get("uncertain_criteria", [])) for r in results)
    return round(min(1.0, unknown / total), 4) if total else None


def run_query(db: Session, qdef: dict, *, offline: bool = True, reasons_enabled: bool = True) -> QueryRecord:
    from app.models import Dataset
    from app.services.search_service import run_connection_search

    dataset_id = db.query(Dataset.id).scalar()

    trace = JudgeTrace()
    audit_transitions: list[dict] = []

    with contextlib.ExitStack() as es:
        es.enter_context(_instrument(trace, audit_transitions))
        es.enter_context(_reasons(reasons_enabled))
        if offline:
            es.enter_context(_offline())
        resp = run_connection_search(db, dataset_id=dataset_id, query=qdef["query"])
        db.rollback()  # never persist eval searches into the pilot DB

    r = resp.model_dump()
    interp = r["interpreted_query"]
    results = [_result_row(x) for x in r["connections"]["results"]]
    near = [_result_row(x) for x in r["connections"]["near_matches"]]

    tally = {b: 0 for b in _TRANSITION_BUCKETS}
    for t in audit_transitions:
        tally[t["bucket"]] = tally.get(t["bucket"], 0) + 1
    audit_changes = [
        {"person_id": t["person_id"], "transition": t["bucket"], "decision": t["audit_decision"],
         "reason": t["reason"], "issues": t["issues"]}
        for t in audit_transitions
        if t["bucket"] not in ("exact_to_exact", "possible_to_possible")
    ]

    fur = _final_uncertainty_rate(interp, results)
    return QueryRecord(
        query_id=qdef["id"],
        query=qdef["query"],
        group=qdef["group"],
        interpretation={
            "intent": interp.get("intent"),
            "context": interp.get("context", {}),
            "target_person_context": interp.get("target_person_context", {}),
            "unresolved": interp.get("unresolved", []),
            "interpretation_summary": interp.get("interpretation_summary", ""),
            "interpretation_confidence": interp.get("interpretation_confidence"),
            "criteria": [
                {"id": c["id"], "type": c["type"], "value": c.get("value"),
                 "values": c.get("values", []), "concept": c.get("concept"),
                 "operator": c.get("operator"), "scope": c.get("scope"),
                 "required": c.get("required"), "modality": c.get("modality"),
                 "weight": c.get("weight")}
                for c in interp.get("criteria", [])
            ],
        },
        funnel={
            **_funnel_from_judge(r.get("judge_metadata")),
            "returned": r["connections"]["returned"],
            "exact": r["connections"].get("exact_match_count", 0),
            "possible": r["connections"].get("possible_match_count", 0),
            "near": len(near),
            "audit_pool": (r.get("audit_metadata") or {}).get("audited_candidates"),
        },
        results=results,
        near_matches=near,
        judge_metadata=r.get("judge_metadata"),
        audit_metadata=r.get("audit_metadata"),
        judge_trace=trace.as_dict(),
        audit_transitions=audit_transitions,
        audit_transition_tally=tally,
        audit_changes=audit_changes,
        final_uncertainty_rate=fur,
        unknown_required_rate=fur,
        reason_generation_enabled=reasons_enabled,
        llm_provider=r.get("llm_provider"),
        llm_model=r.get("llm_model"),
    )
