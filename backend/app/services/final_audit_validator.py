"""Fact-consistency validator for the final result audit (V4 PART 5 §12, PART 5.5).

The final audit is LLM output, so it is validated too — with the SAME evidence-
ref + grounding rules the exhaustive judge uses. The validator:

  * drops criterion reviews for unknown criterion_ids and drops invented
    evidence references
  * SYNTHESIZES a review for every REQUIRED criterion the model omitted
    (status_review="uncertain", audit_missing=true) — an omitted required review
    is never silently "supported" (§1)
  * validates each REQUIRED review's grounding:
      - "unsupported" is treated exactly like a proposed semantic FALSE — it must
        pass ``judge_validator.validate_negative_grounding`` (scope, career-wide
        coverage, deterministic-TRUE authority, backend completeness). Ungrounded
        -> the review becomes "uncertain" and cannot remove the candidate (§5–§9)
      - "supported" must have a scope-valid supporting ref OR an independent
        deterministic / code-authoritative TRUE, else it becomes "uncertain"
        (§10/§11)
  * runs a DETERMINISTIC-ONLY rescore so verified facts, chronology and cached
    company classifications stay authoritative (§10/§14/§29)
  * computes the APPLIED qualification and NEVER produces EXACT from a non-EXACT
    first pass (§9). UNKNOWN over a first-pass EXACT -> POSSIBLE unless the facts
    alone already prove every required criterion (§8)
  * ``APPROVED`` requires COMPLETE, grounded required-criterion coverage (§2);
    otherwise it is downgraded to UNKNOWN
  * ``llm_verified`` is True ONLY on a fully-covered, fully-grounded APPROVED (§3)

Output shape (per person):
  {person_id, decision, confidence, reason, criteria[], missing_required_reviews,
   supporting_evidence_refs, contradicting_evidence_refs, applied_qualification,
   failed_required[], audit_issues[], llm_verified, first_pass_qualification,
   validation{...}}
"""
from __future__ import annotations

from app.constants import AuditDecision, Qualification, TriState
from app.schemas import ParsedSearchQuery
from app.services.judge_packet import packet_experience_current_map, packet_refs
from app.services.judge_validator import _scope_ok_for, validate_negative_grounding
from app.services.scoring import ProfileFacts, ScoringContext, _label, score_candidate

_MISSING_REVIEW_REASON = "required criterion was not reviewed by the final auditor"


def validate_audit(
    raw: dict,
    packet: dict,
    parsed: ParsedSearchQuery,
    facts: ProfileFacts,
    ctx: ScoringContext,
    *,
    first_pass_qualification: str,
    first_pass_uncertain: list[str] | None = None,
) -> dict:
    crits = {c.id: c for c in parsed.criteria}
    required_ids = [c.id for c in parsed.criteria if c.required]
    valid_refs = packet_refs(packet)
    notes: list[str] = []
    decision = raw.get("decision", AuditDecision.UNKNOWN)
    conf = float(raw.get("confidence", 0.5) or 0.5)
    audit_missing = bool(raw.get("audit_missing"))
    pid = raw.get("person_id")
    jr = ctx.judge_results.get(pid, {}) if pid else {}

    if audit_missing:
        decision = AuditDecision.UNKNOWN
        notes.append("candidate not audited (batch failure / oversized packet) — treated as UNKNOWN")

    # ── clean reviews + evidence refs; track drops per review ────────
    reviews: list[dict] = []
    reviews_by_cid: dict[str, dict] = {}
    dropped_refs = 0
    for r in raw.get("criteria", []):
        cid = r.get("criterion_id")
        if cid not in crits:
            continue
        raw_sup = r.get("supporting_evidence_refs") or []
        raw_con = r.get("contradicting_evidence_refs") or []
        sup = [x for x in raw_sup if x in valid_refs]
        con = [x for x in raw_con if x in valid_refs]
        d = (len(raw_sup) - len(sup)) + (len(raw_con) - len(con))
        dropped_refs += d
        rv = {
            "criterion_id": cid,
            "status_review": r.get("status_review", "uncertain"),
            "reason": (r.get("reason") or "")[:200],
            "supporting_evidence_refs": sup,
            "contradicting_evidence_refs": con,
            "dropped_refs": d,
        }
        reviews.append(rv)
        reviews_by_cid[cid] = rv

    # ── §1 — synthesize a review for every omitted REQUIRED criterion ──
    missing_required_reviews = 0
    for cid in required_ids:
        if cid not in reviews_by_cid:
            missing_required_reviews += 1
            rv = {
                "criterion_id": cid, "status_review": "uncertain",
                "reason": _MISSING_REVIEW_REASON, "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [], "dropped_refs": 0, "audit_missing": True,
            }
            reviews.append(rv)
            reviews_by_cid[cid] = rv
    if missing_required_reviews:
        notes.append(f"{missing_required_reviews} required criterion review(s) omitted by the auditor")

    top_sup = [x for x in (raw.get("supporting_evidence_refs") or []) if x in valid_refs]
    top_con = [x for x in (raw.get("contradicting_evidence_refs") or []) if x in valid_refs]
    dropped_refs += (len(raw.get("supporting_evidence_refs") or []) - len(top_sup)) + \
                    (len(raw.get("contradicting_evidence_refs") or []) - len(top_con))
    if dropped_refs:
        notes.append(f"dropped {dropped_refs} invalid evidence ref(s)")

    # ── deterministic-only rescore — facts stay authoritative (§10/§14) ──
    det = score_candidate(facts, parsed, _fresh_ctx(ctx))
    det_unmet = set(det.unmet_required or [])

    # ── §5–§11 — ground every REQUIRED review; convert ungrounded ones ──
    grounded_fail: list[str] = []
    required_supported_ok = True
    required_evidence_lost = False
    det_uncertain = set(det.uncertain_required or [])
    cur_map = packet_experience_current_map(packet)
    for cid in required_ids:
        c = crits[cid]
        rv = reviews_by_cid[cid]
        sr = rv["status_review"]
        label = _label(c)

        if rv.get("audit_missing") or sr == "uncertain":
            required_supported_ok = False
            continue

        if sr == "unsupported":
            grounded, gnote = validate_negative_grounding(
                c, facts, ctx, packet, rv["contradicting_evidence_refs"], rv["reason"],
            )
            if grounded or label in det_unmet:
                grounded_fail.append(label)
            else:
                rv["status_review"] = "uncertain"
                rv["reason"] = (rv["reason"] + f" [downgraded: {gnote}]")[:240]
                notes.append(f"'{label}' unsupported review is not grounded — {gnote}")
            required_supported_ok = False
            continue

        # sr == "supported" — must be grounded (§10/§11). The deterministic-only
        # rescore already says which required criteria hold; a 'supported' review
        # for a still-uncertain semantic concept needs a scope-valid ref or a
        # first-pass judge TRUE.
        if rv["dropped_refs"]:
            required_evidence_lost = True
        sup = rv["supporting_evidence_refs"]
        grounded_true = False
        if label in det_unmet:
            grounded_true = False
        elif label not in det_uncertain:
            grounded_true = True  # deterministically / code-authoritatively met
        elif jr.get(cid, {}).get("status") == TriState.TRUE:
            grounded_true = True
        elif sup and _scope_ok_for(c.scope, sup, packet, cur_map):
            grounded_true = True
        if not grounded_true:
            rv["status_review"] = "uncertain"
            rv["reason"] = (rv["reason"] + " [downgraded: 'supported' with no scope-valid "
                            "evidence ref and no deterministic proof]")[:240]
            notes.append(f"'{label}' supported review has no grounding — treated as uncertain")
            required_supported_ok = False

    # ── §29 — verified facts already disqualify: no APPROVAL possible ──
    if det.qualification == Qualification.NOT_MATCH:
        notes.append("verified facts contradict a required criterion — auditor cannot approve; removed")
        return _out(AuditDecision.INCORRECT, conf, raw, reviews, top_sup, top_con,
                    applied=Qualification.NOT_MATCH,
                    failed=(det.unmet_required or ["a verified fact"]),
                    notes=notes, llm_verified=False,
                    missing_required_reviews=missing_required_reviews,
                    first_pass_qualification=first_pass_qualification)

    # ── decision adjustments (§2/§11) ───────────────────────────────
    # A grounded 'unsupported' required review is a verified required-FALSE —
    # it ALWAYS means NOT_MATCH (§12: "required FALSE -> NOT_MATCH"), regardless
    # of what decision label the model attached. Applying this unconditionally
    # (not just when the model said APPROVED) keeps the badge and the audit
    # decision from ever disagreeing.
    if grounded_fail:
        if decision != AuditDecision.INCORRECT:
            notes.append(f"decision overridden -> INCORRECT: a required criterion review is a "
                         f"grounded 'unsupported' ({', '.join(grounded_fail)})")
        decision = AuditDecision.INCORRECT
    elif decision == AuditDecision.INCORRECT:
        decision = AuditDecision.UNKNOWN
        notes.append("INCORRECT not grounded (no grounded 'unsupported' required review, no "
                     "deterministic contradiction) — downgraded to UNKNOWN")
    if decision == AuditDecision.APPROVED and (missing_required_reviews or not required_supported_ok):
        decision = AuditDecision.UNKNOWN
        notes.append("APPROVED requires a grounded 'supported' review for EVERY required "
                     "criterion — downgraded to UNKNOWN (§2)")

    deterministic_proves_exact = det.qualification == Qualification.EXACT_MATCH
    applied, failed = _apply(decision, first_pass_qualification, deterministic_proves_exact,
                             grounded_fail)

    # ── §3 — llm_verified means FULL, grounded audit coverage ───────
    full_coverage = (
        not audit_missing
        and missing_required_reviews == 0
        and required_supported_ok
        and not required_evidence_lost
        and not grounded_fail
    )
    llm_verified = (
        decision == AuditDecision.APPROVED
        and full_coverage
        and applied == first_pass_qualification
    )

    return _out(decision, conf, raw, reviews, top_sup, top_con, applied=applied, failed=failed,
                notes=notes, llm_verified=llm_verified,
                missing_required_reviews=missing_required_reviews,
                first_pass_qualification=first_pass_qualification)


# ─────────────────────── helpers ───────────────────────


def _fresh_ctx(ctx: ScoringContext) -> ScoringContext:
    return ScoringContext(
        company_class=ctx.company_class,
        company_ids_by_criterion=ctx.company_ids_by_criterion,
    )


def _apply(decision: str, fp: str, det_exact: bool, grounded_fail: list[str]):
    """Allowed transitions (§8/§9). NEVER returns EXACT from a non-EXACT first
    pass — the final auditor cannot manufacture confidence."""
    if decision == AuditDecision.INCORRECT:
        return Qualification.NOT_MATCH, (grounded_fail or ["a required criterion"])
    if decision == AuditDecision.DOWNGRADE:
        return (Qualification.POSSIBLE_MATCH if fp == Qualification.EXACT_MATCH else fp), []
    if decision == AuditDecision.UNKNOWN:
        if fp == Qualification.EXACT_MATCH and not det_exact:
            return Qualification.POSSIBLE_MATCH, []
        return fp, []
    return fp, []  # APPROVED — keep the first-pass qualification unchanged


def _out(decision, conf, raw, reviews, top_sup, top_con, *, applied, failed, notes,
         llm_verified, missing_required_reviews, first_pass_qualification) -> dict:
    unsupported_reasons = [
        r["reason"] for r in reviews if r["status_review"] == "unsupported" and r["reason"]
    ]
    return {
        "person_id": raw.get("person_id"),
        "decision": decision,
        "confidence": round(float(conf), 3),
        "reason": (raw.get("reason") or "")[:400],
        "criteria": [{k: v for k, v in r.items() if k != "dropped_refs"} for r in reviews],
        "missing_required_reviews": missing_required_reviews,
        "supporting_evidence_refs": top_sup,
        "contradicting_evidence_refs": top_con,
        "applied_qualification": applied,
        "failed_required": failed,
        "audit_issues": (notes + unsupported_reasons)[:8],
        "llm_verified": bool(llm_verified),
        "first_pass_qualification": first_pass_qualification,
        "validation": {"notes": notes},
    }
