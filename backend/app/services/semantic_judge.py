"""Exhaustive batched LLM semantic judge (V4 PART 3 §8–§10, §19, §27–§31).

In ``all_viable`` mode EVERY candidate with genuine unresolved REQUIRED
semantic uncertainty is judged — no artificial cap, no reduction in recall.
"Genuine unresolved uncertainty" is a testable decision (``needs_semantic_
judge`` / ``candidate_needs_judge``, hardening PART 4/5): a candidate whose
required judgeable criteria are ALL already TRUE/FALSE from stored facts,
cached company classification, or stored ProfileSemantic v3 data never reaches
the judge at all — nor does one already sealed NOT_MATCH by a different
required criterion, since resolving this one couldn't change that outcome.
This is what keeps a ~1,000-person broad query from generating ~100 judge
requests: most candidates are already decided before any query-time LLM call
is even considered.

Candidates that DO need judging are sent in BATCHES (``semantic_judge_batch_
size``) through the central LLM router, with adaptive splitting on truncation
(``app.services.llm.adaptive_batch`` — shared with the final auditor): a batch
whose response hits ``max_tokens`` is cut in half and retried as two smaller
requests rather than being retried identically. The model judges professional
MEANING against a compact, evidence-referenced packet; it may not invent
employers, roles, dates, skills, education or references.

Every verdict is validated downstream (``judge_validator``) before it is
allowed to change a score. A missing person / criterion becomes UNKNOWN, never
an assumed TRUE/FALSE. A partial batch failure (or an unresolved truncation)
keeps the verdicts already obtained from every other batch.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.constants import (
    CODE_AUTHORITATIVE_CRITERION_TYPES,
    JUDGEABLE_CRITERION_TYPES,
    JudgeMode,
    JudgeStatus,
    Qualification,
    TriState,
)
from app.schemas import CompactJudgeBatch, ParsedSearchQuery
from app.services.judge_packet import build_packets, plan_payload
from app.services.llm.adaptive_batch import run_adaptive
from app.services.llm.router import generate_structured
from app.services.llm.token_estimate import estimate_judge_output_tokens, plan_batch_size

log = logging.getLogger("app.judge")

_SYSTEM = (
    "You are a professional-network analyst. For each PERSON you decide whether they satisfy "
    "each listed CRITERION, using ONLY the evidence packet given for that person.\n\n"
    "FACTS ARE LOCKED. The backend already verified employers, job titles, dates, education "
    "rows, locations and company classifications. You judge MEANING and INTERPRETATION on top "
    "of those facts. You may NOT contradict a fact, and you may NOT invent an employer, role, "
    "date, skill, degree, publication or evidence reference that is not in the packet.\n\n"
    "MISSING EVIDENCE IS NOT FALSE. If the packet does not contain enough to decide, return "
    "\"unknown\". A weak or indirect implication is also \"unknown\", not \"true\". Use "
    "\"false\" only when the evidence CLEARLY contradicts the criterion.\n\n"
    "Ground every verdict: put the packet references that support it in "
    "supporting_refs — valid ref forms are \"exp:<id>\", \"edu:<id>\", \"cert:<id>\", "
    "\"skill:<name>\", \"assertion:<n>\", \"company:<key>\", \"pub:<id>\" (a publication), "
    "\"vol:<id>\" (a volunteering role), \"rec:<id>\" (a recommendation received). Put anything "
    "that argues AGAINST the criterion in contradicting_refs. A \"true\" verdict with no "
    "supporting reference is rejected. A \"false\" verdict needs a contradicting reference OR an "
    "explicit statement that the full history was reviewed and clearly lacks it — otherwise "
    "return \"unknown\".\n\n"
    "SCOPE MATTERS. If a criterion is scoped 'current', it can only be true from a role marked "
    "is_current=true. 'past' needs a non-current role.\n\n"
    "COMMON FALSE POSITIVES — do NOT mark true for these:\n"
    "  - \"used AWS\" / \"built on AWS\"  != employed by Amazon\n"
    "  - \"sold to CXOs\" / \"advises executives\"  != is a CXO\n"
    "  - \"studied at MIT\" / a degree from a university  != a faculty / professor appointment\n"
    "  - \"participated in a mentorship program\" / \"was mentored\"  != is a mentor\n"
    "  - \"technology transformation\" / \"digital transformation\"  != employment in the tech industry\n"
    "  - employer is a healthcare company  != personal HIPAA / compliance expertise\n"
    "  - \"published a paper\" / a single publication  != a professor / career researcher\n"
    "  - a senior or staff title  != evidence of mentoring or people leadership\n"
    "  - a bootcamp / weekend event / hackathon at a startup  != working at a startup\n\n"
    "BOOLEAN STRUCTURE: when a criterion lists several values with operator ALL_OF, every value "
    "must hold; with ANY_OF, one is enough; with NOT, the value must be ABSENT.\n"
    "MODALITY: a criterion with modality 'possible' is a soft signal — 'true' if the evidence "
    "supports it, but its absence must never count against the person.\n\n"
    "Each person object lists an \"unresolved_criteria\" array — those are the ONLY criterion_ids "
    "you must return a verdict for. The rest of that person's criteria were already decided from "
    "stored facts; do not re-judge them and do not invent extra criteria.\n\n"
    "OUTPUT IS TRANSPORT ONLY, not a report: no prose paragraphs, no per-person summary, no "
    "overall fit. A one-word-scale 'reason' of a few words is enough ONLY if it clarifies a "
    "borderline call — omit it otherwise. Return JSON only, in exactly this shape: "
    '{"people":[{"person_id":"...","criteria":[{"criterion_id":"...","status":"true|false|unknown",'
    '"confidence":0-1,"supporting_refs":["exp:.."],"contradicting_refs":[],"reason":""}]}]} '
    "— one entry per person_id, one verdict per criterion_id in that person's unresolved_criteria."
)


@dataclass
class JudgeMetadata:
    mode: str
    network_size: int = 0
    candidate_pool_size: int = 0
    hard_fact_rejected_count: int = 0
    viable_candidate_count: int = 0
    judge_candidate_count: int = 0
    judge_batch_count: int = 0
    judge_successful_batches: int = 0
    judge_failed_batches: int = 0
    judge_status: str = JudgeStatus.NOT_USED
    capped: bool = False
    cap_limit: int = 0
    omitted_people: int = 0
    omitted_criteria: int = 0
    #: packets too large for a single batch — left unjudged (subset of omitted_people)
    oversized_packets: int = 0
    judgeable_criteria: list[str] = field(default_factory=list)
    providers: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    #: hardening PART 3 — batches that came back truncated (max_tokens) and how
    #: many times a truncated batch was cut in half and retried as two.
    truncations: int = 0
    adaptive_splits: int = 0
    #: hardening PART 4 — candidates already fully decided from stored facts /
    #: semantics before any query-time LLM call was even considered.
    candidates_decided_locally: int = 0
    candidates_needing_llm: int = 0
    #: hardening PART 10 — total (person, criterion) verdicts actually asked
    #: for, vs judge_candidate_count * len(judgeable_criteria) which is what
    #: would have been sent before per-criterion filtering.
    judgeable_criteria_sent: int = 0
    #: hardening PART 14 — the search deadline was reached mid-run; some
    #: batches were never attempted (their candidates fall back to UNKNOWN).
    deadline_reached: bool = False

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "status": self.judge_status,
            "network_size": self.network_size,
            "candidate_pool_size": self.candidate_pool_size,
            "hard_fact_rejected_count": self.hard_fact_rejected_count,
            "viable_candidate_count": self.viable_candidate_count,
            "judge_candidate_count": self.judge_candidate_count,
            "judge_batch_count": self.judge_batch_count,
            "judge_successful_batches": self.judge_successful_batches,
            "judge_failed_batches": self.judge_failed_batches,
            "capped": self.capped,
            "cap_limit": self.cap_limit,
            "omitted_people": self.omitted_people,
            "omitted_criteria": self.omitted_criteria,
            "oversized_packets": self.oversized_packets,
            "judgeable_criteria": self.judgeable_criteria,
            "providers": self.providers,
            "models": self.models,
            "prompt_chars": self.prompt_chars,
            "truncations": self.truncations,
            "adaptive_splits": self.adaptive_splits,
            "candidates_decided_locally": self.candidates_decided_locally,
            "candidates_needing_llm": self.candidates_needing_llm,
            "judgeable_criteria_sent": self.judgeable_criteria_sent,
            "deadline_reached": self.deadline_reached,
        }


@dataclass
class JudgeRun:
    verdicts: dict[str, dict[str, dict]]          # person_id -> criterion_id -> raw verdict dict
    packets_by_id: dict[str, dict]
    metadata: JudgeMetadata


def judgeable_criteria(parsed: ParsedSearchQuery) -> list:
    return [
        c for c in parsed.criteria
        if c.type in JUDGEABLE_CRITERION_TYPES and c.type not in CODE_AUTHORITATIVE_CRITERION_TYPES
    ]


def needs_semantic_judge(prescored, crit) -> bool:
    """True when resolving THIS criterion for THIS candidate could actually
    change whether they end up Exact / Possible / Top-N (hardening PART 4/5 —
    the staged decision function the mission asked to make testable).

    Skip judging when:
      * the criterion isn't required — a semantic UNKNOWN on a non-required
        criterion never changes qualification, only a soft score contribution
      * the criterion isn't judgeable, or is code-authoritative (career
        transition / years experience are never judge-overridden)
      * the deterministic prescore for THIS criterion already resolved it to
        TRUE or FALSE — a validated judge verdict cannot do better than a
        grounded fact/company-classification/semantic-assertion already did
      * the candidate is ALREADY sealed NOT_MATCH by a DIFFERENT required
        criterion — this one's own outcome cannot change that result, so
        spending a call on it buys nothing (mission: "candidate could not
        affect the displayed result")
    """
    if not crit.required:
        return False
    if crit.type not in JUDGEABLE_CRITERION_TYPES or crit.type in CODE_AUTHORITATIVE_CRITERION_TYPES:
        return False
    if prescored is None:
        return True  # no prior signal — safest is to judge, never invented
    if prescored.status_by_criterion.get(crit.id) != TriState.UNKNOWN:
        return False
    return prescored.qualification != Qualification.NOT_MATCH


def candidate_needs_judge(prescored, jcrits: list) -> bool:
    return any(needs_semantic_judge(prescored, c) for c in jcrits)


def run_judge(
    query: str,
    parsed: ParsedSearchQuery,
    bundle: list[tuple],
    ctx,
    *,
    network_size: int,
    pool_size: int,
    hard_rejected_count: int,
    local_scored: dict | None = None,
    deadline=None,
) -> JudgeRun:
    """``bundle``: ``[(person, ProfileFacts, {"volunteering":[...], "recommendations":[...]})]``
    — the candidates that passed the hard-fact gate."""
    mode = settings.semantic_judge_mode
    jcrits = judgeable_criteria(parsed)
    meta = JudgeMetadata(
        mode=mode, network_size=network_size, candidate_pool_size=pool_size,
        hard_fact_rejected_count=hard_rejected_count, viable_candidate_count=len(bundle),
        judgeable_criteria=[c.id for c in jcrits],
    )

    if not settings.semantic_judge_enabled or mode == JudgeMode.OFF or not jcrits or not bundle:
        meta.judge_status = JudgeStatus.NOT_USED
        return JudgeRun({}, {}, meta)

    # ── STAGE E/F (hardening PART 4) — drop candidates already fully decided
    #    by stored facts / semantics. Applies in EVERY mode: "all_viable" means
    #    every candidate with genuine unresolved required uncertainty, never
    #    "everyone regardless of whether the judge could change anything." ──
    total_before_local = len(bundle)
    if local_scored is not None:
        bundle = [(p, f, x) for (p, f, x) in bundle
                 if candidate_needs_judge(local_scored.get(p.id), jcrits)]
    meta.candidates_decided_locally = total_before_local - len(bundle)
    meta.candidates_needing_llm = len(bundle)
    meta.viable_candidate_count = len(bundle)
    if not bundle:
        meta.judge_status = JudgeStatus.NOT_USED
        return JudgeRun({}, {}, meta)

    if mode == JudgeMode.UNCERTAIN_ONLY:
        bundle = _uncertain_shortlist(bundle, jcrits, local_scored)
        meta.viable_candidate_count = len(bundle)
        if not bundle:
            meta.judge_status = JudgeStatus.NOT_USED
            return JudgeRun({}, {}, meta)

    cap = settings.semantic_judge_max_candidates
    if cap and cap > 0 and len(bundle) > cap:
        meta.capped = True
        meta.cap_limit = cap
        bundle = _cap_prioritise(bundle, local_scored)[:cap]

    # ── PART 10 — PER-CRITERION unresolved list, not "all jcrits for everyone".
    #    A person with 3/4 criteria already TRUE/FALSE locally is only asked
    #    about the 1 that's still unknown — this is what actually shrinks the
    #    output, on top of who even reaches the judge. ──────────────────────
    unresolved_by_person: dict[str, list[str]] = {
        p.id: [c.id for c in jcrits if needs_semantic_judge(local_scored.get(p.id) if local_scored else None, c)]
        for p, _f, _x in bundle
    }
    meta.judgeable_criteria_sent = sum(len(v) for v in unresolved_by_person.values())

    # ── hardening PART 15 — spend budget/deadline-limited calls on candidates
    #    NEAR THE QUALIFICATION BOUNDARY first: fewer remaining unresolved
    #    required criteria (one verdict away from a decided outcome) and a
    #    higher local match score are judged before a candidate that is still
    #    ambiguous on everything. If a budget cap or the search deadline cuts
    #    the run short, the highest-value candidates were already resolved. ──
    bundle = sorted(
        bundle,
        key=lambda t: (len(unresolved_by_person.get(t[0].id, [])),
                       -(local_scored[t[0].id].match_score if local_scored and t[0].id in local_scored else 0.0)),
    )

    packets = build_packets(bundle, parsed, ctx, query=query, unresolved_by_person=unresolved_by_person)
    packets_by_id = {pkt["person_id"]: pkt for pkt in packets}
    meta.judge_candidate_count = len(packets)

    payload = plan_payload(query, parsed, jcrits)

    # ── PART 4 — proactively size batches from criteria DENSITY instead of
    #    discovering "too big" only after a truncation. ─────────────────────
    counts = [len(unresolved_by_person.get(pkt["person_id"], [])) or 1 for pkt in packets]
    avg_criteria = (sum(counts) / len(counts)) if counts else 1.0
    planned_size = plan_batch_size(settings.semantic_judge_batch_size, avg_criteria)
    batches, oversized = _make_batches(
        packets,
        size=planned_size,
        max_chars=settings.semantic_judge_max_batch_chars,
    )
    meta.oversized_packets = len(oversized)
    if oversized:
        log.warning("judge: %d packet(s) too large for a batch — left unjudged: %s",
                    len(oversized), oversized[:10])

    verdicts: dict[str, dict[str, dict]] = {}
    for batch in batches:
        if deadline is not None and deadline.expired():
            meta.deadline_reached = True
            log.warning("judge: search deadline reached — %d/%d batches skipped, "
                        "their candidates fall back to UNKNOWN", len(batches) - meta.judge_batch_count, len(batches))
            break
        leaves, stats = run_adaptive(batch, lambda pkts: _call_judge(payload, pkts, unresolved_by_person))
        meta.judge_batch_count += stats.batches_attempted
        meta.judge_successful_batches += stats.successful_batches
        meta.judge_failed_batches += stats.failed_batches
        meta.truncations += stats.truncations
        meta.adaptive_splits += stats.adaptive_splits
        for prov, n in stats.providers.items():
            meta.providers[prov] = meta.providers.get(prov, 0) + n
        for m in stats.models:
            if m not in meta.models:
                meta.models.append(m)
        for leaf in leaves:
            if leaf.outcome != "ok":
                continue
            for person_id, crit_verdicts in leaf.payload.items():
                verdicts[person_id] = crit_verdicts

    _fill_missing(verdicts, packets_by_id, jcrits, meta)

    if meta.deadline_reached:
        meta.judge_status = JudgeStatus.PARTIAL
    elif meta.judge_successful_batches == 0:
        meta.judge_status = JudgeStatus.UNAVAILABLE
    elif meta.judge_failed_batches or meta.omitted_people or meta.omitted_criteria:
        meta.judge_status = JudgeStatus.PARTIAL
    else:
        meta.judge_status = JudgeStatus.FULL

    log.info(
        "judge %s: hard_gate_viable=%d locally_resolved=%d judge_candidates=%d "
        "unresolved_criteria=%d batches=%d/%d ok truncations=%d splits=%d status=%s",
        mode, total_before_local, meta.candidates_decided_locally, meta.judge_candidate_count,
        meta.judgeable_criteria_sent, meta.judge_successful_batches, meta.judge_batch_count,
        meta.truncations, meta.adaptive_splits, meta.judge_status,
    )
    return JudgeRun(verdicts, packets_by_id, meta)


# ─────────────────────── batching ───────────────────────


def _pkt_chars(pkt: dict) -> int:
    return len(json.dumps(pkt, ensure_ascii=False, default=str))


def _make_batches(
    packets: list[dict], *, size: int, max_chars: int,
) -> tuple[list[list[dict]], list[str]]:
    """Pack packets into ``<= size`` / ``<= max_chars`` batches. A single packet
    that itself exceeds ``max_chars`` (or was flagged ``_packet_too_large``) is
    NEVER placed in a batch — it would create an oversized provider request. Its
    person_id is returned as ``oversized`` and the candidate is left
    unjudged / UNKNOWN, status PARTIAL (V4 PART 3.6 §8 / PART 5 §40). Shared by
    the exhaustive judge and the final auditor."""
    size = max(1, size)
    max_chars = max(2000, max_chars)
    batches: list[list[dict]] = []
    oversized: list[str] = []
    cur: list[dict] = []
    cur_chars = 0
    for pkt in packets:
        pc = _pkt_chars(pkt)
        if pkt.get("_packet_too_large") or pc > max_chars:
            oversized.append(pkt.get("person_id"))
            continue
        if cur and (len(cur) >= size or cur_chars + pc > max_chars):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(pkt)
        cur_chars += pc
    if cur:
        batches.append(cur)
    return batches, oversized


def _expand_compact(cv, valid_ids: set[str]) -> dict:
    """Compact verdict -> the existing internal shape every downstream
    validator/scorer already expects (hardening PART 1). ``match_strength``
    is DERIVED — never asked of the model: TRUE -> confidence, FALSE -> 0,
    UNKNOWN -> 0. ``experience_ids`` derived from ``exp:`` refs."""
    status = cv.status if cv.status in ("true", "false") else TriState.UNKNOWN
    strength = cv.confidence if status == TriState.TRUE else 0.0
    sup = [r for r in cv.supporting_refs if r in valid_ids] if valid_ids else cv.supporting_refs
    con = [r for r in cv.contradicting_refs if r in valid_ids] if valid_ids else cv.contradicting_refs
    exp_ids = [r.split("exp:", 1)[1] for r in sup if r.startswith("exp:")]
    return {
        "criterion_id": cv.criterion_id, "status": status,
        "match_strength": round(float(strength), 3), "confidence": cv.confidence,
        "reason": cv.reason, "supporting_evidence_refs": sup, "contradicting_evidence_refs": con,
        "experience_ids": exp_ids,
    }


def _call_judge(
    payload: dict, packets: list[dict], unresolved_by_person: dict[str, list[str]] | None = None,
    *, _retry: bool = False,
) -> tuple[str, object, str | None, str | None]:
    """One batched judge request through the router — the ``CallFn`` the
    adaptive splitter drives. Returns ``("ok", {person_id: {criterion_id:
    verdict_dict}}, provider, model)``, ``("truncated", None, None, None)``
    (caller should split and retry the halves), or ``("failed", None, None,
    None)`` (every provider genuinely exhausted / non-truncation error)."""
    counts = [len(pkt.get("unresolved_criteria") or []) or 1 for pkt in packets]
    max_tokens = estimate_judge_output_tokens(len(packets), counts)
    if _retry:
        from app.services.llm.token_estimate import MAX_OUTPUT_TOKENS
        max_tokens = min(MAX_OUTPUT_TOKENS, max_tokens * 2)

    user = (
        "SEARCH PLAN (how to judge — never a phrase to match):\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
        + "\n\nPEOPLE (one evidence packet each — judge ONLY each person's own unresolved_criteria):\n"
        + json.dumps(packets, ensure_ascii=False, default=str)
    )
    result = generate_structured(
        _SYSTEM, user, CompactJudgeBatch,
        max_tokens=max_tokens, operation="semantic_judge", return_meta=True,
    )
    if result[0] is not None:
        batch, provider, model_id, _meta = result
        expanded: dict[str, dict[str, dict]] = {}
        for pv in batch.people:
            allowed = set((unresolved_by_person or {}).get(pv.person_id, [])) or None
            expanded[pv.person_id] = {
                cv.criterion_id: _expand_compact(cv, set())
                for cv in pv.criteria
                if allowed is None or cv.criterion_id in allowed  # never accept an unasked criterion
            }
        return "ok", expanded, provider, model_id

    _, meta = result
    truncated = any(a.get("status") in ("output_truncated", "request_too_large")
                    for a in meta.get("attempts", []))
    if truncated and len(packets) == 1 and not _retry:
        # PART 7 — last line of defense: a single person still truncating at
        # the estimated budget gets ONE retry with double the room before
        # giving up (nothing left to split). Bounded — never a second retry.
        return _call_judge(payload, packets, unresolved_by_person, _retry=True)
    if truncated:
        return "truncated", None, None, None
    return "failed", None, None, None


# ─────────────────────── completeness (§30) ───────────────────────


def _fill_missing(verdicts: dict, packets_by_id: dict, jcrits: list, meta: JudgeMetadata) -> None:
    """Fill UNKNOWN ONLY for criteria that were actually ASKED for a packet
    that was actually attempted (its own ``unresolved_criteria`` stamp, falling
    back to every judgeable criterion for a packet without one — e.g. a test
    double) — a criterion the person was never asked about (already resolved
    locally) is correctly absent from ``verdicts`` and must NOT be flagged
    omitted (hardening PART 10). Driven by ``packets_by_id`` (the packets that
    actually survived oversized-packet filtering), not the pre-filter bundle."""
    judged = set(verdicts)
    all_jcrit_ids = [c.id for c in jcrits]
    for pid, pkt in packets_by_id.items():
        crit_ids = pkt.get("unresolved_criteria")
        if crit_ids is None:
            crit_ids = all_jcrit_ids
        pcrit = verdicts.setdefault(pid, {})
        if pid not in judged and crit_ids:
            meta.omitted_people += 1
        for cid in crit_ids:
            if cid not in pcrit:
                if pid in judged:
                    meta.omitted_criteria += 1
                pcrit[cid] = {
                    "criterion_id": cid, "status": TriState.UNKNOWN,
                    "match_strength": 0.0, "confidence": 0.0, "reason": "",
                    "supporting_evidence_refs": [], "contradicting_evidence_refs": [],
                    "experience_ids": [], "judge_missing": True,
                }


# ─────────────────────── uncertain_only mode ───────────────────────


def _uncertain_shortlist(bundle: list[tuple], jcrits: list, local_scored: dict | None) -> list[tuple]:
    """The cheaper opt-in mode's pool cap. ``bundle`` has ALREADY been through
    the same ``candidate_needs_judge`` staged filter as ``all_viable`` (V4
    hardening PART 4/5) — this only trims that already-ambiguous set further
    when it's larger than ``semantic_judge_pool``, prioritising by prescore.
    (Previously used a raw match_strength ∈ [low,high] band, which silently
    missed a required criterion sitting at strength 0.0 — a real UNKNOWN is
    not always a "medium" number. The tri-state decision function is exact.)"""
    if not local_scored or len(bundle) <= settings.semantic_judge_pool:
        return bundle[: settings.semantic_judge_pool] if not local_scored else bundle
    return _cap_prioritise(bundle, local_scored)[: settings.semantic_judge_pool]


def _cap_prioritise(bundle: list[tuple], local_scored: dict | None) -> list[tuple]:
    if not local_scored:
        return bundle
    return sorted(
        bundle,
        key=lambda t: local_scored[t[0].id].match_score if t[0].id in local_scored else 0.0,
        reverse=True,
    )
