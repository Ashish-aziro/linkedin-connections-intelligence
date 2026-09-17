"""Near Match LLM judge — batched, evidence-grounded (near-match design PART 8).

Answers a DIFFERENT question than the strict semantic judge: not "does this
criterion hold" but "given that this person already failed the strict plan,
are they still worth recommending, and why?" It never writes into
``ScoringContext.judge_results`` and never touches the strict Exact/Possible
pipeline — a completely separate, smaller pass over an already-bounded pool.

A candidate is only ever added to ``verdicts`` (below) when their batch came
back "ok" — a batch that fails or truncates unrecoverably simply leaves its
candidates ABSENT from ``verdicts``, never stamped with a placeholder/UNKNOWN
entry. That absence is what lets a "partial" run (some batches ok, some
failed) stay safe without any extra filtering downstream: an omitted
person_id can never reach ``near_match_validator`` or ``near_match_ranking``
in the first place (near-match design PART 1, v2 — no LLM verdict for a
candidate means no near match for that candidate, full stop; there is no
deterministic/local fallback anywhere in this pipeline).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import settings
from app.constants import CriterionType, GeoRelation
from app.schemas import NearMatchJudgeBatch, ParsedSearchQuery
from app.services.geo import classify_relation_deterministic
from app.services.judge_packet import build_packets
from app.services.llm.adaptive_batch import run_adaptive
from app.services.llm.concurrency import run_concurrent_map
from app.services.llm.router import generate_structured
from app.services.near_match_pool import NearCandidate

log = logging.getLogger("app.near_match")

_SYSTEM = (
    "You review candidates who ALREADY FAILED a strict professional-network search, and decide "
    "whether each is still worth recommending as a NEAR MATCH — someone who does not fully satisfy "
    "the original query but is professionally or geographically close enough to be worth showing.\n\n"
    "FACTS ARE LOCKED. You may not contradict the evidence packet or invent an employer, role, "
    "date, skill, degree, location fact, or evidence reference not present in it.\n\n"
    "For each PERSON you are told: the query's PRIMARY INTENT (the actual point of the search), "
    "the full original criteria, and WHICH criterion this specific person failed or is uncertain "
    "on (their 'gap'). Decide:\n"
    "  useful_near_match: true only if the person is genuinely relevant to the PRIMARY INTENT "
    "despite the gap — never true just because one keyword appears, and never true for someone "
    "with no real connection to the primary intent.\n"
    "  relation_type: pick ONE generic category that best explains why they're still relevant:\n"
    "    geographic_adjacent   — a different but nearby/metro-adjacent location than requested\n"
    "    role_adjacent         — a closely related professional role/function\n"
    "    industry_adjacent     — closely related industry/domain experience\n"
    "    experience_adjacent   — relevant experience that falls just short of the exact ask\n"
    "    seniority_adjacent    — one level off the requested seniority\n"
    "    company_category_adjacent — a related employer category (e.g. adjacent to startup/big tech)\n"
    "    partial_requirement_match — satisfies most, but not all, of a compound requirement\n"
    "    other_relevant        — relevant for a reason not covered above\n"
    "    not_meaningful         — not actually a useful recommendation (use this + useful_near_match=false)\n"
    "  If the gap is a LOCATION criterion, only use geographic_adjacent when the two places are "
    "genuinely close (same metro area, a well-known adjacent suburb, easy commuting distance) — a "
    "'geo_hint' field may already tell you the relation from structured data; if you are not "
    "confident two places are actually close, do not claim geographic_adjacent — prefer "
    "other_relevant or not_meaningful instead. NEVER invent mileage or claim certainty you don't have.\n\n"
    "Ground every true verdict: evidence_refs must cite packet references (exp:<id>, edu:<id>, "
    "cert:<id>, skill:<name>, assertion:<n>, company:<key>) that actually support satisfied_intent. "
    "A useful_near_match=true with no evidence_refs is rejected downstream — always include at "
    "least one when you say true.\n\n"
    "relaxed_criterion_id must be exactly the criterion id given in that person's 'gap' — never a "
    "different id, never invented.\n\n"
    "short_reason: one concise, user-facing sentence (no internal jargon like 'hard gate' or "
    "'judge') explaining why this person is still worth considering, e.g. 'Strong venture-capital "
    "background; located in a nearby Metro Atlanta suburb rather than Atlanta proper.'\n\n"
    "Return JSON only: {\"people\":[{\"person_id\":\"...\",\"useful_near_match\":true|false,"
    "\"confidence\":0-1,\"satisfied_intent\":\"...\",\"relaxed_criterion_id\":\"...\","
    "\"relation_type\":\"...\",\"evidence_refs\":[\"exp:..\"],\"short_reason\":\"...\"}]} "
    "— one entry per person_id."
)


@dataclass
class NearJudgeMetadata:
    enabled: bool
    candidates_considered: int = 0
    candidates_judged: int = 0
    batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    geo_relations: dict[str, int] = field(default_factory=dict)
    providers: dict[str, int] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    status: str = "not_used"  # not_used | unavailable | partial | full

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "candidates_considered": self.candidates_considered,
            "candidates_judged": self.candidates_judged,
            "batches": self.batches,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "geo_relations": self.geo_relations,
            "providers": self.providers,
            "models": self.models,
        }


@dataclass
class NearJudgeRun:
    verdicts: dict[str, dict]           # person_id -> raw verdict dict (schema-validated)
    packets_by_id: dict[str, dict]
    metadata: NearJudgeMetadata


def _gap_context(nc: NearCandidate, parsed: ParsedSearchQuery) -> dict:
    """The compact 'why is this person in the near pool' hint the judge needs —
    never the full plan re-explained per person."""
    crits_by_id = {c.id: c for c in parsed.criteria}
    gaps = []
    for cid in nc.failed_criterion_ids:
        c = crits_by_id.get(cid)
        if c is None:
            continue
        gap = {
            "criterion_id": cid, "type": c.type,
            "value": c.concept or c.value or " or ".join(c.values),
        }
        if c.type == CriterionType.LOCATION:
            geo_hint = classify_relation_deterministic(
                {"city": getattr(nc.person, "city", None), "state": getattr(nc.person, "state", None),
                 "location_text": getattr(nc.person, "location_text", None)},
                c.values or ([c.value] if c.value else []),
            )
            if geo_hint != GeoRelation.UNKNOWN:
                gap["geo_hint"] = geo_hint
        gaps.append(gap)
    return {"primary_intent": parsed.primary_intent, "source": nc.source, "gaps": gaps}


def _call_near_judge(plan_payload: dict, packets: list[dict]) -> tuple[str, object, str | None, str | None]:
    """One batched near-match judge request — the testable seam (mirrors
    ``semantic_judge._call_judge``): tests monkeypatch THIS function directly
    instead of the LLM router, so no real Anthropic call is ever made."""
    user = (
        "SEARCH CONTEXT:\n" + json.dumps(plan_payload, ensure_ascii=False, default=str)
        + "\n\nCANDIDATES (each already carries its own near_match_context.gaps):\n"
        + json.dumps(packets, ensure_ascii=False, default=str)
    )
    max_tokens = min(4000, 300 + 260 * len(packets))
    result = generate_structured(
        _SYSTEM, user, NearMatchJudgeBatch,
        max_tokens=max_tokens, operation="near_match_judge", return_meta=True,
    )
    if result[0] is None:
        _, rmeta = result
        truncated = any(a.get("status") in ("output_truncated", "request_too_large")
                        for a in rmeta.get("attempts", []))
        return ("truncated" if truncated else "failed"), None, None, None
    batch, provider, model_id, _rmeta = result
    return "ok", {v.person_id: v.model_dump() for v in batch.people}, provider, model_id


def run_near_judge(
    query: str, parsed: ParsedSearchQuery, pool: list[NearCandidate], ctx,
    *, vol_by_id: dict | None = None, rec_by_id: dict | None = None,
) -> NearJudgeRun:
    meta = NearJudgeMetadata(enabled=settings.near_match_llm_enabled, candidates_considered=len(pool))
    if not settings.near_match_llm_enabled or not pool:
        meta.status = "not_used"
        return NearJudgeRun({}, {}, meta)

    bundle = [
        (nc.person, nc.facts, {
            "volunteering": (vol_by_id or {}).get(nc.person.id, []),
            "recommendations": (rec_by_id or {}).get(nc.person.id, []),
        })
        for nc in pool
    ]
    packets = build_packets(bundle, parsed, ctx, query=query, full_profile=True,
                            max_packet_chars=settings.near_match_max_packet_chars)
    packets_by_id = {pkt["person_id"]: pkt for pkt in packets}
    by_pid = {nc.person.id: nc for nc in pool}
    for pkt in packets:
        nc = by_pid.get(pkt["person_id"])
        if nc is not None:
            pkt["near_match_context"] = _gap_context(nc, parsed)
            geo_hints = [g["geo_hint"] for g in pkt["near_match_context"]["gaps"] if g.get("geo_hint")]
            for h in geo_hints:
                meta.geo_relations[h] = meta.geo_relations.get(h, 0) + 1
    meta.candidates_judged = len(packets)

    plan_payload = {
        "original_query": query,
        "primary_intent": parsed.primary_intent,
        "criteria": [
            {"id": c.id, "type": c.type, "concept": c.concept or c.value, "required": c.required}
            for c in parsed.criteria
        ],
    }

    def _call(pkts: list[dict]) -> tuple[str, object, str | None, str | None]:
        return _call_near_judge(plan_payload, pkts)

    size = max(1, settings.near_match_judge_batch_size)
    batches = [packets[i : i + size] for i in range(0, len(packets), size)]
    verdicts: dict[str, dict] = {}
    # TASK 3 — independent batches, same bounded concurrency as full_verification
    # (SEMANTIC_JUDGE_CONCURRENCY; 1 = sequential, identical to the pre-TASK-3
    # loop). Results are merged back in the main thread, in batch order.
    batch_results = run_concurrent_map(
        batches, lambda b: run_adaptive(b, _call),
        max_workers=max(1, settings.semantic_judge_concurrency),
        profile_label="near_match_judge",
    )
    for leaves, stats in batch_results:
        meta.batches += stats.batches_attempted
        meta.successful_batches += stats.successful_batches
        meta.failed_batches += stats.failed_batches
        for prov, n in stats.providers.items():
            meta.providers[prov] = meta.providers.get(prov, 0) + n
        for m in stats.models:
            if m not in meta.models:
                meta.models.append(m)
        for leaf in leaves:
            if leaf.outcome == "ok":
                verdicts.update(leaf.payload)

    if meta.successful_batches == 0:
        meta.status = "unavailable"
    elif meta.failed_batches:
        meta.status = "partial"
    else:
        meta.status = "full"

    log.info(
        "near_match_judge: considered=%d judged=%d batches=%d/%d ok status=%s geo_relations=%s",
        meta.candidates_considered, meta.candidates_judged, meta.successful_batches,
        meta.batches, meta.status, meta.geo_relations,
    )
    return NearJudgeRun(verdicts, packets_by_id, meta)
