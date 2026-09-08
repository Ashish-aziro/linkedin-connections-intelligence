"""Orchestrate a connection search (spec §33, §38–§41, §50–§51; V4 PART 3).

Flow (V4 PART 3 §21):

    interpret query
      -> full local network scan (candidate_pool, <= FULL_SCAN_MAX_CONNECTIONS
         means EVERYONE)
      -> bulk-load every fact once
      -> HARD-FACT VIABILITY GATE  (candidate_gate) — reject ONLY on a verified
         contradiction the LLM cannot reasonably overturn
      -> local pre-score the viable set (evidence / prior signal only — it does
         NOT remove anyone, MIN_MATCH_SCORE is not applied here)
      -> EXHAUSTIVE SEMANTIC JUDGE (semantic_judge.run_judge) — in all_viable
         mode EVERY viable candidate is judged, in batches
      -> FACT-CONSISTENCY VALIDATOR (judge_validator) — every verdict checked
         against the packet + locked facts before it can change a score
      -> deterministic rescore with the validated verdicts
      -> qualification tier (exact / possible / not_match); drop not_match
      -> NOW apply MIN_MATCH_SCORE
      -> cross-encoder rerank WITHIN tiers
      -> top results + judge observability metadata
"""
from __future__ import annotations

import logging
from dataclasses import replace

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import (
    _QUALIFICATION_RANK,
    CriterionType,
    JUDGEABLE_CRITERION_TYPES,
    Qualification,
    RESULT_BEARING_STATUSES,
    SearchStatus,
    VerificationStatus,
)
from app.models import SearchQuery
from app.schemas import (
    ConnectionBucket,
    ExternalBucket,
    ParsedSearchQuery,
    SearchResponse,
    SearchResultItem,
)
from app.services import search_verification as _verify
from app.services.candidate_gate import hard_gate
from app.services.candidate_pool import get_candidates
from app.services.deadline import Deadline
from app.services.judge_validator import validate_person
from app.services.llm import budget as llm_budget
from app.services.matching import company_matches, norm_company
from app.services.person_view import education_to_out, experience_to_out, skill_to_out
from app.services.profile_authority import current_employer_from
from app.services.query_interpreter import interpret_query
from app.services.reason_generator import generate_reason, generate_reasons_batch
from app.services import search_profile
from app.services.scoring import ScoredCandidate, ScoringContext, load_facts, score_candidate
from app.services.semantic_judge import run_judge
from app.services.semantic_similarity import compute_semantic_similarity

log = logging.getLogger("app.search")

#: stored-response format version (V4 PART 7 §9). Bump when the persisted snapshot
#: shape changes; ``load_search`` can then branch on ``SearchRunState.response_version``.
RESPONSE_VERSION = 1


def _tier_key(s):
    """Rank by qualification tier FIRST, then match score (V4 §25)."""
    return (_QUALIFICATION_RANK.get(s.qualification, 1), -s.match_score)


def run_connection_search(db: Session, *, dataset_id: str, query: str) -> SearchResponse:
    llm_budget.clear_budget()  # defensive — a prior request on a reused thread must never leak in
    import time as _time

    prof = search_profile.start()
    _wall_start = _time.perf_counter()
    # hardening PART 14 / mission STEP 9 — ONE wall-clock budget for the whole
    # search. Deterministic scoring always runs to completion; every OPTIONAL
    # expensive stage (similarity, judge, rerank, audit, reason) checks it and
    # bails to fast partial finalization once it is spent.
    deadline = Deadline(settings.search_max_seconds)
    with prof.stage("query_interpretation"):
        parsed, provider, model = interpret_query(query)
    log.info("query %r -> %d criteria (intent=%s) via %s",
             query, len(parsed.criteria), parsed.intent, provider)

    # ── hardening PART 6 — soft budget on every LLM call from HERE on (judge /
    #    audit / reason). Interpretation is foundational, not optional, so it
    #    is never metered. Cleared unconditionally before returning below. ──
    llm_budget.start_budget(settings.search_llm_max_calls)
    calls_interpretation = 0 if provider == "deterministic" else 1

    # ── V4 PART 6 B2/B3/B13 — NO LLM ⇒ NO RESULTS. If interpretation fell back
    #    to the deterministic parser (or Anthropic is required and did not run
    #    it), do NOT continue into a normal result response — a keyword-only
    #    plan must never be shown as Exact/Possible matches. ───────────────────
    required_semantic = any(
        c.required and c.type in JUDGEABLE_CRITERION_TYPES for c in parsed.criteria
    )
    _gate = _verify.interpretation_gate(provider)
    if _gate is not None:
        _gs, _gvs, _greason = _gate
        llm_budget.clear_budget()
        search_profile.clear()
        return _no_result_response(
            db, dataset_id=dataset_id, query=query, parsed=parsed,
            provider=provider, model=model, status=_gs,
            verification_status=_gvs, reason=_greason,
            ident=_verify.ai_identity(provider, model),
        )

    with prof.stage("query_embedding"):
        query_embedding = _maybe_embed(query)  # for relevance RANKING only — never gates
    with prof.stage("candidate_fetch"):
        candidates, total = get_candidates(db, dataset_id, parsed, query_embedding)
    pids = [p.id for p in candidates]

    # bulk-load every fact once (spec §31 / PART 3 §58 — no per-candidate N+1)
    with prof.stage("bulk_fact_load"):
        facts_cache = {
            "experiences": repo.bulk_experiences(db, pids),
            "education": repo.bulk_education(db, pids),
            "skills": repo.bulk_skills(db, pids),
            "certifications": repo.bulk_certifications(db, pids),
            "languages": repo.bulk_languages(db, pids),
            "publications": repo.bulk_publications(db, pids),
            "semantics": repo.bulk_semantics(db, pids),
            "embeddings": repo.bulk_embeddings_by_person(db, pids),
        }
        vol_by_id = repo.bulk_volunteering(db, pids)
        rec_by_id = repo.bulk_recommendations(db, pids)

    with prof.stage("company_classification"):
        ctx = ScoringContext(
            query_embedding=query_embedding,
            company_ids_by_criterion=_resolve_company_ids(db, dataset_id, parsed),
            company_class=_pool_company_class(db, parsed, facts_cache["experiences"]),
        )
    facts_by_id: dict = {p.id: load_facts(db, p, facts_cache) for p in candidates}

    # ── hard-fact viability gate (V4 PART 3 §4-§7) ────────────────────
    with prof.stage("hard_gate"):
        decisions = {p.id: hard_gate(facts_by_id[p.id], parsed, ctx) for p in candidates}
    viable = [p for p in candidates if decisions[p.id].viable]
    hard_rejected = [p for p in candidates if not decisions[p.id].viable]
    log.info("hard-fact gate: %d viable, %d rejected (of %d scanned)",
             len(viable), len(hard_rejected), total)

    # ── STEP 7 — ONE vectorised numpy matmul for whole-profile relevance,
    #    instead of a per-candidate dot product inside every score_candidate. ──
    _precompute_relevance(ctx, [facts_by_id[p.id] for p in viable])

    # ── STEP 3/4 — batched concept-vs-career cross-encoder for the viable set
    #    (ONE predict() per distinct concept, not one pair per candidate). Skipped
    #    once the deadline is spent — ranking degrades, the search never hangs. ──
    with prof.stage("semantic_similarity"):
        if not deadline.expired():
            compute_semantic_similarity(
                [facts_by_id[p.id] for p in viable], parsed, ctx, deadline=deadline
            )

    # ── local pre-score the viable set — evidence / prior signal only,
    #    NOT a filter, MIN_MATCH_SCORE deliberately not applied (§21/§22) ──
    with prof.stage("prescore"):
        prescored: dict[str, ScoredCandidate] = {
            p.id: score_candidate(facts_by_id[p.id], parsed, ctx) for p in viable
        }
    prof.incr("prescore_candidates", len(viable))

    # ── exhaustive semantic judge (§8-§10) ───────────────────────────
    bundle = [
        (p, facts_by_id[p.id],
         {"volunteering": vol_by_id.get(p.id, []), "recommendations": rec_by_id.get(p.id, [])})
        for p in viable
    ]
    with prof.stage("judge"):
        judge_run = run_judge(
            query, parsed, bundle, ctx,
            network_size=total, pool_size=len(candidates),
            hard_rejected_count=len(hard_rejected), local_scored=prescored,
            deadline=deadline,
        )

    # ── validate every verdict before it can change a score (§17) ────
    with prof.stage("judge_validation"):
        for pid, person_verdicts in judge_run.verdicts.items():
            packet = judge_run.packets_by_id.get(pid)
            if packet is None or pid not in facts_by_id:
                continue
            validated = validate_person(person_verdicts, packet, parsed, facts_by_id[pid], ctx)
            if validated:
                ctx.judge_results[pid] = validated

    # ── deterministic rescore (STEP 5) — ONLY the candidates whose validated
    #    judge verdicts can actually move something are recomputed; everyone
    #    else reuses their (immutable) pre-score. A verdict changes the score
    #    solely through ``scoring._score_one``'s judge-override branch, which
    #    fires only on a TRUE/FALSE status for a judge-overridable type — an
    #    all-UNKNOWN verdict leaves the deterministic score byte-identical, so
    #    those candidates keep their pre-score (no second full N pass). ──
    from app.constants import TriState as _TriState

    changed_ids = {
        pid for pid, vv in ctx.judge_results.items()
        if any((v or {}).get("status") in (_TriState.TRUE, _TriState.FALSE) for v in vv.values())
    }
    prof.incr("rescore_candidates", len(changed_ids & {p.id for p in viable}))
    with prof.stage("rescore"):
        scored: list[ScoredCandidate] = []
        near_pool: list[ScoredCandidate] = []
        for p in viable:
            r = score_candidate(facts_by_id[p.id], parsed, ctx) if p.id in changed_ids else prescored[p.id]
            if r.qualification == Qualification.NOT_MATCH:
                if len(r.unmet_required) == 1:
                    near_pool.append(r)
                continue
            scored.append(r)

        # hard-rejected candidates that miss only one thing — surfaced as near-matches,
        # never judged, never in the main results
        for p in hard_rejected:
            r = score_candidate(facts_by_id[p.id], parsed, ctx)
            if r.qualification == Qualification.NOT_MATCH and len(r.unmet_required) <= 2:
                near_pool.append(r)

    # ── NOW apply MIN_MATCH_SCORE (never before the judge, §22). A verified
    #    EXACT_MATCH is kept even with a modest numeric score. ──
    scored = [
        s for s in scored
        if s.match_score >= settings.min_match_score or s.qualification == Qualification.EXACT_MATCH
    ]
    scored.sort(key=_tier_key)

    # ── cross-encoder rerank WITHIN tiers (§37) — a small, bounded pool
    #    (rerank_pool), skipped once the deadline is spent (STEP 9/10). ──
    pool = scored[: settings.rerank_pool]
    if settings.reranker_enabled and pool and not deadline.expired():
        from app.services.reranker import cross_encode

        with prof.stage("rerank"):
            texts = [_candidate_text(db, c, facts_by_id) for c in pool]
            for c, ce in zip(pool, cross_encode(query, texts)):
                ctx.reranker_scores[c.person.id] = ce
            rescored = [score_candidate(facts_by_id[c.person.id], parsed, ctx) for c in pool]
            rescored = [r for r in rescored if r.qualification != Qualification.NOT_MATCH]
            rescored.sort(key=_tier_key)
            scored = rescored + scored[settings.rerank_pool :]
            scored.sort(key=_tier_key)  # cross-encoder must NOT reorder across tiers (V4 §25/§37)

    # ── FINAL RESULT AUDIT (V4 PART 5) — one grounded LLM correctness pass over
    #    the TOP_N + BUFFER pool, BEFORE reason generation / persistence. It can
    #    only keep / downgrade / remove, never upgrade POSSIBLE->EXACT. Removed
    #    candidates drop out; the un-audited tail is kept only for the counts and
    #    is NEVER promoted into the shown results (§3/§23). Skipped entirely once
    #    the deadline is spent — fast partial finalization (STEP 10). ──────────
    if deadline.expired():
        log.warning("search %r: deadline spent before final audit — finalizing PARTIAL", query)
        audit_run, audit_by_id, survivors, tail = None, {}, scored, []
    else:
        with prof.stage("audit"):
            audit_run, audit_by_id, survivors, tail = _run_final_audit(
                db, query, parsed, scored, near_pool, ctx, facts_by_id, vol_by_id, rec_by_id, deadline,
            )
    scored = survivors + tail

    total_scored = len(scored)
    exact_n = sum(1 for s in scored if s.qualification == Qualification.EXACT_MATCH)
    possible_n = sum(1 for s in scored if s.qualification == Qualification.POSSIBLE_MATCH)

    judge_metadata = judge_run.metadata.as_dict()
    audit_metadata = audit_run.metadata.as_dict() if audit_run else None

    # ── V4 PART 6 B2/B4/B5/B12 — verification gate. A required semantic judge
    #    that produced no valid verdicts, or a required final audit that could
    #    not complete, means the search returns NO user-visible candidates
    #    (VERIFICATION_INCOMPLETE) rather than unverified deterministic ones. ──
    status, verification_status, fail_reason = _verify.decide(
        interp_provider=provider, required_semantic=required_semantic,
        judge_meta=judge_metadata, audit_meta=audit_metadata,
        audit_ran=audit_run is not None, deadline_reached=deadline.expired(),
    )
    ident = _verify.ai_identity(
        provider, model, judge_metadata.get("providers"),
        (audit_metadata or {}).get("providers"),
    )
    if status not in RESULT_BEARING_STATUSES:
        llm_budget.clear_budget()
        prof.timings_ms["total"] = round((_time.perf_counter() - _wall_start) * 1000.0, 1)
        search_profile.clear()
        log.info("search %r -> %s (%s): %s", query, status, verification_status, fail_reason)
        return _no_result_response(
            db, dataset_id=dataset_id, query=query, parsed=parsed, provider=provider,
            model=model, status=status, verification_status=verification_status,
            reason=fail_reason, ident=ident, judge_metadata=judge_metadata,
            audit_metadata=audit_metadata, total_candidates=total_scored, suppressed=total_scored,
        )
    if ident["fallback_used"]:
        status = SearchStatus.SUCCESS_WITH_FALLBACK
    llm_verified_flag = verification_status == VerificationStatus.COMPLETE or (
        not required_semantic and provider != "deterministic"
    )

    # ONE authoritative user-facing result count (V4 PART 5.5 §20): TOP_CONNECTIONS.
    if audit_run is not None:
        top = survivors[: settings.top_connections]  # audited candidates only
    else:
        top = _maybe_llm_rerank(db, query, scored[: settings.top_connections])

    sq = repo.create_search_query(
        db,
        dataset_id=dataset_id,
        query_text=query,
        interpreted_query_json=parsed.model_dump(),
        llm_provider=provider,
        llm_model=model,
        total_candidates=total_scored,
    )

    # ── batched display-reason generation (hardening PART 10) — ONE LLM call
    #    for the whole top-N instead of one per candidate. Display-only: never
    #    affects ranking / qualification / score. ──────────────────────────
    # skip the LLM reason path once the deadline is spent — a name/company/skill
    # deterministic template still explains every result, it just isn't prose.
    llm_reason_pool = (
        top[: settings.llm_reason_top_n]
        if settings.llm_reason_generation and not deadline.expired() else []
    )
    with prof.stage("reason_generation"):
        reasons_by_id = (
            generate_reasons_batch(llm_reason_pool, query, facts_by_id=facts_by_id) if llm_reason_pool else {}
        )

    results: list[SearchResultItem] = []
    for rank, cand in enumerate(top, start=1):
        reason = reasons_by_id.get(cand.person.id) or generate_reason(cand, query, allow_llm=False)
        item = _to_result_item(
            db, rank, cand, parsed, query, reason=reason,
            audit=audit_by_id.get(cand.person.id),
            facts=facts_by_id.get(cand.person.id),
        )
        results.append(item)
        repo.add_search_result(
            db, search_id=sq.id, person_id=cand.person.id, bucket="connection", rank=rank,
            match_score=item.match_score, data_confidence=item.data_confidence,
            reason=item.reason, payload=item.model_dump(),
        )

    near_pool.sort(key=lambda s: s.match_score, reverse=True)
    seen_near: set[str] = set()
    near_items: list[SearchResultItem] = []
    for i, cand in enumerate(near_pool, start=1):
        if cand.person.id in seen_near or len(near_items) >= 5:
            continue
        seen_near.add(cand.person.id)
        near_reason = generate_reason(cand, query, allow_llm=False)
        item = _to_result_item(db, len(near_items) + 1, cand, parsed, query, reason=near_reason,
                               facts=facts_by_id.get(cand.person.id))
        near_items.append(item)
        # near matches persist in their OWN bucket — same schema, qualification
        # stays not_match, never mixed into the main results (V4 PART 7 §4).
        repo.add_search_result(
            db, search_id=sq.id, person_id=item.person_id, bucket="connection_near",
            rank=item.rank, match_score=item.match_score, data_confidence=item.data_confidence,
            reason=item.reason, payload=item.model_dump(),
        )

    # hardening PART 6 — per-search LLM call tally, no prompts/profile data.
    # judge/audit batch counts already include every adaptive-split attempt.
    reason_calls = 1 if (llm_reason_pool and settings.llm_reason_generation
                        and any(c.evidence for c in llm_reason_pool)) else 0
    llm_calls = {
        "query_interpretation": calls_interpretation,
        "semantic_judge": judge_metadata.get("judge_batch_count", 0),
        "final_audit": (audit_metadata or {}).get("batch_count", 0),
        "reason_generation": reason_calls,
    }
    llm_calls["total"] = calls_interpretation + llm_calls["semantic_judge"] \
        + llm_calls["final_audit"] + reason_calls
    llm_calls["budget"] = {"max_calls": settings.search_llm_max_calls, "used_after_interpretation": llm_budget.used()}
    llm_calls["deadline"] = deadline.as_dict()
    llm_budget.clear_budget()

    prof.timings_ms["total"] = round((_time.perf_counter() - _wall_start) * 1000.0, 1)
    profile_dict = prof.as_dict()
    llm_calls["profile"] = profile_dict
    search_profile.clear()

    verification_metadata = {
        **ident,
        "search_status": status,
        "verification_status": verification_status,
        "llm_verified": llm_verified_flag,
        "unverified_results_suppressed": 0,
        "reason": None,
    }

    # B29 — ONE concise structured summary line (no keys / prompts / profile data).
    log.info(
        "search %r -> %s | verification=%s ai_provider=%s anthropic(att=%s ok=%s) fallback=%s "
        "llm_calls=%d total_ms=%d deadline_reached=%s judge=%s audit=%s "
        "hard_gate(viable=%d rejected=%d) judge(batches=%s trunc=%s splits=%s) "
        "verified_results=%d suppressed=0 stage_ms=%s",
        query, status, verification_status, ident["ai_provider"],
        ident["anthropic_attempted"], ident["anthropic_succeeded"], ident["fallback_used"],
        llm_calls["total"], deadline.elapsed_ms(), deadline.expired(),
        judge_metadata.get("status"), (audit_metadata or {}).get("status"),
        len(viable), len(hard_rejected),
        judge_metadata.get("judge_batch_count"), judge_metadata.get("truncations"),
        judge_metadata.get("adaptive_splits"),
        sum(1 for r in results if r.llm_verified), profile_dict["timings_ms"],
    )

    # FINAL validated search-level snapshot (V4 PART 7 §3) — captured here, AFTER
    # _run_final_audit -> final_auditor.finalize(). load_search rebuilds the whole
    # response from this row + the persisted result payloads, never re-running any
    # LLM / embedding / judge / audit / reason step.
    repo.upsert_search_run_state(
        db, sq.id,
        response_version=RESPONSE_VERSION,
        exact_match_count=exact_n,
        possible_match_count=possible_n,
        returned_count=len(results),
        near_match_count=len(near_items),
        total_candidates=total_scored,
        external_searched=False,
        judge_metadata=judge_metadata,
        audit_metadata=audit_metadata,
        search_status=status,
        verification_metadata=verification_metadata,
    )

    return SearchResponse(
        search_id=sq.id,
        query=query,
        interpreted_query=parsed.model_dump(),
        connections=ConnectionBucket(
            total_candidates=total_scored, returned=len(results), results=results,
            exact_match_count=exact_n, possible_match_count=possible_n, near_matches=near_items,
        ),
        external=ExternalBucket(searched=False),
        llm_provider=provider,
        llm_model=model,
        judge_metadata=judge_metadata,
        audit_metadata=audit_metadata,
        llm_calls=llm_calls,
        search_status=status,
        verification_status=verification_status,
        ai_provider=ident["ai_provider"],
        ai_model=ident["ai_model"],
        anthropic_attempted=ident["anthropic_attempted"],
        anthropic_succeeded=ident["anthropic_succeeded"],
        fallback_used=ident["fallback_used"],
        llm_verified=llm_verified_flag,
        unverified_results_suppressed=0,
    )


def _no_result_response(
    db: Session, *, dataset_id: str, query: str, parsed: ParsedSearchQuery,
    provider: str | None, model: str | None, status: str, verification_status: str,
    reason: str | None, ident: dict, judge_metadata: dict | None = None,
    audit_metadata: dict | None = None, total_candidates: int = 0, suppressed: int = 0,
) -> SearchResponse:
    """V4 PART 6 B2/B3/B22 — a search that could not be AI-verified. Persists the
    diagnostic state but NO ``search_results`` rows, so a reload can never
    resurrect unverified deterministic candidates (B38)."""
    sq = repo.create_search_query(
        db, dataset_id=dataset_id, query_text=query,
        interpreted_query_json=parsed.model_dump(),
        llm_provider=provider, llm_model=model, total_candidates=total_candidates,
    )
    verification_metadata = {
        **ident, "search_status": status, "verification_status": verification_status,
        "llm_verified": False, "unverified_results_suppressed": suppressed, "reason": reason,
    }
    repo.upsert_search_run_state(
        db, sq.id, response_version=RESPONSE_VERSION,
        exact_match_count=0, possible_match_count=0, returned_count=0, near_match_count=0,
        total_candidates=total_candidates, external_searched=False,
        judge_metadata=judge_metadata, audit_metadata=audit_metadata,
        search_status=status, verification_metadata=verification_metadata,
    )
    return SearchResponse(
        search_id=sq.id, query=query, interpreted_query=parsed.model_dump(),
        connections=ConnectionBucket(total_candidates=total_candidates, returned=0, results=[],
                                     exact_match_count=0, possible_match_count=0, near_matches=[]),
        external=ExternalBucket(searched=False),
        llm_provider=provider, llm_model=model,
        judge_metadata=judge_metadata, audit_metadata=audit_metadata,
        search_status=status, verification_status=verification_status,
        ai_provider=ident["ai_provider"], ai_model=ident["ai_model"],
        anthropic_attempted=ident["anthropic_attempted"],
        anthropic_succeeded=ident["anthropic_succeeded"],
        fallback_used=ident["fallback_used"], llm_verified=False,
        unverified_results_suppressed=suppressed,
    )


def load_search(db: Session, search_id: str) -> SearchResponse | None:
    """Reconstruct a completed search's response from persisted state ONLY.

    This is a HISTORICAL SNAPSHOT (V4 PART 7 §1): it never re-runs query
    interpretation, embeddings, the semantic judge, the final auditor, reason
    generation, or Apify — it only reads ``search_queries`` + ``search_results``
    + ``search_run_states``.
    """
    sq: SearchQuery | None = repo.get_search_query(db, search_id)
    if not sq:
        return None

    rows = repo.get_search_results(db, search_id)
    # ``get_search_results`` orders by rank; filtering by bucket keeps that order.
    main_rows = [_item_from_payload(r.payload) for r in rows if r.bucket == "connection"]
    near_rows = [_item_from_payload(r.payload) for r in rows if r.bucket == "connection_near"]

    state = repo.get_search_run_state(db, search_id)
    if state is not None:
        exact_n = state.exact_match_count
        possible_n = state.possible_match_count
        total_candidates = state.total_candidates or sq.total_candidates
        external_searched = state.external_searched
        judge_metadata = state.judge_metadata
        audit_metadata = state.audit_metadata
        vm = getattr(state, "verification_metadata", None) or {}
        search_status = getattr(state, "search_status", None) or SearchStatus.SUCCESS
        # B38 — a persisted failed/incomplete search reloads with NO candidates,
        # even if (defensively) result rows somehow exist.
        if search_status not in RESULT_BEARING_STATUSES:
            main_rows, near_rows = [], []
            exact_n = possible_n = 0
        return SearchResponse(
            search_id=sq.id, query=sq.query_text,
            interpreted_query=sq.interpreted_query_json or {},
            connections=ConnectionBucket(
                total_candidates=total_candidates, returned=len(main_rows), results=main_rows,
                exact_match_count=exact_n, possible_match_count=possible_n, near_matches=near_rows,
            ),
            external=ExternalBucket(searched=bool(external_searched)),
            llm_provider=sq.llm_provider, llm_model=sq.llm_model,
            judge_metadata=judge_metadata, audit_metadata=audit_metadata,
            search_status=search_status,
            verification_status=vm.get("verification_status", VerificationStatus.NOT_REQUIRED),
            ai_provider=vm.get("ai_provider"), ai_model=vm.get("ai_model"),
            anthropic_attempted=bool(vm.get("anthropic_attempted")),
            anthropic_succeeded=bool(vm.get("anthropic_succeeded")),
            fallback_used=bool(vm.get("fallback_used")),
            llm_verified=bool(vm.get("llm_verified")),
            unverified_results_suppressed=int(vm.get("unverified_results_suppressed") or 0),
        )
    # Pre-PART-7 saved search — no snapshot row. Derive counts from the stored
    # payload qualifications; metadata is unrecoverable, so leave it None and
    # near_matches empty (V4 PART 7 §7).
    exact_n = sum(1 for m in main_rows if m.qualification == Qualification.EXACT_MATCH)
    possible_n = sum(1 for m in main_rows if m.qualification == Qualification.POSSIBLE_MATCH)
    total_candidates = sq.total_candidates
    external_searched = bool(sq.external_searched)
    judge_metadata = None
    audit_metadata = None

    return SearchResponse(
        search_id=sq.id,
        query=sq.query_text,
        interpreted_query=sq.interpreted_query_json or {},
        connections=ConnectionBucket(
            total_candidates=total_candidates,
            returned=len(main_rows),
            results=main_rows,
            exact_match_count=exact_n,
            possible_match_count=possible_n,
            near_matches=near_rows,
        ),
        external=ExternalBucket(searched=bool(external_searched)),
        llm_provider=sq.llm_provider,
        llm_model=sq.llm_model,
        judge_metadata=judge_metadata,
        audit_metadata=audit_metadata,
    )


def _item_from_payload(payload: dict) -> SearchResultItem:
    """Tolerant rebuild of a stored result — older payloads may lack V4 fields
    (qualification / audit / uncertainty); pydantic defaults fill those in
    (V4 PART 7 §7)."""
    return SearchResultItem(**(payload or {}))


# ─────────────────────── final audit (V4 PART 5) ───────────────────────


def _run_final_audit(db, query, parsed, scored, near_pool, ctx, facts_by_id, vol_by_id, rec_by_id, deadline=None):
    """Returns ``(audit_run | None, audit_by_id, survivors_sorted, tail)``.

    Audits the TOP_N + BUFFER pool in ONE batched pass, validates every decision,
    applies the allowed qualification transitions — removed candidates drop out
    (or become 1-miss near-matches). ``tail`` = the scored candidates beyond the
    audit pool, kept ONLY for the tier counts, never promoted (§3/§23)."""
    if not settings.final_result_audit_enabled or not scored:
        return None, {}, scored, []

    from app.services.final_audit_validator import validate_audit
    from app.services.final_auditor import finalize as _finalize_audit
    from app.services.final_auditor import run_final_audit

    # audit pool = the user-facing count + a buffer, so a removal can be
    # back-filled from candidates already audited in the same pass (§20).
    pool_n = max(1, settings.top_connections + settings.final_result_audit_buffer)
    audit_pool = scored[:pool_n]
    tail = scored[pool_n:]
    bundle_by_id = {
        c.person.id: (
            c.person, facts_by_id[c.person.id],
            {"volunteering": vol_by_id.get(c.person.id, []), "recommendations": rec_by_id.get(c.person.id, [])},
        )
        for c in audit_pool if c.person.id in facts_by_id
    }
    audit_run = run_final_audit(query, parsed, audit_pool, ctx, bundle_by_id=bundle_by_id, deadline=deadline)

    survivors: list[ScoredCandidate] = []
    audit_by_id: dict[str, dict] = {}
    for cand in audit_pool:
        raw = audit_run.decisions.get(cand.person.id) or {"person_id": cand.person.id, "audit_missing": True}
        packet = audit_run.packets_by_id.get(cand.person.id) or {}
        v = validate_audit(
            raw, packet, parsed, facts_by_id[cand.person.id], ctx,
            first_pass_qualification=cand.qualification,
            first_pass_uncertain=cand.uncertain_required,
        )
        audit_run.decisions[cand.person.id] = v
        audit_by_id[cand.person.id] = v

        applied = v["applied_qualification"]
        if applied == Qualification.NOT_MATCH:
            if len(v["failed_required"]) == 1:
                near_pool.append(replace(cand, qualification=Qualification.NOT_MATCH,
                                         unmet_required=list(v["failed_required"])))
            continue
        nc = replace(cand, qualification=applied)
        if applied == Qualification.POSSIBLE_MATCH and cand.qualification == Qualification.EXACT_MATCH:
            nc = replace(nc, uncertain_required=(cand.uncertain_required
                                                 or list(v["failed_required"])
                                                 or ["downgraded by the final audit"]))
        survivors.append(nc)

    _finalize_audit(audit_run, audit_by_id)  # tally + review-completeness -> status
    survivors.sort(key=_tier_key)
    return audit_run, audit_by_id, survivors, tail


# ─────────────────────── helpers ───────────────────────


def _resolve_company_ids(db: Session, dataset_id: str, parsed: ParsedSearchQuery) -> dict[str, set[str]]:
    """Map each company criterion to the LinkedIn company_ids its value(s)
    resolve to within this dataset (fuzzy name match on the index keys)."""
    company_crits = [
        c for c in parsed.criteria
        if c.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY)
    ]
    if not company_crits or not settings.company_id_matching:
        return {}
    index = repo.company_name_index(db, dataset_id)
    out: dict[str, set[str]] = {}
    for c in company_crits:
        ids: set[str] = set()
        for value in (c.values or [c.value]):
            target = norm_company(value)
            for name_key, cids in index.items():
                if name_key == target or company_matches(name_key, value):
                    ids |= cids
        if ids:
            out[c.id] = ids
    return out


def _pool_company_class(db: Session, parsed: ParsedSearchQuery, exp_by_person: dict) -> dict:
    """CACHE-ONLY employer classification for the candidate pool (V4 §8 / PART 3
    §26/§59). Normal search NEVER launches classification LLM jobs — a missing
    classification is left UNKNOWN. Bulk classification happens only via backfill
    / a maintenance job."""
    from app.services.company_intel import company_key, to_dict

    seen: dict[tuple, tuple] = {}
    for exps in exp_by_person.values():
        for e in exps:
            if e.company_name:
                seen.setdefault((e.company_id, e.company_name),
                                (e.company_id, e.company_name, e.company_linkedin_url))
    if not seen:
        return {}
    keys = [company_key(cid, nm) for cid, nm, _ in seen.values()]
    rows = repo.get_company_semantics(db, keys)
    return {k: to_dict(r) for k, r in rows.items()}


def _candidate_text(db: Session, cand: ScoredCandidate, facts_by_id: dict) -> str:
    """Compact text for the cross-encoder — prefer the stored embedding text."""
    from app.models import ProfileEmbedding

    row = db.query(ProfileEmbedding.search_text).filter(
        ProfileEmbedding.person_id == cand.person.id
    ).first()
    if row and row[0]:
        return row[0][:1200]
    p = cand.person
    return " · ".join(
        filter(None, [p.full_name, p.headline, p.current_title, p.current_company, p.location_text])
    )


def _maybe_llm_rerank(db: Session, query: str, top: list) -> list:
    # V4 PART 3 §36 — the exhaustive criterion-level judge replaces this; it stays
    # disabled unless an operator explicitly opts in.
    if not settings.llm_rerank_enabled or len(top) < 3:
        return top
    from app.services.reranker import llm_rerank

    cands = []
    for c in top:
        p = c.person
        line = " · ".join(filter(None, [
            p.full_name, p.current_title, p.current_company,
            "matched: " + ", ".join(c.matched_criteria) if c.matched_criteria else None,
        ]))
        cands.append({"person_id": p.id, "line": line[:220]})
    res = llm_rerank(query, cands)
    if not res:
        return top
    by_id = {c.person.id: c for c in top}
    reordered = [by_id[pid] for pid in res["order"] if pid in by_id and pid not in res["drop"]]
    seen = {c.person.id for c in reordered}
    reordered += [c for c in top if c.person.id not in seen and c.person.id not in res["drop"]]
    return reordered


def _maybe_embed(query: str) -> bytes | None:
    try:
        from app.services.embeddings import embed_text

        return embed_text(query)
    except Exception:  # noqa: BLE001
        log.warning("query embedding failed — continuing without semantic prefilter", exc_info=False)
        return None


def _precompute_relevance(ctx: ScoringContext, facts_list: list) -> None:
    """STEP 7 — whole-profile relevance for the viable set in ONE numpy matmul,
    stored on ``ctx.relevance_by_person``. Mirrors ``scoring._cosine_norm``
    (normalised vectors -> dot product, /0.6, clamped to [0,1]); a vector whose
    shape disagrees with the query is skipped (never a meaningless score)."""
    if not ctx.query_embedding or not facts_list:
        return
    import numpy as np

    from app.services.embeddings import to_array

    q = to_array(ctx.query_embedding)
    ids: list[str] = []
    mat: list = []
    for f in facts_list:
        if not f.embedding:
            continue
        v = to_array(f.embedding)
        if v.shape != q.shape:
            continue
        ids.append(f.person.id)
        mat.append(v)
    if not mat:
        return
    sims = np.dot(np.vstack(mat), q) / 0.6
    for pid, s in zip(ids, sims):
        ctx.relevance_by_person[pid] = float(max(0.0, min(1.0, s)))


def _to_result_item(
    db: Session,
    rank: int,
    cand: ScoredCandidate,
    parsed: ParsedSearchQuery,
    query: str,
    *,
    reason: str,
    audit: dict | None = None,
    facts=None,
) -> SearchResultItem:
    p = cand.person

    def _terms(*types: str) -> set[str]:
        return {
            v.lower()
            for c in parsed.criteria if c.type in types
            for v in (c.values or [c.value] or [c.concept or ""]) if v
        }

    skill_terms = _terms(CriterionType.SKILL, CriterionType.DOMAIN, CriterionType.SEMANTIC_CONCEPT)
    company_terms = _terms(
        CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY, CriterionType.COMPANY_CATEGORY
    )
    edu_terms = _terms(CriterionType.EDUCATION)

    # hardening PART 13 — the SAME experiences list scoring/judge/audit already
    # bulk-loaded, so the current-employer name shown here can never disagree
    # with what they used (no separate per-candidate re-fetch).
    exps = facts.experiences if facts is not None else repo.get_experiences(db, p.id)
    current_company = current_employer_from(p, exps)
    rel_exp = [
        experience_to_out(e)
        for e in exps
        if e.is_current
        or any(t in (e.company_name or "").lower() or t in (e.position or "").lower() for t in company_terms)
    ][:5]

    rel_skills = [
        skill_to_out(s)
        for s in repo.get_skills(db, p.id)
        if not skill_terms or any(t in s.skill_name_norm or s.skill_name_norm in t for t in skill_terms)
    ][:12]

    edus = repo.get_education(db, p.id)
    rel_edu = [
        education_to_out(e)
        for e in edus
        if not edu_terms
        or any(t in (e.school_name or "").lower() or t in (e.field_of_study or "").lower() for t in edu_terms)
    ]
    if not rel_edu and edus:
        rel_edu = [education_to_out(edus[0])]

    return SearchResultItem(
        rank=rank,
        person_id=p.id,
        name=p.full_name,
        linkedin_url=p.linkedin_url,
        profile_picture_url=p.profile_picture_url,
        current_title=p.current_title,
        current_company=current_company,
        location=p.location_text,
        is_connection=True,
        match_score=cand.match_score,
        data_confidence=p.profile_completeness,
        reason=reason,
        qualification=cand.qualification,
        uncertain_criteria=cand.uncertain_required,
        unmet_criteria=cand.unmet_required,
        matched_criteria=cand.matched_criteria,
        score_breakdown=cand.components,
        evidence=cand.evidence,
        relevant_experience=rel_exp or [experience_to_out(e) for e in exps[:2]],
        relevant_skills=rel_skills,
        relevant_education=rel_edu,
        audit_decision=(audit or {}).get("decision"),
        audit_confidence=(audit or {}).get("confidence"),
        audit_reason=((audit or {}).get("reason") or None),
        audit_issues=(audit or {}).get("audit_issues", []),
        llm_verified=bool((audit or {}).get("llm_verified")),
    )
