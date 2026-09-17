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
from app.constants import _QUALIFICATION_RANK, CriterionType, Qualification
from app.models import SearchQuery
from app.schemas import (
    ConnectionBucket,
    ExternalBucket,
    ParsedSearchQuery,
    SearchResponse,
    SearchResultItem,
)
from app.constants import _RELIABLE_NEAR_RELATIONS
from app.services.candidate_gate import hard_gate
from app.services.candidate_pool import get_candidates
from app.services.deadline import Deadline
from app.services.judge_validator import validate_person
from app.services.llm import budget as llm_budget
from app.services.matching import company_matches, norm_company
from app.services.near_match_service import build_near_matches
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

#: review finding H1 — these ``ParsedSearchQuery`` fields are internal
#: reasoning the schema itself documents as never shown to the frontend
#: (near-match anchoring, an internal validator confidence bound). A bare
#: ``model_dump()`` serialized them anyway, so the raw API response leaked
#: them even though no UI code reads them today. Excluded at every point the
#: parsed query is persisted or returned.
_INTERNAL_QUERY_FIELDS = frozenset({
    "primary_intent", "intent_anchor_criterion_ids", "interpretation_confidence_cap",
})


def _tier_key(s):
    """Rank by qualification tier FIRST, then match score (V4 §25)."""
    return (_QUALIFICATION_RANK.get(s.qualification, 1), -s.match_score)


def run_connection_search(db: Session, *, dataset_id: str, query: str) -> SearchResponse:
    llm_budget.clear_budget()  # defensive — a prior request on a reused thread must never leak in
    import time as _time

    prof = search_profile.start()
    _wall_start = _time.perf_counter()
    full_mode = settings.full_llm_verification
    # hardening PART 14 — wall-clock budget for a search's OPTIONAL LLM work.
    # In FULL SONNET VERIFICATION mode verification is MANDATORY, not optional,
    # so it uses ``full_verification_max_seconds`` (0 = unlimited) — the search
    # never returns partial results because time ran out.
    deadline = Deadline(
        settings.full_verification_max_seconds if full_mode else settings.search_max_seconds
    )
    with prof.stage("query_interpretation"):
        parsed, provider, model = interpret_query(query)
    log.info("query %r -> %d criteria (intent=%s) via %s",
             query, len(parsed.criteria), parsed.intent, provider)

    # ── hardening PART 6 — soft budget on every LLM call from HERE on (judge /
    #    audit / reason). Interpretation is foundational, not optional, so it
    #    is never metered. Cleared unconditionally before returning below. ──
    llm_budget.start_budget(settings.search_llm_max_calls)
    calls_interpretation = 0 if provider == "deterministic" else 1

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
                [facts_by_id[p.id] for p in viable], parsed, ctx, deadline=deadline, db=db,
            )

    # ── local pre-score the viable set — evidence / prior signal only,
    #    NOT a filter, MIN_MATCH_SCORE deliberately not applied (§21/§22) ──
    with prof.stage("prescore"):
        prescored: dict[str, ScoredCandidate] = {
            p.id: score_candidate(facts_by_id[p.id], parsed, ctx) for p in viable
        }
    prof.incr("prescore_candidates", len(viable))

    # ── verification — EVERY hard-gate survivor is reviewed ──────────
    bundle = [
        (p, facts_by_id[p.id],
         {"volunteering": vol_by_id.get(p.id, []), "recommendations": rec_by_id.get(p.id, [])})
        for p in viable
    ]
    judge_run = None
    fv_metadata: dict | None = None
    if full_mode:
        # FULL SONNET VERIFICATION — Sonnet reviews every filtered candidate
        # against the whole plan. Raises VerificationIncompleteError (-> HTTP
        # 503, retryable) if any candidate's required review cannot complete;
        # a partial/unverified result is NEVER returned.
        from app.services.full_verification import run_full_verification

        with prof.stage("full_verification"):
            fv_run = run_full_verification(
                query, parsed, bundle, ctx,
                network_size=total, pool_size=len(candidates),
                hard_rejected_count=len(hard_rejected), local_scored=prescored,
                db=db,
            )
        with prof.stage("verification_validation"):
            newly_validated: dict[str, dict[str, dict]] = {}
            for pid, person_verdicts in fv_run.verdicts.items():
                packet = fv_run.packets_by_id.get(pid)
                if packet is None or pid not in facts_by_id:
                    continue
                validated = validate_person(person_verdicts, packet, parsed, facts_by_id[pid], ctx)
                if validated:
                    ctx.judge_results[pid] = validated
                    newly_validated[pid] = validated
        # TASK 4 — cache ONLY validated verdicts, and only ones NOT already
        # served from the cache this run (re-writing a hit is a harmless no-op
        # upsert but would understate how many calls the cache actually saved).
        with prof.stage("verdict_cache_write"):
            from app.services import semantic_verdict_cache
            from app.services.semantic_judge import judgeable_criteria as _jcrits_fn

            cache_writes = semantic_verdict_cache.store(
                db, newly_validated, fv_run.packets_by_id, _jcrits_fn(parsed), parsed,
                model=settings.anthropic_model, skip_keys=fv_run.cache_hit_keys,
            )
            prof.incr("verdict_cache_writes", cache_writes)
            prof.incr("verdict_cache_hits", fv_run.metadata.get("cache_hits", 0))
        fv_metadata = fv_run.metadata
        # INVARIANT — a successful full-verification search reviewed EVERY filtered
        # candidate. If not, treat it as an incomplete verification (retryable),
        # never a silent partial success.
        if fv_metadata["sonnet_verified_candidate_count"] != len(viable):
            from app.services.full_verification import VerificationIncompleteError

            raise VerificationIncompleteError(
                f"full-verification invariant violated: filtered={len(viable)} "
                f"verified={fv_metadata['sonnet_verified_candidate_count']}",
                metadata=fv_metadata,
            )
    else:
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
        near_pool: list[ScoredCandidate] = []  # legacy list — the final audit still appends to this
        near_raw: list[tuple] = []             # (person, facts, scored, source) — feeds near_match_service
        viable_not_match = 0
        for p in viable:
            r = score_candidate(facts_by_id[p.id], parsed, ctx) if p.id in changed_ids else prescored[p.id]
            if r.qualification == Qualification.NOT_MATCH:
                near_raw.append((p, facts_by_id[p.id], r, "not_match"))
                viable_not_match += 1
                if len(r.unmet_required) == 1:
                    near_pool.append(r)
                continue
            scored.append(r)

        # hard-rejected candidates — surfaced (if at all) via the intent-aware
        # near-match pipeline below, which is what lets a candidate whose ONLY
        # problem is a literal location mismatch still be recognised as
        # professionally/geographically relevant instead of the story ending at
        # the hard gate. This runs in EVERY mode, including FULL SONNET
        # VERIFICATION: "a successful result page only contains reviewed
        # candidates" is an invariant of the MAIN Exact/Possible results only
        # (enforced above, unchanged) — it does not apply to Near Match, which
        # has its OWN independent LLM review (near_match_judge) and its own
        # hard invariant that a shown near match always has a validated
        # useful_near_match=true verdict (near_match_service.build_near_matches).
        # A candidate the hard gate rejected for one verified fact (e.g. a
        # literal location mismatch) was never sent to full_verification
        # either way, so ``score_candidate`` behaves identically in both modes
        # here — this loop is UNCONDITIONAL (no ``if not full_mode`` guard) —
        # see TASK 1/2 hardening below for the counters that make this
        # traceable in production logs, not just trusted by reading the code.
        hard_rejected_total = len(hard_rejected)
        hard_rejected_scored = 0
        hard_rejected_not_match = 0
        hard_rejected_missing_gap_metadata = 0
        hard_rejected_selected_for_near_pool = 0
        for p in hard_rejected:
            hard_rejected_scored += 1
            r = score_candidate(facts_by_id[p.id], parsed, ctx)
            if r.qualification == Qualification.NOT_MATCH:
                hard_rejected_not_match += 1
                if not r.unmet_required_ids:
                    # should never happen (qualification NOT_MATCH implies
                    # ``unmet`` was non-empty in score_candidate) — counted
                    # defensively so a future scoring regression is visible
                    # here instead of silently dropping the candidate's gap.
                    hard_rejected_missing_gap_metadata += 1
                near_raw.append((p, facts_by_id[p.id], r, "hard_gate_reject"))
                if len(r.unmet_required) <= 2:
                    near_pool.append(r)
                    hard_rejected_selected_for_near_pool += 1

        # TASK 1 hardening (near-match candidate-coverage bug report) — one
        # safe, aggregate-only accounting line for the hard-rejected -> near
        # pool boundary. Counts only; never a name, location, or profile field.
        log.info(
            "near_match_candidate_construction: hard_rejected_total=%d "
            "hard_rejected_passed_to_near_builder=%d hard_rejected_scored=%d "
            "hard_rejected_not_match=%d hard_rejected_missing_gap_metadata=%d "
            "hard_rejected_selected_for_near_pool=%d viable_not_match_candidates=%d",
            hard_rejected_total, hard_rejected_total, hard_rejected_scored,
            hard_rejected_not_match, hard_rejected_missing_gap_metadata,
            hard_rejected_selected_for_near_pool, viable_not_match,
        )
        prof.incr("hard_rejected_total", hard_rejected_total)
        prof.incr("hard_rejected_passed_to_near_builder", hard_rejected_total)
        prof.incr("hard_rejected_scored", hard_rejected_scored)
        prof.incr("hard_rejected_not_match", hard_rejected_not_match)
        prof.incr("hard_rejected_missing_gap_metadata", hard_rejected_missing_gap_metadata)
        prof.incr("hard_rejected_selected_for_near_pool", hard_rejected_selected_for_near_pool)
        prof.incr("viable_not_match_candidates", viable_not_match)

    # ── FULL SONNET VERIFICATION display eligibility — only candidates whose
    #    review completed AND every required criterion is TRUE (EXACT) reach the
    #    main results. A required criterion that reviewed as INSUFFICIENT_EVIDENCE
    #    (-> POSSIBLE_MATCH) is EXCLUDED, never shown as "Possible / needs
    #    verification". ──────────────────────────────────────────────────────
    if full_mode:
        # ``scored`` already dropped rescore NOT_MATCH (they went to ``near_pool``);
        # what is left that is not EXACT is a required INSUFFICIENT_EVIDENCE.
        insufficient = [s for s in scored if s.qualification != Qualification.EXACT_MATCH]
        scored = [s for s in scored if s.qualification == Qualification.EXACT_MATCH]
        if fv_metadata is not None:
            fv_metadata["excluded_insufficient_evidence"] = len(insufficient)
            fv_metadata["excluded_false"] = len(near_pool)

    # ── NOW apply MIN_MATCH_SCORE (never before the judge, §22). A verified
    #    EXACT_MATCH is kept even with a modest numeric score. Below-threshold
    #    POSSIBLE candidates are not simply discarded — borderline confidence,
    #    not a failed requirement — they feed the near-match pool instead. ──
    _below_threshold = [
        s for s in scored
        if not (s.match_score >= settings.min_match_score or s.qualification == Qualification.EXACT_MATCH)
    ]
    near_raw.extend((s.person, facts_by_id[s.person.id], s, "below_threshold") for s in _below_threshold)
    prof.incr("below_threshold_candidates", len(_below_threshold))
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

    # ── FULL SONNET VERIFICATION — the final audit is a REMOVAL-ONLY brake here.
    #    Every shown candidate already completed full Sonnet verification with
    #    every required criterion TRUE. The audit may still REMOVE a candidate it
    #    finds a clear contradiction for (already handled — those become
    #    NOT_MATCH / near), but a "downgrade to POSSIBLE" must NOT produce a
    #    "Possible match / needs verification" card — it is excluded instead. ──
    if full_mode:
        audit_downgraded = [s for s in scored if s.qualification != Qualification.EXACT_MATCH]
        scored = [s for s in scored if s.qualification == Qualification.EXACT_MATCH]
        survivors = [s for s in survivors if s.qualification == Qualification.EXACT_MATCH]
        if fv_metadata is not None and audit_downgraded:
            fv_metadata["excluded_audit_downgrade"] = len(audit_downgraded)
            fv_metadata["excluded_insufficient_evidence"] = (
                fv_metadata.get("excluded_insufficient_evidence", 0) + len(audit_downgraded)
            )
        # near-match hardening (bug report TASK 5) — a candidate the audit
        # downgraded from EXACT to POSSIBLE (insufficient evidence at audit
        # time, NOT a factual contradiction — a grounded contradiction is
        # always INCORRECT/NOT_MATCH, handled separately above and already
        # near-match-eligible) used to just vanish here: excluded from main
        # results and NEVER reconsidered for Near Match. This gives them the
        # SAME chance every other excluded-but-plausible candidate gets — a
        # fresh, independently validated near-match verdict, never an
        # automatic promotion (near_match_service still requires its own
        # useful_near_match=true verdict; an audit REMOVAL for a grounded
        # contradiction is untouched by this and stays authoritative).
        for s in audit_downgraded:
            av = audit_by_id.get(s.person.id) or {}
            gap_ids = list(av.get("failed_required_ids") or [])
            if not gap_ids or s.person.id not in facts_by_id:
                continue  # no usable gap to ground a near-match request — skip, never guess
            near_raw.append((
                s.person, facts_by_id[s.person.id],
                replace(s, qualification=Qualification.NOT_MATCH, unmet_required_ids=gap_ids),
                "audit_downgrade_possible",
            ))

    total_scored = len(scored)
    exact_n = sum(1 for s in scored if s.qualification == Qualification.EXACT_MATCH)
    possible_n = sum(1 for s in scored if s.qualification == Qualification.POSSIBLE_MATCH)
    # ONE authoritative user-facing result count (V4 PART 5.5 §20): TOP_CONNECTIONS.
    if audit_run is not None:
        top = survivors[: settings.top_connections]  # audited candidates only
    else:
        top = _maybe_llm_rerank(db, query, scored[: settings.top_connections])

    sq = repo.create_search_query(
        db,
        dataset_id=dataset_id,
        query_text=query,
        interpreted_query_json=parsed.model_dump(exclude=_INTERNAL_QUERY_FIELDS),
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
        if full_mode:
            # every main result completed a full Sonnet review with every
            # required criterion TRUE — a stronger signal than the audit flag.
            item.llm_verified = True
        results.append(item)
        repo.add_search_result(
            db, search_id=sq.id, person_id=cand.person.id, bucket="connection", rank=rank,
            match_score=item.match_score, data_confidence=item.data_confidence,
            reason=item.reason, payload=item.model_dump(),
        )

    # ── INTENT-AWARE NEAR MATCHES — a separate, bounded, evidence-grounded pass
    #    (never touches ``scored`` / ``survivors`` / ``top`` above). A candidate
    #    the final audit itself downgraded is folded in too, tagged distinctly;
    #    ``near_raw`` may contain the same person from more than one stage —
    #    ``build_near_pool`` dedupes by person id, keeping the first. ─────────
    near_raw_ids = {t[0].id for t in near_raw}
    for r in near_pool:
        if r.person.id not in near_raw_ids:
            near_raw.append((r.person, facts_by_id.get(r.person.id), r, "audit_downgrade"))
            near_raw_ids.add(r.person.id)

    # near-match design PART 1/8 (v2): the LLM explaining WHY a person is
    # still useful IS the feature — if it cannot complete (deadline spent,
    # disabled, unavailable, or the pipeline raises for any other reason:
    # transport error, malformed output, a validator bug), near_matches=[]
    # rather than a heuristic/code-only substitute. This must NEVER turn a
    # successful strict search into a failure — only the near-match section
    # is skipped, the main results below are entirely unaffected.
    near_matches_ranked: list[tuple] = []
    near_match_metadata = None
    if deadline.expired():
        log.warning("search %r: deadline spent before the near-match pipeline — near_matches=[]", query)
    else:
        try:
            with prof.stage("near_match"):
                near_matches_ranked, near_match_metadata = build_near_matches(
                    query, parsed, ctx, candidates=near_raw, vol_by_id=vol_by_id, rec_by_id=rec_by_id,
                )
        except Exception:  # noqa: BLE001 — near matches are supplemental; never fail the search for them
            # log.exception() captures the full traceback at ERROR level — this
            # must never be silent, only never fatal to the strict search. The
            # query text itself is not sensitive (already logged unredacted
            # elsewhere in this module); no API key, profile payload, prompt,
            # or auth header is ever included here.
            log.exception("near-match generation failed for search %r; returning strict results without near matches", query)
            near_matches_ranked, near_match_metadata = [], None

    near_items: list[SearchResultItem] = []
    for cand, verdict in near_matches_ranked:
        near_reason = verdict.get("short_reason") or generate_reason(cand, query, allow_llm=False)
        # bug report PART 6 observability — a near match reached via
        # "audit_downgrade_possible" (or the older audit-removal source) has a
        # real audit decision behind it; surface it exactly like a main
        # result does, instead of leaving audit_decision/audit_reason always
        # None. ``audit_by_id`` is empty ``{}`` when the audit never ran
        # (disabled, deadline) or for a candidate never in the audit pool
        # (hard_gate_reject / not_match / below_threshold sources) — safe no-op then.
        item = _to_result_item(db, len(near_items) + 1, cand, parsed, query, reason=near_reason,
                               facts=facts_by_id.get(cand.person.id),
                               audit=audit_by_id.get(cand.person.id))
        relation = verdict.get("relation_type")
        item.near_relation_type = relation if relation in _RELIABLE_NEAR_RELATIONS else None
        item.near_relation_source = verdict.get("relation_source")
        item.near_match_confidence = verdict.get("confidence")
        near_items.append(item)
        # near matches persist in their OWN bucket — same schema, qualification
        # stays not_match, never mixed into the main results (V4 PART 7 §4).
        repo.add_search_result(
            db, search_id=sq.id, person_id=item.person_id, bucket="connection_near",
            rank=item.rank, match_score=item.match_score, data_confidence=item.data_confidence,
            reason=item.reason, payload=item.model_dump(),
        )

    judge_metadata = fv_metadata if full_mode else judge_run.metadata.as_dict()
    audit_metadata = audit_run.metadata.as_dict() if audit_run else None

    # hardening PART 6 — per-search LLM call tally, no prompts/profile data.
    # judge/audit batch counts already include every adaptive-split attempt.
    reason_calls = 1 if (llm_reason_pool and settings.llm_reason_generation
                        and any(c.evidence for c in llm_reason_pool)) else 0
    near_match_calls = near_match_metadata.judge.batches if near_match_metadata else 0
    llm_calls = {
        "query_interpretation": calls_interpretation,
        "semantic_judge": judge_metadata.get("judge_batch_count", 0),
        "final_audit": (audit_metadata or {}).get("batch_count", 0),
        "reason_generation": reason_calls,
        "near_match_judge": near_match_calls,
    }
    llm_calls["total"] = calls_interpretation + llm_calls["semantic_judge"] \
        + llm_calls["final_audit"] + reason_calls + near_match_calls
    llm_calls["budget"] = {"max_calls": settings.search_llm_max_calls, "used_after_interpretation": llm_budget.used()}
    llm_calls["deadline"] = deadline.as_dict()
    llm_budget.clear_budget()

    prof.timings_ms["total"] = round((_time.perf_counter() - _wall_start) * 1000.0, 1)
    profile_dict = prof.as_dict()
    llm_calls["profile"] = profile_dict
    search_profile.clear()

    log.info(
        "search %r done: llm_calls=%d elapsed_ms=%d deadline_s=%s deadline_reached=%s "
        "judge_status=%s audit_status=%s | stage_ms=%s counters=%s",
        query, llm_calls["total"], deadline.elapsed_ms(), deadline.seconds, deadline.expired(),
        judge_metadata.get("status"), (audit_metadata or {}).get("status"),
        profile_dict["timings_ms"], profile_dict["counters"],
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
    )

    return SearchResponse(
        search_id=sq.id,
        query=query,
        interpreted_query=parsed.model_dump(exclude=_INTERNAL_QUERY_FIELDS),
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
    else:
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
                # near-match hardening (bug report TASK 5) — ``unmet_required_ids``
                # was never set here before (``replace`` keeps the PRE-audit
                # value, which for a candidate that was EXACT before the audit
                # is empty), so this candidate could reach the near-match pool
                # with no usable gap id — every relaxed_criterion_id the near-
                # match LLM later proposed would fail validation. Now grounded
                # in the audit's own ``failed_required_ids``.
                near_pool.append(replace(
                    cand, qualification=Qualification.NOT_MATCH,
                    unmet_required=list(v["failed_required"]),
                    unmet_required_ids=list(v.get("failed_required_ids") or []),
                ))
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
