"""Near Match pipeline entry point (near-match design, orchestration).

    local candidate pool (near_match_pool)
        -> batched LLM judge (near_match_judge)
        -> evidence validation (near_match_validator)
        -> ranking (near_match_ranking)
        -> bounded to settings.near_match_max_results

Kept as ONE call from ``search_service`` so the strict Exact/Possible pipeline
stays untouched and easy to audit for "near-match code never influenced a main
result" — this module only ever reads candidates the strict pipeline already
decided NOT to show as a main result.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

from app.config import settings
from app.constants import Qualification
from app.schemas import ParsedSearchQuery
from app.services.near_match_judge import NearJudgeMetadata, run_near_judge
from app.services.near_match_pool import NearCandidate, build_near_pool
from app.services.near_match_ranking import RankedNearMatch, rank_near_matches
from app.services.near_match_validator import validate_near_verdicts
from app.services.scoring import ScoredCandidate

log = logging.getLogger("app.near_match")

#: judge statuses that mean "we got at least some real evidence-grounded
#: output" — anything else (disabled / unavailable / not_used) means
#: near_matches=[] (near-match design PART 1, v2). The whole point of this
#: section is the LLM explaining WHY a person is still useful despite a gap —
#: a heuristic/code-only substitute would misrepresent that as AI judgement,
#: so there is no deterministic fallback ranking here (contrast with the
#: rest of this codebase, where an LLM stage degrades gracefully).
_LLM_USABLE_STATUSES = {"full", "partial"}


@dataclass
class NearMatchMetadata:
    pool_size: int
    judge: NearJudgeMetadata
    validated_count: int
    final_count: int
    llm_used: bool
    #: PART 9 observability — safe counters only, never a profile / prompt.
    rejected_count: int = 0
    relation_source_counts: dict[str, int] = field(default_factory=dict)
    #: PART 9 diagnostics (issue: "all N were judged but all N were rejected,
    #: with no visibility into why") — every key is one of
    #: ``near_match_validator.REJECTION_REASONS`` or "geographic_conflict".
    #: Aggregate counts only; never a name, profile field, or evidence string.
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    total_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "judge": self.judge.as_dict(),
            "validated_count": self.validated_count,
            "final_count": self.final_count,
            "llm_used": self.llm_used,
            "rejected_count": self.rejected_count,
            "relation_source_counts": self.relation_source_counts,
            "rejection_reason_counts": self.rejection_reason_counts,
            "total_ms": self.total_ms,
        }


def build_near_matches(
    query: str,
    parsed: ParsedSearchQuery,
    ctx,
    *,
    candidates: list[tuple],  # [(person, ProfileFacts, ScoredCandidate, source)]
    vol_by_id: dict | None = None,
    rec_by_id: dict | None = None,
) -> tuple[list[tuple[ScoredCandidate, dict]], NearMatchMetadata]:
    """Returns ``[(display_candidate, verdict), ...]`` — already ranked and
    capped at ``settings.near_match_max_results``. ``display_candidate`` is a
    ScoredCandidate copy with ``qualification`` forced to NOT_MATCH so the
    frontend never renders a near match with a Possible/Exact badge (PART 12) —
    the original strict-pipeline objects are never mutated. ``verdict`` is
    always a validated, evidence-grounded dict here (never None) — a
    candidate without one is dropped, never shown (PART 1, v2). If the
    near-match LLM judge could not complete (disabled, unavailable, every
    batch failed), the result is an empty list, not a heuristic substitute."""
    started = time.perf_counter()
    pool = build_near_pool(candidates, parsed, ctx)
    judge_run = run_near_judge(query, parsed, pool, ctx, vol_by_id=vol_by_id, rec_by_id=rec_by_id)
    llm_used = judge_run.metadata.status in _LLM_USABLE_STATUSES

    if not llm_used:
        meta = NearMatchMetadata(
            pool_size=len(pool), judge=judge_run.metadata,
            validated_count=0, final_count=0, llm_used=False, rejected_count=len(pool),
            total_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        _log_summary(meta)
        return [], meta

    pool_by_id: dict[str, NearCandidate] = {nc.person.id: nc for nc in pool}
    validated, rejection_reason_counts = validate_near_verdicts(
        judge_run.verdicts, judge_run.packets_by_id, pool_by_id, parsed,
    )
    ranked = rank_near_matches(pool, validated, parsed)
    capped = ranked[: max(0, settings.near_match_max_results)]

    # near-match design PART 1/3, v3 — the core invariant made EXPLICIT and
    # self-enforcing, not just true by construction: no candidate reaches the
    # user without a real, validated, useful_near_match=true LLM verdict.
    # ``pid_with_verdict`` is the ONLY set of person_ids that could ever have
    # survived run_near_judge (a failed/omitted batch never adds an entry) and
    # validate_near_verdicts (evidence + confidence + relaxed-criterion
    # checks) — asserting against it here means a future edit that tries to
    # rank/display an unjudged candidate fails loudly instead of shipping a
    # silent local-only recommendation.
    # a plain ``assert`` is stripped under ``python -O`` — this invariant must
    # hold even then, so it is an explicit check + raise, not an assertion.
    pid_with_verdict = set(validated)
    for r in capped:
        if r.candidate.person.id not in pid_with_verdict or r.verdict is None \
                or r.verdict.get("useful_near_match") is not True:
            raise RuntimeError(
                "near-match invariant violated: a candidate without a validated "
                "useful_near_match=true verdict reached the output stage "
                f"(person_id={r.candidate.person.id!r})"
            )

    out: list[tuple[ScoredCandidate, dict]] = []
    for r in capped:
        display = replace(r.candidate.scored, qualification=Qualification.NOT_MATCH)
        out.append((display, r.verdict))

    relation_source_counts: dict[str, int] = {}
    for v in validated.values():
        src = v.get("relation_source") or "n/a"
        relation_source_counts[src] = relation_source_counts.get(src, 0) + 1

    meta = NearMatchMetadata(
        pool_size=len(pool), judge=judge_run.metadata,
        validated_count=len(validated), final_count=len(out), llm_used=llm_used,
        rejected_count=len(pool) - len(validated),
        relation_source_counts=relation_source_counts,
        rejection_reason_counts=rejection_reason_counts,
        total_ms=round((time.perf_counter() - started) * 1000.0, 1),
    )
    _log_summary(meta)
    return out, meta


def _log_summary(meta: NearMatchMetadata) -> None:
    """PART 9 — one safe summary line per search. Counters only: never a
    profile payload, prompt, API key, or auth header."""
    log.info(
        "near_match_service: near_pool_candidates=%d near_candidates_judged=%d "
        "near_candidates_accepted=%d near_candidates_rejected=%d near_llm_batches=%d "
        "near_llm_status=%s near_relation_source_counts=%s near_rejection_reasons=%s near_total_ms=%.1f",
        meta.pool_size, meta.judge.candidates_judged, meta.validated_count, meta.rejected_count,
        meta.judge.batches, meta.judge.status, meta.relation_source_counts,
        meta.rejection_reason_counts, meta.total_ms,
    )
    if not meta.llm_used and meta.pool_size:
        log.info(
            "near_match_service: near-match LLM judge unusable (status=%s) — near_matches=[], "
            "no heuristic fallback shown", meta.judge.status,
        )
