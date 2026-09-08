"""Output-token budgeting for batched judge/audit requests.

V4 PART 6 B6/B7/B8 — recalibrated against LIVE Claude Sonnet 5 on real evidence
packets. Findings from the tuning loop:

  * compact JSON is token-inefficient and Sonnet 5 fills confidence + several
    long real evidence refs + a short reason, so a verdict runs ~250 output
    tokens (the pre-PART-6 estimate assumed 55 and truncated on nearly every
    batch);
  * Sonnet 5 generation latency is strongly super-linear in output size — a
    truncated 10-candidate batch cost ~30 s + 3 split-retry calls; a clean
    ~6-candidate batch costs ~15–20 s.

So ``plan_batch_size`` targets a bounded per-call output (``_BATCH_OUTPUT_CEILING``)
— a normal 1-criterion query judges in batches of ~6, a dense multi-criteria
query in batches of ~2–3 — each a single non-truncating request. The adaptive
splitter remains only as an emergency net.
"""
from __future__ import annotations

#: measured compact judge verdict cost (Sonnet 5, live, on REAL evidence
#: packets). Compact JSON is token-INEFFICIENT (~1.2 chars/token) and against a
#: rich packet Sonnet 5 fills confidence + several long evidence refs + a short
#: reason — ~260 output tokens per verdict observed. Budget for that so a batch
#: does NOT truncate + split (a truncated 10-candidate batch cost ~30 s + 3
#: retry calls; four clean 4-candidate batches cost ~15 s each).
_PER_CRITERION_TOKENS = 250
#: per-person shell: {"person_id":"...","criteria":[...]}
_PER_PERSON_TOKENS = 22
_BASE_TOKENS = 60
#: proportional safety margin (natural variance, an extra ref, a short reason)
_MARGIN_FRACTION = 0.28
_MIN_MARGIN = 450
#: floor so even a 1-person/1-criterion request has room
_MIN_TOKENS = 700
#: audit reviews carry status_review + a per-criterion reason + a display_reason
_PER_AUDIT_CRITERION_TOKENS = 320

#: absolute ceiling regardless of estimate — a real overflow is still caught as
#: truncation + adaptively split; this just avoids a wasteful request size.
MAX_OUTPUT_TOKENS = 12000
#: proactive batch-sizing target — keep a batch's estimated OUTPUT near this so
#: each call is small, fast (~10–15 s) and non-truncating. Sonnet 5 generation
#: latency is strongly super-linear in output size, so several small clean
#: batches beat one large batch that truncates and splits.
_BATCH_OUTPUT_CEILING = 2600


def _total_criteria(people_count: int, criteria_counts: list[int] | int) -> int:
    if isinstance(criteria_counts, int):
        return criteria_counts * max(1, people_count)
    return sum(criteria_counts) if criteria_counts else max(1, people_count)


def _estimate(people_count: int, total_criteria: int, per_criterion: int) -> int:
    people_count = max(1, people_count)
    subtotal = _BASE_TOKENS + people_count * _PER_PERSON_TOKENS + total_criteria * per_criterion
    margin = max(_MIN_MARGIN, int(subtotal * _MARGIN_FRACTION))
    return min(MAX_OUTPUT_TOKENS, max(_MIN_TOKENS, subtotal + margin))


def estimate_judge_output_tokens(people_count: int, criteria_counts: list[int] | int) -> int:
    """``criteria_counts``: total unresolved criteria across the batch, either
    a list (one count per person) or a uniform int (criteria per person)."""
    return _estimate(people_count, _total_criteria(people_count, criteria_counts), _PER_CRITERION_TOKENS)


def estimate_audit_output_tokens(people_count: int, criteria_counts: list[int] | int) -> int:
    return _estimate(people_count, _total_criteria(people_count, criteria_counts), _PER_AUDIT_CRITERION_TOKENS)


def plan_batch_size(default_size: int, avg_criteria_per_person: float, *, min_size: int = 1,
                    per_criterion: int | None = None) -> int:
    """Proactively size a batch BEFORE sending it (B7/B8). Pick the largest size
    <= ``default_size`` whose estimated output stays near ``_BATCH_OUTPUT_CEILING``
    — so each call is small, fast and non-truncating. A dense multi-criteria
    query (or the audit, whose per-review cost is higher) gets a smaller batch.
    Pass ``per_criterion=_PER_AUDIT_CRITERION_TOKENS`` from the auditor."""
    default_size = max(min_size, default_size)
    pc = per_criterion if per_criterion is not None else _PER_CRITERION_TOKENS
    avg = max(1.0, avg_criteria_per_person)
    per_person_output = _PER_PERSON_TOKENS + avg * pc
    fit = int((_BATCH_OUTPUT_CEILING - _BASE_TOKENS - _MIN_MARGIN) / max(1.0, per_person_output))
    return max(min_size, min(default_size, fit))
