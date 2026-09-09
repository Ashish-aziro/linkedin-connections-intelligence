"""Output-token budgeting for batched judge/audit requests (hardening PART 2).

The previous formula (``400 + 320 * len(packets)``) depended ONLY on candidate
count. A query with several REQUIRED criteria per person needs proportionally
more output — one person with 4 criteria still needs real room to finish valid
JSON, which is exactly what truncated at ``max_tokens=720`` in the live
failure this fixes. Budget now scales with (people x criteria), using the
COMPACT verdict shape's real per-item cost, plus a safety margin — and batch
SIZE is planned proactively from the same criteria density instead of waiting
for a truncation to discover the batch was too big.
"""
from __future__ import annotations

#: rough JSON-token cost of one compact criterion verdict, e.g.
#: {"criterion_id":"c1","status":"true","confidence":0.9,
#:  "supporting_refs":["exp:123"],"contradicting_refs":[],"reason":"..."}
_PER_CRITERION_TOKENS = 55
#: per-person shell: {"person_id":"...","criteria":[...]}
_PER_PERSON_TOKENS = 20
_BASE_TOKENS = 80
_SAFETY_MARGIN = 300
#: floor so even a 1-person/1-criterion request has room for natural variance
_MIN_TOKENS = 400
#: audit reviews carry a slightly longer per-criterion reason
_PER_AUDIT_CRITERION_TOKENS = 75

#: absolute ceiling regardless of estimate — keep well under typical provider
#: per-request output limits; a real overflow still gets caught as truncation
#: and adaptively split, this just avoids requesting a wasteful/unsupported size.
MAX_OUTPUT_TOKENS = 8000


def _total_criteria(people_count: int, criteria_counts: list[int] | int) -> int:
    if isinstance(criteria_counts, int):
        return criteria_counts * max(1, people_count)
    return sum(criteria_counts) if criteria_counts else max(1, people_count)


def estimate_judge_output_tokens(people_count: int, criteria_counts: list[int] | int) -> int:
    """``criteria_counts``: total unresolved criteria across the batch, either
    as a list (one count per person) or a uniform int (criteria per person)."""
    people_count = max(1, people_count)
    total_criteria = _total_criteria(people_count, criteria_counts)
    est = _BASE_TOKENS + people_count * _PER_PERSON_TOKENS + total_criteria * _PER_CRITERION_TOKENS
    return min(MAX_OUTPUT_TOKENS, max(_MIN_TOKENS, est + _SAFETY_MARGIN))


def estimate_audit_output_tokens(people_count: int, criteria_counts: list[int] | int) -> int:
    people_count = max(1, people_count)
    total_criteria = _total_criteria(people_count, criteria_counts)
    est = _BASE_TOKENS + people_count * _PER_PERSON_TOKENS + total_criteria * _PER_AUDIT_CRITERION_TOKENS
    return min(MAX_OUTPUT_TOKENS, max(_MIN_TOKENS, est + _SAFETY_MARGIN))


def plan_batch_size(default_size: int, avg_criteria_per_person: float, *, min_size: int = 1) -> int:
    """Proactively size a batch BEFORE sending it (hardening PART 4) — a query
    with many criteria per person gets a smaller batch from the first attempt,
    instead of discovering the batch was too big only after a truncation."""
    default_size = max(min_size, default_size)
    if avg_criteria_per_person <= 1.5:
        return default_size
    if avg_criteria_per_person <= 3:
        return max(min_size, min(default_size, 5))
    if avg_criteria_per_person <= 6:
        return max(min_size, min(default_size, 3))
    return max(min_size, min(default_size, 2))
