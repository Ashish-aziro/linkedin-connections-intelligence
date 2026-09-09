"""hardening PART 3/24 — output-token budgeting scales with people x criteria.

The bug this replaces: ``min(4000, 400 + 320 * len(packets))`` depended ONLY on
candidate count, completely ignoring how many required criteria each person
had — a single person with 4 required criteria got no more budget than one
with a single criterion, which is exactly what truncated in the live failure.
These tests prove the new estimator scales with BOTH dimensions, leaves real
headroom over the actual serialized payload size, and stays within
provider-safe bounds.
"""
from __future__ import annotations

import json

from app.services.llm.token_estimate import (
    MAX_OUTPUT_TOKENS,
    estimate_audit_output_tokens,
    estimate_judge_output_tokens,
    plan_batch_size,
)

def _compact_verdict(n_refs: int = 1) -> dict:
    return {
        "criterion_id": "c1", "status": "true", "confidence": 0.91,
        "supporting_refs": [f"exp:{i}" for i in range(n_refs)],
        "contradicting_refs": [], "reason": "",
    }


# ─────────────────────── judge estimator: people x criteria scaling ───────────────────────


def test_single_person_single_criterion_gets_the_floor_or_more():
    est = estimate_judge_output_tokens(1, 1)
    assert est >= 400  # the documented minimum


def test_single_person_four_criteria_has_real_headroom_over_the_actual_payload():
    """The exact live-failure shape: 1 person, 4 required criteria, which
    truncated at the OLD formula's flat max_tokens=720 regardless of criteria
    count. The new estimator must leave generous headroom above the ACTUAL
    serialized size of 4 compact verdicts (not just be numerically bigger than
    the old, criteria-blind formula, which is not itself a meaningful bound)."""
    payload = {"people": [{"person_id": "p0", "criteria": [_compact_verdict() for _ in range(4)]}]}
    actual_tokens = len(json.dumps(payload)) / 3
    new = estimate_judge_output_tokens(1, 4)
    assert new >= actual_tokens * 2  # at least 2x the real need -- real headroom, not a razor's edge


def test_output_estimate_grows_with_criteria_count_for_a_fixed_people_count():
    one = estimate_judge_output_tokens(1, 1)
    four = estimate_judge_output_tokens(1, 4)
    ten = estimate_judge_output_tokens(1, 10)
    assert one < four < ten


def test_output_estimate_grows_with_people_count_for_a_fixed_criteria_count():
    one_person = estimate_judge_output_tokens(1, [1])
    ten_people = estimate_judge_output_tokens(10, [1] * 10)
    assert ten_people > one_person


def test_four_people_four_criteria_each_dwarfs_one_person_four_criteria():
    est_1x4 = estimate_judge_output_tokens(1, 4)
    est_4x4 = estimate_judge_output_tokens(4, [4, 4, 4, 4])
    assert est_4x4 > est_1x4


def test_ten_people_one_criterion_each_is_between_the_extremes():
    est_1x1 = estimate_judge_output_tokens(1, 1)
    est_10x1 = estimate_judge_output_tokens(10, [1] * 10)
    est_1x4 = estimate_judge_output_tokens(1, 4)
    assert est_1x1 < est_10x1
    # 10 people x 1 criterion has more total (person,criterion) pairs than
    # 1 person x 4 criteria (10 vs 4) so it must cost at least as much
    assert est_10x1 >= est_1x4


def test_never_exceeds_the_absolute_ceiling():
    huge = estimate_judge_output_tokens(500, [20] * 500)
    assert huge <= MAX_OUTPUT_TOKENS


def test_criteria_counts_as_uniform_int_matches_equivalent_list():
    as_int = estimate_judge_output_tokens(5, 3)          # 3 criteria/person uniformly
    as_list = estimate_judge_output_tokens(5, [3] * 5)
    assert as_int == as_list


# ─────────────────────── audit estimator ───────────────────────


def test_audit_estimate_also_scales_with_criteria_not_just_people():
    """The audit path had the analogous bug (§3 review): budgeted mainly by
    packet count. Must also scale up as criteria count grows."""
    one = estimate_audit_output_tokens(1, 1)
    four = estimate_audit_output_tokens(1, 4)
    ten = estimate_audit_output_tokens(1, 10)
    assert one < four < ten


def test_audit_per_criterion_cost_exceeds_judge_per_criterion_cost():
    """Audit reviews carry a longer per-criterion reason than compact judge
    verdicts -- for the same shape, audit should never estimate LESS."""
    judge_est = estimate_judge_output_tokens(3, [4, 4, 4])
    audit_est = estimate_audit_output_tokens(3, [4, 4, 4])
    assert audit_est >= judge_est


def test_audit_never_exceeds_the_absolute_ceiling():
    huge = estimate_audit_output_tokens(500, [20] * 500)
    assert huge <= MAX_OUTPUT_TOKENS


# ─────────────────────── proactive batch sizing ───────────────────────


def test_plan_batch_size_unchanged_for_simple_low_criteria_queries():
    assert plan_batch_size(10, 1.0) == 10
    assert plan_batch_size(10, 1.5) == 10


def test_plan_batch_size_shrinks_as_criteria_density_rises():
    simple = plan_batch_size(10, 1.0)
    moderate = plan_batch_size(10, 2.5)
    dense = plan_batch_size(10, 5.0)
    very_dense = plan_batch_size(10, 8.0)
    assert simple > moderate > dense > very_dense
    assert very_dense >= 1  # never below min_size


def test_plan_batch_size_never_exceeds_the_default_size():
    assert plan_batch_size(3, 8.0) <= 3


# ─────────────────────── serialized compact-verdict size sanity ───────────────────────


def test_estimated_budget_comfortably_covers_the_real_compact_serialized_size():
    """A sanity check tying the estimator to reality: the ACTUAL serialized
    size of N compact verdicts (in characters, ~4 chars/token) must fit well
    within the token budget the estimator hands out for that shape."""
    for people, criteria in [(1, 1), (1, 4), (4, 1), (4, 4), (10, 1)]:
        payload = {
            "people": [
                {"person_id": f"p{i}", "criteria": [_compact_verdict() for _ in range(criteria)]}
                for i in range(people)
            ]
        }
        serialized_chars = len(json.dumps(payload))
        approx_tokens = serialized_chars / 3  # conservative (real tokenizers do better)
        budget = estimate_judge_output_tokens(people, criteria)
        assert budget >= approx_tokens, f"people={people} criteria={criteria}: budget {budget} < ~{approx_tokens:.0f} tokens needed"
