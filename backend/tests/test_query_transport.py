"""V4 PART 6 B3 — tolerant LLM transport → strict search plan.

A nullable / alternative field in the model's JSON must be repaired locally and
re-validated against the strict ``ParsedSearchQuery`` — never cause three
identical Anthropic retries, never a silent schema failure.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import LenientSearchPlan, ParsedSearchQuery
from app.services.query_transport import repair_plan


def _strict(raw: dict) -> ParsedSearchQuery:
    lenient = LenientSearchPlan.model_validate(raw)
    repaired, _notes = repair_plan(lenient)
    return ParsedSearchQuery.model_validate(repaired)


def test_company_category_concept_with_value_null():
    p = _strict({"criteria": [
        {"id": "c1", "type": "company_category", "concept": "nonprofit",
         "value": None, "required": True, "weight": 50},
    ]})
    c = p.criteria[0]
    assert c.type == "company_category"
    assert c.concept == "nonprofit"
    assert c.value == "nonprofit"  # cross-filled, strict schema satisfied


def test_industry_experience_concept_with_value_and_values_null():
    p = _strict({"criteria": [
        {"id": "c2", "type": "industry_experience",
         "concept": "nonprofit sector experience", "value": None, "values": None,
         "required": True, "weight": 50},
    ]})
    c = p.criteria[0]
    assert c.type == "industry_experience"
    assert c.concept == "nonprofit sector experience"
    assert c.values == ["nonprofit sector experience"]


def test_location_with_plain_string_value():
    p = _strict({"criteria": [
        {"id": "loc", "type": "location", "value": "Chicago", "required": True, "weight": 100},
    ]})
    assert p.criteria[0].value == "Chicago"
    assert p.criteria[0].type == "location"


def test_multi_value_any_of_preserved():
    p = _strict({"criteria": [
        {"id": "co", "type": "past_company", "values": ["Amazon", "Google"],
         "operator": "ANY_OF", "value": None, "required": True, "weight": 100},
    ]})
    c = p.criteria[0]
    assert c.values == ["Amazon", "Google"]
    assert c.operator == "ANY_OF"


def test_all_of_and_not_operators_preserved():
    p = _strict({"criteria": [
        {"id": "a", "type": "past_company", "values": ["Amazon", "Microsoft"],
         "operator": "ALL_OF", "value": None, "required": True, "weight": 60},
        {"id": "b", "type": "current_company", "value": "Amazon", "operator": "NOT",
         "required": True, "weight": 40},
    ]})
    assert p.criteria[0].operator == "ALL_OF"
    assert p.criteria[1].operator == "NOT"


def test_missing_id_is_synthesized():
    p = _strict({"criteria": [
        {"type": "role_function", "concept": "software engineering", "value": None,
         "required": True, "weight": 100},
    ]})
    assert p.criteria[0].id  # non-empty


def test_values_as_bare_string_is_split():
    p = _strict({"criteria": [
        {"id": "x", "type": "location", "values": "Memphis, Nashville", "value": None,
         "operator": "ANY_OF", "required": True, "weight": 100},
    ]})
    assert p.criteria[0].values == ["Memphis", "Nashville"]


def test_empty_criteria_are_dropped():
    lenient = LenientSearchPlan.model_validate({"criteria": [
        {"type": "keyword"}, {"type": "skill", "value": None, "values": None, "concept": None},
    ]})
    repaired, notes = repair_plan(lenient)
    assert repaired["criteria"] == []
    assert any("dropped" in n for n in notes)


def test_truly_invalid_plan_raises_for_the_caller_to_fall_back():
    lenient = LenientSearchPlan.model_validate({"criteria": ["garbage", None, 7]})
    repaired, _ = repair_plan(lenient)
    with pytest.raises(ValidationError):
        ParsedSearchQuery.model_validate(repaired)  # -> caller does deterministic fallback (B2: AI_UNAVAILABLE)


def test_non_list_criteria_tolerated():
    lenient = LenientSearchPlan.model_validate({"criteria": {"type": "location", "value": "Denver"}})
    repaired, _ = repair_plan(lenient)
    p = ParsedSearchQuery.model_validate(repaired)
    assert p.criteria[0].value == "Denver"


def test_weights_renormalised_to_100_after_repair():
    p = _strict({"criteria": [
        {"id": "a", "type": "location", "value": "Chicago", "required": True, "weight": 3},
        {"id": "b", "type": "industry_experience", "concept": "nonprofit", "value": None,
         "required": True, "weight": 1},
    ]})
    assert abs(sum(c.weight for c in p.criteria) - 100.0) < 0.5
