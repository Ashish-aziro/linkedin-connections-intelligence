"""General query-understanding robustness — the tolerant LLM transport layer.

A mostly-correct Anthropic interpretation must survive harmless JSON transport
imperfections (operator/value/modality = null, missing id, string weight,
`values` as a bare string, alias type spellings) instead of failing strict
validation, retrying three times, and silently falling back to the regex parser.

No live LLM calls: only the Anthropic TRANSPORT OUTPUT is mocked (as a
``LenientSearchPlan``). Everything after it — structural repair, strict
``ParsedSearchQuery`` validation, the deterministic fact safety-net, intent
finalization — runs for real.

Nothing here is keyed to a literal example word (Atlanta / Chicago / nonprofit /
engineer). The mock plans are hand-built structural shapes; the assertions check
structure, not query strings.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.constants import CriterionType, Modality, Operator
from app.schemas import LenientSearchPlan, ParsedSearchQuery
from app.services import query_interpreter as qi
from app.services.query_transport import repair_plan


@pytest.fixture(autouse=True)
def _llm_on(monkeypatch):
    monkeypatch.setattr(settings, "llm_query_interpretation", True)


def _mock_llm(monkeypatch, plan_dict: dict, *, provider="anthropic:paid",
              model="claude-haiku-4-5-20251001", calls: list | None = None):
    """Patch the query-interpretation transport call. The router hands us the
    schema (``LenientSearchPlan``); we return a parsed instance of it, exactly
    like a real successful call."""
    def fake(system, user, schema, **kw):  # noqa: ARG001
        if calls is not None:
            calls.append(1)
        return schema.model_validate(plan_dict), provider, model
    monkeypatch.setattr("app.services.query_interpreter.generate_structured", fake)


def _by_type(parsed: ParsedSearchQuery, ctype: str):
    return [c for c in parsed.criteria if c.type == ctype]


# ─────────────────────────── TEST A — nullable transport fields ───────────────────────────


def test_A_nullable_fields_do_not_retry_or_fall_back(monkeypatch):
    calls: list = []
    _mock_llm(monkeypatch, {
        "intent": "find_people",
        "criteria": [
            {"type": "role_function", "concept": "software engineering",
             "operator": None, "value": None, "id": None, "required": True, "weight": "70"},
            {"type": "skill", "value": "kubernetes", "operator": None,
             "modality": None, "required": False, "weight": "30"},
        ],
    }, calls=calls)

    parsed, provider, model = qi.interpret_query("people who build platforms and know kubernetes")

    assert provider == "anthropic:paid"        # §13 — Anthropic drove the plan
    assert model == "claude-haiku-4-5-20251001"
    assert len(calls) == 1                     # §7/§26 — exactly one LLM call, no retry loop
    assert len(parsed.criteria) == 2
    assert all(c.operator in (Operator.ANY_OF, Operator.ALL_OF, Operator.NOT) for c in parsed.criteria)
    assert all(c.modality in (Modality.CERTAIN, Modality.POSSIBLE) for c in parsed.criteria)
    assert all(c.id for c in parsed.criteria)  # synthesized where missing
    assert abs(sum(c.weight for c in parsed.criteria) - 100.0) < 0.5


def test_A_unit_repair_is_purely_structural(monkeypatch):
    lp = LenientSearchPlan.model_validate({"criteria": [
        {"type": "role", "concept": "data engineering", "operator": None, "value": None},
    ]})
    repaired, notes = repair_plan(lp)
    p = ParsedSearchQuery.model_validate(repaired)
    assert p.criteria[0].type == CriterionType.ROLE_FUNCTION   # alias "role" -> role_function
    assert p.criteria[0].concept == "data engineering"
    assert p.criteria[0].value == "data engineering"           # back-compat single value
    assert p.criteria[0].operator == Operator.ANY_OF
    assert notes  # repair notes exist (log-only)


# ─────────────────────────── TEST B — location + role ───────────────────────────


def test_B_role_and_location_both_survive(monkeypatch):
    # conceptually "senior engineers in <city>" — one harmless null field
    _mock_llm(monkeypatch, {
        "intent": "find_people",
        "criteria": [
            {"id": "role", "type": "role_function", "concept": "software engineering",
             "required": True, "weight": 45, "operator": None},
            {"id": "sen", "type": "seniority", "value": "senior", "required": True, "weight": 25},
            {"id": "loc", "type": "location", "value": "atlanta", "required": True, "weight": 30},
        ],
    })
    parsed, provider, _ = qi.interpret_query("senior engineers in atlanta")

    assert provider == "anthropic:paid"
    loc = _by_type(parsed, CriterionType.LOCATION)
    role = _by_type(parsed, CriterionType.ROLE_FUNCTION)
    assert loc and role                                  # both dimensions survived
    assert loc[0].required is True                       # meaning made it a hard constraint
    assert "atlanta" in [v.lower() for v in (loc[0].values or [loc[0].value])]


# ─────────────────────────── TEST C — company history ───────────────────────────


def test_C_two_company_and_semantics(monkeypatch):
    _mock_llm(monkeypatch, {
        "criteria": [{
            "id": "co", "type": "past_company", "values": ["Amazon", "Microsoft"],
            "operator": "ALL_OF", "scope": "any_experience", "required": True, "weight": 100,
        }],
    })
    parsed, _, _ = qi.interpret_query("people who worked at both Amazon and Microsoft")
    co = _by_type(parsed, CriterionType.PAST_COMPANY)
    assert co
    assert set(v.lower() for v in co[0].values) == {"amazon", "microsoft"}
    assert co[0].operator == Operator.ALL_OF             # AND, not OR
    assert co[0].required is True


def test_C_company_or_semantics(monkeypatch):
    _mock_llm(monkeypatch, {
        "criteria": [{
            "id": "co", "type": "current_company", "values": "Google, Meta",
            "operator": "ANY_OF", "scope": "current_company", "required": True, "weight": 100,
        }],
    })
    parsed, _, _ = qi.interpret_query("engineers currently at Google or Meta")
    co = _by_type(parsed, CriterionType.CURRENT_COMPANY)
    assert co and set(v.lower() for v in co[0].values) == {"google", "meta"}
    assert co[0].operator == Operator.ANY_OF


# ─────────────────────────── TEST D — semantic recommendation ───────────────────────────


def test_D_semantic_recommendation_preserved(monkeypatch):
    _mock_llm(monkeypatch, {
        "intent": "professional_recommendation",
        "criteria": [
            {"id": "fund", "type": "professional_concept",
             "concept": "helping founders raise venture funding", "required": True,
             "weight": 55, "operator": None, "value": None},
            {"id": "ai", "type": "industry_experience",
             "concept": "experience in the AI / machine-learning industry", "required": False,
             "weight": 45},
        ],
    })
    parsed, provider, _ = qi.interpret_query("who could help me raise funding for an AI startup")
    assert provider == "anthropic:paid"
    concepts = [c for c in parsed.criteria
                if c.type in (CriterionType.PROFESSIONAL_CONCEPT, CriterionType.INDUSTRY_EXPERIENCE,
                              CriterionType.SEMANTIC_CONCEPT)]
    assert len(concepts) == 2
    assert any("fund" in (c.concept or "").lower() for c in concepts)   # concept text kept verbatim
    assert all(c.type != CriterionType.KEYWORD for c in parsed.criteria)  # not a literal-text search


# ─────────────────────────── TEST E — cross-domain AND ───────────────────────────


def test_E_cross_domain_stays_two_required_criteria(monkeypatch):
    _mock_llm(monkeypatch, {
        "criteria": [
            {"id": "a", "type": "industry_experience", "concept": "healthcare industry experience",
             "required": True, "weight": 50, "operator": None},
            {"id": "b", "type": "professional_concept", "concept": "cybersecurity expertise",
             "required": True, "weight": 50, "operator": None},
        ],
    })
    parsed, _, _ = qi.interpret_query("people with healthcare and cybersecurity experience")
    req = [c for c in parsed.criteria if c.required]
    # each domain is its OWN required criterion (not one blended concept)
    assert any(c.required and "healthcare" in (c.concept or "").lower()
               and "cyber" not in (c.concept or "").lower() for c in parsed.criteria)
    assert any(c.required and "cyber" in (c.concept or "").lower()
               and "healthcare" not in (c.concept or "").lower() for c in parsed.criteria)
    assert len(req) >= 2
    # never collapsed into a single ANY_OF criterion carrying both domains
    for c in parsed.criteria:
        vt = " ".join(c.values or []).lower()
        assert not (c.operator == Operator.ANY_OF and "healthcare" in vt and "cyber" in vt)


# ─────────────────────────── TEST F — exclusions ───────────────────────────


def test_F_not_operator_survives_transport(monkeypatch):
    _mock_llm(monkeypatch, {
        "criteria": [
            {"id": "skill", "type": "professional_concept", "concept": "distributed systems",
             "required": True, "weight": 60, "operator": None},
            {"id": "excl", "type": "current_company", "value": "Amazon", "operator": "NOT",
             "scope": "current_company", "required": True, "weight": 40},
        ],
    })
    parsed, _, _ = qi.interpret_query("distributed systems engineers not currently at Amazon")
    nots = [c for c in parsed.criteria if c.operator == Operator.NOT]
    assert nots
    assert "amazon" in [v.lower() for v in (nots[0].values or [nots[0].value])]


# ─────────────────────────── TEST G — modality ───────────────────────────


def test_G_possible_modality_is_a_soft_criterion(monkeypatch):
    _mock_llm(monkeypatch, {
        "criteria": [
            {"id": "core", "type": "role_function", "concept": "compliance / legal role",
             "required": True, "weight": 70},
            {"id": "soft", "type": "professional_concept", "concept": "HIPAA compliance familiarity",
             "modality": "possible", "required": False, "weight": 30, "operator": None},
        ],
    })
    parsed, _, _ = qi.interpret_query("compliance people who might know HIPAA")
    soft = [c for c in parsed.criteria if "hipaa" in (c.concept or "").lower()]
    assert soft
    assert soft[0].modality == Modality.POSSIBLE
    assert soft[0].required is False


# ─────────────────────────── TEST H — capitalization invariance ───────────────────────────


@pytest.mark.parametrize("q", [
    "senior engineers in Atlanta",
    "senior engineers in atlanta",
    "SENIOR ENGINEERS IN ATLANTA",
])
def test_H_capitalization_does_not_change_explicit_constraints(monkeypatch, q):
    # identical mock plan for every casing — the ONLY variable is the query string
    _mock_llm(monkeypatch, {
        "criteria": [
            {"id": "role", "type": "role_function", "concept": "software engineering",
             "required": True, "weight": 60},
            {"id": "loc", "type": "location", "value": "Atlanta", "required": True, "weight": 40},
        ],
    })
    parsed, _, _ = qi.interpret_query(q)
    loc = _by_type(parsed, CriterionType.LOCATION)
    assert loc, q
    assert loc[0].required is True, q
    assert "atlanta" in [v.lower() for v in (loc[0].values or [loc[0].value])], q


def test_H_deterministic_path_also_casing_invariant():
    # no LLM at all — the fact safety-net must still catch a lower-case place
    from app.services.query_facts import extract_facts
    hi = extract_facts("senior engineers in Atlanta")
    lo = extract_facts("senior engineers in atlanta")
    up = extract_facts("SENIOR ENGINEERS IN ATLANTA")
    def loc_vals(fs):
        return {v.lower() for c in fs.criteria if c.type == CriterionType.LOCATION
                for v in (c.values or [c.value])}
    assert loc_vals(hi) == loc_vals(lo) == loc_vals(up) == {"atlanta"}


def test_H_in_domain_is_not_mistaken_for_a_place():
    from app.services.query_facts import extract_facts
    for q in ("people in sales", "leaders in engineering", "experts in security",
              "people who work in big tech", "someone in product management"):
        locs = [c for c in extract_facts(q).criteria if c.type == CriterionType.LOCATION]
        assert not locs, q


# ─────────────────────────── TEST I — ambiguous query ───────────────────────────


def test_I_ambiguity_lowers_confidence_not_invents_specifics(monkeypatch):
    _mock_llm(monkeypatch, {
        "criteria": [{"id": "vague", "type": "semantic_concept",
                      "concept": "generally strong / impressive professional profile",
                      "required": False, "weight": 100, "operator": None, "value": None}],
        "interpretation_confidence": 0.9,
    })
    parsed, _, _ = qi.interpret_query("who are the best people in my network")
    assert parsed.interpretation_confidence <= 0.7          # not a confident reading
    # no invented concrete facts
    assert not _by_type(parsed, CriterionType.LOCATION)
    assert not _by_type(parsed, CriterionType.CURRENT_COMPANY)
    assert not _by_type(parsed, CriterionType.PAST_COMPANY)


# ─────────────────────────── TEST J — unrecoverable LLM output ───────────────────────────


def test_J_no_criteria_falls_back_to_deterministic(monkeypatch):
    _mock_llm(monkeypatch, {"criteria": []})
    parsed, provider, model = qi.interpret_query("people who previously worked at Amazon")
    assert provider == "deterministic" and model is None
    assert parsed.criteria                                  # deterministic parser produced something
    assert any(c.type == CriterionType.PAST_COMPANY for c in parsed.criteria)


def test_J_llm_returns_none_falls_back(monkeypatch):
    monkeypatch.setattr("app.services.query_interpreter.generate_structured",
                        lambda *a, **k: None)
    parsed, provider, _ = qi.interpret_query("engineers currently at Google who know Java")
    assert provider == "deterministic"
    assert any(c.type == CriterionType.CURRENT_COMPANY for c in parsed.criteria)


def test_J_garbage_criteria_are_dropped_then_fallback(monkeypatch):
    # every criterion is structurally empty -> repair drops them all -> no criteria
    _mock_llm(monkeypatch, {"criteria": [
        {"type": None, "value": None, "values": None, "concept": None, "operator": None},
        {"type": "skill"},
    ]})
    parsed, provider, _ = qi.interpret_query("who could be a CTO")
    # falls back safely, does not crash
    assert provider == "deterministic"
    assert parsed.criteria


# ─────────────────────────── §13 — Anthropic drives the plan for ordinary queries ───────────────────────────


def test_ordinary_valid_query_logs_anthropic_not_deterministic(monkeypatch, caplog):
    _mock_llm(monkeypatch, {
        "criteria": [{"id": "k8s", "type": "skill", "value": "kubernetes",
                      "required": False, "weight": 100, "operator": None}],
    })
    import logging
    with caplog.at_level(logging.INFO, logger="app.query"):
        parsed, provider, _ = qi.interpret_query("who knows kubernetes")
    assert provider == "anthropic:paid"
    assert "falling back to deterministic parser" not in caplog.text
