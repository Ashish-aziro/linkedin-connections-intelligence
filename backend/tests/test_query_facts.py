"""Query Understanding V4 — deterministic layer (V4 §14–§19, §53, §61, §64).

All tests run the deterministic path (conftest sets LLM_QUERY_INTERPRETATION=false),
so they prove the fallback + fact-merge behaviour with no LLM at all.
"""
from __future__ import annotations

from app.constants import CriterionType, Operator
from app.services.query_facts import extract_facts, strip_context
from app.services.query_interpreter import interpret_query


def _plan(q):
    parsed, _provider, _ = interpret_query(q)
    return parsed


def _by_type(plan, t):
    return [c for c in plan.criteria if c.type == t]


def _vals(c):
    return {v.lower() for v in (c.values or ([c.value] if c.value else []))}


# ─────────────────────── §14 context vs candidate ───────────────────────


def test_networking_event_is_context_not_a_skill():
    plan = _plan("Who should I invite to a CXO networking event in Memphis or Nashville?")
    assert "networking" not in {c.value for c in plan.criteria}
    assert not _by_type(plan, CriterionType.SKILL)
    assert "networking" in plan.context.get("purpose", "").lower()


def test_networking_expertise_IS_a_criterion():
    plan = _plan("people with professional networking expertise")
    hits = _by_type(plan, CriterionType.SKILL) + _by_type(plan, CriterionType.SEMANTIC_CONCEPT)
    assert any("networking" in (c.value or c.concept or "").lower() for c in hits)


def test_strip_context_keeps_real_requirements():
    cleaned, ctx = strip_context("invite to a CXO networking event in Nashville")
    assert "cxo" in cleaned.lower()
    assert "networking" not in cleaned.lower()
    assert ctx["purpose"]


# ─────────────────────── §16 deterministic fallback ───────────────────────


def test_big_tech_in_bay_area():
    plan = _plan("people working in big tech in Bay Area")
    cat = _by_type(plan, CriterionType.COMPANY_CATEGORY)
    loc = _by_type(plan, CriterionType.LOCATION)
    assert cat and "big tech" in (cat[0].concept or "").lower() and cat[0].required
    assert loc and "bay area" in _vals(loc[0]) and loc[0].required


def test_former_amazon_now_at_startups():
    plan = _plan("Former Amazon people now at startups")
    past = _by_type(plan, CriterionType.PAST_COMPANY)
    cat = _by_type(plan, CriterionType.COMPANY_CATEGORY)
    assert past and "amazon" in _vals(past[0]) and past[0].required
    assert cat and "startup" in (cat[0].concept or "") and cat[0].required


def test_cxos_in_nashville_both_required():
    plan = _plan("CXOs in Nashville")
    sen = _by_type(plan, CriterionType.SENIORITY)
    loc = _by_type(plan, CriterionType.LOCATION)
    assert sen and sen[0].required
    assert loc and loc[0].required and "nashville" in _vals(loc[0])


def test_memphis_or_nashville_is_any_of():
    plan = _plan("executives in Memphis or Nashville")
    loc = _by_type(plan, CriterionType.LOCATION)[0]
    assert _vals(loc) == {"memphis", "nashville"} and loc.operator == Operator.ANY_OF


def test_former_google_or_meta():
    plan = _plan("former Google or Meta engineers")
    past = _by_type(plan, CriterionType.PAST_COMPANY)[0]
    assert _vals(past) == {"google", "meta"} and past.operator == Operator.ANY_OF and past.required


def test_not_currently_at_amazon():
    plan = _plan("engineers not currently at Amazon")
    nots = [c for c in plan.criteria if c.operator == Operator.NOT]
    assert nots and "amazon" in _vals(nots[0])


def test_cxo_event_keeps_cxo_and_location_required():
    plan = _plan("Who should I invite to a CXO networking event in Memphis or Nashville?")
    sen = _by_type(plan, CriterionType.SENIORITY)
    loc = _by_type(plan, CriterionType.LOCATION)
    assert sen and sen[0].required and "cxo" in _vals(sen[0])
    assert loc and loc[0].required and _vals(loc[0]) == {"memphis", "nashville"}


# ─────────────────────── §18 interpretation summary + confidence ───────────────────────


def test_summary_and_confidence_present():
    plan = _plan("Former Amazon people now at startups")
    assert plan.interpretation_summary.startswith("Interpreted as")
    assert 0.0 <= plan.interpretation_confidence <= 1.0


def test_ambiguous_query_has_lower_confidence():
    vague = _plan("people who worked in tech")
    precise = _plan("former Google or Meta engineers")
    assert vague.interpretation_confidence < precise.interpretation_confidence


# ─────────────────────── §53 ambiguity — different plans ───────────────────────


def test_worked_in_tech_never_becomes_keyword_tech():
    a = _plan("people who worked in tech")
    b = _plan("people who worked at tech companies")
    assert not any(c.type == CriterionType.KEYWORD and (c.value or "").lower() == "tech"
                   for c in a.criteria + b.criteria)


# ─────────────────────── extract_facts unit ───────────────────────


def test_extract_facts_scopes_former_vs_current():
    fs = extract_facts("former Amazon people currently at Google")
    past = [c for c in fs.criteria if c.type == CriterionType.PAST_COMPANY]
    cur = [c for c in fs.criteria if c.type == CriterionType.CURRENT_COMPANY]
    assert past and "amazon" in _vals(past[0])
    assert cur and "google" in _vals(cur[0])


# ─────────────────────── B.5 — AND vs OR (V4 §3/§8/§34) ───────────────────────


def test_google_or_meta_is_any_of():
    past = _by_type(_plan("former Google or Meta engineers"), CriterionType.PAST_COMPANY)[0]
    assert _vals(past) == {"google", "meta"} and past.operator == Operator.ANY_OF and past.required


def test_amazon_and_microsoft_is_all_of():
    past = _by_type(_plan("people with Amazon and Microsoft experience"), CriterionType.PAST_COMPANY)[0]
    assert _vals(past) == {"amazon", "microsoft"} and past.operator == Operator.ALL_OF and past.required


def test_security_or_cloud_is_semantic_any_of():
    plan = _plan("security or cloud experts")
    sem = [c for c in plan.criteria if c.type in (CriterionType.PROFESSIONAL_CONCEPT, CriterionType.ROLE_FUNCTION)]
    assert sem and sem[0].operator == Operator.ANY_OF
    assert {"security", "cloud"} <= _vals(sem[0])
    assert not any(c.type == CriterionType.KEYWORD for c in plan.criteria)


def test_ai_and_security_is_semantic_all_of():
    plan = _plan("AI and security leaders")
    sem = [c for c in plan.criteria if c.type in (CriterionType.PROFESSIONAL_CONCEPT, CriterionType.ROLE_FUNCTION)]
    assert sem and sem[0].operator == Operator.ALL_OF
    assert {"ai", "security"} <= _vals(sem[0])


def test_not_currently_at_amazon_is_not_operator():
    nots = [c for c in _plan("engineers not currently at Amazon").criteria if c.operator == Operator.NOT]
    assert nots and "amazon" in _vals(nots[0]) and nots[0].required


# ─────────────────────── B.5 — role requiredness (V4 §7/§34) ───────────────────────


def test_software_engineers_at_fintech_both_required():
    plan = _plan("software engineers at fintech companies")
    role = [c for c in plan.criteria if c.type == CriterionType.ROLE_FUNCTION]
    fin = [c for c in plan.criteria if c.type in (CriterionType.COMPANY_CATEGORY, CriterionType.INDUSTRY_EXPERIENCE)
           and "fintech" in (c.concept or c.value or "").lower()]
    assert role and role[0].required, "role_function must be required without a 'must' word"
    assert fin and fin[0].required, "fintech employer must be required"
    assert not any(c.type == CriterionType.KEYWORD for c in plan.criteria)


def test_role_becomes_role_function_not_generic_title():
    plan = _plan("data scientists in Nashville")
    assert any(c.type == CriterionType.ROLE_FUNCTION for c in plan.criteria)


# ─────────────────────── B.5 — transition + years (deterministic emit) ───────────────────────


def test_moved_from_consulting_to_tech_emits_transition():
    tr = _by_type(_plan("people who moved from consulting to tech"), CriterionType.CAREER_TRANSITION)
    assert tr and tr[0].required and "consulting" in tr[0].concept.lower()


def test_ten_plus_years_backend_emits_years_experience():
    ye = _by_type(_plan("people with 10+ years of backend experience"), CriterionType.YEARS_EXPERIENCE)
    assert ye and ye[0].required and "10" in (ye[0].value + ye[0].concept)


# ─────────────────────── review findings D1/D2/F3 — geographic intent ───────────────────────
# "Anthropic is the only place that decides geographic intent; when geography
# isn't mentioned, it has zero influence" — this deterministic fact layer must
# never fabricate a required LOCATION from a skill/tool/domain word, must
# never force a hedged location back into a hard requirement, and must let an
# explicit "strictly/only in X" survive as a non-relaxable signal.


def test_experience_in_python_is_never_a_location():
    """review finding D1 — a skill/tool name after "in" is not a place."""
    for q in (
        "software engineers with experience in Python",
        "experts in Kubernetes networking",
        "tenure-track faculty in computational biology",
        "engineers in Salesforce, Docker",
    ):
        assert not _by_type(_plan(q), CriterionType.LOCATION), q


def test_ai_engineers_no_location_mentioned_has_no_location_criterion():
    """A query with no geography must introduce zero LOCATION criteria."""
    assert not _by_type(_plan("AI engineers"), CriterionType.LOCATION)


def test_ai_engineers_in_atlanta_preferably_is_not_required():
    """review finding D2 — hedging language keeps a mentioned location as a
    preference, never silently promoted back to a hard requirement."""
    loc = _by_type(_plan("AI engineers in Atlanta, preferably"), CriterionType.LOCATION)
    assert loc and loc[0].required is False and "atlanta" in _vals(loc[0])


def test_ai_engineers_strictly_in_atlanta_stays_required_and_geo_strict():
    """review finding F3 — an explicit "strictly in X" both stays required and
    is marked non-relaxable for Near Match purposes."""
    loc = _by_type(_plan("AI engineers strictly in Atlanta"), CriterionType.LOCATION)
    assert loc and loc[0].required is True and loc[0].geo_strict is True
    assert "atlanta" in _vals(loc[0])


def test_ai_engineers_in_atlanta_or_nearby_cities_allows_expansion():
    """An ordinary required location (even one that explicitly invites nearby
    cities) must NOT be geo_strict — nearby-city Near Match relaxation stays
    available."""
    loc = _by_type(_plan("AI engineers in Atlanta or nearby cities"), CriterionType.LOCATION)
    assert loc and loc[0].required is True and loc[0].geo_strict is False


def test_ordinary_in_atlanta_is_required_and_not_geo_strict():
    loc = _by_type(_plan("AI engineers in Atlanta"), CriterionType.LOCATION)
    assert loc and loc[0].required is True and loc[0].geo_strict is False


def test_fictional_city_name_still_recognised_as_a_location():
    """The gazetteer/denylist fix must not regress recall for a real place
    name simply because it is not in any finite reference list."""
    loc = _by_type(_plan("senior engineers in Fernbrook"), CriterionType.LOCATION)
    assert loc and "fernbrook" in _vals(loc[0])


def test_llm_preferred_location_is_not_overridden_by_the_fact_layer(monkeypatch):
    """review finding D2 — when the LLM correctly reads a location as
    preferred, the deterministic fact layer must never force it back to
    required, even though the raw text contains a bare 'in <place>'."""
    import app.services.query_interpreter as qi
    from app.config import settings

    monkeypatch.setattr(settings, "llm_query_interpretation", True)

    def fake_generate_structured(*args, **kwargs):
        from app.schemas import LenientSearchPlan

        return LenientSearchPlan.model_validate({
            "criteria": [
                {"id": "role", "type": "role_function", "concept": "AI engineering",
                 "required": True, "weight": 60},
                {"id": "loc", "type": "location", "value": "Atlanta",
                 "required": False, "weight": 40},
            ],
        }), "anthropic:paid", "claude-test"

    monkeypatch.setattr(qi, "generate_structured", fake_generate_structured)
    parsed, provider, _ = qi.interpret_query("AI engineers in Atlanta, preferably")
    assert provider == "anthropic:paid"
    loc = _by_type(parsed, CriterionType.LOCATION)
    assert loc and loc[0].required is False, "the LLM's preferred=false must survive the fact merge"
