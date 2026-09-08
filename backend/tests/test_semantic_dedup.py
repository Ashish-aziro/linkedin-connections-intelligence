"""V4 PART 6 B4 — generic semantic-criterion deduplication.

Two criteria that express ONE user requirement collapse into one (across
scopes, broader scope surviving); genuinely distinct dimensions never collapse.
No query words or criterion-type pairs are hardcoded.
"""
from __future__ import annotations

from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.query_interpreter import _dedupe_semantic_duplicates, _same_requirement


def _c(**kw):
    kw.setdefault("weight", 25)
    kw.setdefault("id", "c" + str(abs(hash(str(sorted(kw.items())))) % 9999))
    return SearchCriterion(**kw)


def _dedupe(*crits):
    p = ParsedSearchQuery(criteria=list(crits))
    _dedupe_semantic_duplicates(p)
    return p.criteria


def test_two_nonprofit_industry_criteria_collapse():
    out = _dedupe(
        _c(id="n1", type="industry_experience",
           concept="professional experience working in the nonprofit sector",
           scope="any_experience", required=True, weight=50),
        _c(id="n2", type="industry_experience", concept="nonprofit sector experience",
           scope="career", required=True, weight=50),
    )
    assert len(out) == 1
    assert out[0].required is True


def test_fintech_industry_collapses_but_role_function_stays():
    out = _dedupe(
        _c(id="r", type="role_function", concept="software engineering",
           scope="career", required=True, weight=40),
        _c(id="f1", type="industry_experience",
           concept="professional experience in the fintech industry",
           scope="current_company", required=True, weight=30),
        _c(id="f2", type="industry_experience", concept="fintech sector experience",
           scope="career", required=True, weight=30),
    )
    types = sorted(c.type for c in out)
    assert types == ["industry_experience", "role_function"]
    fintech = next(c for c in out if c.type == "industry_experience")
    assert fintech.scope == "career"  # broader scope survives
    assert fintech.weight == 60.0


def test_cybersecurity_and_healthcare_stay_separate():
    out = _dedupe(
        _c(id="c1", type="industry_experience",
           concept="professional experience in the cybersecurity industry",
           scope="any_experience", required=True, weight=33),
        _c(id="c2", type="professional_concept",
           concept="people with cybersecurity experience", scope="career",
           required=True, weight=34),
        _c(id="h1", type="industry_experience", concept="healthcare experience",
           scope="career", required=True, weight=33),
    )
    concepts = " ".join(c.concept or "" for c in out).lower()
    assert "cyber" in concepts and "healthcare" in concepts
    # cyber's two representations merge; healthcare is untouched
    assert len(out) == 2


def test_research_plus_industry_experience_are_distinct():
    out = _dedupe(
        _c(id="res", type="professional_concept", concept="research experience",
           scope="career", required=True, weight=50),
        _c(id="ind", type="industry_experience",
           concept="professional experience working in industry (applied, non-academic)",
           scope="any_experience", required=True, weight=50),
    )
    assert len(out) == 2


def test_same_requirement_helper():
    a = _c(type="industry_experience", concept="fintech industry experience")
    b = _c(type="industry_experience", concept="professional experience in the fintech sector")
    assert _same_requirement(a, b) is True
    c = _c(type="industry_experience", concept="healthcare experience")
    assert _same_requirement(a, c) is False


def test_weights_renormalise_to_100_after_dedup():
    p = ParsedSearchQuery(criteria=[
        _c(id="n1", type="industry_experience", concept="nonprofit sector experience",
           scope="any_experience", required=True, weight=60),
        _c(id="n2", type="industry_experience", concept="nonprofit industry experience",
           scope="career", required=True, weight=20),
        _c(id="loc", type="location", value="Chicago", required=True, weight=20),
    ])
    _dedupe_semantic_duplicates(p)
    # dedup does not itself renormalise; interpret_query's _renorm / the schema
    # model-validator does. Assert the merged weight is the sum.
    merged = next(c for c in p.criteria if c.type == "industry_experience")
    assert merged.weight == 80.0
