"""Deterministic + semantic evidence-backed match scoring (spec §14–§20, §27–§28).

Hybrid (spec: "Use deterministic matching for facts. Use semantic reasoning for
concepts."):

* EXACT FACTS (company / school / location / certification / language / title /
  skill) — computed in code against normalized rows. ``match_strength ∈ [0,1]``,
  attached evidence points at a real row. A ``required`` fact that isn't found is
  a hard exclude (we have the person's complete LinkedIn experience list, so
  "not found" is a real negative).

* SEMANTIC CONCEPTS (semantic_concept / company_category) — evaluated strongest→
  weakest: profile semantic assertion → company classification against a real
  experience row → semantic fields (industries / job_families / …) → concept-vs-
  career cross-encoder. Returns a TRUE / FALSE / UNKNOWN status. A ``required``
  semantic concept excludes ONLY on a confident FALSE — never on UNKNOWN
  (spec §15/§28: missing data ≠ verified absence).

No LLM decides a numeric score here. (An optional batched LLM *judge* runs
upstream in search_service for ambiguous semantic concepts and feeds its
verdict back in via ``ScoringContext.judge_results``.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import repositories as repo
from app.config import settings
from app.constants import (
    CriterionType,
    Operator,
    Qualification,
    Scope,
    SkillSource,
    TriState,
)
from app.models import Person
from app.schemas import EvidenceItem, ParsedSearchQuery, ScoreComponent, SearchCriterion
from app.services import career_chronology as _career
from app.services.company_intel import company_key
from app.services.geo import expand_values, location_matches
from app.services.matching import (
    category_field,
    company_matches,
    concept_overlap,
    experience_weight,
    is_cxo_title,
    norm,
    phrase_matches,
    seniority_rank,
    token_overlap,
)

_REQUIRED_MIN = 0.15   # below this a required non-semantic criterion is a hard miss
_EXACT_MIN = 0.55      # a required FUZZY non-semantic criterion (title/seniority) counts as
                       # TRUE only above this — "Director" does not satisfy "CXO"
_MATCHED_MIN = 0.2
#: structured facts whose match is effectively binary in-scope — recency/duration
#: weighting affects the *score* but not whether the fact is TRUE (review #1)
_BINARY_FACT_TYPES = {
    CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY, CriterionType.LOCATION,
    CriterionType.EDUCATION, CriterionType.CERTIFICATION, CriterionType.LANGUAGE,
}
#: types evaluated as meaning (tri-state), never phrase matches (V4 §6/§29)
_SEMANTIC_TYPES = {
    CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY,
    CriterionType.PROFESSIONAL_CONCEPT, CriterionType.INDUSTRY_EXPERIENCE,
    CriterionType.ROLE_FUNCTION, CriterionType.CAREER_TRANSITION,
    CriterionType.YEARS_EXPERIENCE,
}
#: semantic types the LLM judge may OVERRIDE with a validated TRUE/FALSE verdict
#: (V4 PART 3 §15). CAREER_TRANSITION / YEARS_EXPERIENCE stay code-authoritative
#: — the judge never decides verified ordering or durations (§16).
_JUDGE_OVERRIDABLE = _SEMANTIC_TYPES - {
    CriterionType.CAREER_TRANSITION, CriterionType.YEARS_EXPERIENCE,
}


@dataclass
class ScoringContext:
    """Per-search signals shared across candidates."""

    query_embedding: bytes | None = None
    reranker_scores: dict[str, float] = field(default_factory=dict)
    company_ids_by_criterion: dict[str, set[str]] = field(default_factory=dict)
    #: company_key -> classification dict (from company_intel.get_or_classify)
    company_class: dict[str, dict] = field(default_factory=dict)
    #: person_id -> {criterion_id: {"status","match_strength","confidence","reason","evidence"}}
    judge_results: dict[str, dict[str, dict]] = field(default_factory=dict)
    #: (person_id, concept) -> concept-vs-career cross-encoder score in [0,1].
    #: PRECOMPUTED in batches by ``semantic_similarity.compute_semantic_similarity``
    #: — ``score_candidate`` only ever READS this, never invokes a model (STEP 2/3).
    semantic_similarity: dict[tuple[str, str], float] = field(default_factory=dict)
    #: person_id -> whole-profile embedding cosine in [0,1]. Precomputed in ONE
    #: vectorised numpy matmul (STEP 7); ``_relevance_component`` reads it and
    #: only falls back to a per-pair dot product for direct unit-test calls.
    relevance_by_person: dict[str, float] = field(default_factory=dict)


@dataclass
class ProfileFacts:
    person: Person
    experiences: list
    education: list
    skills: list
    semantic: dict
    embedding: bytes | None
    certifications: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    publications: list = field(default_factory=list)

    @property
    def skill_norms(self) -> dict[str, object]:
        return {s.skill_name_norm: s for s in self.skills}


@dataclass
class ScoredCandidate:
    person: Person
    match_score: float
    components: list[ScoreComponent]
    evidence: list[EvidenceItem]
    matched_criteria: list[str] = field(default_factory=list)
    excluded_reason: str | None = None
    #: V4 §22-25 — ranked BEFORE match_score. not_match candidates are kept out
    #: of the normal results bucket (may surface as near-matches).
    qualification: str = Qualification.POSSIBLE_MATCH
    #: required criteria that were FALSE or unmet (for near-match explanations)
    unmet_required: list[str] = field(default_factory=list)
    #: required semantic criteria that are UNKNOWN (why this is only POSSIBLE)
    uncertain_required: list[str] = field(default_factory=list)
    #: criterion_id -> TriState, for every tri-state (semantic / structured-NOT)
    #: criterion — lets the judge-triage decide, PER CRITERION, whether a call is
    #: even worth spending (hardening PART 4/5: staged local resolution).
    status_by_criterion: dict[str, str] = field(default_factory=dict)


def load_facts(db: Session, person: Person, facts_cache: dict | None = None) -> ProfileFacts:
    """Build a ProfileFacts. ``facts_cache`` (from repositories.bulk_*) avoids
    the per-candidate N+1 (spec §31) — pass ``{"experiences": {...}, ...}``."""
    fc = facts_cache or {}
    pid = person.id

    if fc:
        sem_row = fc.get("semantics", {}).get(pid)
        return ProfileFacts(
            person=person,
            experiences=fc.get("experiences", {}).get(pid, []),
            education=fc.get("education", {}).get(pid, []),
            skills=fc.get("skills", {}).get(pid, []),
            semantic=(sem_row.data if sem_row and sem_row.data else {}),
            embedding=fc.get("embeddings", {}).get(pid),
            certifications=fc.get("certifications", {}).get(pid, []),
            languages=fc.get("languages", {}).get(pid, []),
            publications=fc.get("publications", {}).get(pid, []),
        )

    sem = repo.get_semantic(db, pid)
    from app.models import ProfileEmbedding

    row = db.query(ProfileEmbedding).filter(ProfileEmbedding.person_id == pid).first()
    return ProfileFacts(
        person=person,
        experiences=repo.get_experiences(db, pid),
        education=repo.get_education(db, pid),
        skills=repo.get_skills(db, pid),
        semantic=(sem.data if sem and sem.data else {}),
        embedding=row.vector if row else None,
        certifications=repo.get_certifications(db, pid),
        languages=repo.get_languages(db, pid),
        publications=repo.get_publications(db, pid),
    )


# ─────────────────────── exact-fact strategies ───────────────────────


def _experiences_in_scope(experiences: list, scope: str | None) -> list:
    if scope in (Scope.CURRENT, Scope.CURRENT_COMPANY):
        return [e for e in experiences if e.is_current]
    if scope in (Scope.PAST, Scope.PAST_COMPANY):
        return [e for e in experiences if not e.is_current]
    return list(experiences)


def _want_current_for(crit: SearchCriterion) -> bool | None:
    """Resolve current/past/any for an employment criterion (V4 §1). An explicit
    ``scope`` wins; otherwise the criterion type's default."""
    if crit.scope in (Scope.ANY_EXPERIENCE, Scope.CAREER):
        return None
    if crit.scope in (Scope.CURRENT, Scope.CURRENT_COMPANY):
        return True
    if crit.scope in (Scope.PAST, Scope.PAST_COMPANY):
        return False
    return crit.type == CriterionType.CURRENT_COMPANY


def _score_company(
    facts: ProfileFacts, value: str, *, want_current: bool | None, resolved_ids: set[str] | None = None
) -> tuple[float, list[EvidenceItem]]:
    """STRICT scope (V4 §1): ``want_current=True`` only current roles satisfy,
    ``want_current=False`` only NON-current roles satisfy, ``None`` any role.
    A current Amazon role does NOT partially satisfy "former Amazon"."""
    best = 0.0
    ev: list[EvidenceItem] = []
    recency_on = settings.recency_weighting_enabled
    for e in facts.experiences:
        if want_current is True and not e.is_current:
            continue
        if want_current is False and e.is_current:
            continue
        by_id = bool(resolved_ids) and bool(e.company_id) and e.company_id in resolved_ids
        if not by_id and not company_matches(e.company_name, value):
            continue
        item = EvidenceItem(
            type="experience",
            text=f"{e.position or 'Role'} at {e.company_name}"
            + (f" ({e.start_year}–{e.end_year or 'present'})" if e.start_year else "")
            + (" — verified company" if by_id else ""),
            detail={
                "company": e.company_name, "title": e.position, "start_year": e.start_year,
                "end_year": e.end_year, "is_current": e.is_current, "verified": by_id,
            },
        )
        strength = experience_weight(e, enabled=recency_on)  # 1.0 base, recency/duration only
        if strength > best:
            best, ev = strength, [item]
    return best, ev


def _score_skill(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    target = norm(value)
    for s in facts.skills:
        if s.skill_name_norm == target or phrase_matches(s.skill_name_norm, target):
            if not s.is_inferred and s.source in (SkillSource.PROFILE, SkillSource.EXPERIENCE, SkillSource.EDUCATION):
                return 1.0, [EvidenceItem(type="skill", text=f"{s.skill_name} — listed on LinkedIn", detail={"source": s.source})]
    sem = facts.semantic
    for sk in sem.get("explicit_skills", []):
        if phrase_matches(sk, value):
            return 0.9, [EvidenceItem(type="skill", text=f"{sk} — stated on profile", detail={"source": "semantic_explicit"})]
    for isk in sem.get("inferred_skills", []):
        name = isk.get("skill") if isinstance(isk, dict) else None
        if name and phrase_matches(name, value):
            conf = float(isk.get("confidence", 0.7))
            return min(0.9, 0.5 + conf * 0.4), [
                EvidenceItem(type="semantic", text=f"{name} — inferred: {isk.get('evidence', '')[:160]}", detail={"confidence": conf, "inferred": True})
            ]
    for e in facts.experiences:
        exp_skill_text = " ".join(e.skills_json) if e.skills_json else ""
        if phrase_matches(e.description, value) or phrase_matches(exp_skill_text, value):
            w = experience_weight(e, enabled=settings.recency_weighting_enabled)
            return round(0.6 * w, 3), [EvidenceItem(type="experience", text=f"referenced in the {e.company_name or 'role'} description", detail={})]
    for dom in sem.get("technical_domains", []) + sem.get("domain_expertise", []):
        if phrase_matches(dom, value):
            return 0.6, [EvidenceItem(type="semantic", text=f"domain: {dom}", detail={"inferred": True})]
    hay = " ".join(filter(None, [facts.person.headline, facts.person.about]))
    if phrase_matches(hay, value):
        return 0.5, [EvidenceItem(type="headline", text="mentioned in headline / about", detail={})]
    return 0.0, []


def _score_domain(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    # legacy type — route to the concept evaluator so it uses the full semantic
    # profile instead of phrase_matches (spec §6)
    strength, ev, _status = _score_semantic_concept(facts, _synthetic_crit(value), ScoringContext())
    if strength > 0:
        return strength, ev
    return _score_skill(facts, value)


def _title_strength(title: str | None, value: str) -> float:
    if not title:
        return 0.0
    if phrase_matches(title, value):
        return 1.0
    return 0.6 if token_overlap(title, value) >= 0.34 else 0.0


def _score_title(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    p = facts.person
    cur = _title_strength(p.current_title, value)
    if cur > 0:
        return cur, [EvidenceItem(type="experience", text=f"current title: {p.current_title}", detail={"current": True})]
    best, ev = 0.0, []
    for e in facts.experiences:
        s = _title_strength(e.position, value) * (1.0 if e.is_current else 0.9)
        if s > best:
            best, ev = s, [EvidenceItem(type="experience", text=f"{e.position} at {e.company_name}", detail={})]
    return best, ev


def _score_education(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    v = norm(value)
    for ed in facts.education:
        if ed.school_name and phrase_matches(ed.school_name, value):
            return 1.0, [EvidenceItem(type="education", text=f"{ed.degree or 'Studied'} at {ed.school_name}", detail={"school": ed.school_name})]
    for ed in facts.education:
        if ed.field_of_study and phrase_matches(ed.field_of_study, value):
            return 0.9, [EvidenceItem(type="education", text=f"{ed.field_of_study} at {ed.school_name or 'a school'}", detail={})]
        if ed.degree and (v in norm(ed.degree) or norm(ed.degree) in v):
            return 0.85, [EvidenceItem(type="education", text=f"{ed.degree} — {ed.school_name or ''}".strip(" —"), detail={})]
    for kw in facts.semantic.get("education_keywords", []):
        if phrase_matches(kw, value):
            return 0.7, [EvidenceItem(type="semantic", text=f"education: {kw}", detail={"inferred": True})]
    return 0.0, []


def _score_location(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    p = facts.person
    fields = [p.location_text, p.city, p.state, p.country]
    for candidate in expand_values([value]):
        if location_matches(fields, candidate):
            return 1.0, [EvidenceItem(type="location", text=f"located in {p.location_text or candidate}", detail={"matched": candidate})]
    return 0.0, []


def _score_seniority(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    v = norm(value)
    title = facts.person.current_title
    # "CXO"/executive — accept any real C-suite title, not a rank-gap game
    if "cxo" in v or v in {"executive", "c-suite", "c-level", "chief"}:
        if is_cxo_title(title) or seniority_rank(facts.semantic.get("seniority_level")) == 8:
            return 1.0, [EvidenceItem(type="experience", text=f"executive title: {title or facts.semantic.get('seniority_level')}", detail={})]
        rank = seniority_rank(facts.semantic.get("seniority_level")) or seniority_rank(title)
        return (0.5 if rank == 7 else 0.25 if rank == 6 else 0.0), (
            [EvidenceItem(type="semantic", text=f"seniority: {facts.semantic.get('seniority_level') or title}", detail={})] if rank and rank >= 6 else []
        )
    want = seniority_rank(value)
    have = seniority_rank(facts.semantic.get("seniority_level")) or seniority_rank(title)
    if want is None or have is None:
        return 0.0, []
    gap = abs(want - have)
    strength = 1.0 if gap == 0 else 0.6 if gap == 1 else 0.0
    if strength:
        return strength, [EvidenceItem(type="semantic", text=f"seniority: {facts.semantic.get('seniority_level') or title}", detail={})]
    return 0.0, []


def _score_certification(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    for c in facts.certifications:
        hay = " ".join(filter(None, [c.name, c.issuer]))
        if phrase_matches(hay, value) or phrase_matches(value, c.name or ""):
            return 1.0, [EvidenceItem(type="certification", text=f"{c.name}" + (f" — {c.issuer}" if c.issuer else "") + " (certification)", detail={"issuer": c.issuer, "issued_at": c.issued_at})]
    for c in facts.certifications:
        toks = [t for t in norm(value).split() if t not in {"certification", "certified", "cert", "certificate"}]
        if toks and all(t in norm(f"{c.name} {c.issuer}") for t in toks):
            return 0.9, [EvidenceItem(type="certification", text=f"{c.name} (certification)", detail={})]
    return 0.0, []


def _score_language(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    wanted = {t for t in norm(value).replace(" or ", " ").replace(" and ", " ").split() if len(t) > 1}
    wanted -= {"speaks", "speak", "language", "languages", "fluent", "native"}
    for lang in facts.languages:
        if lang.name_norm in wanted or any(w in lang.name_norm for w in wanted):
            return 1.0, [EvidenceItem(type="language", text=f"speaks {lang.name}" + (f" ({lang.proficiency})" if lang.proficiency else ""), detail={"proficiency": lang.proficiency})]
    return 0.0, []


def _score_publication(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    if not facts.publications:
        return 0.0, []
    generic = {"published", "publication", "publications", "research", "paper", "papers", "author", "wrote"}
    topical = [t for t in norm(value).split() if t not in generic and len(t) > 2]
    if not topical:
        p = facts.publications[0]
        return 0.85, [EvidenceItem(type="publication", text=f'published "{(p.title or "")[:90]}"', detail={"count": len(facts.publications)})]
    for p in facts.publications:
        hay = norm(f"{p.title} {p.description or ''}")
        if sum(1 for t in topical if t in hay) / len(topical) >= 0.5:
            return 1.0, [EvidenceItem(type="publication", text=f'published "{(p.title or "")[:90]}"', detail={})]
    return 0.4, [EvidenceItem(type="publication", text=f"has {len(facts.publications)} publication(s)", detail={})]


def _score_keyword(facts: ProfileFacts, value: str) -> tuple[float, list[EvidenceItem]]:
    parts = [
        facts.person.headline, facts.person.about,
        " ".join(e.description or "" for e in facts.experiences),
        " ".join(s.skill_name for s in facts.skills),
        " ".join(facts.semantic.get("searchable_keywords", [])),
    ]
    if phrase_matches(" ".join(filter(None, parts)), value):
        return 0.8, [EvidenceItem(type="headline", text=f"profile mentions '{value}'", detail={})]
    return 0.0, []


# ─────────────────────── semantic-concept strategies ───────────────────────


def _synthetic_crit(concept: str) -> SearchCriterion:
    return SearchCriterion(id="_syn", type=CriterionType.SEMANTIC_CONCEPT, concept=concept, weight=1)


def _career_snippet(facts: ProfileFacts) -> str:
    bits = [facts.person.headline or ""]
    for e in facts.experiences[:6]:
        bits.append(f"{e.position or ''} at {e.company_name or ''}. {(e.description or '')[:200]}")
    sem = facts.semantic
    bits.append(" ".join(sem.get("industries", []) + sem.get("job_families", []) + sem.get("domain_expertise", [])))
    if sem.get("career_summary"):
        bits.append(sem["career_summary"])
    return "  ".join(b for b in bits if b.strip())[:1500]


def _score_company_category(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    """Tri-state on the ACTUAL employer(s) in scope (V4 §7). A classification
    below ``company_category_confidence_min`` is treated as UNKNOWN, so a
    low-confidence TRUE cannot create an EXACT_MATCH."""
    concept = (crit.concept or crit.value or "").strip()
    field_name = category_field(concept)
    scope = crit.scope or Scope.CURRENT_COMPANY  # "startup" defaults to "now at a startup"
    scoped = _experiences_in_scope(facts.experiences, scope)
    cmin = settings.company_category_confidence_min

    saw_confident_false = False
    saw_low_conf_true = False
    for e in scoped:
        row = ctx.company_class.get(
            company_key(getattr(e, "company_id", None), getattr(e, "company_name", None))
        )
        if not row:
            continue
        conf = float(row.get("confidence") or 0.0)
        val = row.get(field_name) if field_name else None
        if val is True and conf >= cmin:
            return 1.0, [
                EvidenceItem(
                    type="company_inference",
                    text=f"{e.company_name} classified as {concept}"
                    + (f" (“{row.get('reason')}”)" if row.get("reason") else ""),
                    detail={"confidence": conf, "provenance": row.get("provenance"),
                            "company": e.company_name, "role": e.position},
                )
            ], TriState.TRUE
        if val is True:
            saw_low_conf_true = True
        if val is False and conf >= cmin:
            saw_confident_false = True
        if field_name is None:  # loose industry/category text fallback
            for x in (row.get("industries") or []) + (row.get("categories") or []):
                if concept_overlap(x, concept) >= 0.5 and conf >= cmin:
                    return 0.85, [
                        EvidenceItem(type="company_inference", text=f"{e.company_name} is in {x}",
                                     detail={"confidence": conf, "provenance": row.get("provenance")})
                    ], TriState.TRUE

    if saw_low_conf_true:
        return 0.4, [EvidenceItem(type="company_inference",
                                  text=f"employer may be {concept} (low-confidence classification)", detail={})], TriState.UNKNOWN
    if saw_confident_false:
        return 0.0, [
            EvidenceItem(type="company_inference", text=f"employer(s) in scope are not classified as {concept}", detail={})
        ], TriState.FALSE
    return 0.0, [], TriState.UNKNOWN  # not classified yet / ambiguous — NOT a verified false


def _exp_current_map(facts: ProfileFacts) -> dict:
    return {getattr(e, "id", None): bool(getattr(e, "is_current", False)) for e in facts.experiences}


def _scope_allows(scope: str | None, exp_ids, is_current_by_id: dict) -> bool:
    """Whether an assertion / experience-semantic in scope (V4 §2/§3). Verified
    structured history wins: an assertion claiming 'current' but linked only to
    NON-current experiences is not current."""
    if scope in (None, Scope.CAREER, Scope.ANY_EXPERIENCE):
        return True
    ids = [i for i in (exp_ids or []) if i in is_current_by_id]
    if not ids:
        return True  # no linked rows to contradict — leave to other signals
    currents = [is_current_by_id[i] for i in ids]
    if scope in (Scope.CURRENT, Scope.CURRENT_COMPANY):
        return any(currents)
    if scope in (Scope.PAST, Scope.PAST_COMPANY):
        return any(not c for c in currents)
    return True


def _score_semantic_concept(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    concept = (crit.concept or crit.value or "").strip()
    if not concept:
        return 0.0, [], TriState.UNKNOWN
    sem = facts.semantic
    scope = crit.scope
    cur_by_id = _exp_current_map(facts)

    best_strength, best_ev, best_status = 0.0, [], TriState.UNKNOWN

    # 0. experience-level semantics (V4 §2/§29 — most specific, scope-aware).
    #    ROLE_FUNCTION -> role_function/professional_domain/role_domains;
    #    INDUSTRY_EXPERIENCE -> employer_industries. Only experiences IN SCOPE
    #    for the criterion are eligible.
    want_industry = crit.type == CriterionType.INDUSTRY_EXPERIENCE
    exp_sem = _career.exp_semantics_by_id(sem)
    if exp_sem:
        for eid, es in exp_sem.items():
            if not _scope_allows(scope, [eid], cur_by_id):
                continue
            fields = (["employer_industries", "employer_categories"] if want_industry
                      else ["role_function", "professional_domain", "role_domains"])
            hit = 0.0
            for f in fields:
                v = es.get(f)
                for item in ([v] if isinstance(v, str) else (v or [])):
                    hit = max(hit, concept_overlap(str(item), concept))
            if hit >= 0.55:
                strength = min(0.9, 0.55 + hit * 0.35) * (0.6 + 0.4 * float(es.get("confidence", 0.6)))
                if strength > best_strength:
                    best_strength, best_status = strength, TriState.TRUE
                    best_ev = [EvidenceItem(type="semantic",
                                            text=f"{'industry' if want_industry else 'role'}: "
                                                 f"{es.get('role_function') or (es.get('employer_industries') or ['?'])[0]}",
                                            detail={"experience_id": eid, "inferred": True})]

    # 1. a matching semantic assertion (strong — LLM-derived + evidence + source ids).
    #    Assertion scope is validated against the linked experiences (V4 §3).
    for a in sem.get("semantic_assertions", []):
        if not isinstance(a, dict):
            continue
        if not _scope_allows(scope, a.get("experience_ids"), cur_by_id):
            continue
        ov = concept_overlap(a.get("concept", ""), concept)
        if ov >= 0.5:
            conf = float(a.get("confidence", 0.6))
            strength = min(0.95, 0.5 + conf * 0.45) * (0.6 + 0.4 * ov)
            ev = [EvidenceItem(
                type="semantic",
                text=f"{a.get('concept')} — {'; '.join(a.get('evidence', [])[:2])}",
                detail={"confidence": conf, "category": a.get("category"), "inferred": True},
            )]
            if strength > best_strength:
                best_strength, best_ev, best_status = strength, ev, TriState.TRUE

    # 2. profile-level semantic fields — no per-experience scope, so for a
    #    current/past-scoped criterion they are only a weak UNKNOWN-tier signal.
    _profile_can_be_true = scope in (None, Scope.CAREER, Scope.ANY_EXPERIENCE)
    if crit.type == CriterionType.INDUSTRY_EXPERIENCE:
        _fields = (("industries", "industry"),)
    elif crit.type == CriterionType.ROLE_FUNCTION:
        _fields = (("job_families", "role"), ("role_keywords", "role"))
    else:  # SEMANTIC_CONCEPT / PROFESSIONAL_CONCEPT — any dimension
        _fields = (("industries", "industry"), ("job_families", "role"),
                   ("domain_expertise", "domain expertise"),
                   ("leadership_experience", "leadership"), ("role_keywords", "role"))
    if best_strength < 0.8:
        for fld, label in _fields:
            for v in sem.get(fld, []):
                if concept_overlap(v, concept) >= 0.55 and 0.72 > best_strength:
                    best_strength = 0.72 if _profile_can_be_true else 0.45
                    best_status = TriState.TRUE if _profile_can_be_true else TriState.UNKNOWN
                    best_ev = [EvidenceItem(type="semantic", text=f"{label}: {v}", detail={"inferred": True})]

    # 3. concept-vs-career similarity (criterion-level, not whole query, spec §26).
    #    PURE: reads the PRECOMPUTED, batched cross-encoder score from ctx —
    #    score_candidate never loads or calls a model (STEP 2/3). Skipped for
    #    role_function / industry_experience when we HAVE experience-level
    #    semantics: structured role vs industry data is authoritative there, and
    #    a fuzzy similarity must not turn a clean "no" into a maybe (V4 §H.5).
    _structured_authoritative = bool(exp_sem) and crit.type in (
        CriterionType.ROLE_FUNCTION, CriterionType.INDUSTRY_EXPERIENCE)
    if best_strength < 0.6 and not _structured_authoritative:
        ce = ctx.semantic_similarity.get((facts.person.id, concept))
        if ce is not None and ce >= 0.5:
            strength = 0.35 + ce * 0.3
            if strength > best_strength:
                best_strength, best_ev, best_status = strength, [
                    EvidenceItem(type="semantic_relevance",
                                 text="career profile is semantically consistent with this concept",
                                 detail={"cross_encoder": round(ce, 3)})
                ], TriState.UNKNOWN  # a similarity signal alone isn't a confident TRUE

    if best_strength < _REQUIRED_MIN:
        return 0.0, [], TriState.UNKNOWN
    return round(best_strength, 3), best_ev, best_status


def _invert_tristate(status: str) -> str:
    """Tri-state NOT (V4 §5): TRUE->FALSE, FALSE->TRUE, UNKNOWN->UNKNOWN. Missing
    information is NOT proof of a negation."""
    if status == TriState.TRUE:
        return TriState.FALSE
    if status == TriState.FALSE:
        return TriState.TRUE
    return TriState.UNKNOWN


def _score_semantic_multi(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    """Score a semantic criterion honouring ``values`` + ``operator`` (V4 §4/§5).
    Each value keeps the parent's ``type`` + ``scope`` (review #2). ANY_OF -> best
    value. ALL_OF -> all values must hold. NOT -> tri-state invert."""
    vals = [v for v in (crit.values or ([crit.value] if crit.value else [])) if v]

    if len(vals) <= 1:
        # single value / rich free-text concept — score whole, then invert for NOT
        s, ev, status = _score_semantic_concept(facts, crit, ctx)
        if crit.operator == Operator.NOT:
            return round(1.0 - s, 3), [], _invert_tristate(status)
        return s, ev, status

    def _child(i: int, v: str) -> SearchCriterion:
        return SearchCriterion(id=f"{crit.id}#{i}", type=crit.type, concept=v,
                               value=v, scope=crit.scope, weight=1)

    results = [_score_semantic_concept(facts, _child(i, v), ctx) for i, v in enumerate(vals)]
    if crit.operator == Operator.NOT:
        best = max(results, key=lambda r: r[0])
        return round(1.0 - best[0], 3), [], _invert_tristate(best[2])
    if crit.operator == Operator.ALL_OF:
        strength = min(r[0] for r in results)
        if any(r[2] == TriState.FALSE for r in results):
            status = TriState.FALSE
        elif all(r[2] == TriState.TRUE for r in results):
            status = TriState.TRUE
        else:
            status = TriState.UNKNOWN
        ev = [e for r in results for e in r[1]][:4]
        return round(strength, 3), ev, status
    # ANY_OF
    best = max(results, key=lambda r: r[0])
    return best


def _score_chronology(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    """career_transition / years_experience — real implementation in PART D
    (career_chronology). Until then these stay UNKNOWN so a required transition
    criterion yields POSSIBLE_MATCH, never a false EXACT_MATCH or exclusion."""
    from app.services.career_chronology import score_transition, score_years_experience

    if crit.type == CriterionType.CAREER_TRANSITION:
        return score_transition(facts, crit, ctx)
    return score_years_experience(facts, crit)


# ─────────────────────── dispatch ───────────────────────

_STRATEGIES = {
    CriterionType.SKILL: _score_skill,
    CriterionType.DOMAIN: _score_domain,
    CriterionType.TITLE: _score_title,
    CriterionType.EDUCATION: _score_education,
    CriterionType.LOCATION: _score_location,
    CriterionType.SENIORITY: _score_seniority,
    CriterionType.KEYWORD: _score_keyword,
    CriterionType.CERTIFICATION: _score_certification,
    CriterionType.LANGUAGE: _score_language,
    CriterionType.PUBLICATION: _score_publication,
}


def _score_one(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str | None]:
    """Returns ``(strength, evidence, status)``. ``status`` is a TriState for
    semantic criteria, else ``None``."""
    try:
        # validated judge verdict (V4 PART 3) — overrides local scoring ONLY for a
        # confident TRUE / FALSE on a judgeable semantic type. UNKNOWN / omitted
        # falls through to the deterministic scorer (§30/§48); chronology stays
        # code-authoritative (§16).
        jr = ctx.judge_results.get(facts.person.id, {}).get(crit.id)
        if jr and crit.type in _JUDGE_OVERRIDABLE and jr.get("status") in (TriState.TRUE, TriState.FALSE):
            return (
                float(jr.get("match_strength", 0.0)),
                [EvidenceItem(type="semantic", text=(jr.get("reason") or "")[:220],
                              detail={"confidence": jr.get("confidence"), "judge": True,
                                      "evidence": (jr.get("evidence") or [])[:3],
                                      "supporting_refs": jr.get("supporting_evidence_refs", [])})]
                if jr.get("status") == TriState.TRUE else [],
                jr.get("status"),
            )

        if crit.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.PROFESSIONAL_CONCEPT,
                         CriterionType.INDUSTRY_EXPERIENCE, CriterionType.ROLE_FUNCTION):
            return _score_semantic_multi(facts, crit, ctx)
        if crit.type == CriterionType.COMPANY_CATEGORY:
            return _score_company_category(facts, crit, ctx)
        if crit.type in (CriterionType.CAREER_TRANSITION, CriterionType.YEARS_EXPERIENCE):
            return _score_chronology(facts, crit, ctx)

        # tri-state NOT for a verified structured fact (V4 PART 3.5 §3) — absence
        # of DATA is UNKNOWN, never a satisfied NOT.
        if crit.operator == Operator.NOT and crit.type in _NOT_TRISTATE_TYPES:
            return _score_structured_not(facts, crit, ctx)

        if crit.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY):
            want = _want_current_for(crit)
            rids = ctx.company_ids_by_criterion.get(crit.id) if settings.company_id_matching else None
            s, e = _combine_over_values(
                crit, lambda v: _score_company(facts, v, want_current=want, resolved_ids=rids)
            )
            return s, e, None

        strat = _STRATEGIES.get(crit.type, _score_keyword)
        s, e = _combine_over_values(crit, lambda v: strat(facts, v))
        return s, e, None
    except Exception:  # noqa: BLE001 — a matcher bug must not kill a search
        return 0.0, [], None


#: structured fact types where a required NOT must be tri-state, not "absent
#: data == satisfied" (V4 PART 3.5 §3 / PART 3.6 §3).
_NOT_TRISTATE_TYPES = {
    CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY,
    CriterionType.LOCATION, CriterionType.EDUCATION,
}


def _score_structured_not(
    facts: ProfileFacts, crit: SearchCriterion, ctx: ScoringContext
) -> tuple[float, list[EvidenceItem], str]:
    """Tri-state NOT for a verified structured fact:
      excluded value verifiably PRESENT (in scope)  -> FALSE
      verifiably ABSENT + the relevant section is backend-authoritative -> TRUE
      section unknown / not authoritative           -> UNKNOWN

    Section authority is the SHARED ``profile_authority`` policy — a NOT past
    company and the hard gate use the exact same rule (V4 PART 3.6 §3).
    """
    from app.services.profile_authority import (
        current_employer_known,
        education_history_authoritative,
        location_known,
        work_history_authoritative,
    )

    vals = [v for v in (crit.values or ([crit.value] if crit.value else [])) if v]

    if crit.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY):
        want = _want_current_for(crit)
        rids = ctx.company_ids_by_criterion.get(crit.id) if settings.company_id_matching else None
        present = max(
            (_score_company(facts, v, want_current=want, resolved_ids=rids)[0] for v in vals),
            default=0.0,
        )
        if present >= _REQUIRED_MIN:
            return 0.0, [EvidenceItem(type="experience",
                                      text=f"is at an excluded company ({' / '.join(vals)})", detail={})], TriState.FALSE
        reliable = (current_employer_known(facts) if crit.type == CriterionType.CURRENT_COMPANY
                    else work_history_authoritative(facts))
        return _not_true_or_unknown(reliable, f"no excluded-company ({' / '.join(vals)}) role")

    if crit.type == CriterionType.LOCATION:
        present = max((_score_location(facts, v)[0] for v in vals), default=0.0)
        if present >= _REQUIRED_MIN:
            return 0.0, [EvidenceItem(type="location", text="located in an excluded place", detail={})], TriState.FALSE
        return _not_true_or_unknown(location_known(facts), "location is not an excluded place")

    # EDUCATION
    present = max((_score_education(facts, v)[0] for v in vals), default=0.0)
    if present >= _REQUIRED_MIN:
        return 0.0, [EvidenceItem(type="education", text="attended an excluded school", detail={})], TriState.FALSE
    return _not_true_or_unknown(education_history_authoritative(facts),
                                "no excluded school in the education history")


def _not_true_or_unknown(reliable: bool, true_text: str) -> tuple[float, list[EvidenceItem], str]:
    if reliable:
        return 1.0, [EvidenceItem(type="experience", text=true_text, detail={})], TriState.TRUE
    return 0.5, [], TriState.UNKNOWN


def _combine_over_values(crit: SearchCriterion, scorer) -> tuple[float, list[EvidenceItem]]:
    """Apply ``scorer(value)`` across ``crit.values`` and combine per operator
    (spec §13). ANY_OF: best. ALL_OF: min (all must hold). NOT: 1 - best."""
    vals = crit.values or ([crit.value] if crit.value else [])
    if not vals:
        return 0.0, []
    results = [scorer(v) for v in vals]
    if crit.operator == Operator.ALL_OF:
        s = min(r[0] for r in results)
        ev = [item for r in results for item in r[1]]
        return s, ev
    if crit.operator == Operator.NOT:
        best = max(results, key=lambda r: r[0])
        # matching the value is BAD here — invert. Strong presence -> 0; absence -> 1.
        return (1.0 - best[0]), (
            [EvidenceItem(type="experience", text="does not match the excluded criterion", detail={})]
            if best[0] < _REQUIRED_MIN else best[1]
        )
    best = max(results, key=lambda r: r[0])  # ANY_OF
    return best


# ─────────────────────── relevance component ───────────────────────


def _cosine_norm(query_emb: bytes | None, profile_emb: bytes | None) -> float:
    if not query_emb or not profile_emb:
        return 0.0
    import numpy as np

    from app.services.embeddings import to_array

    q, v = to_array(query_emb), to_array(profile_emb)
    if q.shape != v.shape:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(q, v)) / 0.6))


def _relevance_component(facts: ProfileFacts, ctx: ScoringContext, weight: float) -> ScoreComponent:
    # STEP 7 — prefer the vectorised, precomputed cosine; fall back to a single
    # dot product only for direct unit-test calls that never ran the batch pass.
    emb = ctx.relevance_by_person.get(facts.person.id)
    if emb is None:
        emb = _cosine_norm(ctx.query_embedding, facts.embedding)
    ce = ctx.reranker_scores.get(facts.person.id)
    if ce is not None:
        strength = settings.rerank_blend * ce + (1 - settings.rerank_blend) * emb
        note = "embedding + reranker"
    else:
        strength = emb
        note = "embedding similarity"
    strength = round(max(0.0, min(1.0, strength)), 3)
    return ScoreComponent(
        criterion="Overall relevance to your query", criterion_id="relevance", type="relevance",
        weight=round(weight, 2), match_strength=strength, score=round(weight * strength, 2), required=False,
        evidence=[EvidenceItem(type="semantic_relevance", text=f"Whole-profile semantic match to the query ({note})",
                               detail={"embedding": round(emb, 3), "reranker": round(ce, 3) if ce is not None else None})]
        if strength > 0 else [],
    )


_EXACT_TYPES = {
    CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY, CriterionType.EDUCATION,
    CriterionType.CERTIFICATION, CriterionType.LANGUAGE, CriterionType.LOCATION,
}


def _effective_relevance_weight(parsed: ParsedSearchQuery) -> float:
    base = max(0.0, min(60.0, settings.relevance_weight))
    if not parsed.criteria or base == 0.0:
        return 0.0
    strongest_exact = max((c.weight for c in parsed.criteria if c.type in _EXACT_TYPES), default=0.0)
    if strongest_exact >= 55:
        return base * 0.25
    if strongest_exact >= 35:
        return base * 0.55
    return base


def score_candidate(
    facts: ProfileFacts, parsed: ParsedSearchQuery, ctx: ScoringContext | None = None
) -> ScoredCandidate:
    from app.services import search_profile

    search_profile.incr("score_candidate_calls")
    ctx = ctx or ScoringContext()
    components: list[ScoreComponent] = []
    all_evidence: list[EvidenceItem] = []
    matched: list[str] = []
    total = 0.0

    rel_w = _effective_relevance_weight(parsed)
    scale = (100.0 - rel_w) / 100.0

    unmet: list[str] = []          # required + confidently FALSE / below the fact bar
    uncertain: list[str] = []      # required semantic + UNKNOWN
    status_by_criterion: dict[str, str] = {}

    for crit in parsed.criteria:
        strength, ev, status = _score_one(facts, crit, ctx)
        if status is not None:
            status_by_criterion[crit.id] = status
        eff_weight = crit.weight * scale
        score = round(eff_weight * strength, 2)
        components.append(ScoreComponent(
            criterion=_label(crit), criterion_id=crit.id, type=crit.type,
            weight=round(eff_weight, 2), match_strength=round(strength, 3), score=score,
            required=crit.required, evidence=ev,
        ))

        if crit.required:
            if status is not None:  # semantic tri-state OR structured-NOT tri-state (§3)
                if status == TriState.FALSE:
                    unmet.append(_label(crit))
                elif status != TriState.TRUE:
                    uncertain.append(_label(crit))
            elif crit.type in _BINARY_FACT_TYPES:
                # a structured fact either matched (in scope) or it didn't —
                # recency weighting must not demote a real match (review #1)
                if strength < _REQUIRED_MIN:
                    unmet.append(_label(crit))
            elif strength < _REQUIRED_MIN:
                unmet.append(_label(crit))          # clear miss
            elif strength < _EXACT_MIN:
                unmet.append(_label(crit))          # partial (Director when CXO asked) -> near-match

        total += score
        if strength >= _MATCHED_MIN:
            matched.append(_label(crit))
            all_evidence.extend(ev)

    # ── qualification tier (V4 §22-25) ────────────────────────────────
    if unmet:
        qualification = Qualification.NOT_MATCH
    elif uncertain:
        qualification = Qualification.POSSIBLE_MATCH
    else:
        qualification = Qualification.EXACT_MATCH

    if qualification == Qualification.NOT_MATCH:
        return ScoredCandidate(
            person=facts.person, match_score=round(min(100.0, total), 1), components=components,
            evidence=[], matched_criteria=matched,
            excluded_reason=f"required criterion not met: {unmet[0]}",
            qualification=qualification, unmet_required=unmet, uncertain_required=uncertain,
            status_by_criterion=status_by_criterion,
        )

    if rel_w > 0:
        rc = _relevance_component(facts, ctx, rel_w)
        components.append(rc)
        total += rc.score
        if rc.match_strength >= 0.55:
            all_evidence.extend(rc.evidence)

    return ScoredCandidate(
        person=facts.person, match_score=round(min(100.0, total), 1), components=components,
        evidence=_dedupe_evidence(all_evidence), matched_criteria=matched,
        qualification=qualification, uncertain_required=uncertain,
        status_by_criterion=status_by_criterion,
    )


def _label(c: SearchCriterion) -> str:
    val = " or ".join(c.values) if len(c.values) > 1 else (c.value or c.concept or "")
    pretty = {
        CriterionType.CURRENT_COMPANY: f"Currently at {val}",
        CriterionType.PAST_COMPANY: f"Previously at {val}",
        CriterionType.SKILL: val,
        CriterionType.DOMAIN: val,
        CriterionType.TITLE: f"{val} role",
        CriterionType.EDUCATION: f"Studied / attended {val}",
        CriterionType.LOCATION: f"Located in {val}",
        CriterionType.SENIORITY: f"{val.title()} level",
        CriterionType.KEYWORD: val,
        CriterionType.CERTIFICATION: f"{val} certification",
        CriterionType.LANGUAGE: f"Speaks {val}",
        CriterionType.PUBLICATION: f"Published on {val}" if val else "Has publications",
        CriterionType.SEMANTIC_CONCEPT: c.concept or val,
        CriterionType.COMPANY_CATEGORY: f"Works at a {c.concept or val} company"
        + ({Scope.PAST_COMPANY: " (past)", Scope.CURRENT_COMPANY: " (current)"}.get(c.scope, "")),
    }
    if c.operator == Operator.NOT:
        return f"NOT {pretty.get(c.type, val)}"
    return pretty.get(c.type, val)


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen, out = set(), []
    for it in items:
        key = (it.type, it.text)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
