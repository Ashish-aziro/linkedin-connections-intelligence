"""Pydantic models for API I/O and validated LLM output (spec §67).

Never trust free-form LLM output — every LLM call is parsed into one of the
``*Data`` / ``Parsed*`` models here and rejected on validation failure.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.constants import (
    _QUERY_INTENT_ALIASES,
    ALL_CRITERION_TYPES,
    ALL_MODALITIES,
    ALL_OPERATORS,
    ALL_QUERY_INTENTS,
    ALL_SCOPES,
    Modality,
    Operator,
    QueryIntent,
)

# ─────────────────────────── datasets ───────────────────────────


class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    connection_count: int
    status: str
    created_at: datetime
    updated_at: datetime


class UploadReport(BaseModel):
    dataset: DatasetSummary
    total_rows: int
    imported: int
    duplicates_removed: int
    skipped_no_url: int
    skipped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []


class DatasetStatusReport(BaseModel):
    dataset_id: str
    name: str
    status: str
    connections: int
    counts: dict[str, int]
    ready: int
    partial: int
    failed: int
    pending: int
    waiting_for_llm: int
    progress_done: int
    progress_total: int
    progress_pct: int
    last_updated: datetime | None = None
    job: dict[str, Any] | None = None


# ─────────────────────────── people ───────────────────────────


class ExperienceOut(BaseModel):
    position: str | None = None
    company_name: str | None = None
    company_linkedin_url: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    is_current: bool = False
    duration_text: str | None = None
    description: str | None = None
    location: str | None = None


class EducationOut(BaseModel):
    school_name: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    end_year: int | None = None


class SkillOut(BaseModel):
    skill_name: str
    source: str
    is_inferred: bool
    confidence: float
    evidence: str | None = None


class PersonOut(BaseModel):
    person_id: str
    is_connection: bool
    linkedin_url: str
    public_identifier: str | None = None
    full_name: str | None = None
    headline: str | None = None
    about: str | None = None
    location_text: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    profile_picture_url: str | None = None
    connections_count: int | None = None
    followers_count: int | None = None
    profile_completeness: int = 0
    enrichment_state: str
    last_scraped_at: datetime | None = None
    experiences: list[ExperienceOut] = []
    education: list[EducationOut] = []
    skills: list[SkillOut] = []
    semantics: dict[str, Any] | None = None


class PersonListItem(BaseModel):
    person_id: str
    full_name: str | None = None
    headline: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    linkedin_url: str
    profile_completeness: int = 0
    enrichment_state: str


# ─────────────────── validated LLM output: profile ───────────────────


class InferredSkill(BaseModel):
    skill: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str

    @field_validator("skill")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("skill must be non-empty")
        return v.strip()


def _coerce_str_list(v):
    """Tolerate the model returning [{'name': x}] / [{'skill': x}] / 'a, b'."""
    if v is None:
        return []
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    out = []
    for item in v if isinstance(v, list) else [v]:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            val = item.get("name") or item.get("skill") or item.get("value") or item.get("keyword")
            if val:
                out.append(str(val).strip())
    return [x for x in out if x]


class SemanticAssertion(BaseModel):
    """One derived professional-concept assertion (spec §7, V4 §13). Provenance-
    tagged, never presented as a verified LinkedIn fact. Links back to the
    normalized rows it was derived from (source IDs beat matching evidence text)."""

    concept: str
    category: str = "industry_experience"
    scope: str = "career"
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    evidence: list[str] = []
    #: normalized row ids this assertion is grounded in (V4 §13) — validated in
    #: semantic_llm._ground(); invalid ids are dropped, never trusted.
    experience_ids: list[str] = []
    education_ids: list[str] = []
    certification_ids: list[str] = []

    @field_validator("concept")
    @classmethod
    def _nonempty_concept(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("concept must be non-empty")
        return v.strip()

    _norm_ids = field_validator(
        "evidence", "experience_ids", "education_ids", "certification_ids", mode="before",
    )(staticmethod(_coerce_str_list))


class ExperienceSemantic(BaseModel):
    """Meaning of ONE work experience (V4 §10/§11). Role function is tracked
    separately from employer industry so "Accountant at Google" and "Software
    Engineer at JPMorgan" behave differently per query."""

    experience_id: str
    role_function: str | None = None          # "software engineering", "accounting"
    professional_domain: str | None = None    # "software systems", "finance"
    role_domains: list[str] = []
    role_seniority: str | None = None
    employer_industries: list[str] = []       # "technology", "financial services"
    employer_categories: list[str] = []       # "big tech", "large bank" (advisory; company_intel is authoritative)
    leadership_signals: list[str] = []
    mentoring_signals: list[str] = []
    founder_signals: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)

    _norm_lists = field_validator(
        "role_domains", "employer_industries", "employer_categories",
        "leadership_signals", "mentoring_signals", "founder_signals", mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("role_function", "professional_domain", "role_seniority", mode="before")
    @classmethod
    def _first_if_list(cls, v):
        # LLMs sometimes wrap a scalar in a list — don't let that sink the whole
        # profile's enrichment (review #7)
        if isinstance(v, (list, tuple)):
            return (str(v[0]).strip() if v else None) or None
        return v


class ProfileSemanticData(BaseModel):
    seniority_level: str | None = None
    job_families: list[str] = []
    technical_domains: list[str] = []
    industries: list[str] = []
    explicit_skills: list[str] = []
    inferred_skills: list[InferredSkill] = []
    leadership_experience: list[str] = []
    domain_expertise: list[str] = []
    career_summary: str | None = None
    years_of_experience: float | None = None
    current_role_summary: str | None = None
    past_company_names: list[str] = []
    current_company_names: list[str] = []
    education_keywords: list[str] = []
    role_keywords: list[str] = []
    searchable_keywords: list[str] = []
    #: derived professional-concept assertions (spec §7) — NOT verified LinkedIn
    #: facts. Each keeps its own evidence + confidence so provenance survives
    #: into search results.
    semantic_assertions: list["SemanticAssertion"] = []
    #: per-experience meaning (V4 §10) — role function vs employer industry, etc.
    experience_semantics: list["ExperienceSemantic"] = []

    _norm_lists = field_validator(
        "job_families",
        "technical_domains",
        "industries",
        "explicit_skills",
        "leadership_experience",
        "domain_expertise",
        "past_company_names",
        "current_company_names",
        "education_keywords",
        "role_keywords",
        "searchable_keywords",
        mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("inferred_skills", mode="before")
    @classmethod
    def _coerce_inferred(cls, v):
        if not v:
            return []
        out = []
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str):
                out.append({"skill": item, "confidence": 0.5, "evidence": ""})
        return out

    @field_validator("semantic_assertions", mode="before")
    @classmethod
    def _coerce_assertions(cls, v):
        if not v:
            return []
        out = []
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, dict) and item.get("concept"):
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append({"concept": item.strip(), "evidence": []})
        return out

    @field_validator("experience_semantics", mode="before")
    @classmethod
    def _coerce_exp_sem(cls, v):
        if not v:
            return []
        return [item for item in (v if isinstance(v, list) else [v])
                if isinstance(item, dict) and item.get("experience_id")]

    @field_validator("years_of_experience", mode="before")
    @classmethod
    def _sane_yoe(cls, v) -> float | None:
        if v is None or v == "":
            return None
        try:
            return max(0.0, min(60.0, float(v)))
        except (TypeError, ValueError):
            return None


# ─────────────────── validated LLM output: company classification ───────────────────


class CompanyClassificationItem(BaseModel):
    key: str  # echoes the request key so we can match batched responses back up
    industries: list[str] = []
    categories: list[str] = []
    is_technology_company: bool | None = None
    is_startup: bool | None = None
    is_big_tech: bool | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: str = ""

    _norm = field_validator("industries", "categories", mode="before")(staticmethod(_coerce_str_list))


class CompanyClassificationBatch(BaseModel):
    companies: list[CompanyClassificationItem] = []


# ─────────────────── validated LLM output: exhaustive semantic judge (V4 PART 3 §14) ───────────────────


class JudgeCriterionVerdict(BaseModel):
    """The judge's read of ONE professional-meaning criterion for one person.
    Grounded in packet evidence references — the backend validates every ref and
    rejects invented ones (V4 PART 3 §13/§17)."""

    criterion_id: str
    #: true (evidence clearly supports it) / false (evidence clearly contradicts
    #: it) / unknown (packet insufficient — NOT the same as false).
    status: str = "unknown"
    match_strength: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""
    #: packet refs ("exp:<id>", "edu:<id>", "cert:<id>", "skill:<norm>",
    #: "assertion:<index>", "company:<key>") that SUPPORT the verdict.
    supporting_evidence_refs: list[str] = []
    contradicting_evidence_refs: list[str] = []
    #: experience_ids the verdict is grounded in (subset of supporting refs).
    experience_ids: list[str] = []

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("true", "false", "unknown") else "unknown"

    _norm_refs = field_validator(
        "supporting_evidence_refs", "contradicting_evidence_refs", "experience_ids",
        mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("match_strength", "confidence", mode="before")
    @classmethod
    def _clamp01(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class JudgePersonVerdict(BaseModel):
    person_id: str
    criteria: list[JudgeCriterionVerdict] = []
    #: informational only — MUST NOT override required criteria (V4 PART 3 §14).
    overall_fit: str = "moderate"
    overall_reason: str = ""

    @field_validator("overall_fit", mode="before")
    @classmethod
    def _norm_fit(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in ("strong", "moderate", "weak", "not_fit") else "moderate"

    @field_validator("criteria", mode="before")
    @classmethod
    def _drop_bad_criteria(cls, v):
        if not v:
            return []
        out = []
        for c in (v if isinstance(v, list) else [v]):
            if isinstance(c, dict):
                if c.get("criterion_id"):
                    out.append(c)
            else:  # already a JudgeCriterionVerdict — let pydantic pass it through
                out.append(c)
        return out


class FinalAuditCriterionReview(BaseModel):
    """The final auditor's consistency check of ONE criterion (V4 PART 5 §13)."""

    criterion_id: str
    #: supported | unsupported | uncertain — is the first-pass status defensible?
    status_review: str = "uncertain"
    reason: str = ""
    supporting_evidence_refs: list[str] = []
    contradicting_evidence_refs: list[str] = []

    @field_validator("status_review", mode="before")
    @classmethod
    def _norm_review(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in ("supported", "unsupported", "uncertain") else "uncertain"

    _norm_refs = field_validator(
        "supporting_evidence_refs", "contradicting_evidence_refs", mode="before",
    )(staticmethod(_coerce_str_list))


class FinalAuditPersonDecision(BaseModel):
    """The final auditor's overall verdict for one candidate (V4 PART 5 §5)."""

    person_id: str
    decision: str = "unknown"          # approved | downgrade | incorrect | unknown
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""
    criteria: list[FinalAuditCriterionReview] = []
    supporting_evidence_refs: list[str] = []
    contradicting_evidence_refs: list[str] = []
    #: advisory ONLY — the backend computes the applied qualification and never
    #: upgrades POSSIBLE -> EXACT from the auditor (§9).
    suggested_qualification: str | None = None

    @field_validator("decision", mode="before")
    @classmethod
    def _norm_decision(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in ("approved", "downgrade", "incorrect", "unknown") else "unknown"

    @field_validator("suggested_qualification", mode="before")
    @classmethod
    def _norm_sq(cls, v):
        if not v:
            return None
        v = str(v).strip().lower().replace(" ", "_")
        return v if v in ("exact_match", "possible_match", "not_match") else None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp01(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    _norm_refs = field_validator(
        "supporting_evidence_refs", "contradicting_evidence_refs", mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("criteria", mode="before")
    @classmethod
    def _drop_bad(cls, v):
        if not v:
            return []
        return [c for c in (v if isinstance(v, list) else [v])
                if not isinstance(c, dict) or c.get("criterion_id")]


class FinalAuditBatch(BaseModel):
    people: list[FinalAuditPersonDecision] = []

    @field_validator("people", mode="before")
    @classmethod
    def _drop_bad_people(cls, v):
        if not v:
            return []
        return [p for p in (v if isinstance(v, list) else [v])
                if not isinstance(p, dict) or p.get("person_id")]


class JudgeBatch(BaseModel):
    people: list[JudgePersonVerdict] = []

    @field_validator("people", mode="before")
    @classmethod
    def _drop_bad_people(cls, v):
        if not v:
            return []
        out = []
        for p in (v if isinstance(v, list) else [v]):
            if isinstance(p, dict):
                if p.get("person_id"):
                    out.append(p)
            else:  # already a JudgePersonVerdict
                out.append(p)
        return out


# ─────────────────── compact LLM TRANSPORT schemas (hardening PART 1/2) ───────────────────
#
# The judge/audit's ONLY job at query time is "does this criterion hold, with
# grounded evidence" — not user-facing prose. These are what the model
# actually returns; the backend expands them into the existing internal
# JudgeCriterionVerdict / FinalAuditCriterionReview shape (match_strength
# derived from status+confidence, experience_ids derived from exp: refs) so
# every downstream validator/scorer is unchanged.


class CompactCriterionVerdict(BaseModel):
    criterion_id: str
    status: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    supporting_refs: list[str] = []
    contradicting_refs: list[str] = []
    #: short ONLY — kept because judge_validator's mentor/passive-mentee check
    #: reads verdict text; NOT a user-facing explanation.
    reason: str = Field(default="", max_length=100)

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("true", "false", "unknown") else "unknown"

    _norm_refs = field_validator(
        "supporting_refs", "contradicting_refs", mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp01(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("reason", mode="before")
    @classmethod
    def _short(cls, v):
        return (str(v or ""))[:100]


class CompactPersonVerdict(BaseModel):
    person_id: str
    criteria: list[CompactCriterionVerdict] = []

    @field_validator("criteria", mode="before")
    @classmethod
    def _drop_bad_criteria(cls, v):
        if not v:
            return []
        return [c for c in (v if isinstance(v, list) else [v])
                if not isinstance(c, dict) or c.get("criterion_id")]


class CompactJudgeBatch(BaseModel):
    people: list[CompactPersonVerdict] = []

    @field_validator("people", mode="before")
    @classmethod
    def _drop_bad_people(cls, v):
        if not v:
            return []
        return [p for p in (v if isinstance(v, list) else [v])
                if not isinstance(p, dict) or p.get("person_id")]


class CompactAuditCriterionReview(BaseModel):
    criterion_id: str
    status_review: str = "uncertain"
    supporting_refs: list[str] = []
    contradicting_refs: list[str] = []
    reason: str = Field(default="", max_length=100)

    @field_validator("status_review", mode="before")
    @classmethod
    def _norm_review(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in ("supported", "unsupported", "uncertain") else "uncertain"

    _norm_refs = field_validator(
        "supporting_refs", "contradicting_refs", mode="before",
    )(staticmethod(_coerce_str_list))

    @field_validator("reason", mode="before")
    @classmethod
    def _short(cls, v):
        return (str(v or ""))[:100]


class CompactAuditPersonDecision(BaseModel):
    person_id: str
    decision: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    criteria: list[CompactAuditCriterionReview] = []

    @field_validator("decision", mode="before")
    @classmethod
    def _norm_decision(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in ("approved", "downgrade", "incorrect", "unknown") else "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp01(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("criteria", mode="before")
    @classmethod
    def _drop_bad(cls, v):
        if not v:
            return []
        return [c for c in (v if isinstance(v, list) else [v])
                if not isinstance(c, dict) or c.get("criterion_id")]


class CompactAuditBatch(BaseModel):
    people: list[CompactAuditPersonDecision] = []

    @field_validator("people", mode="before")
    @classmethod
    def _drop_bad_people(cls, v):
        if not v:
            return []
        return [p for p in (v if isinstance(v, list) else [v])
                if not isinstance(p, dict) or p.get("person_id")]


# ─────────────────── validated LLM output: query ───────────────────


#: unknown types the LLM sometimes emits, mapped to a real first-class type (V4 §6)
_CRITERION_TYPE_ALIASES = {
    "industry": "industry_experience", "sector": "industry_experience",
    "industry_sector": "industry_experience", "employer_category": "company_category",
    "employer_industry": "industry_experience",
    "role": "role_function", "function": "role_function", "job_function": "role_function",
    "profession": "role_function", "domain": "role_function",
    "seniority_level": "seniority", "years_of_experience": "years_experience",
    "tenure": "years_experience", "experience_years": "years_experience",
    "transition": "career_transition", "career_change": "career_transition",
    "concept": "professional_concept", "leadership": "professional_concept",
    "mentorship": "professional_concept", "capability": "professional_concept",
}


class SearchCriterion(BaseModel):
    id: str
    type: str
    weight: float = Field(ge=0)
    required: bool = False

    # backward-compatible single value (every existing scorer reads .value)
    value: str = ""
    # v3 additions — optional, all existing code that only reads .value/.type
    # keeps working unchanged.
    values: list[str] = []
    operator: str = Operator.ANY_OF
    scope: str | None = None
    #: free-text semantic concept description — required for semantic_concept /
    #: company_category, ignored for exact-fact types.
    concept: str | None = None
    #: certainty the query attaches to this requirement (V4 PART 2 §5).
    #: "possible" ("might have X", "possible X experience") keeps the concept as
    #: a soft ranking signal only — it is never a hard filter regardless of
    #: ``required``.
    modality: str = Modality.CERTAIN

    @field_validator("modality")
    @classmethod
    def _known_modality(cls, v: str) -> str:
        v = (v or Modality.CERTAIN).strip().lower()
        return v if v in ALL_MODALITIES else Modality.CERTAIN

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if v in ALL_CRITERION_TYPES:
            return v
        if v in _CRITERION_TYPE_ALIASES:
            return _CRITERION_TYPE_ALIASES[v]
        # NEVER silently fall to `keyword` (V4 §6). An unrecognised semantic-ish
        # type is a professional_concept; only an explicit "keyword"/"text" type
        # produces a literal text search.
        if v in ("text", "phrase", "mention", "literal"):
            return "keyword"
        return "professional_concept"

    @field_validator("operator")
    @classmethod
    def _known_operator(cls, v: str) -> str:
        v = (v or Operator.ANY_OF).strip().upper()
        return v if v in ALL_OPERATORS else Operator.ANY_OF

    @field_validator("scope")
    @classmethod
    def _known_scope(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip().lower()
        return v if v in ALL_SCOPES else None

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v):
        return _coerce_str_list(v)

    @model_validator(mode="after")
    def _sync_value_and_values(self) -> "SearchCriterion":
        if not self.values and self.value:
            self.values = [self.value]
        if not self.value and self.values:
            self.value = self.values[0]
        if not self.value and not self.values and self.concept:
            self.value = self.concept
            self.values = [self.concept]
        return self


class ParsedSearchQuery(BaseModel):
    #: reusable search intent (V4 PART 2 §1) — one of ``ALL_QUERY_INTENTS``.
    #: Shapes which criteria matter; never a search phrase itself.
    intent: str = QueryIntent.PROFESSIONAL_RECOMMENDATION
    #: candidate requirements — what a PERSON must / should look like. Distinct
    #: from ``context`` (event framing) and ``target_person_context`` (the
    #: mentee / the searcher). V4 PART 2 §2: the sentence is NOT turned into one
    #: search phrase.
    criteria: list[SearchCriterion]
    #: non-candidate framing (V4 §14) — e.g. {"purpose": "networking event"}. These
    #: words must NOT become criteria.
    context: dict[str, str] = {}
    #: relational context that SHAPES candidate criteria but is NEVER a search
    #: phrase (V4 PART 2 §3) — e.g. {"field": "backend engineering",
    #: "current_role": "backend engineer", "goal": "engineering management"}.
    #: For "my field" this is filled from the configured current-user profile;
    #: if that is not configured the key is added to ``unresolved`` instead.
    target_person_context: dict[str, str] = {}
    #: context keys the interpreter could NOT resolve (e.g. "field" when the
    #: query said "my field" but no current-user profile is configured). Lowers
    #: interpretation_confidence; the value is never hallucinated.
    unresolved: list[str] = []
    #: one plain-English sentence describing how the query was read (V4 §18)
    interpretation_summary: str = ""

    @field_validator("intent", mode="before")
    @classmethod
    def _known_intent(cls, v) -> str:
        v = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
        if v in ALL_QUERY_INTENTS:
            return v
        if v in _QUERY_INTENT_ALIASES:
            return _QUERY_INTENT_ALIASES[v]
        return QueryIntent.PROFESSIONAL_RECOMMENDATION

    @field_validator("unresolved", mode="before")
    @classmethod
    def _norm_unresolved(cls, v):
        return _coerce_str_list(v)
    #: 0..1 — lower for semantically ambiguous queries ("people who worked in tech")
    interpretation_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    #: hard upper bound set by the plan validator on an unresolved structural
    #: problem (V4 §9) — _finalize() must not raise confidence above this.
    interpretation_confidence_cap: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalize_weights(self) -> "ParsedSearchQuery":
        if not self.criteria:
            raise ValueError("query produced no criteria")
        total = sum(c.weight for c in self.criteria)
        if total <= 0:
            equal = 100.0 / len(self.criteria)
            for c in self.criteria:
                c.weight = equal
        elif abs(total - 100.0) > 0.5:
            for c in self.criteria:
                c.weight = round(c.weight / total * 100, 2)
        return self


# ─────────────────────── search results ───────────────────────


class EvidenceItem(BaseModel):
    type: str  # experience | education | skill | headline | about | semantic
    text: str
    detail: dict[str, Any] = {}


class ScoreComponent(BaseModel):
    criterion: str
    criterion_id: str
    type: str
    weight: float
    match_strength: float
    score: float
    required: bool = False
    evidence: list[EvidenceItem] = []


class SearchResultItem(BaseModel):
    rank: int
    person_id: str
    name: str | None = None
    linkedin_url: str
    profile_picture_url: str | None = None
    current_title: str | None = None
    current_company: str | None = None
    location: str | None = None
    is_connection: bool = True
    match_score: float
    data_confidence: int
    reason: str
    #: V4 §22-25 — exact_match / possible_match / not_match (near-matches only)
    qualification: str = "possible_match"
    #: required criteria still uncertain (why this is possible_match not exact)
    uncertain_criteria: list[str] = []
    #: required criteria this person FAILS (near-match rows only)
    unmet_criteria: list[str] = []
    matched_criteria: list[str] = []
    score_breakdown: list[ScoreComponent] = []
    evidence: list[EvidenceItem] = []
    relevant_experience: list[ExperienceOut] = []
    relevant_skills: list[SkillOut] = []
    relevant_education: list[EducationOut] = []
    #: V4 PART 5 §26 — final-audit outcome (optional, backward-compatible).
    audit_decision: str | None = None
    audit_confidence: float | None = None
    audit_reason: str | None = None
    audit_issues: list[str] = []
    #: True only when the final auditor APPROVED this candidate against validated
    #: evidence — a stronger signal than the deterministic qualification alone.
    llm_verified: bool = False


class ConnectionBucket(BaseModel):
    total_candidates: int
    returned: int
    results: list[SearchResultItem]
    #: V4 §22-26
    exact_match_count: int = 0
    possible_match_count: int = 0
    #: candidates that miss exactly one required criterion — clearly labelled,
    #: never mixed into `results` (V4 §26)
    near_matches: list[SearchResultItem] = []


class ExternalBucket(BaseModel):
    searched: bool = False
    total_candidates: int = 0
    returned: int = 0
    results: list[SearchResultItem] = []


class SearchRequest(BaseModel):
    dataset_id: str
    query: str = Field(min_length=2)


class SearchResponse(BaseModel):
    search_id: str
    query: str
    interpreted_query: dict[str, Any]
    connections: ConnectionBucket
    external: ExternalBucket
    llm_provider: str | None = None
    llm_model: str | None = None
    #: V4 PART 3 §32 — optional observability block for the exhaustive judge run
    #: (mode, status, candidate/batch counts, providers). Backward-compatible:
    #: absent on searches that did not run the judge.
    judge_metadata: dict[str, Any] | None = None
    #: V4 PART 5 §25 — optional observability block for the final result audit
    #: (enabled, status, pool/batch counts, decision tally, providers). Persisted
    #: post-``finalize()`` in ``search_run_states`` and rebuilt by ``load_search``
    #: (V4 PART 7).
    audit_metadata: dict[str, Any] | None = None
    #: hardening PART 6 — per-search LLM call tally (query_interpretation,
    #: semantic_judge, final_audit, reason_generation, total) + budget info. No
    #: prompts, no profile data. Live-response only for now (not yet persisted —
    #: same bootstrapping step judge/audit metadata went through before PART 7).
    llm_calls: dict[str, Any] | None = None
