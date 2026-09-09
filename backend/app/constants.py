"""Shared string constants for state machines and enumerations.

Kept as plain strings (not DB enums) so SQLite stays flexible and Postgres can
adopt real enums later without a data migration.
"""
from __future__ import annotations


class EnrichmentState:
    """Per-person enrichment progress (spec §54)."""

    PENDING = "PENDING"
    APIFY_RUNNING = "APIFY_RUNNING"
    APIFY_COMPLETE = "APIFY_COMPLETE"
    NORMALIZED = "NORMALIZED"
    LLM_PENDING = "LLM_PENDING"
    LLM_COMPLETE = "LLM_COMPLETE"
    READY = "READY"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    WAITING_FOR_FREE_LLM = "WAITING_FOR_FREE_LLM"


#: States the worker treats as "done, do not reprocess this run".
TERMINAL_STATES = {
    EnrichmentState.READY,
    EnrichmentState.PARTIAL,
    EnrichmentState.FAILED,
}

#: States that still need the Apify scrape step.
NEEDS_APIFY = {EnrichmentState.PENDING, EnrichmentState.APIFY_RUNNING}


class JobStatus:
    """Enrichment job status (spec §8)."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class DatasetStatus:
    CREATED = "CREATED"
    READY_FOR_ENRICHMENT = "READY_FOR_ENRICHMENT"
    ENRICHING = "ENRICHING"
    READY = "READY"


class SkillSource:
    """Provenance of a skill row (spec §17)."""

    PROFILE = "linkedin_profile_skill"
    EXPERIENCE = "linkedin_experience_skill"
    EDUCATION = "linkedin_education_skill"
    DESCRIPTION = "linkedin_description"
    LLM = "llm_inference"


class RawSource:
    APIFY_HARVESTAPI = "apify_harvestapi"


class CriterionType:
    """Types the query interpreter may emit (spec §30, extended v3)."""

    CURRENT_COMPANY = "current_company"
    PAST_COMPANY = "past_company"
    SKILL = "skill"
    DOMAIN = "domain"
    TITLE = "title"
    EDUCATION = "education"
    LOCATION = "location"
    SENIORITY = "seniority"
    KEYWORD = "keyword"
    CERTIFICATION = "certification"
    LANGUAGE = "language"
    PUBLICATION = "publication"
    #: a professional concept that is NOT a literal string to search for —
    #: industry experience, role/function, leadership, mentorship, etc.
    #: Evaluated via semantic_assertions -> company classification -> semantic
    #: fields -> cross-encoder -> LLM judge (never phrase_matches alone).
    SEMANTIC_CONCEPT = "semantic_concept"
    #: a company-level classification (startup / big_tech / fintech / consulting
    #: / healthcare_tech / …) evaluated against the person's ACTUAL employer(s)
    #: via CompanySemantic, not against literal text in their profile.
    COMPANY_CATEGORY = "company_category"

    # ── V4 first-class professional types (V4 §6) ──────────────
    #: "worked in tech" / "fintech experience" — the INDUSTRY of the person's
    #: employers, evaluated from experience-level employer_industries + assertions.
    INDUSTRY_EXPERIENCE = "industry_experience"
    #: "software engineers" / "product managers" — the FUNCTION of the person's
    #: role, independent of employer industry (V4 §11).
    ROLE_FUNCTION = "role_function"
    #: any other professional concept that is not literal text and not one of the
    #: above (leadership, mentorship, "enterprise sales experience", …). The safe
    #: landing type for an unrecognised semantic-ish type (never `keyword`).
    PROFESSIONAL_CONCEPT = "professional_concept"
    #: "moved from consulting to tech" — ordered from→to over the career (V4 §19).
    CAREER_TRANSITION = "career_transition"
    #: "10+ years of backend experience" — minimum duration in a domain (V4 §21).
    YEARS_EXPERIENCE = "years_experience"


ALL_CRITERION_TYPES = {
    v for k, v in vars(CriterionType).items() if not k.startswith("_") and isinstance(v, str)
}

#: criterion types that must be evaluated as semantic concepts, never phrase_matches
SEMANTIC_CRITERION_TYPES = {
    CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY,
    CriterionType.INDUSTRY_EXPERIENCE, CriterionType.ROLE_FUNCTION,
    CriterionType.PROFESSIONAL_CONCEPT, CriterionType.CAREER_TRANSITION,
}


class Qualification:
    """Candidate-level match tier (V4 §22–§25). Ranked before match_score."""

    EXACT_MATCH = "exact_match"      # every required criterion is TRUE
    POSSIBLE_MATCH = "possible_match"  # no required FALSE, but a required semantic is UNKNOWN
    NOT_MATCH = "not_match"          # a required criterion is confidently FALSE / unmet


_QUALIFICATION_RANK = {
    Qualification.EXACT_MATCH: 0,
    Qualification.POSSIBLE_MATCH: 1,
    Qualification.NOT_MATCH: 2,
}


class JudgeMode:
    """How many candidates reach the LLM semantic judge (V4 PART 3 §8)."""

    OFF = "off"                        # no semantic judge at all
    UNCERTAIN_ONLY = "uncertain_only"  # only ambiguity-band candidates (cheap mode)
    ALL_VIABLE = "all_viable"          # EVERY candidate past the hard-fact gate


ALL_JUDGE_MODES = {JudgeMode.OFF, JudgeMode.UNCERTAIN_ONLY, JudgeMode.ALL_VIABLE}


class JudgeStatus:
    """Outcome of the semantic-judge run for a search (V4 PART 3 §29)."""

    FULL = "full"                # every viable candidate got a validated verdict
    PARTIAL = "partial"          # some batches failed / model omitted people or criteria
    NOT_USED = "not_used"        # mode=off, judging disabled, or no judgeable criteria
    UNAVAILABLE = "unavailable"  # every provider failed — no verdicts at all


#: criterion types the LLM judge is allowed to decide (professional MEANING).
#: Everything else is deterministic / code-authoritative (V4 PART 3 §15/§16).
JUDGEABLE_CRITERION_TYPES = {
    CriterionType.SEMANTIC_CONCEPT,
    CriterionType.PROFESSIONAL_CONCEPT,
    CriterionType.ROLE_FUNCTION,
    CriterionType.INDUSTRY_EXPERIENCE,
    CriterionType.COMPANY_CATEGORY,
}

#: the judge may be SHOWN the deterministic verdict for these as context, but it
#: can never change verified ordering / durations (V4 PART 3 §16).
CODE_AUTHORITATIVE_CRITERION_TYPES = {
    CriterionType.CAREER_TRANSITION,
    CriterionType.YEARS_EXPERIENCE,
}


class AuditDecision:
    """The final result auditor's verdict for one candidate (V4 PART 5 §8).

    A correctness BRAKE only — it can keep, downgrade or remove a candidate, it
    can never upgrade POSSIBLE -> EXACT or invent a numeric score.
    """

    APPROVED = "approved"      # the existing qualification is supported
    DOWNGRADE = "downgrade"    # relevant, but something "verified" is actually uncertain
    INCORRECT = "incorrect"    # a required condition clearly fails — remove from results
    UNKNOWN = "unknown"        # the audit could not reliably decide — be conservative


ALL_AUDIT_DECISIONS = {
    AuditDecision.APPROVED, AuditDecision.DOWNGRADE, AuditDecision.INCORRECT, AuditDecision.UNKNOWN,
}


class AuditStatus:
    """Outcome of the final-audit pass for a search (V4 PART 5 §20/§25)."""

    FULL = "full"                # every audit-pool candidate got a validated decision
    PARTIAL = "partial"          # a batch failed / the model omitted people
    NOT_USED = "not_used"        # audit disabled or nothing to audit
    UNAVAILABLE = "unavailable"  # every provider failed — no decisions at all


class QueryIntent:
    """What the searcher is actually trying to accomplish (V4 PART 2 §1).

    Reusable across arbitrary professional-network questions — NOT one branch per
    phrasing. The intent shapes which candidate criteria matter and how strict
    they are; it never becomes a search phrase itself.
    """

    #: "find <people matching a description>" — an interrogative / imperative
    #: lookup with concrete-ish criteria ("who are the data scientists in NYC").
    FIND_PEOPLE = "find_people"
    #: "who should I reach out to about X" — a soft, ranked recommendation with
    #: no single hard filter. The default for a declarative noun phrase.
    PROFESSIONAL_RECOMMENDATION = "professional_recommendation"
    #: "who could mentor / advise / coach <someone>" — relational: evidence of
    #: developing people, judged against a mentee context, never a literal
    #: "mentor" match and never "senior => mentor".
    MENTOR_RECOMMENDATION = "mentor_recommendation"
    #: "experts in X" / "deep knowledge of X" — depth in a subject, not just
    #: incidental exposure.
    SUBJECT_MATTER_EXPERTISE = "subject_matter_expertise"
    #: "people who moved from A to B" / "A -> B transitions" — an ordered career
    #: change is the point of the query.
    CAREER_TRANSITION = "career_transition"
    #: "who should I invite to <event>" — the event is context; candidate
    #: criteria are whatever describes the guest, minus the event framing.
    NETWORKING_INVITATION = "networking_invitation"


ALL_QUERY_INTENTS = {
    v for k, v in vars(QueryIntent).items() if not k.startswith("_") and isinstance(v, str)
}

#: intent words / near-synonyms an LLM (or a loose caller) might emit
_QUERY_INTENT_ALIASES = {
    "find": "find_people", "search": "find_people", "lookup": "find_people",
    "people_search": "find_people", "discovery": "find_people",
    "recommend": "professional_recommendation", "recommendation": "professional_recommendation",
    "referral": "professional_recommendation", "reach_out": "professional_recommendation",
    "mentor": "mentor_recommendation", "mentorship": "mentor_recommendation",
    "advice": "mentor_recommendation", "advising": "mentor_recommendation",
    "coaching": "mentor_recommendation", "guidance": "mentor_recommendation",
    "expert": "subject_matter_expertise", "expertise": "subject_matter_expertise",
    "sme": "subject_matter_expertise", "subject_matter_expert": "subject_matter_expertise",
    "specialist": "subject_matter_expertise",
    "transition": "career_transition", "career_change": "career_transition",
    "career_move": "career_transition", "pivot": "career_transition",
    "networking": "networking_invitation", "invite": "networking_invitation",
    "invitation": "networking_invitation", "event": "networking_invitation",
    "introduction": "networking_invitation",
}


class Modality:
    """Certainty the query attaches to a requirement (V4 PART 2 §5).

    "HIPAA compliance experts" is CERTAIN; "might have HIPAA compliance
    experience" / "possible HIPAA experience" is POSSIBLE — a soft signal that
    can lift a ranking but must never be a hard filter.
    """

    CERTAIN = "certain"
    POSSIBLE = "possible"


ALL_MODALITIES = {Modality.CERTAIN, Modality.POSSIBLE}


class Operator:
    """How a criterion's ``values`` combine (spec §13)."""

    ANY_OF = "ANY_OF"
    ALL_OF = "ALL_OF"
    NOT = "NOT"


ALL_OPERATORS = {Operator.ANY_OF, Operator.ALL_OF, Operator.NOT}


class Scope:
    """Which part of a career a criterion applies to (spec §1)."""

    CURRENT = "current"
    PAST = "past"
    ANY_EXPERIENCE = "any_experience"
    CAREER = "career"
    CURRENT_COMPANY = "current_company"
    PAST_COMPANY = "past_company"


ALL_SCOPES = {v for k, v in vars(Scope).items() if not k.startswith("_") and isinstance(v, str)}


class TriState:
    """Three-valued fact status — missing data is UNKNOWN, never FALSE (spec §15)."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class AssertionCategory:
    """``semantic_assertions[].category`` on ProfileSemanticData (spec §7)."""

    INDUSTRY_EXPERIENCE = "industry_experience"
    ROLE_FUNCTION = "role_function"
    LEADERSHIP = "leadership"
    MENTORSHIP = "mentorship"
    COMPANY_CATEGORY = "company_category"
    DOMAIN_EXPERTISE = "domain_expertise"


class CompanyClassProvenance:
    LLM_INFERENCE = "ai_company_inference"
    UNKNOWN = "unknown"


class LLMProviderName:
    ANTHROPIC = "anthropic:paid"
    GROQ_PRIMARY = "groq:primary"
    GROQ_FALLBACK = "groq:fallback"
    OPENROUTER = "openrouter:free"
    NONE = "none"
