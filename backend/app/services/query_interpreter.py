"""Natural-language query → a semantic search plan (spec §1–§3, §13, §30–§32, v3).

Primary path: free-LLM chain producing a ``ParsedSearchQuery`` whose criteria
can be exact facts (company/school/location/...) OR semantic concepts
(industry experience, company category, leadership, mentorship, ...).
MEANING over word occurrence: "worked in tech" must become a
``semantic_concept``, never a literal company/keyword search; "startup" must
become a ``company_category`` evaluated against the person's actual employer,
never a company named "startup"; "Memphis or Nashville" must become one
criterion with two values and ANY_OF, never a single mangled string.

Fallback: a deterministic regex parser that always runs when every LLM
provider is down. It preserves the concrete, checkable parts of a query
(locations incl. OR, companies, schools, seniority, titles) — it does NOT try
to fabricate semantic-concept understanding without a model; unresolvable
concepts become low-weight ``semantic_concept``/``keyword`` criteria rather
than being silently dropped.
"""
from __future__ import annotations

import logging
import re

from app.config import settings
from app.constants import CriterionType, Operator, Scope
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.llm.router import generate_structured
from app.services.matching import concept_overlap, seniority_rank
from app.services.query_facts import (
    build_summary,
    extract_facts,
    strip_context,
    validate_and_repair,
)
from app.services.query_intent import augment_plan

log = logging.getLogger("app.query")

_KNOWN_SKILLS = {
    "aws", "gcp", "azure", "java", "python", "golang", "go", "rust", "c++", "typescript",
    "javascript", "react", "node", "node.js", "kubernetes", "docker", "terraform", "kafka",
    "spark", "sql", "postgresql", "mysql", "redis", "mongodb", "graphql", "distributed systems",
    "machine learning", "ml", "deep learning", "nlp", "llm", "llms", "pytorch", "tensorflow",
    "data engineering", "mlops", "devops", "security", "cryptography", "blockchain",
    "microservices", "system design", "cloud", "cloud infrastructure", "networking",
    "product management", "design", "ui", "ux", "fintech", "payments", "fraud",
}

_SYSTEM = (
    "You convert a recruiter/networking search into a structured search plan. Output JSON only.\n\n"
    "Each criterion has: id (short slug), type, weight (points, all criteria sum to 100), "
    "required (bool). Depending on type, also set:\n"
    "  - value OR values (a list) — for exact facts: current_company, past_company, education, "
    "location, certification, language, publication, title, skill, keyword. Use `values` with "
    "several entries + operator=ANY_OF for an OR ('Memphis or Nashville' -> values: "
    "[\"Memphis\",\"Nashville\"], operator: ANY_OF). Use operator=ALL_OF when the query clearly "
    "wants both ('Amazon and Microsoft experience' -> past_company values [Amazon, Microsoft], "
    "ALL_OF). operator=NOT excludes ('not currently at Amazon').\n"
    "  - concept + scope — for semantic_concept and company_category ONLY (see below).\n\n"
    "CRITICAL - distinguish FACTS from CONCEPTS:\n"
    "A fact is something that either literally appears on a LinkedIn profile as a field: a company "
    "name, a school, a city, a certification, a language, an explicit skill. Use the concrete "
    "current_company/past_company/education/location/certification/language/skill types for these, "
    "with the real-world value the user means (e.g. 'Google', 'Stanford', 'Nashville').\n"
    "A CONCEPT is something that must be judged from someone's career as a whole - it is NEVER a "
    "literal string to search for on the profile:\n"
    "  - 'startup' / 'big tech' / 'FAANG' / 'consulting firm' as a description of WHERE someone "
    "works -> type: company_category, concept: the category name ('startup', 'big tech', "
    "'consulting'), scope: current_company or past_company or any_experience depending on the "
    "query ('now at a startup' -> scope current_company; 'ex-big-tech' -> scope past_company). "
    "NEVER emit current_company with value 'startup' or 'FAANG' - those are not company names.\n"
    "  - 'engineering leader' / 'mentor' / 'advisor' / 'coach' -> type: professional_concept, "
    "concept ('engineering leadership', 'mentors other engineers'), scope: career.\n"
    "  - 'software engineers' / 'ML engineers' / 'product managers' / 'data scientists' / "
    "'technical backgrounds' / 'technical people' -> type: role_function, concept: the canonical "
    "function ('software engineering', 'a technical/engineering role'), scope: career. This is the "
    "FUNCTION the person actually performs, independent of their employer's industry: a software "
    "engineer at a bank still has a technical role_function even though the bank is not a tech "
    "company.\n"
    "  - 'worked in tech' / 'technology industry experience' / 'fintech experience' / 'healthcare "
    "experience' (an industry/domain framing - 'experience IN <industry>') -> type: "
    "industry_experience, concept: a short description of the industry (e.g. 'professional "
    "experience in the technology industry'), scope: any_experience or career. This is about the "
    "EMPLOYER'S industry, independent of what the person's own role_function was: an accountant at "
    "a healthcare company has healthcare industry_experience without being a clinician.\n"
    "  - THREE DIFFERENT DIMENSIONS, do not collapse them into one type: role_function = what the "
    "person DOES; industry_experience = what industry their EMPLOYER is in; company_category = a "
    "classification of the SPECIFIC EMPLOYER (startup/big tech/etc). 'worked in tech' is "
    "industry_experience (the employer's industry). 'worked AT tech companies' / 'tech-company "
    "employment' is company_category. 'technical background' / 'technical people' is role_function. "
    "NEVER emit current_company/keyword with value 'tech' or 'FAANG' for any of these three.\n"
    "  - if a concept is genuinely broad/ambiguous and does not cleanly fit role_function, "
    "industry_experience, or company_category, fall back to semantic_concept with a short "
    "description, scope: any_experience or career.\n"
    "  - 'moved from consulting to tech' / 'left big tech for startups' -> type: career_transition, "
    "concept: 'from <A> to <B>'. '10+ years of backend experience' -> type: years_experience, "
    "value: '10', concept: 'at least 10 years in backend engineering'.\n"
    "  - AND vs OR: 'Google or Meta' -> operator ANY_OF; 'Amazon and Microsoft experience' (both) "
    "-> operator ALL_OF. This applies to semantic concepts too: 'security or cloud experts' -> "
    "role_function/professional_concept values [security, cloud] ANY_OF.\n"
    "  - NEVER emit type 'keyword' unless the user literally asks for text matching ('profiles "
    "mentioning Kubernetes'). An unusual concept is 'professional_concept', never 'keyword'.\n"
    "  - seniority words (CXO, senior, staff, director, VP, founder) -> type: seniority, value: the "
    "word. 'CXO' covers CEO/CTO/CFO/COO/CMO/CPO/Chief-anything - do not require the literal word "
    "'CXO' on the profile.\n\n"
    "Weights reflect the query's own emphasis. A criterion is required only when the query clearly "
    "demands it as a filter, not a preference - 'currently at Google who knows Java' -> Google "
    "required, Java preferred. 'Former Amazon people now at startups' -> BOTH past_company(Amazon) "
    "and company_category(startup, scope current_company) are required - it is a single query about "
    "two facts that must both hold, not a nice-to-have. Weights MUST sum to 100.\n\n"
    "REQUIREDNESS FROM MEANING (do not wait for the words 'must'/'only'): the plural noun that is "
    "the SUBJECT of the query is always required. 'CXOs in Nashville' -> seniority(CXO) AND "
    "location(Nashville) BOTH required. 'former Amazon employees' -> past_company(Amazon) required. "
    "'software engineers at fintech companies' -> role and fintech-employer both required.\n\n"
    "CONTEXT vs CANDIDATE REQUIREMENT: words describing the EVENT or PURPOSE are not candidate "
    "criteria. 'people to invite to a networking event' -> networking is context, NOT skill=networking. "
    "'speak at an AI conference' -> AI expertise may be a criterion, 'conference' is context. Put "
    "event/purpose framing in the top-level `context` object ({\"purpose\": \"...\"}), never as a "
    "criterion. Contrast: 'people with professional networking expertise' -> networking IS a criterion.\n\n"
    "Also set `interpretation_summary` (one sentence: how you read the query) and "
    "`interpretation_confidence` (0..1; lower it for vague queries like 'people who worked in tech').\n\n"
    "UNIVERSAL REPRESENTATION (V4 PART 2) — the plan must generalize beyond "
    "company/title/location/skill:\n"
    "  - `intent`: one of find_people, professional_recommendation, "
    "mentor_recommendation, subject_matter_expertise, career_transition, "
    "networking_invitation. Pick what the searcher is trying to DO.\n"
    "  - `target_person_context`: {field, current_role, goal} for the person the "
    "candidate must help — the mentee in 'mentor for a backend engineer moving "
    "into management' ({current_role: 'backend engineer', field: 'backend "
    "engineering', goal: 'engineering management'}), or the searcher for 'anyone "
    "in my field'. This is NEVER a criterion / search phrase. For 'my field' use "
    "the searcher profile line below if present; if it is absent, leave the field "
    "out and add 'field' to `unresolved` — do NOT guess it.\n"
    "  - Do NOT turn a relational sentence into one search phrase. Decompose it "
    "into candidate criteria: 'who could mentor a backend engineer moving into "
    "management' -> criteria: engineering-management/leadership experience "
    "(required, professional_concept), evidence of mentoring/coaching/advising "
    "others (required), familiarity with backend engineering (preferred), similar "
    "IC->manager trajectory (preferred). Never require the literal word 'mentor'. "
    "Seniority alone does NOT imply mentor.\n"
    "  - CROSS-DOMAIN AND: 'cybersecurity and healthcare', 'AI and healthcare', "
    "'research plus industry experience' -> TWO separate REQUIRED criteria (both "
    "must hold), not one blended concept and not ANY_OF.\n"
    "  - MODALITY: 'might have HIPAA experience' / 'possible HIPAA compliance "
    "experience' -> set `modality: possible` on that criterion, required=false, "
    "low weight. It must NOT be identical to 'HIPAA compliance experts'.\n"
    "  - ACADEMIA: keep these distinct — professor/faculty EMPLOYMENT "
    "(professional_concept 'faculty appointment'), research EMPLOYMENT, a "
    "university DEGREE (education), and PUBLICATIONS (publication). Studying at a "
    "university is NOT working in academia; one paper is NOT a professor. "
    "'professors in AI' -> faculty-appointment (required) + research/teaching "
    "focus on AI (required), NOT education=AI."
)


_MODAL_RE = re.compile(r"\b(must|only|required|has to|need to have|exclusively)\b", re.I)
#: types the model tends to over-require; soften unless the query uses modal language.
#: Semantic/company-category/location/company/education criteria are NOT auto-softened -
#: whether they are hard filters is a real per-query judgment, not a blanket rule
#: (spec §2/§3 — "Former Amazon people now at startups" needs BOTH required).
_SOFT_TYPES = {CriterionType.SKILL, CriterionType.KEYWORD, CriterionType.TITLE}


def _soften_requirements(parsed: ParsedSearchQuery, query: str) -> ParsedSearchQuery:
    """Skills/keywords/titles are hard requirements only when the query says so
    ('must know X'). Cap total required criteria so an over-eager model can't
    exclude everyone."""
    hard_modal = bool(_MODAL_RE.search(query))
    for c in parsed.criteria:
        if c.type in _SOFT_TYPES and not hard_modal:
            c.required = False
    required = [c for c in parsed.criteria if c.required]
    if len(required) > 3:
        for c in sorted(required, key=lambda x: x.weight)[:-3]:
            c.required = False
    return parsed


def interpret_query(query: str) -> tuple[ParsedSearchQuery, str, str | None]:
    """Return ``(parsed, provider_name, model)``. provider = 'deterministic' on
    fallback. Every path runs the deterministic fact layer (V4 §15) so explicit
    locations / companies / OR / NOT survive even a perfect-looking LLM plan and
    even a total LLM outage."""
    cleaned, context = strip_context(query)
    facts = extract_facts(cleaned, context=context)

    if settings.llm_query_interpretation:
        result = generate_structured(
            _SYSTEM,
            f"Search query: {cleaned!r}\n"
            + (f"Event/context (NOT a candidate requirement): {context['purpose']!r}\n" if context.get("purpose") else "")
            + (
                f"Searcher profile (only for resolving 'my ...' references): "
                f"field={settings.user_field!r}"
                + (f", current_role={settings.user_current_role!r}" if settings.user_current_role else "")
                + (f", goal={settings.user_goal!r}" if settings.user_goal else "")
                + "\n"
                if settings.user_field else ""
            )
            + "Produce the search-plan JSON.",
            ParsedSearchQuery,
            max_tokens=1200,
            operation="query_interpretation",
            timeout=settings.query_interpretation_timeout_seconds,
        )
        if result is not None:
            parsed, provider, model = result
            if parsed.criteria:
                parsed = _soften_requirements(parsed, query)
                parsed, issues = validate_and_repair(parsed, query, facts, context=context)
                _finalize(parsed, query, issues)
                return parsed, provider, model
        log.info("query interpreter: falling back to deterministic parser")

    parsed = _soften_requirements(_deterministic_parse(cleaned), query)
    parsed, issues = validate_and_repair(parsed, query, facts, context=context)
    _finalize(parsed, query, issues)
    return parsed, "deterministic", None


#: the plural noun that is the SUBJECT of a query is always a hard filter, even
#: without "must"/"only" (V4 §19). "CXOs in Nashville" -> CXO required.
_SUBJECT_RE = re.compile(
    r"^\s*(?:the\s+|former\s+|ex[- ]|current\s+|senior\s+|top\s+|good\s+)*"
    r"(cxos?|ceos?|ctos?|cfos?|coos?|executives?|founders?|co-?founders?|"
    r"engineers?|developers?|architects?|managers?|directors?|vps?|"
    r"consultants?|researchers?|scientists?|designers?|leaders?|mentors?|"
    r"analysts?|recruiters?|marketers?|salespeople|advisors?)\b",
    re.I,
)
_SUBJECT_TYPES = {
    CriterionType.SENIORITY, CriterionType.TITLE, CriterionType.SEMANTIC_CONCEPT,
    CriterionType.ROLE_FUNCTION, CriterionType.INDUSTRY_EXPERIENCE,
}


_CXO_WORDS_RE = re.compile(r"\b(cxo|ceo|cto|cfo|coo|cmo|cpo|ciso|cio|chief|executive)s?\b", re.I)
#: "<subject> at fintech companies" / "in financial services" -> the employer
#: constraint is part of the subject, so it is required too (V4 §7).
_SUBJECT_EMPLOYER_RE = re.compile(
    r"\b(?:at|in|for|with)\s+(?:a\s+|an\s+)?[\w /-]*?\b"
    r"(fintech|consulting|healthcare|biotech|banking|financial services|"
    r"insurance|retail|manufacturing|enterprise software|big tech|startups?)\b",
    re.I,
)


#: query opens with (fillers) then a professional-role phrase
_ROLE_SUBJECT_RE = re.compile(
    r"^\s*(?:the\s+|find\s+|list\s+|show\s+(?:me\s+)?|people\s+(?:who\s+are\s+)?|"
    r"senior\s+|staff\s+|lead\s+|principal\s+|former\s+|ex[- ]|top\s+)*"
    r"(software engineers?|ml engineers?|machine learning engineers?|data scientists?|"
    r"data engineers?|security engineers?|backend engineers?|frontend engineers?|"
    r"product managers?|engineering managers?|sales leaders?|designers?|researchers?|"
    r"engineers?|developers?|scientists?|managers?|consultants?|analysts?)\b",
    re.I,
)


def _promote_subject(parsed: ParsedSearchQuery, query: str) -> None:
    # a seniority/role word in the query's SUBJECT or event PURPOSE is a filter
    subject = _SUBJECT_RE.match(query) or _ROLE_SUBJECT_RE.match(query)
    purpose = parsed.context.get("purpose", "")
    cxo_context = bool(_CXO_WORDS_RE.search(purpose))
    promoted = False
    if subject or cxo_context:
        word = subject.group(1).rstrip("s").lower() if subject else ""
        for c in parsed.criteria:
            text = f"{c.value} {c.concept} {' '.join(c.values)}".lower()
            if c.type == CriterionType.SENIORITY and (cxo_context or subject):
                c.required = True
                promoted = True
                break
            if c.type in _SUBJECT_TYPES and word and (word in text or c.type == CriterionType.ROLE_FUNCTION):
                c.required = True
                promoted = True
                break
        # a role_function anywhere counts as the subject even if the sentence
        # starts with a filler ("people who are software engineers ...")
        if not promoted:
            for c in parsed.criteria:
                if c.type == CriterionType.ROLE_FUNCTION:
                    c.required = True
                    promoted = True
                    break

    # "<role> at fintech companies" -> the employer category is required too
    emp = _SUBJECT_EMPLOYER_RE.search(query)
    if emp and (promoted or any(c.type == CriterionType.ROLE_FUNCTION for c in parsed.criteria)):
        cat = emp.group(1).lower().rstrip("s")
        for c in parsed.criteria:
            if c.type in (CriterionType.COMPANY_CATEGORY, CriterionType.INDUSTRY_EXPERIENCE) \
                    and cat in (c.concept or c.value or "").lower():
                c.required = True
                return


#: types the judge/audit evaluate as MEANING, not a literal profile field (mirrors
#: scoring._SEMANTIC_TYPES minus the code-authoritative career_transition /
#: years_experience, which are never semantic duplicates of anything else).
_MEANING_TYPES = {
    CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY,
    CriterionType.PROFESSIONAL_CONCEPT, CriterionType.INDUSTRY_EXPERIENCE,
    CriterionType.ROLE_FUNCTION,
}
#: the generic catch-all type — dropped first when it overlaps a more specific one
_GENERIC_MEANING_TYPE = CriterionType.SEMANTIC_CONCEPT
#: concept_overlap() >= this is treated as "the same underlying meaning" (1.0 for
#: substring containment, else Jaccard token overlap)
_DUPLICATE_OVERLAP_MIN = 0.6


def _concept_text(c: SearchCriterion) -> str:
    return c.concept or c.value or " ".join(c.values or [])


def _pick_primary(a: SearchCriterion, b: SearchCriterion) -> SearchCriterion:
    """Which of two duplicate criteria survives — prefer a specific type (e.g.
    professional_concept, industry_experience) over the generic semantic_concept
    catch-all, then the larger weight, tie-broken by insertion order (``a``)."""
    if a.type == _GENERIC_MEANING_TYPE and b.type != _GENERIC_MEANING_TYPE:
        return b
    if b.type == _GENERIC_MEANING_TYPE and a.type != _GENERIC_MEANING_TYPE:
        return a
    return a if a.weight >= b.weight else b


def _dedupe_semantic_duplicates(parsed: ParsedSearchQuery) -> None:
    """Collapse two criteria that judge the SAME underlying meaning (V4
    hardening PART 8) — e.g. a "research experience" professional_concept and a
    "research" role_function for the same query are one requirement asked
    twice, not two. This doubles judge/audit load for nothing and can silently
    over-tighten an AND query. Detected GENERICALLY by concept-text overlap
    within the same type-family + scope — never by hardcoded query words. A
    real cross-domain AND ("cybersecurity AND healthcare") has near-zero
    concept-text overlap and is untouched, preserving legitimate multi-criteria
    requirements."""
    kept: list[SearchCriterion] = []
    for c in parsed.criteria:
        if c.type not in _MEANING_TYPES:
            kept.append(c)
            continue
        dup_idx = next(
            (i for i, k in enumerate(kept)
             if k.type in _MEANING_TYPES and k.scope == c.scope
             and concept_overlap(_concept_text(k), _concept_text(c)) >= _DUPLICATE_OVERLAP_MIN),
            None,
        )
        if dup_idx is None:
            kept.append(c)
            continue
        dup = kept[dup_idx]
        primary = _pick_primary(dup, c)
        primary.required = dup.required or c.required
        primary.modality = dup.modality if dup.required else (c.modality if c.required else primary.modality)
        primary.weight = round(dup.weight + c.weight, 2)
        kept[dup_idx] = primary
    parsed.criteria[:] = kept


def _finalize(parsed: ParsedSearchQuery, query: str, issues: list[str]) -> None:
    # V4 PART 2 — universal representation: intent, relational context,
    # cross-domain / modality / academia / mentor shape, fallback cleanup.
    augment_plan(parsed, query)
    _promote_subject(parsed, query)
    _dedupe_semantic_duplicates(parsed)
    if parsed.unresolved:
        # "my field" with no configured profile, etc. — never guessed, and it
        # must not read as a confident interpretation (V4 PART 2 §3/§9).
        parsed.interpretation_confidence_cap = min(parsed.interpretation_confidence_cap, 0.55)
    summary, confidence = build_summary(parsed, query)
    parsed.interpretation_summary = summary
    # a cap set by validate_and_repair (unrepaired OR/NOT mismatch, V4 §9) must
    # NOT be raised again here.
    parsed.interpretation_confidence = round(min(confidence, parsed.interpretation_confidence_cap), 2)
    if issues:
        log.info("query plan repairs for %r: %s", query, "; ".join(issues))


# ─────────────────────── deterministic parser ───────────────────────

_PAST_RE = re.compile(r"(?:previously|formerly|ex[- ]|used to (?:work|be)|before)\s+(?:worked\s+(?:at|for)\s+)?", re.I)
_CURRENT_RE = re.compile(r"currently\s+(?:works?\s+)?(?:at|for|with)\s+", re.I)
_AT_RE = re.compile(r"\b(?:works?\s+at|worked\s+at|at)\s+([A-Z][\w&.\- ]+?)(?:\s+(?:who|that|and|,|\.|$))", re.I)
_SCHOOL_RE = re.compile(r"(?:studied at|went to|graduated from|degree from|alum(?:ni|nus)? of|attended)\s+([A-Z][\w&.\- ]+)", re.I)
_STUDY_RE = re.compile(r"stud(?:ied|ying)\s+([a-z][\w ]+?)(?:\s+(?:at|and|,|\.|$))", re.I)

#: "in Memphis or Nashville" / "in Memphis, Nashville, or Atlanta" — a
#: preposition, then 2+ capitalized place names joined by commas/or.
_LOCATION_OR_RE = re.compile(
    r"\b(?:in|from|near|based in)\s+((?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*)"
    r"(?:\s*,\s*(?:or\s+)?(?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*)"
    r"|\s+or\s+(?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*))+)",
)
_COMPANY_CATEGORY_WORDS = {
    "startup": "startup", "startups": "startup",
    "big tech": "big tech", "faang": "big tech", "big technology": "big tech",
    "fintech": "fintech", "consulting firm": "consulting", "consultancy": "consulting",
    "consulting firms": "consulting", "big four": "consulting", "big 4": "consulting",
    "healthcare company": "healthcare", "biotech": "biotech", "enterprise software": "enterprise software",
}

#: query words that name a professional ROLE FUNCTION (meaning, not exact title).
#: The value is the canonical role phrase used as the criterion concept.
_ROLE_FUNCTION_WORDS = {
    "software engineer": "software engineering", "software engineers": "software engineering",
    "swe": "software engineering", "backend engineer": "backend engineering",
    "frontend engineer": "frontend engineering", "full stack engineer": "full-stack engineering",
    "ml engineer": "machine learning engineering", "machine learning engineer": "machine learning engineering",
    "data scientist": "data science", "data scientists": "data science",
    "data engineer": "data engineering", "security engineer": "security engineering",
    "product manager": "product management", "product managers": "product management",
    "sales leader": "sales leadership", "sales leaders": "sales leadership",
    "engineering manager": "engineering management", "engineering leader": "engineering leadership",
    "designer": "design", "designers": "design", "researcher": "research", "researchers": "research",
}


def _clean_name(s: str) -> str:
    return re.sub(r"\s+(who|that|and|with)\b.*$", "", s.strip(" .,")).strip()


def _split_or_list(blob: str) -> list[str]:
    parts = re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+", blob.strip())
    return [_clean_name(p) for p in parts if _clean_name(p)]


def _deterministic_parse(query: str) -> ParsedSearchQuery:
    q = query.strip()
    ql = q.lower()
    crits: list[SearchCriterion] = []
    used_spans: list[str] = []

    # locations FIRST, with OR support — this was silently dropped before (spec §11/§13)
    loc_m = _LOCATION_OR_RE.search(q)
    if loc_m:
        places = _split_or_list(loc_m.group(1))
        if places:
            crits.append(
                SearchCriterion(
                    id="location", type=CriterionType.LOCATION, values=places,
                    operator=Operator.ANY_OF, weight=35, required=True,
                )
            )
            used_spans.extend(p.lower() for p in places)

    from app.services.query_facts import parse_value_group

    for m in re.finditer(
        r"(?i:previously|formerly|former|ex[- ])\s*(?i:worked\s+(?:at|for)\s+)?"
        r"([A-Z][\w&.\-]*(?:\s+(?:[A-Z][\w&.\-]*|of|and|or|&))*)",
        q,
    ):
        blob = re.sub(r"\s+(?:of|and|or|&)$", "", _clean_name(m.group(1))).strip()
        names, op = parse_value_group(blob)
        names = [n for n in names if n and n.lower() not in _COMPANY_CATEGORY_WORDS]
        if names:
            crits.append(SearchCriterion(id=f"past_{_slug(names[0])}", type=CriterionType.PAST_COMPANY,
                                         values=names, operator=op, weight=30, required=False,
                                         scope=Scope.PAST_COMPANY))
            used_spans.extend(n.lower() for n in names)

    for m in _CURRENT_RE.finditer(q):
        tail = q[m.end():]
        name = _clean_name(re.split(r"\b(who|that|and)\b", tail)[0])
        if name and name.lower() not in _COMPANY_CATEGORY_WORDS:
            crits.append(SearchCriterion(id=f"cur_{_slug(name)}", type=CriterionType.CURRENT_COMPANY, value=name, weight=35, required="currently" in ql and " who " in ql, scope=Scope.CURRENT_COMPANY))
            used_spans.append(name.lower())

    # company categories ("now at startups", "ex-big-tech") — NOT a company name.
    # Dedupe by resulting CATEGORY (not matched phrase) — "startup"/"startups"
    # both map to the same category and must not double-add.
    seen_categories: set[str] = set()
    for phrase, category in _COMPANY_CATEGORY_WORDS.items():
        if phrase in ql and category not in seen_categories:
            seen_categories.add(category)
            scope = Scope.PAST_COMPANY if re.search(rf"(ex|former|previously)[- ]\w*\s*{re.escape(phrase)}", ql) else Scope.CURRENT_COMPANY
            crits.append(SearchCriterion(id=f"cat_{_slug(category)}", type=CriterionType.COMPANY_CATEGORY, concept=category, scope=scope, weight=35, required=True))
            used_spans.append(phrase)

    for m in _SCHOOL_RE.finditer(q):
        name = _clean_name(m.group(1))
        if name and name.lower() not in used_spans:
            crits.append(SearchCriterion(id=f"edu_{_slug(name)}", type=CriterionType.EDUCATION, value=name, weight=40, required=False))
            used_spans.append(name.lower())

    for m in _STUDY_RE.finditer(q):
        field = m.group(1).strip()
        if field and len(field) > 2:
            crits.append(SearchCriterion(id=f"field_{_slug(field)}", type=CriterionType.EDUCATION, value=field, weight=25, required=False))

    for m in re.finditer(r"([A-Za-z][\w+.\- ]{1,30}?)\s+(?:certified|certification|cert)\b", q, re.I):
        val = _clean_name(m.group(1))
        if val and val.lower() not in used_spans:
            crits.append(SearchCriterion(id=f"cert_{_slug(val)}", type=CriterionType.CERTIFICATION, value=val, weight=35, required=False))
            used_spans.append(val.lower())
    for m in re.finditer(r"\bspeaks?\s+([A-Za-z]+(?:\s+(?:or|and)\s+[A-Za-z]+)*)", q, re.I):
        val = m.group(1).strip()
        crits.append(SearchCriterion(id=f"lang_{_slug(val)}", type=CriterionType.LANGUAGE, value=val, weight=40, required=False))
    if re.search(r"\b(publish(?:ed|es)?|publication|research paper|wrote a paper)\b", ql):
        topic_m = re.search(r"publish(?:ed)?(?:\s+(?:research|papers?|on))?\s+(?:on\s+|about\s+)?([a-z][\w ]{2,40})", ql)
        topic = topic_m.group(1).strip() if topic_m else ""
        crits.append(SearchCriterion(id="pub", type=CriterionType.PUBLICATION, value=topic, weight=30, required=False))

    if not any(c.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY) for c in crits):
        for m in _AT_RE.finditer(q + " "):
            name = _clean_name(m.group(1))
            nl = name.lower()
            # "at fintech companies" / "at a consulting firm" -> category, handled
            # elsewhere; never a company literally named "fintech companies"
            if re.search(r"\b(compan(?:y|ies)|firms?)\b", nl) or any(w in nl for w in _COMPANY_CATEGORY_WORDS):
                continue
            if name and nl not in used_spans and nl not in _KNOWN_SKILLS:
                # "worked at X" / bare "at X" is ANY experience; only "works at X" is current
                verb = m.group(0)[: m.start(1) - m.start(0)].lower()
                current = "works at" in verb or "work at" in verb
                crits.append(SearchCriterion(
                    id=f"co_{_slug(name)}", type=CriterionType.CURRENT_COMPANY, value=name,
                    scope=None if current else Scope.ANY_EXPERIENCE, weight=30, required=False,
                ))
                used_spans.append(nl)

    for skill in sorted(_KNOWN_SKILLS, key=lambda s: -len(s)):
        if re.search(rf"\b{re.escape(skill)}\b", ql):
            if any(skill in u for u in used_spans):
                continue
            crits.append(SearchCriterion(id=f"skill_{_slug(skill)}", type=CriterionType.SKILL, value=skill, weight=20, required=False))
            used_spans.append(skill)

    # seniority — singularise so "CXOs" / "founders" register; treat a bare
    # "executive(s)" query word as CXO-level intent (it never means it on a real
    # profile, so it only lives here on the query side)
    ql_sing = re.sub(r"\b(cxo|ceo|cto|cfo|coo|founder|director|manager|"
                     r"vp|principal|lead|head|architect)s\b", r"\1", ql)
    if re.search(r"\bexecutives?\b", ql):
        ql_sing += " cxo"
    rank = seniority_rank(ql_sing)
    if rank is not None and rank >= 3:
        label = next((w for w in ["cxo", "executive", "principal", "staff", "senior", "director", "vp", "lead", "founder", "head", "ceo", "cto", "cfo", "coo"] if re.search(rf"\b{w}s?\b", ql)), "senior")
        crits.append(SearchCriterion(id="seniority", type=CriterionType.SENIORITY, value=label, weight=15, required=False))

    _company_words = {w for c in crits if c.type in (CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY)
                      for w in (c.values or [c.value]) for w in w.lower().split()}
    #: a recognised role phrase -> role_function (meaning), else generic title
    _seen_roles: set[str] = set()
    for phrase, canon in sorted(_ROLE_FUNCTION_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", ql) and canon not in _seen_roles:
            if phrase.split()[0] in _company_words:
                continue
            _seen_roles.add(canon)
            crits.append(SearchCriterion(id=f"role_{_slug(canon)}", type=CriterionType.ROLE_FUNCTION,
                                         concept=canon, scope=Scope.CAREER, weight=25, required=False))
    for m in re.finditer(r"\b([a-z]+)\s+(engineer|engineers|developer|developers|manager|managers|scientist|scientists|designer|designers|architect|architects)\b", ql):
        if m.group(1) in _company_words or f"{m.group(1)} {m.group(2)}".rstrip("s") in _ROLE_FUNCTION_WORDS:
            continue
        if any(m.group(1) in (c.concept or "") for c in crits if c.type == CriterionType.ROLE_FUNCTION):
            continue
        title = f"{m.group(1)} {m.group(2)}".rstrip("s")
        crits.append(SearchCriterion(id=f"title_{_slug(title)}", type=CriterionType.TITLE, value=title, weight=20, required=False))

    # anything left over that looks like an industry/concept word becomes a
    # low-weight semantic_concept rather than a literal keyword search — weak,
    # but never silently dropped, and never a false literal-text match either.
    if not crits or all(c.type == CriterionType.SENIORITY for c in crits):
        for kw in _keywords(q):
            crits.append(SearchCriterion(id=f"concept_{_slug(kw)}", type=CriterionType.SEMANTIC_CONCEPT, concept=kw, weight=15, required=False))
    if not crits:
        crits.append(SearchCriterion(id="kw_all", type=CriterionType.KEYWORD, value=q[:60], weight=100, required=False))

    return ParsedSearchQuery(intent="professional_recommendation", criteria=crits)


_STOP = {
    "who", "should", "i", "reach", "out", "to", "for", "about", "people", "person", "in", "my",
    "network", "and", "the", "a", "an", "with", "know", "knows", "at", "of", "that", "currently",
    "previously", "worked", "works", "both", "someone", "find", "list", "give", "me",
    "moved", "move", "from", "now", "then", "later", "into", "invite", "recommend",
    "cxo", "cxos", "ceo", "cto", "cfo", "coo", "executive", "executives",
    "experience", "experiences", "expertise", "background", "backgrounds", "skills",
    "years", "year", "focused", "working",
}


def _keywords(q: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z+.#-]{2,}", q.lower())
    return list(dict.fromkeys(w for w in words if w not in _STOP))[:6]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:24] or "x"
