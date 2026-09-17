"""V4 PART 2 — universal query representation.

Represents an arbitrary professional-network question as
``intent + candidate criteria + relational context`` WITHOUT collapsing the
sentence into a keyword phrase.

Runs after the fact layer and the (LLM or deterministic) criteria parser, so it
post-processes whatever plan already exists:

  * classify the search INTENT (find_people / professional_recommendation /
    mentor_recommendation / subject_matter_expertise / career_transition /
    networking_invitation) — reusable, not one branch per phrasing
  * pull relational context ("mentor for a backend engineer moving into
    management", "anyone in my field") into ``target_person_context`` /
    ``unresolved`` — never into a criterion, never hallucinated
  * expand cross-domain AND ("cybersecurity and healthcare", "research plus
    industry experience") into SEPARATE required semantic dimensions
  * weaken modality ("might have HIPAA experience" != "HIPAA experts")
  * give academia queries the right shape (faculty employment vs research
    employment vs a university degree vs publications — studying somewhere is
    not working in academia; one paper is not a professor)
  * give mentor/advice queries relational, evidence-based criteria (never a
    literal "mentor" match, never "senior => mentor")
  * strip an accidental keyword / low-value fallback once a real concept covers
    the same ground

Deterministic and LLM-free. When an LLM answered, its criteria are kept; this
layer only adds structure it is sure about and removes fallback noise. If it
cannot resolve a relational concept it lowers confidence rather than guessing.
"""
from __future__ import annotations

import re

from app.config import settings
from app.constants import (
    SEMANTIC_CRITERION_TYPES,
    CriterionType,
    Modality,
    Operator,
    QueryIntent,
    Scope,
)
from app.schemas import ParsedSearchQuery, SearchCriterion

_SEMANTIC_TYPES = SEMANTIC_CRITERION_TYPES


# ─────────────────────────── intent classification (§1) ───────────────────────────

_MENTOR_RE = re.compile(
    r"\b(mentor|mentors|mentorship|mentoring|mentored|"
    r"advis(?:e|es|or|ors|ory|ing)|coach(?:es|ing|ed)?|"
    r"career (?:advice|guidance|coaching)|give (?:me |some )?advice|"
    r"someone to talk to about my career)\b",
    re.I,
)
_EXPERT_RE = re.compile(
    r"\b(experts?|expertise|specialis(?:t|ts|ing)|subject[- ]matter|"
    r"deep (?:knowledge|experience|expertise) (?:in|of|on)|authority on|"
    r"guru|thought leaders?|world[- ]class in)\b",
    re.I,
)
_TRANSITION_INTENT_RE = re.compile(
    r"(→|-->|➔|=>)|"
    r"\b(transition(?:s|ing|ed)?|moving into|move into|moved into|"
    r"break into|breaking into|broke into|pivot(?:ed|ing)?|"
    r"switch(?:ing|ed)? (?:careers?|into|from)|"
    r"went from .+ to |from .+ (?:to|into) (?:industry|management|tech))\b",
    re.I,
)
_NETWORKING_RE = re.compile(
    r"\b(networking event|invite|invitation|conference|meetup|meet-up|panel|"
    r"summit|mixer|introduce me|introduction to|reception|roundtable)\b",
    re.I,
)
_FIND_RE = re.compile(
    r"^\s*(who\s+(?:are|is)|find\s+me|find\b|list\b|show me|which\b|search for|"
    r"anyone\b|any one\b|people who\b)",
    re.I,
)


def classify_intent(query: str, context: dict) -> str:
    """Deterministic intent classification. Mentor/advice beats everything (it is
    the most specific relational ask); event framing beats a bare find; expertise
    beats a bare transition; interrogative/imperative => find_people; otherwise a
    declarative noun phrase is a soft professional_recommendation."""
    q = query or ""
    purpose = context.get("purpose", "")
    if _MENTOR_RE.search(q):
        return QueryIntent.MENTOR_RECOMMENDATION
    if (purpose or _NETWORKING_RE.search(q)) and _NETWORKING_RE.search(f"{purpose} {q}"):
        return QueryIntent.NETWORKING_INVITATION
    if _EXPERT_RE.search(q):
        return QueryIntent.SUBJECT_MATTER_EXPERTISE
    if _TRANSITION_INTENT_RE.search(q):
        return QueryIntent.CAREER_TRANSITION
    if _FIND_RE.search(q):
        return QueryIntent.FIND_PEOPLE
    return QueryIntent.PROFESSIONAL_RECOMMENDATION


# ─────────────────────────── relational context (§3) ───────────────────────────

_MY_FIELD_RE = re.compile(
    r"\bmy (?:own )?(?:field|industry|space|area|domain|line of work|profession|discipline)\b",
    re.I,
)
_GENERIC_ROLE_STOP = {
    "person", "someone", "somebody", "people", "professional", "professionals",
    "one", "friend", "colleague", "connection",
}
#: "mentor for a backend engineer …" and "mentor a backend engineer …" (verb
#: object, no "for") — the mentee described inside the query.
_MENTEE_ROLE_RE = re.compile(
    r"\b(?:mentor|mentorship|coach|advise|help)\w*\b\s+"
    r"(?:for\s+)?(?:a|an|my|someone|somebody|some)\s+"
    r"([a-z][a-z /+.\-]*?)"
    r"(?=\s+(?:who|that|trying|looking|wanting|hoping|about|moving|movin|move|"
    r"transition|going|step|grow|now|,|\.|\?|$))",
    re.I,
)
_MENTEE_ROLE_FALLBACK_RE = re.compile(
    r"\bfor\s+(?:a|an|my)\s+"
    r"([a-z][a-z /+.\-]*?\b(?:engineer|manager|developer|designer|scientist|"
    r"analyst|consultant|lead|architect|marketer|founder|pm|product manager|"
    r"researcher|recruiter))\b",
    re.I,
)
_GOAL_RE = re.compile(
    r"\b(?:into|to|toward|towards|for)\s+(?:a\s+|an\s+)?"
    r"((?:engineering |people |product |eng |technical )?"
    r"(?:management|leadership)(?:\s+(?:role|position|track))?)\b",
    re.I,
)
_GOAL_VERB_RE = re.compile(
    r"\b(?:become|becoming|move into|moving into|movin(?:g)? into|step into|"
    r"transition(?:ing)? (?:in)?to|get into|breaking into)\s+"
    r"(?:a\s+|an\s+|the\s+)?([a-z][\w /+.\-]{2,40}?)(?=\s*(?:role|position|,|\.|$))",
    re.I,
)


def _clean_phrase(s: str) -> str:
    s = re.sub(r"[^\w /+.\-]", " ", s or "")
    return re.sub(r"\s{2,}", " ", s).strip(" .,-").strip().lower()


def _role_to_field(role: str) -> str:
    r = f" {role.lower()} "
    table = [
        ("backend", "backend engineering"), ("back-end", "backend engineering"),
        ("frontend", "frontend engineering"), ("front-end", "frontend engineering"),
        ("full stack", "full-stack engineering"), ("fullstack", "full-stack engineering"),
        ("data scien", "data science"), ("data eng", "data engineering"),
        ("machine learning", "machine learning"), (" ml ", "machine learning"),
        ("security", "security engineering"), ("infra", "infrastructure engineering"),
        ("platform", "platform engineering"), ("mobile", "mobile engineering"),
        ("devops", "devops / SRE"), ("sre", "site reliability engineering"),
        ("product manager", "product management"), (" pm ", "product management"),
        ("design", "design"), ("market", "marketing"), ("sales", "sales"),
        ("recruit", "recruiting"), ("research", "research"),
        ("software", "software engineering"), ("engineer", "software engineering"),
        ("developer", "software engineering"),
    ]
    for needle, field in table:
        if needle in r:
            return field
    return role.strip().lower()


def _normalize_goal(goal: str, field: str = "") -> str:
    g = goal.lower().strip()
    g = re.sub(r"\s+(role|position|track)$", "", g).strip()
    if "engineering manage" in g or "engineering leader" in g or "eng manage" in g:
        return "engineering management"
    if "product manage" in g:
        return "product management"
    if g in ("management", "people management", "a management", "an management"):
        return "engineering management" if "engineering" in field else "people management"
    if "leadership" in g:
        return "engineering leadership" if "engineering" in field else "leadership"
    return g


def resolve_relational_context(query: str) -> tuple[dict[str, str], list[str]]:
    """Return ``(target_person_context, unresolved)``.

    ``target_person_context`` = {field?, current_role?, goal?} describing the
    person the candidate must be able to help — the mentee in "mentor for a
    backend engineer…", or the searcher in "anyone in my field". NEVER a search
    phrase. ``unresolved`` lists keys we refused to guess (e.g. "field" when the
    query said "my field" and no current-user profile is configured)."""
    ctx: dict[str, str] = {}
    unresolved: list[str] = []
    q = (query or "").strip()

    if _MY_FIELD_RE.search(q):
        if settings.user_field.strip():
            ctx["field"] = settings.user_field.strip()
            if settings.user_current_role.strip():
                ctx["current_role"] = settings.user_current_role.strip()
            if settings.user_goal.strip():
                ctx["goal"] = settings.user_goal.strip()
        else:
            unresolved.append("field")

    role = None
    m = _MENTEE_ROLE_RE.search(q) or _MENTEE_ROLE_FALLBACK_RE.search(q)
    if m:
        role = _clean_phrase(m.group(1))
    if role and role not in _GENERIC_ROLE_STOP and len(role) >= 3:
        ctx["current_role"] = role
        ctx.setdefault("field", _role_to_field(role))

    gm = _GOAL_RE.search(q)
    goal = _clean_phrase(gm.group(1)) if gm else ""
    if not goal:
        gm2 = _GOAL_VERB_RE.search(q)
        if gm2:
            goal = _clean_phrase(gm2.group(1))
    if goal:
        ctx["goal"] = _normalize_goal(goal, ctx.get("field", ""))

    return ctx, unresolved


# ─────────────────────────── cross-domain AND (§4) ───────────────────────────

_DOMAIN_WORDS = {
    "ai", "artificial intelligence", "ml", "machine learning", "nlp", "llm", "llms",
    "healthcare", "health", "health tech", "digital health", "medtech", "biotech",
    "genomics", "pharma", "pharmaceutical", "life sciences", "clinical",
    "cybersecurity", "security", "infosec", "appsec", "privacy", "compliance",
    "hipaa", "gdpr", "soc2", "fintech", "finance", "financial services", "banking",
    "payments", "insurance", "crypto", "blockchain", "web3", "defi",
    "research", "academia", "academic", "industry", "nonprofit", "non-profit",
    "public sector", "government", "govtech", "defense", "aerospace", "space",
    "climate", "climate tech", "clean energy", "renewables", "sustainability",
    "robotics", "hardware", "semiconductors", "iot", "autonomous vehicles",
    "gaming", "media", "entertainment", "edtech", "education", "e-commerce",
    "retail", "logistics", "supply chain", "manufacturing", "legal", "policy",
    "data", "data engineering", "data science", "devops", "cloud", "distributed systems",
}
_NON_DOMAIN_TOKENS = {
    "people", "person", "someone", "engineers", "engineer", "managers", "manager",
    "leaders", "leader", "experts", "expert", "professionals", "pros", "folks",
    "who", "that", "them", "us", "me", "my", "the", "a", "an",
}
_ROLE_NOUN_AFTER_RE = re.compile(
    r"\s*(leaders?|engineers?|managers?|experts?|professionals?|pros|people|"
    r"specialists?|scientists?|developers?|founders?|executives?|architects?)\b",
    re.I,
)
_DOMAIN_AND_RE = re.compile(
    r"\b([a-z][\w+.\-]*(?:\s+[a-z][\w+.\-]*){0,2}?)\s+(?:and|&|\+|plus)\s+"
    r"([a-z][\w+.\-]*(?:\s+[a-z][\w+.\-]*){0,2}?)"
    r"(?:\s+(experiences?|backgrounds?|expertise|domains?|knowledge|exposure|skills?))?",
    re.I,
)
_DOMAIN_EXPAND = {
    "ai": "artificial intelligence", "ml": "machine learning", "nlp": "natural language processing",
    "infosec": "cybersecurity", "appsec": "application security",
    "non-profit": "nonprofit", "academic": "academia",
}


def _is_domain(s: str) -> bool:
    s = s.lower().strip()
    if s in _DOMAIN_WORDS:
        return True
    return any(re.search(rf"\b{re.escape(w)}\b", s) for w in _DOMAIN_WORDS)


def _expand_domain(s: str) -> str:
    s = s.lower().strip()
    return _DOMAIN_EXPAND.get(s, s)


def _looks_industry(dom: str) -> bool:
    return dom.lower() in {
        "healthcare", "health", "fintech", "finance", "banking", "insurance",
        "biotech", "pharma", "nonprofit", "non-profit", "government", "defense",
        "aerospace", "gaming", "media", "retail", "manufacturing", "education",
        "industry", "academia", "climate", "energy",
    }


def _expand_cross_domain(plan: ParsedSearchQuery, query: str) -> None:
    """"cybersecurity and healthcare" / "research plus industry experience" ->
    TWO separate required semantic dimensions (both must hold), NOT one blended
    phrase and NOT one ANY_OF criterion. A trailing role noun ("AI and security
    leaders") is left to the single ALL_OF criterion the fact layer produced."""
    for m in _DOMAIN_AND_RE.finditer(query or ""):
        a, b = m.group(1).strip().lower(), m.group(2).strip().lower()
        noun = (m.group(3) or "").lower()
        if a in _NON_DOMAIN_TOKENS or b in _NON_DOMAIN_TOKENS:
            continue
        if _ROLE_NOUN_AFTER_RE.match((query or "")[m.end():]):
            continue  # "... leaders/engineers" — role-noun form, not cross-domain
        a_dom, b_dom = _is_domain(a), _is_domain(b)
        if not ((a_dom and b_dom) or (noun and (a_dom or b_dom))):
            continue
        tail_noun = noun or "experience"
        # replace whatever the fact layer / fallback made for these two tokens
        # (a blended ALL_OF criterion, or thin 1-word concepts) with two clean,
        # independently-required semantic dimensions.
        _drop_domain_criteria(plan, {a, b})
        for dom in (a, b):
            plan.criteria.append(SearchCriterion(
                id=f"xd_{_slug(dom)}",
                type=CriterionType.INDUSTRY_EXPERIENCE if _looks_industry(dom)
                else CriterionType.PROFESSIONAL_CONCEPT,
                concept=f"{_expand_domain(dom)} {tail_noun}".strip(),
                scope=Scope.CAREER, weight=25, required=True,
            ))
        return


def _drop_domain_criteria(plan: ParsedSearchQuery, tokens: set[str]) -> None:
    toks = {t.lower() for t in tokens}
    plan.criteria = [
        c for c in plan.criteria
        if not (
            c.type in _SEMANTIC_TYPES
            and (
                ({v.lower() for v in c.values} and {v.lower() for v in c.values} <= toks)
                or (c.concept or c.value or "").strip().lower() in toks
            )
        )
    ]


# ─────────────────────────── modality (§5) ───────────────────────────

_MODAL_HEDGE_RE = re.compile(
    r"\b(might|may|maybe|possibly|possible|perhaps|potential(?:ly)?|"
    r"could (?:have|possibly|potentially)|some exposure to|any exposure to|"
    r"ideally|nice to have|open to someone (?:who|with)|preferably)\b",
    re.I,
)
_MODAL_CONCEPT_RE = re.compile(
    r"\b(?:might|may|maybe|possibly|possible|perhaps|potential(?:ly)?|could)\b\s*"
    r"(?:have\s+|had\s+|has\s+)?(?:some\s+|any\s+)?"
    r"(?:experience\s+(?:with|in|of)\s+|worked\s+(?:with|on)\s+|"
    r"familiarity\s+with\s+|exposure\s+to\s+|knowledge\s+of\s+|done\s+|"
    r"background\s+in\s+)?"
    r"([A-Za-z0-9][\w /&+.\-]*?)"
    r"(?:\s+(?:experience|expertise|background|knowledge|exposure|work|skills?))?"
    r"\s*[.?!]?\s*$",
    re.I,
)
_MODAL_CONCEPT_STOP = {"experience", "it", "that", "this", "someone", "them", "something"}


def _modal_concept(query: str) -> str:
    m = _MODAL_CONCEPT_RE.search(query or "")
    if not m:
        return ""
    phrase = re.sub(r"\s{2,}", " ", m.group(1).strip(" .,-")).strip()
    if not phrase or len(phrase) < 3 or len(phrase.split()) > 5 \
            or phrase.lower() in _MODAL_CONCEPT_STOP:
        return ""
    return f"{phrase} experience"


def _apply_modality(plan: ParsedSearchQuery, query: str) -> None:
    """"might have HIPAA experience" / "possible HIPAA compliance experience" ->
    ONE concept, ``modality=possible``, never required, lower weight. It must NOT
    read the same as "HIPAA compliance experts". A concept another rule already
    marked required (e.g. a cross-domain dimension) is left alone."""
    if not _MODAL_HEDGE_RE.search(query or ""):
        return
    touched = False
    for c in plan.criteria:
        if c.type in _SEMANTIC_TYPES and c.operator != Operator.NOT and not c.required:
            c.modality = Modality.POSSIBLE
            c.weight = max(5.0, round(c.weight * 0.6, 2))
            touched = True
    concept = _modal_concept(query)
    if concept:
        phrase_toks = {t.lower() for t in re.findall(r"[a-z0-9]+", concept.lower())}
        # remove the thin 1-word fallback concepts this phrase now subsumes
        plan.criteria = [
            c for c in plan.criteria
            if not (
                c.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.KEYWORD)
                and not c.required
                and (c.concept or c.value or "").strip().lower() in phrase_toks
            )
        ]
        if not any(
            c.type in _SEMANTIC_TYPES and (c.concept or "").lower() == concept.lower()
            for c in plan.criteria
        ):
            plan.criteria.append(SearchCriterion(
                id=f"maybe_{_slug(concept)}", type=CriterionType.PROFESSIONAL_CONCEPT,
                concept=concept, scope=Scope.CAREER, weight=20, required=False,
                modality=Modality.POSSIBLE,
            ))
        touched = True
    if touched:
        plan.interpretation_confidence_cap = min(plan.interpretation_confidence_cap, 0.7)


# ─────────────────────────── academia (§6) ───────────────────────────

_FACULTY_RE = re.compile(
    r"\b(professors?|faculty|lecturers?|tenured|tenure[- ]track|"
    r"assistant professors?|associate professors?|full professors?|"
    r"department chairs?|deans?|principal investigators?|"
    r"adjuncts?|postdocs?|post-docs?|academic staff)\b",
    re.I,
)
_RESEARCHER_RE = re.compile(
    r"\b(research scientists?|researchers?|research staff|research fellows?|"
    r"research engineers?|scientific staff|r&d)\b",
    re.I,
)
_UNI_EDU_RE = re.compile(
    r"\b(studied|degree|graduated|alum(?:ni|nus)?|ph\.?d|doctorate|"
    r"master'?s|bachelor'?s|undergrad|majored|went to college|university education)\b",
    re.I,
)
_PUBS_RE = re.compile(
    r"\b(publications?|published|papers?|peer[- ]reviewed|co-authored|"
    r"citations?|cited|h-index)\b",
    re.I,
)
_ACADEMIA_WORD_RE = re.compile(r"\bacadem(?:ia|ic|e)\b", re.I)
_FACULTY_TOPIC_RE = re.compile(
    r"\b(?:professors?|faculty|lecturers?|researchers?|research scientists?|phd\w*)\b"
    r"\s+(?:of|in|on|specializing in|focused on|working on|studying)\s+"
    r"([a-z][\w /&+.\-]{1,40})",
    re.I,
)
_ARROW_RE = re.compile(
    r"\b([a-z][\w /\-]{2,30}?)\s*(?:→|-->|->|➔|=>)\s*([a-z][\w /\-]{2,30}?)\b",
    re.I,
)
#: "academia to industry transitions", "consulting into tech moves" — an ordered
#: A→B change named by a transition noun, with no "moved from" verb.
_WORD_TRANSITION_RE = re.compile(
    r"\b([a-z][\w ]{2,25}?)\s+(?:to|into)\s+([a-z][\w ]{2,25}?)\s+"
    r"(transitions?|moves?|switch(?:es)?|pivots?|career changes?|shifts?)\b",
    re.I,
)


def _is_subject(query: str, word: str) -> bool:
    return (query or "").lower().find(word.lower()) <= 25


def _shape_academia(plan: ParsedSearchQuery, query: str) -> None:
    q = query or ""
    fac = _FACULTY_RE.search(q)
    if fac:
        _add_concept(
            plan, concept="a professor / faculty appointment at a university or college",
            token="faculty-employment", weight=30,
            required=_is_subject(q, fac.group(0)),
        )
        _strip_education(plan, q)  # "professors in AI" is NOT "studied AI at a university"
    elif _RESEARCHER_RE.search(q):
        _add_concept(
            plan, concept="a formal research position (industry lab or academic institution)",
            token="research-employment", weight=25,
            required=_is_subject(q, "research"),
        )

    topic_m = _FACULTY_TOPIC_RE.search(q)
    if topic_m:
        topic = _clean_phrase(topic_m.group(1))
        if topic and topic not in _MODAL_CONCEPT_STOP:
            _add_concept(
                plan, concept=f"a research / teaching focus on {topic}",
                token=f"acad-topic-{topic}", weight=25, required=True,
            )

    if _ACADEMIA_WORD_RE.search(q) and not fac:
        if not any(c.type == CriterionType.CAREER_TRANSITION for c in plan.criteria):
            _add_concept(
                plan, concept="employment in academia (a university or research institution)",
                token="academia-employment", weight=20, required=False,
            )

    if _PUBS_RE.search(q) and not any(c.type == CriterionType.PUBLICATION for c in plan.criteria):
        plan.criteria.append(SearchCriterion(
            id="pubs", type=CriterionType.PUBLICATION, value="", weight=15, required=False,
        ))


def _ordered_transition(plan: ParsedSearchQuery, query: str) -> None:
    """"academia -> industry transitions" / "academia to industry transitions" —
    an ordered A->B career change even without the "moved from" verb the fact
    layer looks for."""
    if any(c.type == CriterionType.CAREER_TRANSITION for c in plan.criteria):
        return
    m = _ARROW_RE.search(query or "") or _WORD_TRANSITION_RE.search(query or "")
    if not m:
        return
    frm = re.sub(r"\b(people|professionals?|folks|those|the)\b", "", _clean_phrase(m.group(1))).strip()
    to = re.sub(r"\b(transitions?|moves?|shifts?|pivots?|people|folks)\b", "", _clean_phrase(m.group(2))).strip()
    if len(frm) < 2 or len(to) < 2:
        return
    plan.criteria.append(SearchCriterion(
        id="transition", type=CriterionType.CAREER_TRANSITION,
        concept=f"from {frm} to {to}", scope=Scope.CAREER, weight=40, required=True,
    ))


def _strip_education(plan: ParsedSearchQuery, query: str) -> None:
    if _UNI_EDU_RE.search(query or ""):
        return
    plan.criteria = [c for c in plan.criteria if c.type != CriterionType.EDUCATION]


# ─────────────────────────── mentor / advice (§7) ───────────────────────────

_MENTOR_EVIDENCE = (
    "evidence of mentoring, coaching, advising, people management, team "
    "leadership, or guiding others' career growth"
)
_EXPLICIT_SENIORITY_RE = re.compile(
    r"\b(senior|staff|principal|distinguished|director|vp|vice president|"
    r"cxo|c-level|chief|executive|head of|founding)\b",
    re.I,
)
_LITERAL_MENTOR_RE = re.compile(r"mentor|advice|advis|coach|guidance", re.I)


def _shape_mentor(plan: ParsedSearchQuery, query: str) -> None:
    q = query or ""
    has_evidence = any(
        c.type in _SEMANTIC_TYPES
        and len((c.concept or "").split()) >= 3
        and re.search(r"mentor|coach|advis|manage|leadership", (c.concept or "").lower())
        for c in plan.criteria
    )
    if not has_evidence:
        plan.criteria.append(SearchCriterion(
            id="mentor_evidence", type=CriterionType.PROFESSIONAL_CONCEPT,
            concept=_MENTOR_EVIDENCE, scope=Scope.CAREER, weight=35, required=True,
        ))

    tpc = plan.target_person_context
    goal, field = tpc.get("goal"), tpc.get("field")
    if goal:
        _add_concept(
            plan, concept=f"{goal} or equivalent people-leadership experience",
            token=f"goal-{_slug(goal)}", weight=25, required=True,
        )
    if field:
        _add_concept(
            plan, concept=f"familiarity with {field}",
            token=f"field-{_slug(field)}", weight=15, required=False,
        )
        _add_concept(
            plan,
            concept=f"has personally moved from an individual-contributor {field} "
            f"role into management (similar trajectory)",
            token="ic-to-management", weight=10, required=False,
        )

    # "being senior alone must not automatically imply mentor"
    if not _EXPLICIT_SENIORITY_RE.search(q):
        for c in plan.criteria:
            if c.type == CriterionType.SENIORITY:
                c.required = False

    # never a literal "mentor" / "advice" text match
    plan.criteria = [
        c for c in plan.criteria
        if not (
            c.type in (CriterionType.KEYWORD, CriterionType.SKILL, CriterionType.TITLE)
            and _LITERAL_MENTOR_RE.search(c.value or c.concept or "")
        )
    ]


# ─────────────────────────── subject-matter expertise ───────────────────────────

_EXPERT_STRIP_RE = re.compile(
    r"\b(experts?|expertise|specialis(?:t|ts|ing)|subject[- ]matter|deep|"
    r"knowledge|authority|guru|thought leaders?|world[- ]class|"
    r"professionals?|people|folks|who|are|is|find|list|show me|search for|"
    r"anyone|someone|somebody|in|on|with|of|the|a|an|about|possible)\b",
    re.I,
)


def _shape_expertise(plan: ParsedSearchQuery, query: str) -> None:
    topic = _EXPERT_STRIP_RE.sub(" ", query or "")
    topic = _clean_phrase(topic)
    if not topic or len(topic) < 2 or len(topic.split()) > 6:
        return
    topic_toks = {t for t in re.findall(r"[a-z0-9]+", topic) if len(t) > 2}
    # if the fact layer already produced a semantic concept for this topic
    # (e.g. "security or cloud" ANY_OF), just make it a hard requirement rather
    # than adding a paraphrase alongside it.
    for c in plan.criteria:
        if c.type in _SEMANTIC_TYPES:
            hay = f"{c.concept or ''} {c.value or ''} {' '.join(c.values)}".lower()
            if topic_toks and topic_toks <= set(re.findall(r"[a-z0-9]+", hay)):
                if c.modality != Modality.POSSIBLE:
                    c.required = True
                return
    _add_concept(
        plan, concept=f"deep subject-matter expertise in {topic}",
        token=f"sme-{_slug(topic)}", weight=45, required=True,
    )


# ─────────────────────────── named industry (nonprofit etc.) ───────────────────────────

_INDUSTRY_PHRASE_RE = re.compile(
    r"\b(nonprofit|non-profit|not-for-profit|ngo|social impact|public sector|"
    r"government|govtech|civic|defense|aerospace|biotech|fintech|healthtech|"
    r"health tech|climate tech|clean energy|cleantech|edtech|agtech|"
    r"pharmaceutical|insurance|consumer goods|hospitality)\b"
    r"(?:\s+(?:sector|industry|space|experience|work|background|companies|"
    r"orgs?|organi[sz]ations?|world))?",
    re.I,
)


def _named_industry(plan: ParsedSearchQuery, query: str) -> None:
    m = _INDUSTRY_PHRASE_RE.search(query or "")
    if not m:
        return
    name = _clean_phrase(m.group(1))
    if name in ("non-profit", "not-for-profit", "ngo"):
        name = "nonprofit"
    ql = (query or "").lower()
    is_subject = ql.find(m.group(1).lower()) <= 30
    wants_experience = bool(re.search(r"\b(experience|background|worked|work|sector|history)\b", ql))
    if not (is_subject or wants_experience):
        return
    _add_concept(
        plan, concept=f"professional experience in the {name} sector",
        token=f"ind-{_slug(name)}", weight=30,
        required=is_subject or wants_experience,
        ctype=CriterionType.INDUSTRY_EXPERIENCE,
    )


# ─────────────────────────── fallback cleanup ───────────────────────────

_FALLBACK_STOP = {
    "anyone", "field", "advice", "mentor", "mentors", "give", "can", "people",
    "someone", "somebody", "who", "my", "professional", "professionals",
    "experience", "background", "backgrounds", "help", "need", "looking",
    "possible", "might", "maybe", "into", "trying", "move", "moving",
}


def _drop_keyword_fallback(plan: ParsedSearchQuery) -> None:
    """Remove the deterministic parser's last-resort keyword / 1-word
    semantic_concept criteria once real structure explains the query — never
    turn an unexplained professional concept into keyword matching (§9)."""
    real = [
        c for c in plan.criteria
        if c.type not in (CriterionType.SEMANTIC_CONCEPT, CriterionType.KEYWORD)
    ]
    kept: list[SearchCriterion] = []
    for c in plan.criteria:
        if c.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.KEYWORD):
            tok = _clean_phrase(c.concept or c.value or "")
            if tok in _FALLBACK_STOP:
                continue
            if (
                real
                and c.id.startswith(("concept_", "kw_", "kw-", "concept-"))
                and len(tok.split()) <= 1
                and not c.required
            ):
                continue
        kept.append(c)
    if kept:
        plan.criteria = kept


def _ensure_nonempty(plan: ParsedSearchQuery, query: str) -> None:
    if plan.criteria:
        return
    plan.criteria.append(SearchCriterion(
        id="unresolved_intent", type=CriterionType.PROFESSIONAL_CONCEPT,
        concept=f"professional relevance to: {(query or '').strip()[:80]}",
        scope=Scope.CAREER, weight=100, required=False,
    ))
    plan.interpretation_confidence_cap = min(plan.interpretation_confidence_cap, 0.4)


# ─────────────────────────── orchestration ───────────────────────────


def augment_plan(plan: ParsedSearchQuery, query: str) -> None:
    """Post-process a plan (from the LLM or the deterministic parser) into the
    universal representation. Mutates ``plan`` in place. Safe with no LLM."""
    plan.intent = classify_intent(query, plan.context)

    tctx, unresolved = resolve_relational_context(query)
    if tctx:
        plan.target_person_context = {**plan.target_person_context, **tctx}
    for u in unresolved:
        if u not in plan.unresolved:
            plan.unresolved.append(u)

    _ordered_transition(plan, query)
    _expand_cross_domain(plan, query)
    _named_industry(plan, query)
    _apply_modality(plan, query)
    _shape_academia(plan, query)
    if plan.intent == QueryIntent.MENTOR_RECOMMENDATION:
        _shape_mentor(plan, query)
    elif plan.intent == QueryIntent.SUBJECT_MATTER_EXPERTISE:
        _shape_expertise(plan, query)

    _drop_keyword_fallback(plan)
    _ensure_nonempty(plan, query)
    _renorm(plan)
    _identify_primary_intent(plan)


# ─────────────────────────── primary intent (near-match design PART 5) ───────────────────────────
#
# Which required criterion is the QUERY'S OWN POINT versus a CONSTRAINT on top
# of it? "VCs in Atlanta" is fundamentally about finding investors; Atlanta
# narrows that down. Generic by TYPE, never by matching specific query words —
# a constraint type describes something that is checked as a fact about a
# candidate but is rarely what a recruiter query is fundamentally "about";
# everything else (what the person professionally DOES / IS) is eligible to
# anchor the primary intent. INTERNAL ONLY — never rendered to the user, only
# used to decide what a Near Match is allowed to relax (PART 3) and how Near
# Matches are ranked against each other (PART 10).
_CONSTRAINT_TYPES = {
    CriterionType.LOCATION, CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY,
    CriterionType.EDUCATION, CriterionType.CERTIFICATION, CriterionType.LANGUAGE,
    CriterionType.PUBLICATION, CriterionType.YEARS_EXPERIENCE,
}


def _intent_text(c: SearchCriterion) -> str:
    return c.concept or c.value or " or ".join(c.values) or c.type.replace("_", " ")


def _identify_primary_intent(plan: ParsedSearchQuery) -> None:
    """Prefer the query-interpretation LLM's own ``primary_intent`` /
    ``intent_anchor_criterion_ids`` (it reads MEANING, not just criterion
    type) — but NEVER trust it blindly: the anchor ids are re-validated here
    against the FINAL criteria list (after every ``augment_plan`` mutation —
    cross-domain expansion, modality, academia shaping, fallback cleanup —
    since an id the LLM named may no longer exist by this point). An invalid
    or missing anchor is dropped; if NONE survive, the deterministic
    type-based derivation below is the fallback of record."""
    valid_ids = {c.id for c in plan.criteria}
    llm_anchor_ids = [cid for cid in plan.intent_anchor_criterion_ids if cid in valid_ids]
    if plan.primary_intent.strip() and llm_anchor_ids:
        plan.intent_anchor_criterion_ids = llm_anchor_ids
        return

    required = [c for c in plan.criteria if c.required]
    anchors = [c for c in required if c.type not in _CONSTRAINT_TYPES]
    if not anchors:
        # every required criterion is a "constraint" type (e.g. "people in
        # Austin at Google") — the constraint IS the point of the query, so it
        # anchors itself. A query with no required criteria at all anchors on
        # nothing; Near Match then has no secondary requirement to relax.
        anchors = required
    anchors = sorted(anchors, key=lambda c: c.weight, reverse=True)
    plan.intent_anchor_criterion_ids = [c.id for c in anchors]
    plan.primary_intent = "; ".join(dict.fromkeys(_intent_text(c) for c in anchors))


# ─────────────────────────── helpers ───────────────────────────


def _add_concept(
    plan: ParsedSearchQuery,
    *,
    concept: str,
    token: str,
    weight: float,
    required: bool,
    ctype: str = CriterionType.PROFESSIONAL_CONCEPT,
) -> None:
    """Add a semantic concept criterion unless one already covers ``token``.
    Upgrades an existing match to required when needed."""
    tok = token.lower()
    low = concept.lower()
    for c in plan.criteria:
        if c.type in _SEMANTIC_TYPES:
            hay = f"{c.concept or ''} {c.value or ''} {' '.join(c.values)}".lower()
            if (c.concept or "").lower() == low or (len(tok) >= 3 and tok in hay):
                if required and not c.required and c.modality != Modality.POSSIBLE:
                    c.required = True
                return
    plan.criteria.append(SearchCriterion(
        id=f"cc_{_slug(token)}", type=ctype, concept=concept,
        scope=Scope.CAREER, weight=weight, required=required,
    ))


def _renorm(plan: ParsedSearchQuery) -> None:
    total = sum(c.weight for c in plan.criteria) or 1.0
    if abs(total - 100.0) > 0.5:
        for c in plan.criteria:
            c.weight = round(c.weight / total * 100, 2)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:24] or "x"
