"""Deterministic query-fact layer (V4 §14–§18).

Runs on EVERY query, regardless of whether an LLM answered:

  strip_context()      pull out non-candidate framing ("... networking event")
  extract_facts()      high-precision structured facts: locations (incl. OR),
                       named companies with current/past/NOT scope, boolean
                       operators, "N+ years"
  merge_into_plan()    fold those hard facts into the LLM's SearchPlan — the LLM
                       may enrich meaning, it may not delete an explicit
                       location / company / NOT (V4 §15/§17)
  validate_and_repair()catch a plan that dropped or inverted an explicit
                       constraint and fix it deterministically (V4 §17)
  build_summary()      one-sentence interpretation + a confidence score (V4 §18)

MEANING over words still holds: this layer only asserts the parts of a query
that are literally checkable (a city, a company name, an operator). Industry /
role / seniority *concepts* are left to the LLM + semantic scorer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.constants import CriterionType, KNOWN_TECH_AND_DOMAIN_TERMS, Operator, Scope
from app.schemas import ParsedSearchQuery, SearchCriterion

# ─────────────────────────── context (V4 §14) ───────────────────────────

_EVENT_NOUNS = (
    "networking event", "event", "conference", "meetup", "meet-up", "panel",
    "webinar", "summit", "mixer", "dinner", "luncheon", "gathering", "gala",
    "reception", "roundtable", "fireside chat", "workshop", "session", "party",
    "happy hour",
)
_EVENT_RE = re.compile(
    r"\b(?:a|an|the|our|my|this|for)\s+([a-z][\w&/-]*(?:\s+[a-z][\w&/-]*){0,3}?)\s+"
    r"(" + "|".join(re.escape(n) for n in _EVENT_NOUNS) + r")\b",
    re.I,
)
_PURPOSE_VERB_RE = re.compile(r"\b(invite|speak|present|recommend|nominate|refer)\b", re.I)


#: modifier words that are pure event framing (drop with the noun); anything
#: else in front of the event noun (e.g. "CXO") stays a candidate requirement.
_FRAMING_WORDS = {"networking", "social", "professional", "industry", "alumni", "community"}


def strip_context(query: str) -> tuple[str, dict[str, str]]:
    """Return ``(query_without_context_phrase, context)``. ``context['purpose']``
    holds the event description; only the event noun (and any pure-framing word
    right before it) is removed — real requirements like 'CXO' are left in."""
    context: dict[str, str] = {}
    m = _EVENT_RE.search(query)
    if m:
        modifier, noun = m.group(1).strip(), m.group(2).strip()
        context["purpose"] = f"{modifier} {noun}".strip()
        cut_start = m.start(2)
        # also swallow an immediately-preceding pure-framing word
        pre = query[m.start(1):m.start(2)].strip().split()
        if pre and pre[-1].lower() in _FRAMING_WORDS:
            cut_start = m.start(1) + query[m.start(1):m.start(2)].rfind(pre[-1])
        query = (query[:cut_start] + " " + query[m.end(2):]).strip()
    elif _PURPOSE_VERB_RE.search(query):
        context["purpose"] = "professional recommendation"
    return re.sub(r"\s{2,}", " ", query).strip(" ,."), context


def context_terms(context: dict[str, str]) -> set[str]:
    """Lower-cased words that came from context and must never be a criterion."""
    out: set[str] = set()
    for v in context.values():
        out.update(re.findall(r"[a-z][a-z+.#-]{2,}", v.lower()))
    return out


# ─────────────────────────── fact extraction (V4 §15/§16) ───────────────────────────

_REGION_WORDS = (
    "bay area", "silicon valley", "sf bay area", "san francisco bay area",
    "greater new york", "new york city", "nyc", "greater boston", "greater seattle",
    "greater los angeles", "greater chicago", "dfw", "research triangle",
    "pacific northwest", "tri-state area", "socal", "norcal",
)
_LOC_OR_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+"
    r"((?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*)"
    r"(?:\s*,\s*(?:or\s+)?(?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*)"
    r"|\s+or\s+(?:[A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)*))+)",
)
#: single "in Nashville" / "in Austin, Texas" — capitalised, case-SENSITIVE so
#: "in big tech" is not mistaken for a place. Regions handled separately.
_LOC_ONE_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+"
    r"(?:the\s+)?([A-Z][\w.\-]+(?:\s+[A-Z][\w.\-]+){0,3})"
)
_REGION_IN_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+(?:the\s+)?("
    + "|".join(re.escape(r) for r in _REGION_WORDS) + r")\b",
    re.I,
)

#: case-INSENSITIVE variants, used only when the query carries no casing signal
#: (all-lower / all-upper). Precision then comes from ``_NOT_A_PLACE`` + shape,
#: never a city allowlist.
_LOC_ONE_CI_RE = re.compile(
    r"\b(?:in|from|near|around|based in|located in)\s+"
    r"(?:the\s+)?([a-z][\w.\-]+(?:\s+[a-z][\w.\-]+){0,3})",
    re.I,
)
_LOC_OR_CI_RE = re.compile(_LOC_OR_RE.pattern, re.I)

#: common objects of "in <X>" that are professions / domains / states-of-being,
#: NOT geographic places. Generic (not query-specific, not a city list) — it lets
#: "in atlanta" resolve to a location while "in leadership" / "in sales" do not.
#: Also includes the shared technology/skill/academic-domain reference set
#: (KNOWN_TECH_AND_DOMAIN_TERMS) so "experience in Python" / "experts in
#: Kubernetes" / "faculty in Computational Biology" are never mistaken for a
#: LOCATION criterion — bare capitalization after "in"/"near"/"from" is not
#: proof of a place (review finding D1).
_NOT_A_PLACE = frozenset({
    "leadership", "management", "sales", "marketing", "research", "operations",
    "finance", "accounting", "design", "product", "engineering", "security",
    "compliance", "recruiting", "hr", "legal", "support", "strategy", "consulting",
    "academia", "industry", "government", "policy", "healthcare", "biotech",
    "fintech", "tech", "big tech", "faang", "startup", "startups", "enterprise",
    "software", "hardware", "data", "analytics", "devops", "infrastructure",
    "the industry", "the field", "the space", "the sector", "my field", "my network",
}) | KNOWN_TECH_AND_DOMAIN_TERMS


def _casing_uninformative(query: str) -> bool:
    """True when the query's capitalization carries no signal — it is entirely
    lower-case or entirely upper-case (ignoring the first character). In that
    state a lower-case place name is as strong a location cue as a Title-cased
    one would be, so the case-insensitive location pass may run."""
    body = query[1:] if query else ""
    return body == body.lower() or query == query.upper()


def _plausible_ci_place(cand: str) -> bool:
    c = cand.strip().lower()
    if c in _NOT_A_PLACE or c in _NOT_A_COMPANY:
        return False
    if any(tok in _NOT_A_PLACE for tok in c.split()):
        return False
    # a place is a short proper-noun phrase; reject long descriptive spans
    return 1 <= len(c.split()) <= 4 and len(c) >= 3


_FORMER_RE = re.compile(
    r"(?i:\b(?:former|formerly|previously|ex[- ]|used to work (?:at|for))\s*"
    r"(?:employees?\s+of\s+|people\s+(?:at|from)\s+|worked\s+(?:at|for)\s+)?)"
    r"([A-Z][\w&.\-]*(?:\s+(?:[A-Z][\w&.\-]*|or|and|&|of))*)"
)
_CURRENT_RE = re.compile(
    r"(?i:\b(?:currently|now|presently)\s+(?:at|with|working at|employed at)\s+|"
    r"\bworks?\s+at\s+)"
    r"([A-Z][\w&.\-]*(?:\s+(?:[A-Z][\w&.\-]*|or|and|&))*)"
)
_NOT_AT_RE = re.compile(
    r"(?i:\bnot\s+(?:currently\s+)?(?:at|working at|employed at|with)\s+)"
    r"([A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)*)"
)
_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*years?\b", re.I)
_LAST_N_RE = re.compile(r"\b(?:last|past|recent)\s+(\d{1,2})\s+years?\b", re.I)

_NOT_A_COMPANY = {
    "tech", "big tech", "faang", "startup", "startups", "consulting", "fintech",
    "healthcare", "industry", "product", "engineering",
}
_STOP_TAIL_RE = re.compile(r"\s+(?:who|that|which|and now|,|\.).*$", re.I)


def _clean_company(s: str) -> str:
    s = _STOP_TAIL_RE.sub("", s).strip(" .,&")
    s = re.sub(r"\s+(?:of|and|&|or)$", "", s).strip()
    return s


#: split points that DON'T tell us the connective (comma) vs. ones that DO
_SPLIT_RE = re.compile(r"\s*,\s*|\s+or\s+|\s+and\s+|\s*&\s*|\s*\+\s*|\s+as well as\s+", re.I)
_AND_RE = re.compile(r"\s+and\s+|\s*&\s*|\s*\+\s*|\s+as well as\s+|\bboth\b", re.I)
_OR_RE = re.compile(r"\s+or\s+|\beither\b", re.I)


def parse_value_group(blob: str) -> tuple[list[str], str]:
    """Split a value list AND report the Boolean connective (V4 §3/§4).

    "Google or Meta"          -> (["Google","Meta"], ANY_OF)
    "Amazon and Microsoft"    -> (["Amazon","Microsoft"], ALL_OF)
    "Amazon, Microsoft"       -> (["Amazon","Microsoft"], ANY_OF)   # comma = OR by default
    """
    blob = blob.strip(" .,")
    values = [p.strip(" .,") for p in _SPLIT_RE.split(blob) if p.strip(" .,")]
    has_and = bool(_AND_RE.search(f" {blob} "))
    has_or = bool(_OR_RE.search(f" {blob} "))
    operator = Operator.ALL_OF if (has_and and not has_or) else Operator.ANY_OF
    return values, operator


def _split_or(blob: str) -> list[str]:  # kept for callers that only want the values
    return parse_value_group(blob)[0]


@dataclass
class FactSet:
    criteria: list[SearchCriterion] = field(default_factory=list)
    consumed: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def has_type(self, *types: str) -> bool:
        return any(c.type in types for c in self.criteria)


def extract_facts(query: str, *, context: dict[str, str] | None = None) -> FactSet:
    fs = FactSet()
    ctx_words = context_terms(context or {})
    q = query

    loc_m = _LOC_OR_RE.search(q)
    places: list[str] = []
    if loc_m:
        candidates = _split_or(loc_m.group(1))
        if candidates and all(_looks_like_place(c) for c in candidates):
            places = candidates
    else:
        region_m = _REGION_IN_RE.search(q)
        if region_m:
            places = [region_m.group(1).strip()]
        else:
            for one in _LOC_ONE_RE.finditer(q):
                cand = one.group(1).strip()
                if _looks_like_place(cand):
                    places = [cand]
                    break
    # capitalization invariance (§9): "... in atlanta" / "... IN ATLANTA" must
    # behave like "... in Atlanta". Only when the query itself uses no casing.
    if not places and _casing_uninformative(q):
        ci_or = _LOC_OR_CI_RE.search(q)
        if ci_or:
            cand_list = [p for p in _split_or(ci_or.group(1)) if _plausible_ci_place(p)]
            if len(cand_list) == len(_split_or(ci_or.group(1))) and cand_list:
                places = [p.title() for p in cand_list]
        if not places:
            for one in _LOC_ONE_CI_RE.finditer(q):
                cand = one.group(1).strip()
                if _plausible_ci_place(cand):
                    places = [cand.title()]
                    break
    if places:
        fs.criteria.append(SearchCriterion(
            id="loc", type=CriterionType.LOCATION, values=places,
            operator=Operator.ANY_OF, weight=30, required=location_required(query),
            geo_strict=location_geo_strict(query),
        ))
        fs.consumed.update(p.lower() for p in places)

    for m in _NOT_AT_RE.finditer(q):
        name = _clean_company(m.group(1))
        if name and name.lower() not in _NOT_A_COMPANY:
            fs.criteria.append(SearchCriterion(
                id=f"not_{_slug(name)}", type=CriterionType.CURRENT_COMPANY, value=name,
                operator=Operator.NOT, scope=Scope.CURRENT_COMPANY, weight=25, required=True,
            ))
            fs.consumed.add(name.lower())

    for m in _FORMER_RE.finditer(q):
        blob = _clean_company(m.group(1))
        if not blob:
            continue
        vals, op = parse_value_group(blob)
        names = [n for n in vals if n.lower() not in _NOT_A_COMPANY]
        if not names or any(n.lower() in fs.consumed for n in names):
            continue
        fs.criteria.append(SearchCriterion(
            id=f"past_{_slug(names[0])}", type=CriterionType.PAST_COMPANY,
            values=names, operator=op, scope=Scope.PAST_COMPANY,
            weight=35, required=True,
        ))
        fs.consumed.update(n.lower() for n in names)

    for m in _CURRENT_RE.finditer(q):
        blob = _clean_company(m.group(1))
        vals, op = parse_value_group(blob)
        names = [n for n in vals if n.lower() not in _NOT_A_COMPANY]
        if not names or any(n.lower() in fs.consumed for n in names):
            continue
        fs.criteria.append(SearchCriterion(
            id=f"cur_{_slug(names[0])}", type=CriterionType.CURRENT_COMPANY,
            values=names, operator=op, scope=Scope.CURRENT_COMPANY,
            weight=35, required=True,
        ))
        fs.consumed.update(n.lower() for n in names)

    # "people with Amazon and Microsoft experience" — no former/current cue, but
    # an explicit conjunction of company names before "experience"
    exp_m = _COMPANY_EXPERIENCE_RE.search(q)
    if exp_m and not fs.has_type(CriterionType.PAST_COMPANY, CriterionType.CURRENT_COMPANY):
        vals, op = parse_value_group(exp_m.group(1))
        names = [n for n in vals if _looks_like_company(n)]
        if len(names) >= 1 and not any(n.lower() in fs.consumed for n in names):
            fs.criteria.append(SearchCriterion(
                id=f"exp_{_slug(names[0])}", type=CriterionType.PAST_COMPANY,
                values=names, operator=op, scope=Scope.ANY_EXPERIENCE,
                weight=35, required=True,
            ))
            fs.consumed.update(n.lower() for n in names)

    _extract_semantic_boolean(q, fs)
    _extract_years_experience(q, fs)
    _extract_transition(q, fs)

    fs.criteria = [c for c in fs.criteria if (c.concept or c.value or "").lower() not in ctx_words]
    return fs


#: explicit transition verbs — "from X to Y" only counts as a transition when a
#: transition verb is present, so "moved to Seattle to join a startup" (review #8)
#: is NOT parsed as career_transition(from "to seattle" to "join a startup").
_TRANSITION_RES = (
    re.compile(r"\b(?:moved|transitioned|switched|shifted|pivoted|went|transition)\s+"
               r"from\s+(.+?)\s+(?:to|into)\s+(.+?)(?:\s*$|[,.])", re.I),
    re.compile(r"\b(?:moved|transitioned|switched|shifted|pivoted)\s+(?:into|to)\s+(.+?)\s+from\s+(.+?)(?:\s*$|[,.])", re.I),
    re.compile(r"\bleft\s+(.+?)\s+for\s+(.+?)(?:\s*$|[,.])", re.I),
    re.compile(r"\bformer\s+(.+?)\s+(?:now|currently|later)\s+(?:working\s+)?"
               r"(?:in|as)\s+(.+?)(?:\s*$|[,.])", re.I),
)
_TRANSITION_STOP = re.compile(r"\b(people|person|someone|who|that|which|those|profiles?)\b", re.I)
#: a from/to that is really a company name or a company category is already
#: covered by past_company / company_category — don't add a redundant transition
_NOT_A_TRANSITION_END = {
    "amazon", "google", "meta", "microsoft", "apple", "netflix", "startup", "startups",
    "big tech", "faang", "a startup", "startups now", "industry",
}


def _extract_transition(q: str, fs: FactSet) -> None:
    for i, rx in enumerate(_TRANSITION_RES):
        m = rx.search(q)
        if not m:
            continue
        # pattern 1 (into/to ... from ...) has the groups reversed
        g1, g2 = (m.group(2), m.group(1)) if i == 1 else (m.group(1), m.group(2))
        frm = _TRANSITION_STOP.sub("", g1).strip(" .,?!;:")
        to = _TRANSITION_STOP.sub("", g2).strip(" .,?!;:")
        to = re.sub(r"\btech\b", "technology", to, flags=re.I)
        frm = re.sub(r"\btech\b", "technology", frm, flags=re.I)
        if len(frm) < 2 or len(to) < 2:
            continue
        if frm.lower() in _NOT_A_TRANSITION_END or to.lower() in _NOT_A_TRANSITION_END:
            fs.notes.append(f"transition '{frm} -> {to}' left to past_company/company_category")
            return
        fs.criteria.append(SearchCriterion(
            id="transition", type=CriterionType.CAREER_TRANSITION,
            concept=f"from {frm} to {to}", scope=Scope.CAREER, weight=45, required=True,
        ))
        fs.notes.append(f"career transition: {frm} -> {to}")
        return


# "X and Y experience" / "X, Y and Z experience"
_COMPANY_EXPERIENCE_RE = re.compile(
    r"\b(?:with\s+)?([A-Z][\w&.\-]*(?:\s+(?:and|or|&|,)\s*[A-Z][\w&.\-]*)+)\s+experience\b"
)
#: "security or cloud experts", "AI and security leaders", "sales and marketing pros"
_SEM_BOOL_RE = re.compile(
    r"\b([a-z][\w/ +.-]*?)\s+(and|or)\s+([a-z][\w/ +.-]*?)\s+"
    r"(experts?|specialists?|leaders?|professionals?|engineers?|managers?|people|pros|executives?|"
    r"backgrounds?|experience|skills?|expertise)\b",
    re.I,
)
_ROLE_NOUNS = {"engineer", "engineers", "manager", "managers", "developer", "developers"}


def _extract_semantic_boolean(q: str, fs: FactSet) -> None:
    m = _SEM_BOOL_RE.search(q)
    if not m:
        return
    a, conn, b, noun = m.group(1).strip(), m.group(2).lower(), m.group(3).strip(), m.group(4).lower()
    vals = [v for v in (a, b) if v and len(v) > 1]
    if len(vals) < 2:
        return
    if any(v.lower() in fs.consumed for v in vals):
        return
    op = Operator.ALL_OF if conn == "and" else Operator.ANY_OF
    ctype = CriterionType.ROLE_FUNCTION if noun in _ROLE_NOUNS else CriterionType.PROFESSIONAL_CONCEPT
    fs.criteria.append(SearchCriterion(
        id=f"sem_{_slug(vals[0])}", type=ctype, values=vals, operator=op,
        concept=f"{(' and ' if op == Operator.ALL_OF else ' or ').join(vals)} ({noun})",
        scope=Scope.CAREER, weight=30, required=False,
    ))
    fs.consumed.update(v.lower() for v in vals)


def _extract_years_experience(q: str, fs: FactSet) -> None:
    ym = _YEARS_RE.search(q)
    if not ym or "year" not in q.lower() or _LAST_N_RE.search(q):
        if _LAST_N_RE.search(q):
            fs.notes.append(f"recency window: last {_LAST_N_RE.search(q).group(1)} years")
        return
    n = int(ym.group(1))
    # optional trailing domain: "10+ years of backend experience"
    dom_m = re.search(rf"{re.escape(ym.group(0))}\s+(?:of\s+)?([a-z][\w /-]{{2,40}}?)\s*(?:experience|exp\b|$)", q, re.I)
    domain = (dom_m.group(1).strip() if dom_m else "").strip(" .,")
    fs.criteria.append(SearchCriterion(
        id="years_exp", type=CriterionType.YEARS_EXPERIENCE,
        value=str(n), concept=f"at least {n} years" + (f" in {domain}" if domain else ""),
        scope=Scope.CAREER, weight=20, required=True,
    ))


def _looks_like_company(s: str) -> bool:
    return bool(re.match(r"^[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*)?$", s.strip())) \
        and s.strip().lower() not in _NOT_A_COMPANY


#: hedging language that turns a mentioned location into a preference rather
#: than a hard filter (review finding D2 — "AI engineers in Atlanta,
#: preferably" must NOT become a required location). Scanned over the whole
#: query rather than tied to the exact location span: these phrases only ever
#: describe a location preference in a candidate-search query, so a whole-query
#: match carries no false-positive risk for other criteria types. The absence
#: of any of these phrases keeps the pre-existing default (mentioned == required),
#: which "strictly in Atlanta" / "must be in Atlanta" / a bare "in Atlanta"
#: still hit unchanged.
_LOCATION_PREFERENCE_RE = re.compile(
    r"\b(preferably|ideally|if possible|nice to have|a plus|a bonus|not required|"
    r"not a requirement|not necessary|not mandatory|optional(?:ly)?|open to remote|"
    r"remote[- ](?:ok|okay|friendly|welcome)|willing to relocate|flexible on location|"
    r"no location requirement|doesn't matter|does not matter)\b",
    re.I,
)


def location_required(query: str) -> bool:
    """Whether an explicit location mention in ``query`` should be a hard
    requirement (the default) or a preference — decided from hedging language
    only, never from the location text itself. Shared by the deterministic
    fact layer and the deterministic parser's own location criterion so both
    apply the identical rule (review finding D2)."""
    return not _LOCATION_PREFERENCE_RE.search(query)


#: unambiguous literal phrasing that forbids a nearby-city substitute for the
#: named location — "strictly/only/exclusively in X" or an explicit "not
#: nearby cities/suburbs" — as opposed to an ordinary "in X" mention, which
#: still allows a labeled nearby-city Near Match (review finding F3: a query
#: had no way to say this at all, so a strict location was relaxed exactly
#: like an ordinary one).
_LOCATION_STRICT_RE = re.compile(
    r"\b(?:strictly|only|exclusively)\s+(?:in|near|around|based in|located in)\b"
    r"|\bin\s+[\w .,-]+?\s+only\b"
    r"|\b(?:not|no)\s+nearby\s+(?:cit(?:y|ies)|suburbs?|areas?|substitutes?)\b",
    re.I,
)


def location_geo_strict(query: str) -> bool:
    """Whether an explicit location mention in ``query`` forbids nearby-city
    substitution, as opposed to an ordinary required location (which may
    still surface a nearby-city candidate as a labeled Near Match). Detected
    from unambiguous literal phrasing only, never inferred from the place
    name itself."""
    return bool(_LOCATION_STRICT_RE.search(query))


def _looks_like_place(s: str) -> bool:
    sl = s.lower()
    if sl in _NOT_A_PLACE or sl in _NOT_A_COMPANY:
        return False
    if any(tok in _NOT_A_PLACE for tok in sl.split()):
        return False
    if sl in _REGION_WORDS:
        return True
    return bool(re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*[A-Z][a-z]+| [A-Z]{2})?$", s))


# ─────────────────────────── merge + repair (V4 §15/§17) ───────────────────────────

_FACT_TYPES = {
    CriterionType.LOCATION, CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY,
}


def _values_of(c: SearchCriterion) -> set[str]:
    vs = set(c.values or [])
    if c.value:
        vs.add(c.value)
    return {v.strip().lower() for v in vs if v.strip()}


_SEMANTIC_FAMILY = {
    CriterionType.SEMANTIC_CONCEPT, CriterionType.PROFESSIONAL_CONCEPT,
    CriterionType.ROLE_FUNCTION, CriterionType.INDUSTRY_EXPERIENCE,
}


def _same_fact(a: SearchCriterion, b: SearchCriterion) -> bool:
    same_family = (
        a.type == b.type
        or {a.type, b.type} <= {CriterionType.CURRENT_COMPANY, CriterionType.PAST_COMPANY}
        or {a.type, b.type} <= _SEMANTIC_FAMILY
    )
    if not same_family:
        return False
    return bool(_values_of(a) & _values_of(b))


def merge_into_plan(plan: ParsedSearchQuery, facts: FactSet, *, context: dict[str, str]) -> tuple[ParsedSearchQuery, list[str]]:
    """Fold deterministic hard facts into an LLM plan. The LLM keeps its semantic
    criteria; explicit facts it dropped are re-added; explicit facts it mangled
    (wrong scope / OR->AND / lost NOT) are corrected toward the deterministic
    reading."""
    issues: list[str] = []
    crits = list(plan.criteria)
    ctx_words = context_terms(context)

    fact_values = {v for f in facts.criteria for v in _values_of(f)} | facts.consumed
    #: words already covered by a structured transition / years / boolean fact
    structured_words: set[str] = set()
    for f in facts.criteria:
        if f.type in (CriterionType.CAREER_TRANSITION, CriterionType.YEARS_EXPERIENCE,
                      CriterionType.PROFESSIONAL_CONCEPT, CriterionType.ROLE_FUNCTION):
            structured_words.update(re.findall(r"[a-z][a-z+.#-]{2,}", (f.concept or "").lower()))
    structured_words -= {"least", "years", "professional", "career"}

    kept: list[SearchCriterion] = []
    for c in crits:
        cv = (c.value or c.concept or "").lower()
        if c.type in (CriterionType.SKILL, CriterionType.KEYWORD) and cv in ctx_words:
            issues.append(f"dropped context word '{c.value}' parsed as a {c.type}")
            continue
        weak = c.type in (CriterionType.SKILL, CriterionType.KEYWORD, CriterionType.SEMANTIC_CONCEPT) and not c.required
        if weak and cv in fact_values and len(cv.split()) <= 2:
            issues.append(f"dropped redundant {c.type} '{cv}' (already a hard fact)")
            continue
        if weak and len(cv.split()) == 1 and cv in structured_words:
            issues.append(f"dropped redundant {c.type} '{cv}' (covered by a structured criterion)")
            continue
        kept.append(c)
    crits = kept

    for f in facts.criteria:
        if f.type == CriterionType.LOCATION and f.operator != Operator.NOT:
            # A query has one "where" dimension. The deterministic regex layer
            # only ever sees the raw phrase ("southeast us"), while the LLM may
            # legitimately expand a region into member states/cities that share
            # no literal text with it ("southeast us" -> Georgia, Florida, ...).
            # Matching by VALUE overlap (as _same_fact does) would then treat
            # the LLM's correct expansion as "not the same fact" and bolt on a
            # second required LOCATION criterion with only the unexpanded raw
            # phrase as its value — which no real profile ever literally
            # contains, so the hard gate rejects every candidate outright. Match
            # by type alone here so an LLM-expanded location is recognised as
            # already covering the explicit fact instead of being duplicated.
            match = next((c for c in crits if c.type == CriterionType.LOCATION
                          and c.operator != Operator.NOT), None)
        else:
            match = next((c for c in crits if _same_fact(c, f)), None)
        if match is None:
            issues.append(f"re-added explicit {f.type} {sorted(_values_of(f))} the plan omitted")
            crits.append(f)
            continue
        # union any explicit values the plan missed ("Google or Meta" -> both) —
        # but not the raw unexpanded region phrase into an ALREADY-expanded
        # multi-value location (adds nothing: it can never literally appear on
        # a profile, it would only make the interpretation tag noisier).
        already_expanded_region = (
            f.type == CriterionType.LOCATION and len(_values_of(match)) > 1
            and not _same_fact(match, f)
        )
        missing = set() if already_expanded_region else _values_of(f) - _values_of(match)
        if missing:
            have = match.values or ([match.value] if match.value else [])
            add = [v for v in (f.values or [f.value]) if v.lower() in missing]
            match.values = have + add
            match.value = ""
            issues.append(f"added missing values {sorted(missing)} to {match.type}")
        if f.scope and match.scope != f.scope and f.type in (CriterionType.PAST_COMPANY, CriterionType.CURRENT_COMPANY):
            issues.append(f"fixed scope of {sorted(_values_of(match))}: {match.scope} -> {f.scope}")
            match.scope = f.scope
            match.type = f.type
        # the deterministic connective is authoritative when the query was explicit
        if f.operator in (Operator.ANY_OF, Operator.ALL_OF) and len(_values_of(f)) > 1 \
                and match.operator != f.operator:
            issues.append(f"set operator of {sorted(_values_of(match))}: {match.operator} -> {f.operator}")
            match.operator = f.operator
        if f.operator == Operator.NOT and match.operator != Operator.NOT:
            issues.append(f"restored NOT for {sorted(_values_of(match))}")
            match.operator = Operator.NOT
        # LOCATION is excluded here (review finding D2): mandatory-vs-preferred
        # for a location is Anthropic's call (or, with no LLM at all, the
        # hedge-aware ``location_required()`` value already on the fact) — this
        # deterministic layer must never one-directionally force a location the
        # LLM correctly read as a preference back into a hard requirement.
        # Every other fact type (company / transition / years) keeps the prior
        # behaviour: an explicit fact the query stated plainly is required.
        if f.required and not match.required and f.type != CriterionType.LOCATION:
            issues.append(f"made {sorted(_values_of(match))} required (explicit in the query)")
            match.required = True
        # geo_strict only ever turns ON from an unambiguous literal phrase
        # ("strictly"/"only"/"exclusively") — never off, and never inferred by
        # this layer from the place name itself (review finding F3).
        if f.type == CriterionType.LOCATION and f.geo_strict and not match.geo_strict:
            issues.append(f"marked {sorted(_values_of(match))} geo_strict (explicit 'strictly/only' wording)")
            match.geo_strict = True

    plan.criteria = crits
    plan.context = {**plan.context, **context}
    return _renorm(plan), issues


def validate_and_repair(plan: ParsedSearchQuery, query: str, facts: FactSet, *, context: dict[str, str]) -> tuple[ParsedSearchQuery, list[str]]:
    """Final safety net (V4 §17). Repairs when it can, only warns when it can't;
    an unrepairable Boolean mismatch lowers interpretation_confidence."""
    plan, issues = merge_into_plan(plan, facts, context=context)
    ql = query.lower()
    unrepaired = False

    if re.search(r"\b[\w/-]+\s+or\s+[\w/-]+", ql) and not any(
        c.operator == Operator.ANY_OF and len(_values_of(c)) > 1 for c in plan.criteria
    ):
        # try a deterministic reconstruction from the raw "X or Y <noun>" shape
        m = _SEM_BOOL_RE.search(query)
        if m and m.group(2).lower() == "or":
            fs2 = FactSet()
            _extract_semantic_boolean(query, fs2)
            if fs2.criteria:
                plan.criteria.append(fs2.criteria[0])
                _renorm(plan)
                issues.append(f"reconstructed semantic ANY_OF for {fs2.criteria[0].values}")
            else:
                unrepaired = True
        else:
            unrepaired = True
        if unrepaired:
            issues.append("query contains 'or' but no OR criterion could be produced")

    if re.search(r"\bnot\b|\bnever\b|\bexcept\b|\bexcluding\b", ql) and not any(
        c.operator == Operator.NOT for c in plan.criteria
    ):
        issues.append("query contains a negation but no NOT criterion was produced")
        unrepaired = True

    if unrepaired:
        plan.interpretation_confidence_cap = min(plan.interpretation_confidence_cap, 0.5)
        plan.interpretation_confidence = round(min(plan.interpretation_confidence, 0.5), 2)
    return plan, issues


# ─────────────────────────── summary + confidence (V4 §18) ───────────────────────────

_AMBIGUOUS_RE = re.compile(
    r"\b(worked in tech|in tech|technical people|technology executives?|good people|"
    r"startup experience|strong profile|the best people)\b", re.I,
)


def build_summary(plan: ParsedSearchQuery, query: str) -> tuple[str, float]:
    parts: list[str] = []
    for c in plan.criteria:
        vals = sorted(_values_of(c)) or ([c.concept] if c.concept else [])
        joiner = " or " if c.operator == Operator.ANY_OF else " and "
        label = joiner.join(v for v in vals if v) or c.type
        prefix = "not " if c.operator == Operator.NOT else ""
        scope = f" ({c.scope})" if c.scope and c.scope != Scope.CAREER else ""
        tag = "required" if c.required else "preferred"
        parts.append(f"{prefix}{c.type.replace('_', ' ')}: {label}{scope} [{tag}]")

    head = f"[intent: {plan.intent.replace('_', ' ')}]"
    summary = "Interpreted as — " + "; ".join([head, *parts]) if parts else \
        f"Interpreted as — {head}; no structured criteria found."
    if plan.context.get("purpose"):
        summary += f". Context (not a filter): {plan.context['purpose']}."
    if plan.target_person_context:
        tp = ", ".join(f"{k}={v}" for k, v in plan.target_person_context.items())
        summary += f". Relational context (not a filter): {tp}."
    if plan.unresolved:
        summary += f". Unresolved (not guessed): {', '.join(plan.unresolved)}."

    confidence = 0.85
    if _AMBIGUOUS_RE.search(query):
        confidence -= 0.25
    if plan.unresolved:
        confidence -= 0.3
    required_facts = sum(1 for c in plan.criteria if c.required and c.type in _FACT_TYPES)
    if required_facts == 0 and any(
        c.type in (CriterionType.SEMANTIC_CONCEPT, CriterionType.COMPANY_CATEGORY) for c in plan.criteria
    ):
        confidence -= 0.1
    return summary, round(max(0.2, min(1.0, confidence)), 2)


# ─────────────────────────── helpers ───────────────────────────

def _renorm(plan: ParsedSearchQuery) -> ParsedSearchQuery:
    total = sum(c.weight for c in plan.criteria) or 1.0
    if abs(total - 100.0) > 0.5:
        for c in plan.criteria:
            c.weight = round(c.weight / total * 100, 2)
    return plan


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:24] or "x"
