"""Query-aware, evidence-grounded packet builder for the semantic judge
(V4 PART 3 §11–§13).

One compact packet per candidate. It carries the STRUCTURED facts (role, dates,
company, experience-level semantics, classifications, education, assertions) —
not raw JSON, not image URLs — with a stable evidence REFERENCE on every item so
the judge cites ``exp:<id>`` / ``edu:<id>`` / ``assertion:<n>`` / ``company:<key>``
rather than free text, and the backend can reject an invented reference.

Query-aware: the SearchPlan decides what to prioritise (a mentor query pulls up
career progression, leadership, volunteering, recommendations; a "professors in
AI" query pulls up faculty roles, publications, education, AI evidence) — but
never trims so far that the judge cannot spot a contradiction (§12).

Operates purely from bulk-loaded facts — no per-candidate DB query (§58).
"""
from __future__ import annotations

import json
import re

from app.config import settings
from app.schemas import ParsedSearchQuery
from app.services.career_chronology import exp_semantics_by_id, ordered_experiences
from app.services.company_intel import company_key
from app.services.matching import norm
from app.services.scoring import ProfileFacts, ScoringContext

_STOP = {
    "the", "a", "an", "of", "in", "at", "to", "and", "or", "not", "who", "with",
    "for", "into", "from", "my", "me", "is", "are", "people", "person", "someone",
    "any", "experience", "background", "professional", "professionals",
}
_MENTOR_TOKENS = {"mentor", "mentoring", "mentored", "coach", "coaching", "advis",
                  "advisor", "advising", "leadership", "lead", "manage", "management",
                  "manager", "guide", "guiding", "sponsor", "career", "grow"}
_ACADEMIA_TOKENS = {"professor", "faculty", "academia", "academic", "research",
                    "researcher", "publication", "published", "paper", "papers",
                    "phd", "postdoc", "lecturer", "tenure", "university"}


def priority_tokens(query: str, parsed: ParsedSearchQuery) -> set[str]:
    toks: set[str] = set()
    for c in parsed.criteria:
        for piece in [c.concept or "", c.value or "", *(c.values or [])]:
            toks |= {t for t in re.findall(r"[a-z0-9]+", piece.lower()) if len(t) > 2}
    toks |= {t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2}
    toks |= {t for t in re.findall(r"[a-z0-9]+",
                                   " ".join(parsed.target_person_context.values()).lower()) if len(t) > 2}
    toks |= {t for t in re.findall(r"[a-z0-9]+", parsed.intent.replace("_", " ")) if len(t) > 2}
    return {t for t in toks if t not in _STOP}


def _matches(text: str | None, toks: set[str]) -> int:
    if not text:
        return 0
    low = text.lower()
    return sum(1 for t in toks if t in low)


def _ref(prefix: str, row, idx: int) -> str:
    """Stable evidence reference — the real normalized id when available,
    otherwise a packet-local index (V4 PART 3.5 §2)."""
    rid = getattr(row, "id", None)
    return f"{prefix}:{rid}" if rid else f"{prefix}:{idx}"


def build_packets(
    bundle: list[tuple], parsed: ParsedSearchQuery, ctx: ScoringContext, *, query: str,
    max_packet_chars: int | None = None,
    unresolved_by_person: dict[str, list[str]] | None = None,
    full_profile: bool = False,
) -> list[dict]:
    """``bundle``: ``[(person, ProfileFacts, {"volunteering": [...], "recommendations": [...]})]``.
    Returns one packet dict per candidate (same order). ``max_packet_chars``
    overrides ``settings.semantic_judge_max_packet_chars`` (the final auditor
    uses its own budget, V4 PART 5 §4/§40). ``unresolved_by_person`` — when
    given, stamps each packet with ONLY the criterion ids that person still
    needs judged (hardening PART 10) — everything else on their profile was
    already resolved from stored facts/semantics and must not be re-asked."""
    toks = priority_tokens(query, parsed)
    want_mentor_context = bool(toks & _MENTOR_TOKENS) or parsed.intent == "mentor_recommendation"
    want_academia_context = bool(toks & _ACADEMIA_TOKENS)
    packets = [
        _one_packet(person, facts, extras, ctx, toks,
                    want_mentor_context=want_mentor_context or full_profile,
                    want_academia_context=want_academia_context or full_profile,
                    max_packet_chars=max_packet_chars, full_profile=full_profile)
        for person, facts, extras in bundle
    ]
    if unresolved_by_person is not None:
        for pkt in packets:
            pkt["unresolved_criteria"] = unresolved_by_person.get(pkt["person_id"], [])
    return packets


def packet_refs(packet: dict) -> set[str]:
    """Every evidence reference the packet legitimately contains (V4 PART 3 §17
    — the validator drops any ref outside this set)."""
    refs: set[str] = set()
    cur = packet.get("current") or {}
    if cur.get("experience_id"):
        refs.add(f"exp:{cur['experience_id']}")
    for e in packet.get("past", []):
        if e.get("experience_id"):
            refs.add(f"exp:{e['experience_id']}")
    for es in packet.get("experience_semantics", []):
        if es.get("experience_id"):
            refs.add(f"exp:{es['experience_id']}")
    for ed in packet.get("education", []):
        if ed.get("education_id"):
            refs.add(f"edu:{ed['education_id']}")
    for c in packet.get("certifications", []):
        if c.get("certification_id"):
            refs.add(f"cert:{c['certification_id']}")
    for s in packet.get("skills", []):
        refs.add(f"skill:{norm(s)}")
    for i, _a in enumerate(packet.get("semantic_assertions", [])):
        refs.add(f"assertion:{i}")
    for cc in packet.get("company_classifications", []):
        if cc.get("ref"):
            refs.add(cc["ref"])
    # V4 PART 3.5 §2 — publications / volunteering / recommendations are grounded
    for key in ("publications", "volunteering", "recommendations_received"):
        for item in packet.get(key, []):
            if isinstance(item, dict) and item.get("ref"):
                refs.add(item["ref"])
    return refs


def packet_experience_current_map(packet: dict) -> dict[str, bool]:
    """experience_id -> is_current, for the scope validator (§51)."""
    out: dict[str, bool] = {}
    cur = packet.get("current") or {}
    if cur.get("experience_id"):
        out[cur["experience_id"]] = True
    for e in packet.get("past", []):
        if e.get("experience_id"):
            out[e["experience_id"]] = False
    return out


# ─────────────────────── single packet ───────────────────────


def _one_packet(person, facts: ProfileFacts, extras: dict, ctx: ScoringContext, toks: set[str],
                *, want_mentor_context: bool, want_academia_context: bool,
                max_packet_chars: int | None = None, full_profile: bool = False) -> dict:
    exps = list(reversed(ordered_experiences(list(facts.experiences))))  # newest first
    current = next((e for e in exps if getattr(e, "is_current", False)), None)
    past = [e for e in exps if not getattr(e, "is_current", False)]
    past.sort(key=lambda e: _matches(f"{e.position} {e.company_name} {e.description}", toks), reverse=True)
    # FULL SONNET VERIFICATION — keep EVERY career row (compaction, not omission,
    # handles size); the legacy judge keeps only the 8 most query-relevant.
    keep_past = past if full_profile else past[:8]

    def _exp_row(e, *, full: bool) -> dict:
        rel = _matches(f"{e.position} {e.company_name} {e.description}", toks) > 0
        limit = 900 if (full or rel) else 200
        return {
            "ref": f"exp:{e.id}" if getattr(e, "id", None) else None,
            "experience_id": getattr(e, "id", None),
            "title": e.position,
            "company": e.company_name,
            "start_year": e.start_year,
            "end_year": e.end_year if not getattr(e, "is_current", False) else None,
            "is_current": bool(getattr(e, "is_current", False)),
            "description": (e.description or "")[:limit] or None,
        }

    sem = facts.semantic or {}
    exp_sem = exp_semantics_by_id(sem)
    kept_ids = {getattr(e, "id", None) for e in ([current] if current else []) + keep_past}
    exp_sem_rows = [
        {
            "experience_id": eid,
            "role_function": es.get("role_function"),
            "professional_domain": es.get("professional_domain"),
            "role_domains": es.get("role_domains", []),
            "role_seniority": es.get("role_seniority"),
            "employer_industries": es.get("employer_industries", []),
            "employer_categories": es.get("employer_categories", []),
            "leadership_signals": es.get("leadership_signals", []),
            "mentoring_signals": es.get("mentoring_signals", []),
            "founder_signals": es.get("founder_signals", []),
            "confidence": es.get("confidence", 0.6),
        }
        for eid, es in exp_sem.items() if eid in kept_ids
    ]

    cc_rows: list[dict] = []
    seen_keys: set[str] = set()
    for e in ([current] if current else []) + keep_past:
        if not e.company_name:
            continue
        key = company_key(getattr(e, "company_id", None), e.company_name)
        if key in seen_keys:
            continue
        row = ctx.company_class.get(key)
        if row and (row.get("is_startup") is not None or row.get("is_big_tech") is not None
                    or row.get("is_technology_company") is not None or row.get("industries")):
            seen_keys.add(key)
            cc_rows.append({
                "ref": f"company:{key}",
                "company": e.company_name,
                "is_startup": row.get("is_startup"),
                "is_big_tech": row.get("is_big_tech"),
                "is_technology_company": row.get("is_technology_company"),
                "industries": (row.get("industries") or [])[:4],
                "confidence": row.get("confidence"),
                "provenance": row.get("provenance"),
            })

    assertions = [
        {
            "concept": a.get("concept"),
            "category": a.get("category", "industry_experience"),
            "scope": a.get("scope", "career"),
            "confidence": a.get("confidence", 0.6),
            "experience_ids": a.get("experience_ids", []),
            "education_ids": a.get("education_ids", []),
            "certification_ids": a.get("certification_ids", []),
            "evidence": (a.get("evidence", []) or [])[:2],
        }
        for a in sem.get("semantic_assertions", []) if isinstance(a, dict) and a.get("concept")
    ][:12]

    all_skill_names = [s.skill_name for s in facts.skills]
    skills_rel = [s for s in all_skill_names if _matches(s, toks)] or all_skill_names[:15]
    skills_rel = skills_rel[:25]

    vols = extras.get("volunteering", []) or []
    recs = extras.get("recommendations", []) or []
    packet = {
        "person_id": person.id,
        "headline": person.headline,
        "location": person.location_text,
        "career_summary": sem.get("career_summary"),
        **({
            "name": person.full_name,
            "about": (person.about or "")[:1200] or None,
            "current_title": person.current_title,
            "current_company": person.current_company,
            "languages": [
                {"name": getattr(lg, "name", None), "proficiency": getattr(lg, "proficiency", None)}
                for lg in (getattr(facts, "languages", []) or [])
            ] or None,
            "industries": (sem.get("industries") or [])[:8] or None,
            "job_families": (sem.get("job_families") or [])[:8] or None,
            "role_functions": (sem.get("role_functions") or sem.get("technical_domains") or [])[:8] or None,
            "domain_expertise": (sem.get("domain_expertise") or [])[:8] or None,
            "leadership_experience": (sem.get("leadership_experience") or [])[:6] or None,
        } if full_profile else {}),
        "current": _exp_row(current, full=True) if current else None,
        "past": [_exp_row(e, full=False) for e in keep_past],
        "experience_semantics": exp_sem_rows,
        "company_classifications": cc_rows,
        "education": [
            {"ref": f"edu:{ed.id}", "education_id": ed.id, "school": ed.school_name,
             "degree": ed.degree, "field": ed.field_of_study}
            for ed in facts.education[:6]
        ],
        "skills": skills_rel,
        "certifications": [
            {"ref": f"cert:{c.id}", "certification_id": c.id, "name": c.name}
            for c in facts.certifications if c.name
        ][:15],
        "publications": [
            {"ref": _ref("pub", p, i), "publication_id": getattr(p, "id", None),
             "title": p.title, "description": (getattr(p, "description", "") or "")[:240] or None}
            for i, p in enumerate(facts.publications[:8]) if p.title
        ],
        "semantic_assertions": assertions,
    }

    vol_hay = " ".join(f"{v.role} {v.organization} {v.description or ''}" for v in vols)
    if want_mentor_context or _matches(vol_hay, toks):
        packet["volunteering"] = [
            {"ref": _ref("vol", v, i), "volunteering_id": getattr(v, "id", None),
             "role": v.role, "organization": v.organization, "description": (v.description or "")[:300]}
            for i, v in enumerate(vols[:8])
        ]
        packet["recommendations_received"] = [
            {"ref": _ref("rec", r, i), "recommendation_id": getattr(r, "id", None),
             "relationship": getattr(r, "relationship", None), "text": (r.text or "")[:400]}
            for i, r in enumerate(recs[:5]) if r.text
        ]
    elif vols:
        packet["volunteering"] = [
            {"ref": _ref("vol", v, i), "volunteering_id": getattr(v, "id", None),
             "role": v.role, "organization": v.organization}
            for i, v in enumerate(vols[:6])
        ]

    if want_academia_context:
        packet["education"] = [
            {"ref": f"edu:{ed.id}", "education_id": ed.id, "school": ed.school_name,
             "degree": ed.degree, "field": ed.field_of_study,
             "start_year": ed.start_year, "end_year": ed.end_year}
            for ed in facts.education[:8]
        ]

    return _fit_size(packet, max_packet_chars)


def _size(packet: dict) -> int:
    return len(json.dumps(packet, ensure_ascii=False, default=str))


def _fit_size(packet: dict, cap: int | None = None) -> dict:
    """HARD-ENFORCE the per-packet char budget (V4 PART 3.6 §7).

    Progressive trimming, least-critical evidence first. On return the invariant
    ``_size(packet) <= semantic_judge_max_packet_chars`` HOLDS (unless the cap is
    <= 0). If the mandatory-minimum packet still cannot fit, ``_packet_too_large``
    is set and the candidate is left unjudged rather than an oversized request
    being sent (§8).

    NEVER removed: person_id, current-role identity, company classifications
    relevant to the query, and the evidence refs of any retained evidence."""
    if cap is None:
        cap = settings.semantic_judge_max_packet_chars
    if cap <= 0 or _size(packet) <= cap:
        return packet

    def done() -> bool:
        return _size(packet) <= cap

    #: each stage trims the least-critical evidence first; return as soon as the
    #: packet fits. Ordered per V4 PART 3.6 §7.
    stages = [
        lambda: _trunc_recs(packet, 120),
        lambda: packet.__setitem__("recommendations_received", []),
        lambda: [v.pop("description", None) for v in packet.get("volunteering", [])],
        lambda: [e.update(description=(e["description"][:160])) for e in packet.get("past", []) if e.get("description")],
        lambda: [a.__setitem__("evidence", []) for a in packet.get("semantic_assertions", [])],
        lambda: [p.pop("description", None) for p in packet.get("publications", []) if isinstance(p, dict)],
        lambda: _cap_lists(packet, skills=8, certifications=5, publications=4,
                           semantic_assertions=6, experience_semantics=6),
        lambda: (packet.__setitem__("career_summary", (packet.get("career_summary") or "")[:200] or None)),
        lambda: _cap_lists(packet, skills=3, certifications=2, publications=1,
                           semantic_assertions=2, experience_semantics=2, education=2),
        lambda: [e.pop("description", None) for e in packet.get("past", [])],
        lambda: _drop_past_to(packet, 2),
        lambda: (packet.__setitem__("career_summary", None), packet.pop("headline", None),
                 packet.pop("location", None), packet.pop("volunteering", None)),
        lambda: _cap_lists(packet, skills=0, certifications=0, publications=0,
                           semantic_assertions=0, experience_semantics=0, education=0),
        lambda: _drop_past_to(packet, 0),
        lambda: _minimise_current(packet),
    ]
    for step in stages:
        step()
        if done():
            return _mark(packet)

    # mandatory minimum (person_id + current identity + company classifications)
    # still does not fit — do NOT exceed the cap silently (§8)
    packet["_packet_too_large"] = True
    return _mark(packet)


def _mark(packet: dict) -> dict:
    packet["_truncated"] = True
    return packet


def _trunc_recs(packet: dict, n: int) -> None:
    for r in packet.get("recommendations_received", []):
        if isinstance(r, dict):
            r["text"] = (r.get("text") or "")[:n]


def _cap_lists(packet: dict, **limits: int) -> None:
    for key, n in limits.items():
        if key in packet and isinstance(packet[key], list):
            packet[key] = packet[key][:n]


def _drop_past_to(packet: dict, keep: int) -> None:
    past = packet.get("past")
    if isinstance(past, list) and len(past) > keep:
        packet["past"] = past[:keep]


def _minimise_current(packet: dict) -> None:
    cur = packet.get("current")
    if isinstance(cur, dict):
        packet["current"] = {k: cur.get(k) for k in ("ref", "experience_id", "title", "company", "is_current")}


def plan_payload(query: str, parsed: ParsedSearchQuery, judge_crits: list) -> dict:
    """The universal SearchPlan context the judge needs (V4 PART 3 §20). Boolean
    structure is preserved — values + operator, not a flattened phrase (§57)."""
    return {
        "original_query": query,
        "intent": parsed.intent,
        "context": parsed.context,
        "target_person_context": parsed.target_person_context,
        "unresolved": parsed.unresolved,
        "interpretation_summary": parsed.interpretation_summary,
        "criteria_to_judge": [
            {
                "id": c.id,
                "type": c.type,
                "concept": c.concept or c.value,
                "values": c.values,
                "operator": c.operator,
                "scope": c.scope,
                "required": c.required,
                "modality": c.modality,
                "weight": round(c.weight, 1),
            }
            for c in judge_crits
        ],
    }
