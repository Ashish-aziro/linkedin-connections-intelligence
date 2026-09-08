"""Tolerant LLM transport → strict search plan (V4 PART 6 B3).

The query-interpretation call parses into the permissive ``LenientSearchPlan``
(never fails on a nullable field). ``repair_plan`` then normalizes each raw
criterion — cross-filling ``value`` / ``values`` / ``concept`` by criterion
type, dropping criteria with no usable content, synthesizing a missing ``id`` —
and returns a plain dict that ``ParsedSearchQuery.model_validate`` accepts.

The strict ``SearchCriterion`` / ``ParsedSearchQuery`` schemas are unchanged.
This layer only removes the class of failure where a model returned
``criteria[i].value = null`` (because it used ``concept``) and the strict
``value: str`` field rejected it, causing three identical retries then a
silent deterministic fallback.
"""
from __future__ import annotations

import re

from app.constants import SEMANTIC_CRITERION_TYPES
from app.schemas import LenientSearchPlan

_ALIAS = {
    "industry": "industry_experience", "sector": "industry_experience",
    "employer_industry": "industry_experience", "employer_category": "company_category",
    "role": "role_function", "function": "role_function", "job_function": "role_function",
    "profession": "role_function", "transition": "career_transition",
    "career_change": "career_transition", "leadership": "professional_concept",
    "mentorship": "professional_concept", "capability": "professional_concept",
    "concept": "professional_concept", "years_of_experience": "years_experience",
    "tenure": "years_experience",
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:24] or "x"


def _as_str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    if isinstance(v, dict):
        for k in ("name", "value", "concept", "skill", "keyword"):
            if v.get(k):
                return [str(v[k]).strip()]
        return []
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                for k in ("name", "value", "concept", "skill", "keyword"):
                    if x.get(k):
                        out.append(str(x[k]).strip())
                        break
        return out
    return []


def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float, bool)):
        return str(v)
    lst = _as_str_list(v)
    return lst[0] if lst else ""


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _canon_type(v) -> str:
    t = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIAS.get(t, t)


def repair_plan(lenient: LenientSearchPlan) -> tuple[dict, list[str]]:
    """Return ``(plain_dict_for_strict_validation, repair_notes)``."""
    raw = lenient.model_dump()
    notes: list[str] = []
    crits_out: list[dict] = []

    for i, c in enumerate(raw.get("criteria") or []):
        if not isinstance(c, dict):
            notes.append(f"criteria[{i}] was not an object — dropped")
            continue
        ctype = _canon_type(c.get("type"))
        values = _as_str_list(c.get("values"))
        value = _as_str(c.get("value"))
        concept = c.get("concept")
        concept = concept.strip() if isinstance(concept, str) and concept.strip() else None

        if c.get("value") is None and (values or concept):
            notes.append(f"criteria[{i}] value=null — filled from {'values' if values else 'concept'}")

        if not value and values:
            value = values[0]
        if not values and value:
            values = [value]
        if not value and not values and concept:
            value, values = concept, [concept]
        if concept is None and value and (ctype in SEMANTIC_CRITERION_TYPES or not ctype):
            concept = value

        if not value and not values and not concept:
            notes.append(f"criteria[{i}] had no value / values / concept — dropped")
            continue

        cid = c.get("id") or f"c{i}_{_slug(ctype or value or concept or 'x')}"
        crits_out.append({
            "id": str(cid),
            "type": ctype or ("professional_concept" if concept else "keyword"),
            "weight": _num(c.get("weight"), 0.0),
            "required": bool(c["required"]) if c.get("required") is not None else False,
            "value": value,
            "values": values,
            "operator": str(c.get("operator") or "ANY_OF"),
            "scope": c.get("scope"),
            "concept": concept,
            "modality": str(c.get("modality") or "certain"),
        })

    repaired: dict = {"criteria": crits_out}
    for k in ("intent", "context", "target_person_context", "unresolved",
              "interpretation_summary", "interpretation_confidence"):
        v = raw.get(k)
        if v is not None:
            repaired[k] = v
    # weight totals are re-normalized to 100 by ParsedSearchQuery._normalize_weights
    return repaired, notes
