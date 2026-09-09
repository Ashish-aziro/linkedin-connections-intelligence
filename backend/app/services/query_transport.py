"""Tolerant LLM transport → strict search plan.

The query-interpretation call parses into the permissive ``LenientSearchPlan``
(which never fails on a nullable / missing / string-typed field). ``repair_plan``
then normalizes each raw criterion — STRUCTURAL repair only — and returns a plain
dict that ``ParsedSearchQuery.model_validate`` accepts.

What this layer repairs (representation, never meaning):

* ``operator: null``            -> "ANY_OF" (the strict schema's own default)
* ``value: null`` + concept     -> value filled from concept (semantic types)
* ``value: null`` + values      -> value = values[0]  (backward-compat single value)
* ``values`` missing + value    -> values = [value]
* ``values: "Atlanta"`` (str)   -> ["Atlanta"]
* ``id`` missing                -> synthesized stable slug
* ``weight: "30"`` / missing    -> numeric (0.0 -> re-normalized to 100 by the schema)
* ``modality`` missing / null   -> "certain"
* ``scope`` missing / null      -> None
* criterion-type spelling       -> existing generic alias map
* a criterion with no value / values / concept at all -> dropped (unusable)

What it must NOT do: invent a value the model never supplied (no query-specific
semantic guessing). The STRICT ``SearchCriterion`` / ``ParsedSearchQuery`` schemas
are unchanged and still do the final validation.
"""
from __future__ import annotations

import re

from app.constants import SEMANTIC_CRITERION_TYPES
from app.schemas import _CRITERION_TYPE_ALIASES, LenientSearchPlan

_PASS_THROUGH_KEYS = (
    "intent", "context", "target_person_context", "unresolved",
    "interpretation_summary", "interpretation_confidence",
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:24] or "x"


def _as_str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [p.strip() for p in re.split(r"\s*,\s*", v) if p.strip()]
    if isinstance(v, dict):
        for k in ("name", "value", "concept", "skill", "keyword", "label"):
            if v.get(k):
                return [str(v[k]).strip()]
        return []
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, (int, float, bool)):
                out.append(str(x))
            elif isinstance(x, dict):
                for k in ("name", "value", "concept", "skill", "keyword", "label"):
                    if x.get(k):
                        out.append(str(x[k]).strip())
                        break
        return out
    if isinstance(v, (int, float, bool)):
        return [str(v)]
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
    if isinstance(v, str):
        v = v.strip().rstrip("%")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _canon_type(v) -> str:
    t = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _CRITERION_TYPE_ALIASES.get(t, t)


def repair_plan(lenient: LenientSearchPlan) -> tuple[dict, list[str]]:
    """Return ``(plain_dict_for_strict_validation, repair_notes)``.

    ``notes`` are for the query-plan-repairs log line only — never user-facing.
    """
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
            notes.append(
                f"criteria[{i}] value=null — filled from {'values' if values else 'concept'}"
            )
        if "operator" in c and c.get("operator") is None:
            notes.append(f"criteria[{i}] operator=null — defaulted to ANY_OF")

        if not value and values:
            value = values[0]
        if not values and value:
            values = [value]
        if not value and not values and concept:
            value, values = concept, [concept]
        # a semantic-type criterion with only a literal value gets a concept so
        # the semantic scorer/judge has text to reason over (structural, not a guess)
        if concept is None and value and (ctype in SEMANTIC_CRITERION_TYPES or not ctype):
            concept = value

        if not value and not values and not concept:
            notes.append(f"criteria[{i}] had no value / values / concept — dropped")
            continue

        cid = c.get("id") or f"c{i}-{_slug(ctype or value or concept or 'x')}"
        w = _num(c.get("weight"), default=float("nan"))
        crits_out.append({
            "id": str(cid),
            "type": ctype or ("professional_concept" if concept else "keyword"),
            "weight": w,
            "required": bool(c["required"]) if c.get("required") is not None else False,
            "value": value,
            "values": values,
            "operator": _as_str(c.get("operator")) or "ANY_OF",
            "scope": c.get("scope") if isinstance(c.get("scope"), str) else None,
            "concept": concept,
            "modality": _as_str(c.get("modality")) or "certain",
        })

    # a missing / unparseable weight becomes the average of the explicit ones
    # (or an equal share) so it isn't crushed to 0 by weight normalization —
    # structural, never a semantic-importance guess.
    explicit = [c["weight"] for c in crits_out if c["weight"] == c["weight"]]  # not NaN
    fill = (sum(explicit) / len(explicit)) if explicit else (100.0 / max(1, len(crits_out)))
    for c in crits_out:
        if c["weight"] != c["weight"]:
            c["weight"] = round(fill, 2)
            notes.append(f"criterion {c['id']} weight missing — set to {c['weight']}")

    repaired: dict = {"criteria": crits_out}
    for k in _PASS_THROUGH_KEYS:
        v = raw.get(k)
        if v is not None:
            repaired[k] = v
    # weight totals are re-normalized to 100 by ParsedSearchQuery._normalize_weights
    return repaired, notes
