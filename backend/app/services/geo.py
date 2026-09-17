"""Geographic interpretation — a small normalization/ontology layer (spec §12).

Not a paid API, not a giant per-query dictionary. The query planner is asked to
expand a metro/region into its cities in the criterion's ``values``; this module
provides (a) a compact static fallback for the most common US metros so the
deterministic path and an under-expanding LLM still work, and (b) a
token-subset location matcher so "San Jose" matches
"San Jose, California, United States" (not the strict 75%-phrase rule that
broke "Bay Area" vs "Palo Alto").
"""
from __future__ import annotations

from app.constants import GeoRelation
from app.services.matching import norm

#: region alias -> canonical member cities/areas (lower-cased). Kept deliberately
#: small — the LLM planner does the general case; this covers the metros that
#: show up constantly and must never silently fail.
_REGIONS: dict[str, list[str]] = {
    "bay area": [
        "san francisco", "san jose", "oakland", "palo alto", "mountain view", "sunnyvale",
        "santa clara", "menlo park", "redwood city", "cupertino", "fremont", "berkeley",
        "san mateo", "foster city", "south san francisco", "emeryville", "burlingame",
        "silicon valley",
    ],
    "silicon valley": [
        "san jose", "palo alto", "mountain view", "sunnyvale", "santa clara", "menlo park",
        "cupertino", "redwood city", "san francisco",
    ],
    "sf bay area": ["san francisco", "san jose", "oakland", "palo alto", "mountain view", "sunnyvale"],
    "greater new york": ["new york", "brooklyn", "jersey city", "newark", "manhattan", "queens"],
    "nyc": ["new york", "brooklyn", "manhattan", "jersey city"],
    "new york city": ["new york", "brooklyn", "manhattan"],
    "greater seattle": ["seattle", "bellevue", "redmond", "kirkland", "tacoma"],
    "greater boston": ["boston", "cambridge", "somerville", "waltham"],
    "greater los angeles": ["los angeles", "santa monica", "pasadena", "burbank", "el segundo"],
    "greater chicago": ["chicago", "evanston", "naperville"],
    "dmv": ["washington", "arlington", "alexandria", "bethesda", "reston", "mclean"],
    "washington dc area": ["washington", "arlington", "alexandria", "bethesda"],
    "research triangle": ["raleigh", "durham", "chapel hill", "cary"],
    "dfw": ["dallas", "fort worth", "plano", "irving"],
}


def expand_region(value: str) -> list[str]:
    """Return member cities for a region name, or ``[value]`` if it isn't a
    known region. Always includes the original value too."""
    v = norm(value)
    members = _REGIONS.get(v)
    if members:
        return [value, *members]
    return [value]


def expand_values(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values or []:
        for x in expand_region(v):
            if x not in out:
                out.append(x)
    return out


def location_matches(location_fields: list[str | None], value: str) -> bool:
    """True if ``value``'s significant tokens are all present in any of the
    person's location strings (token-subset, order-free). 'San Jose' matches
    'San Jose, California, United States'; 'Bay Area' matches 'San Francisco
    Bay Area'."""
    val_tokens = {t for t in norm(value).split() if len(t) > 1}
    if not val_tokens:
        return False
    for field in location_fields:
        hay = norm(field)
        if hay and val_tokens <= set(hay.split()):
            return True
    return False


#: value -> the OTHER members of every region it belongs to (built once from
#: ``_REGIONS`` — never a query-specific list). "san jose" -> {"san francisco",
#: "oakland", ...} because both are Bay Area members; a region's own name maps
#: to its members too so "atlanta" style single-city queries with no matching
#: alias key simply get no metro siblings (deterministically UNKNOWN, not FAR).
def _metro_siblings_index() -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for region, members in _REGIONS.items():
        group = {region, *members}
        for m in group:
            idx.setdefault(m, set()).update(group - {m})
    return idx


_METRO_SIBLINGS = _metro_siblings_index()


def _metro_siblings(value: str) -> set[str]:
    v = norm(value)
    out: set[str] = set()
    for tok, siblings in _METRO_SIBLINGS.items():
        if tok == v or tok in v.split() or v in tok.split():
            out |= siblings
    return out


#: the fixed, universal set of US state names — structural reference data
#: (like the metro alias map above), never a per-query or per-city list. Used
#: ONLY to detect a CONFIDENT geographic mismatch: the candidate's own state
#: is known, and the query's location value(s) explicitly name a different
#: state (e.g. an expanded state-level location criterion) — never to guess a
#: bare city's state (that stays UNKNOWN, left to the LLM).
_US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}

#: TASK 5 (near-match hardening) — fixed, structural US state-border data,
#: same class of reference fact as ``_US_STATES`` above (never a per-query or
#: per-city list). Many real metro areas straddle a state line (Chicago/NW
#: Indiana, NYC/NJ/CT, DC/VA/MD, Kansas City MO/KS, ...) that the static
#: ``_REGIONS`` alias map above cannot exhaustively list. "Different state"
#: alone is therefore NOT sufficient evidence of real distance — it only
#: becomes a confident ``FAR`` call when the two states do not even share a
#: border; an adjacent-state pairing stays ``UNKNOWN`` (deferred to the
#: near-match LLM judge, whose accepted claim is then correctly labeled
#: "llm_inference", never "deterministic" — see near_match_validator).
_ADJACENT_STATES: dict[str, set[str]] = {
    "alabama": {"florida", "georgia", "mississippi", "tennessee"},
    "arizona": {"california", "colorado", "nevada", "new mexico", "utah"},
    "arkansas": {"louisiana", "mississippi", "missouri", "oklahoma", "tennessee", "texas"},
    "california": {"arizona", "nevada", "oregon"},
    "colorado": {"arizona", "kansas", "nebraska", "new mexico", "oklahoma", "utah", "wyoming"},
    "connecticut": {"massachusetts", "new york", "rhode island"},
    "delaware": {"maryland", "new jersey", "pennsylvania"},
    "florida": {"alabama", "georgia"},
    "georgia": {"alabama", "florida", "north carolina", "south carolina", "tennessee"},
    "idaho": {"montana", "nevada", "oregon", "utah", "washington", "wyoming"},
    "illinois": {"indiana", "iowa", "kentucky", "missouri", "wisconsin"},
    "indiana": {"illinois", "kentucky", "michigan", "ohio"},
    "iowa": {"illinois", "minnesota", "missouri", "nebraska", "south dakota", "wisconsin"},
    "kansas": {"colorado", "missouri", "nebraska", "oklahoma"},
    "kentucky": {"illinois", "indiana", "missouri", "ohio", "tennessee", "virginia", "west virginia"},
    "louisiana": {"arkansas", "mississippi", "texas"},
    "maine": {"new hampshire"},
    "maryland": {"delaware", "pennsylvania", "virginia", "west virginia", "district of columbia"},
    "massachusetts": {"connecticut", "new hampshire", "new york", "rhode island", "vermont"},
    "michigan": {"indiana", "ohio", "wisconsin"},
    "minnesota": {"iowa", "north dakota", "south dakota", "wisconsin"},
    "mississippi": {"alabama", "arkansas", "louisiana", "tennessee"},
    "missouri": {"arkansas", "illinois", "iowa", "kansas", "kentucky", "nebraska", "oklahoma", "tennessee"},
    "montana": {"idaho", "north dakota", "south dakota", "wyoming"},
    "nebraska": {"colorado", "iowa", "kansas", "missouri", "south dakota", "wyoming"},
    "nevada": {"arizona", "california", "idaho", "oregon", "utah"},
    "new hampshire": {"maine", "massachusetts", "vermont"},
    "new jersey": {"delaware", "new york", "pennsylvania"},
    "new mexico": {"arizona", "colorado", "oklahoma", "texas", "utah"},
    "new york": {"connecticut", "massachusetts", "new jersey", "pennsylvania", "vermont"},
    "north carolina": {"georgia", "south carolina", "tennessee", "virginia"},
    "north dakota": {"minnesota", "montana", "south dakota"},
    "ohio": {"indiana", "kentucky", "michigan", "pennsylvania", "west virginia"},
    "oklahoma": {"arkansas", "colorado", "kansas", "missouri", "new mexico", "texas"},
    "oregon": {"california", "idaho", "nevada", "washington"},
    "pennsylvania": {"delaware", "maryland", "new jersey", "new york", "ohio", "west virginia"},
    "rhode island": {"connecticut", "massachusetts"},
    "south carolina": {"georgia", "north carolina"},
    "south dakota": {"iowa", "minnesota", "montana", "nebraska", "north dakota", "wyoming"},
    "tennessee": {"alabama", "arkansas", "georgia", "kentucky", "mississippi", "missouri",
                  "north carolina", "virginia"},
    "texas": {"arkansas", "louisiana", "new mexico", "oklahoma"},
    "utah": {"arizona", "colorado", "idaho", "nevada", "new mexico", "wyoming"},
    "vermont": {"massachusetts", "new hampshire", "new york"},
    "virginia": {"kentucky", "maryland", "north carolina", "tennessee", "west virginia",
                 "district of columbia"},
    "washington": {"idaho", "oregon"},
    "west virginia": {"kentucky", "maryland", "ohio", "pennsylvania", "virginia"},
    "wisconsin": {"illinois", "iowa", "michigan", "minnesota"},
    "wyoming": {"colorado", "idaho", "montana", "nebraska", "south dakota", "utah"},
}


#: redesign PART 4 — configurable proximity policy, real straight-line miles
#: (haversine), never a driving-distance claim. Kept as small module-level
#: constants (not settings) since they express a geographic FACT threshold,
#: not a search-tuning knob — override via monkeypatch in tests if needed.
SAME_METRO_MAX_MILES = 30.0
NEARBY_CITY_MAX_MILES = 75.0
SAME_REGION_MAX_MILES = 250.0


def classify_relation_with_evidence(
    candidate_fields: dict[str, str | None], wanted_values: list[str],
) -> dict:
    """Redesign PART 4/8/10 — the SAME decision as
    ``classify_relation_deterministic`` (identical, called from there), but
    returns the actual evidence behind it: ``{"relation", "distance_miles",
    "basis"}``. ``basis`` is one of "distance" (real haversine miles — the
    ONLY basis allowed to justify NEARBY_CITY/SAME_REGION), "metro_list"
    (the curated ``_REGIONS`` alias map), "state_adjacency", "state_match",
    or "none" (UNKNOWN — nothing resolved). Never invents a distance for a
    city this process's local reference table doesn't know."""
    cand_city = norm(candidate_fields.get("city") or "")
    cand_state = norm(candidate_fields.get("state") or "")
    cand_loc = norm(candidate_fields.get("location_text") or "")
    cand_tokens = set((cand_loc or f"{cand_city} {cand_state}").split())

    # ── REAL DISTANCE FIRST — the actual PART 4 fix. If both the candidate's
    #    city and (any) wanted value resolve to real coordinates, distance
    #    decides the relation outright; no state name, adjacency, or metro
    #    alias list is even consulted. This is what correctly tells a
    #    same-metro cross-state pair (Chicago/Hammond IN) from a same-state
    #    pair that's actually hundreds of miles apart. ──────────────────────
    from app.services.geo_reference import haversine_miles, resolve_coords, resolve_coords_from_text

    cand_coords = resolve_coords(candidate_fields.get("city"), candidate_fields.get("state")) \
        or resolve_coords_from_text(candidate_fields.get("location_text"))
    if cand_coords is not None:
        best_relation = None
        best_miles = None
        for value in wanted_values:
            wanted_coords = resolve_coords_from_text(value) or resolve_coords(value, None)
            if wanted_coords is None:
                continue
            miles = haversine_miles(cand_coords, wanted_coords)
            if best_miles is None or miles < best_miles:
                best_miles = miles
                if miles <= SAME_METRO_MAX_MILES:
                    best_relation = GeoRelation.SAME_METRO
                elif miles <= NEARBY_CITY_MAX_MILES:
                    best_relation = GeoRelation.NEARBY_CITY
                elif miles <= SAME_REGION_MAX_MILES:
                    best_relation = GeoRelation.SAME_REGION
                else:
                    best_relation = GeoRelation.FAR
        if best_relation is not None:
            return {"relation": best_relation, "distance_miles": best_miles, "basis": "distance"}

    # ── fall back to the curated metro-alias list / state logic below when
    #    this process's local coordinate table doesn't know the city. ──────
    for value in wanted_values:
        val_norm = norm(value)
        siblings = _metro_siblings(value)
        if siblings and (cand_city in siblings or siblings & cand_tokens):
            return {"relation": GeoRelation.SAME_METRO, "distance_miles": None, "basis": "metro_list"}
        # substring containment, NOT a token-set intersection — a multi-word
        # state name ("new york", "north carolina") is one phrase and would
        # never appear as a single element of a space-split token set.
        if cand_state and cand_state in val_norm:
            return {"relation": GeoRelation.SAME_STATE_NOT_NEAR, "distance_miles": None, "basis": "state_match"}

    if cand_state in _US_STATES:
        wanted_text = " | ".join(norm(v) for v in wanted_values)
        wanted_states = {s for s in _US_STATES if s in wanted_text}
        if wanted_states and cand_state not in wanted_states and (not cand_city or cand_city not in wanted_text):
            # Adjacent-state relaxation (near-match hardening) — a state
            # mismatch is confident FAR only when the states don't even share
            # a border; an ADJACENT state may still be the same metro area
            # (Chicago/NW Indiana, DC/VA, NYC/NJ, ...) the static ``_REGIONS``
            # list doesn't happen to enumerate. Only reached when the LOCAL
            # coordinate table above couldn't resolve real distance for
            # either city — real distance always wins when it's available.
            #
            # This relaxation is only meaningful when ``wanted_values`` names
            # ONE specific target state: "Illinois borders Indiana" is real
            # evidence a candidate near that border could be in the same
            # metro. It stops being meaningful once ``wanted_values`` is a
            # BROAD multi-state region expansion (e.g. "Southeast US" -> 12
            # states) — a large state like Texas borders SOME state in almost
            # any big region list, but that says nothing about whether the
            # candidate's specific city is actually near it. So adjacency is
            # only applied against a SINGLE named target state; a multi-state
            # wanted list falls back to the state-difference default (FAR).
            if len(wanted_states) == 1 and any(
                cand_state in _ADJACENT_STATES.get(ws, ()) for ws in wanted_states
            ):
                return {"relation": GeoRelation.UNKNOWN, "distance_miles": None, "basis": "state_adjacency"}
            return {"relation": GeoRelation.FAR, "distance_miles": None, "basis": "state_adjacency"}
    return {"relation": GeoRelation.UNKNOWN, "distance_miles": None, "basis": "none"}


def classify_relation_deterministic(
    candidate_fields: dict[str, str | None], wanted_values: list[str],
) -> str:
    """Near-match design PART 6 — the STRUCTURED-DATA-FIRST geo relation check.
    Thin wrapper around ``classify_relation_with_evidence`` for callers that
    only need the ``GeoRelation`` constant, not the supporting evidence.
    Returns a ``GeoRelation`` constant, never a claim it cannot actually
    support — see ``classify_relation_with_evidence`` for exactly what each
    value requires. Never overrides / duplicates the exact matcher in
    ``scoring._score_location`` — called ONLY for a candidate whose strict
    location match already failed."""
    return classify_relation_with_evidence(candidate_fields, wanted_values)["relation"]
