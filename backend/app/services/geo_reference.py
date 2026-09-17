"""A REAL geographic distance engine, built from a local reference dataset —
no paid geocoding API, no external call during a search (redesign PART 4).

``_CITY_COORDS`` holds real, publicly-known latitude/longitude for major US
cities (the kind of fact printed on any atlas or Wikipedia infobox — this is
reference data, not a per-query or per-person invention; a city absent from
this table is simply UNKNOWN, never guessed). ``haversine_miles`` computes
actual great-circle distance between two such points. This is what makes
"is Austin near Louisiana" answerable by real distance instead of "Texas
touches Louisiana on a map" — the failure mode PART 4/8 asks to fix.

This module has ONE job: resolve a place name to coordinates, and compute
real distance between two resolved points. It does not decide what counts as
"nearby" for a search — that policy (thresholds, metro grouping) lives in
``geo.py``, which is free to change without touching this data.
"""
from __future__ import annotations

import math

from app.services.matching import norm

#: real, public lat/lon (decimal degrees) for major US cities. Deliberately
#: not exhaustive — a city missing here resolves to UNKNOWN (never guessed),
#: which is the correct, honest outcome for a local-reference-only engine.
#: Keyed by normalized "city, state" AND bare "city" (only when the city name
#: is not ambiguous across states in this table) so lookups tolerate either.
_CITY_COORDS: dict[str, tuple[float, float]] = {
    # Georgia / Atlanta metro (incl. commonly-searched suburbs)
    "atlanta, georgia": (33.7490, -84.3880),
    "sandy springs, georgia": (33.9304, -84.3733),
    "marietta, georgia": (33.9526, -84.5499),
    "alpharetta, georgia": (34.0754, -84.2941),
    "decatur, georgia": (33.7748, -84.2963),
    "lawrenceville, georgia": (33.9562, -83.9880),
    "roswell, georgia": (34.0232, -84.3616),
    "duluth, georgia": (34.0029, -84.1446),
    "savannah, georgia": (32.0809, -81.0912),
    "augusta, georgia": (33.4735, -81.9748),
    "athens, georgia": (33.9519, -83.3576),
    "columbus, georgia": (32.4610, -84.9877),
    "macon, georgia": (32.8407, -83.6324),
    "chattanooga, tennessee": (35.0456, -85.3097),
    "birmingham, alabama": (33.5186, -86.8104),
    "greenville, south carolina": (34.8526, -82.3940),
    "columbia, south carolina": (34.0007, -81.0348),
    "charlotte, north carolina": (35.2271, -80.8431),
    # Texas
    "austin, texas": (30.2672, -97.7431),
    "dallas, texas": (32.7767, -96.7970),
    "houston, texas": (29.7604, -95.3698),
    "san antonio, texas": (29.4241, -98.4936),
    "fort worth, texas": (32.7555, -97.3308),
    "el paso, texas": (31.7619, -106.4850),
    # Illinois / Indiana (Chicago metro straddles the state line)
    "chicago, illinois": (41.8781, -87.6298),
    "evanston, illinois": (42.0451, -87.6877),
    "naperville, illinois": (41.7508, -88.1535),
    "hammond, indiana": (41.5834, -87.5000),
    "gary, indiana": (41.5934, -87.3464),
    "munster, indiana": (41.5484, -87.5077),
    "indianapolis, indiana": (39.7684, -86.1581),
    # Northeast
    "new york, new york": (40.7128, -74.0060),
    "jersey city, new jersey": (40.7178, -74.0431),
    "newark, new jersey": (40.7357, -74.1724),
    "brooklyn, new york": (40.6782, -73.9442),
    "boston, massachusetts": (42.3601, -71.0589),
    "cambridge, massachusetts": (42.3736, -71.1097),
    "philadelphia, pennsylvania": (39.9526, -75.1652),
    "pittsburgh, pennsylvania": (40.4406, -79.9959),
    "washington, district of columbia": (38.9072, -77.0369),
    "arlington, virginia": (38.8799, -77.1068),
    "alexandria, virginia": (38.8048, -77.0469),
    "bethesda, maryland": (38.9847, -77.0947),
    "baltimore, maryland": (39.2904, -76.6122),
    # Midwest
    "detroit, michigan": (42.3314, -83.0458),
    "minneapolis, minnesota": (44.9778, -93.2650),
    "st. louis, missouri": (38.6270, -90.1994),
    "kansas city, missouri": (39.0997, -94.5786),
    "kansas city, kansas": (39.1147, -94.6275),
    "columbus, ohio": (39.9612, -82.9988),
    "cincinnati, ohio": (39.1031, -84.5120),
    "cleveland, ohio": (41.4993, -81.6944),
    "milwaukee, wisconsin": (43.0389, -87.9065),
    # South / Southeast
    "miami, florida": (25.7617, -80.1918),
    "orlando, florida": (28.5383, -81.3792),
    "tampa, florida": (27.9506, -82.4572),
    "jacksonville, florida": (30.3322, -81.6557),
    "nashville, tennessee": (36.1627, -86.7816),
    "memphis, tennessee": (35.1495, -90.0490),
    "raleigh, north carolina": (35.7796, -78.6382),
    "durham, north carolina": (35.9940, -78.8986),
    "chapel hill, north carolina": (35.9132, -79.0558),
    "cary, north carolina": (35.7915, -78.7811),
    "richmond, virginia": (37.5407, -77.4360),
    "new orleans, louisiana": (29.9511, -90.0715),
    "baton rouge, louisiana": (30.4515, -91.1871),
    "little rock, arkansas": (34.7465, -92.2896),
    # West
    "los angeles, california": (34.0522, -118.2437),
    "san francisco, california": (37.7749, -122.4194),
    "san jose, california": (37.3382, -121.8863),
    "oakland, california": (37.8044, -122.2712),
    "palo alto, california": (37.4419, -122.1430),
    "san diego, california": (32.7157, -117.1611),
    "sacramento, california": (38.5816, -121.4944),
    "seattle, washington": (47.6062, -122.3321),
    "bellevue, washington": (47.6101, -122.2015),
    "tacoma, washington": (47.2529, -122.4443),
    "portland, oregon": (45.5152, -122.6784),
    "denver, colorado": (39.7392, -104.9903),
    "boulder, colorado": (40.0150, -105.2705),
    "phoenix, arizona": (33.4484, -112.0740),
    "las vegas, nevada": (36.1699, -115.1398),
    "salt lake city, utah": (40.7608, -111.8910),
}


def resolve_coords(city: str | None, state: str | None) -> tuple[float, float] | None:
    """Resolve a (city, state) pair to real coordinates via the LOCAL
    reference table only. Returns ``None`` (never a guess) when the city is
    not in the table — this is the honest UNKNOWN outcome, not a bug."""
    city_n = norm(city or "")
    state_n = norm(state or "")
    if not city_n:
        return None
    key = f"{city_n}, {state_n}" if state_n else city_n
    if key in _CITY_COORDS:
        return _CITY_COORDS[key]
    if state_n:
        for k, v in _CITY_COORDS.items():
            if k.startswith(f"{city_n},") and state_n in k:
                return v
        return None
    # a BARE city name with no state (common for a search's wanted value,
    # e.g. "Atlanta") — match against the table's city portion. Only
    # ambiguous when two DIFFERENT cities share the same bare name; this
    # table has no such collisions today, but guard for it anyway (an
    # ambiguous bare name resolves to UNKNOWN, never an arbitrary pick).
    matches = {v for k, v in _CITY_COORDS.items() if k.split(",", 1)[0] == city_n}
    if len(matches) == 1:
        return next(iter(matches))
    return None


def resolve_coords_from_text(location_text: str | None) -> tuple[float, float] | None:
    """Best-effort resolution from a free-text location string (e.g.
    "Atlanta, Georgia, United States") by matching the table's "city, state"
    keys as a substring — used when structured city/state fields are absent."""
    text = norm(location_text or "")
    if not text:
        return None
    for key, coords in _CITY_COORDS.items():
        if key in text:
            return coords
    return None


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Real great-circle distance in miles between two (lat, lon) points.
    Note (PART 4): this is straight-line distance, not driving distance —
    callers must not present it as a road-trip estimate."""
    lat1, lon1 = a
    lat2, lon2 = b
    r_miles = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    hav = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return round(2 * r_miles * math.asin(math.sqrt(hav)), 1)
