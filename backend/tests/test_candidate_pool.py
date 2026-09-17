"""Candidate retrieval pool — large-dataset union + truncation (review finding
E1/E2).

The retrieval union merges several channels (structured SQL matches, semantic
matches, geographic matches, embedding nearest-neighbours) and, above
``full_scan_max_connections``, truncates the result to ``candidate_pool_size``.
The union previously collapsed into a bare Python ``set`` before truncation,
which discards every channel's own relevance ordering — a high-relevance
candidate could be dropped purely by hash-bucket iteration order, and the
surviving set was not even reproducible across process restarts. These tests
lock in the fix: every channel now contributes a real relevance score, the
union is ranked before it is sliced, and residual ties break on a stable key.
"""
from __future__ import annotations

import uuid

from app import repositories as repo
from app.config import settings
from app.constants import CriterionType, EnrichmentState
from app.models import Person
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services.candidate_pool import get_candidates


def _make_person(db, dataset_id: str, *, title: str | None = None, location_text: str | None = None) -> str:
    p = Person(
        dataset_id=dataset_id,
        linkedin_url=f"https://linkedin.com/in/{uuid.uuid4().hex}",
        is_connection=True,
        enrichment_state=EnrichmentState.READY,
        current_title=title,
        location_text=location_text,
    )
    db.add(p)
    db.commit()
    return p.id


def _plan() -> ParsedSearchQuery:
    return ParsedSearchQuery(criteria=[
        SearchCriterion(id="title", type=CriterionType.TITLE, value="Backend Engineer",
                        weight=60, required=False),
        SearchCriterion(id="loc", type=CriterionType.LOCATION, value="Nashville",
                        weight=40, required=False),
    ])


def test_high_relevance_candidate_survives_large_dataset_truncation(db, monkeypatch):
    """A single structured SQL title match (strong signal) must survive a
    pool cap smaller than the total number of eligible candidates, even
    though every OTHER eligible candidate only matched on a weaker
    (geographic) channel — never dropped by Python `set` iteration order."""
    monkeypatch.setattr(settings, "full_scan_max_connections", 2)
    monkeypatch.setattr(settings, "candidate_pool_size", 3)

    ds = repo.create_dataset(db, "candidate-pool-test")
    needle_id = _make_person(db, ds.id, title="Distinguished Backend Engineer")
    noise_ids = [
        _make_person(db, ds.id, location_text="Nashville, Tennessee, United States")
        for _ in range(6)
    ]
    db.commit()

    kept, total = get_candidates(db, ds.id, _plan(), None)

    assert total == 7
    kept_ids = {p.id for p in kept}
    assert len(kept_ids) == 3
    assert needle_id in kept_ids, "the SQL-matched candidate must survive truncation regardless of hash order"
    # the pool must never contain more than the configured cap
    assert kept_ids <= {needle_id, *noise_ids}


def test_candidate_pool_retrieval_is_deterministic_across_repeated_calls(db, monkeypatch):
    """review finding E2 — identical query + identical DB state must return
    the identical candidate set on every call, not just "on average"."""
    monkeypatch.setattr(settings, "full_scan_max_connections", 2)
    monkeypatch.setattr(settings, "candidate_pool_size", 3)

    ds = repo.create_dataset(db, "candidate-pool-determinism-test")
    _make_person(db, ds.id, title="Distinguished Backend Engineer")
    for _ in range(6):
        _make_person(db, ds.id, location_text="Nashville, Tennessee, United States")
    db.commit()

    first, _ = get_candidates(db, ds.id, _plan(), None)
    second, _ = get_candidates(db, ds.id, _plan(), None)

    assert [p.id for p in first] == [p.id for p in second]


def test_small_dataset_returns_everyone_no_truncation(db, monkeypatch):
    """At or below ``full_scan_max_connections`` every connection is scored —
    no channel union, no truncation, no recall loss at all."""
    monkeypatch.setattr(settings, "full_scan_max_connections", 5000)

    ds = repo.create_dataset(db, "candidate-pool-small-test")
    ids = [_make_person(db, ds.id, title=f"Engineer {i}") for i in range(5)]
    db.commit()

    kept, total = get_candidates(db, ds.id, _plan(), None)

    assert total == 5
    assert {p.id for p in kept} == set(ids)
