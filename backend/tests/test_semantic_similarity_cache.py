"""TASK 5 — semantic similarity (cross-encoder) cache tests.

Uses REAL seeded ``Person`` rows (not lightweight stand-ins) — the cache
table's ``person_id`` is a real foreign key, matching how ``facts_list``
always comes from bulk-loaded real people in production.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.database import SessionLocal
from app.models import SemanticSimilarityCache
from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services import reranker, semantic_similarity
from app.services.scoring import ScoringContext, load_facts
from tests.test_full_verification import _enriched, _seed
from tests.test_search_perf_regressions import _SpyModel

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


@pytest.fixture()
def spy_reranker(monkeypatch):
    spy = _SpyModel()
    monkeypatch.setattr(reranker, "_model", spy)
    monkeypatch.setattr(reranker.settings, "reranker_enabled", True)
    return spy


def _parsed():
    return ParsedSearchQuery(criteria=[
        SearchCriterion(id="a", type="semantic_concept", concept="research experience",
                        weight=50, required=True),
    ])


def _real_facts(db, ds, n: int, *, desc="Runs research projects and publishes papers.") -> list:
    from app import repositories as repo

    pids = [_seed(db, ds, name=f"Person {i}", desc=desc) for i in range(n)]
    db.commit()
    return [load_facts(db, repo.get_person(db, pid), {}) for pid in pids]


def test_second_search_reuses_cached_scores_zero_new_predicts(client, spy_reranker):
    ds = _enriched(client)
    db = SessionLocal()
    try:
        facts = _real_facts(db, ds, 10)
        parsed = _parsed()

        ctx1 = ScoringContext()
        semantic_similarity.compute_semantic_similarity(facts, parsed, ctx1, db=db)
        db.commit()
        assert len(spy_reranker.batch_sizes) == 1  # first call: everyone is a cache miss

        ctx2 = ScoringContext()
        semantic_similarity.compute_semantic_similarity(facts, parsed, ctx2, db=db)
        assert len(spy_reranker.batch_sizes) == 1  # UNCHANGED — no new predict() calls
        assert ctx1.semantic_similarity == ctx2.semantic_similarity  # identical scores, from cache
    finally:
        db.close()


def test_changed_evidence_only_recomputes_that_candidate(client, spy_reranker):
    ds = _enriched(client)
    db = SessionLocal()
    try:
        facts = _real_facts(db, ds, 10)
        parsed = _parsed()

        ctx1 = ScoringContext()
        semantic_similarity.compute_semantic_similarity(facts, parsed, ctx1, db=db)
        db.commit()

        from app import repositories as repo

        repo.replace_experiences(db, facts[0].person.id, [{
            "position": "Totally Different Role", "company_name": "New Co",
            "is_current": True, "start_year": 2024, "description": "Brand new evidence.",
        }])
        db.commit()
        facts2 = [load_facts(db, repo.get_person(db, f.person.id), {}) for f in facts]

        ctx2 = ScoringContext()
        semantic_similarity.compute_semantic_similarity(facts2, parsed, ctx2, db=db)
        assert spy_reranker.batch_sizes[-1] == 1  # only the CHANGED candidate needed a fresh score
    finally:
        db.close()


def test_disabled_setting_skips_cache_entirely(client, spy_reranker, monkeypatch):
    monkeypatch.setattr(settings, "semantic_similarity_cache_enabled", False)
    ds = _enriched(client)
    db = SessionLocal()
    try:
        facts = _real_facts(db, ds, 10)
        parsed = _parsed()
        semantic_similarity.compute_semantic_similarity(facts, parsed, ScoringContext(), db=db)
        db.commit()
        semantic_similarity.compute_semantic_similarity(facts, parsed, ScoringContext(), db=db)
        assert len(spy_reranker.batch_sizes) == 2  # cache never consulted -> every search recomputes
        assert db.query(SemanticSimilarityCache).count() == 0
    finally:
        db.close()


def test_no_db_falls_back_to_pre_task5_behaviour(spy_reranker):
    """``db=None`` (unit tests / callers that don't pass one) must behave
    EXACTLY like before TASK 5 — always compute, never touch a cache."""
    from tests.test_search_perf_regressions import _facts

    facts = [_facts(f"p{i}") for i in range(400)]
    parsed = ParsedSearchQuery(criteria=[
        SearchCriterion(id="a", type="semantic_concept", concept="research experience",
                        weight=50, required=True),
        SearchCriterion(id="b", type="semantic_concept", concept="industry experience",
                        weight=50, required=True),
    ])
    ctx = ScoringContext()
    semantic_similarity.compute_semantic_similarity(facts, parsed, ctx)
    assert len(spy_reranker.batch_sizes) == 2
    assert all(n == 400 for n in spy_reranker.batch_sizes)
