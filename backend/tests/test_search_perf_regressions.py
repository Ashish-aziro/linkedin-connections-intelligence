"""Mission STEP 17 — search-performance regression tests.

Fully offline (cross-encoder + judge + audit mocked at the usual seams). These
lock in the fixes for the real 149-240s searches, whose root cause was
per-candidate ML inference inside ``score_candidate`` run over two full
987-candidate passes:

  1. ``score_candidate`` performs ZERO ML inference.
  2. concept-vs-career similarity is BATCH-called: one ``predict`` per concept,
     never one pair per candidate.
  3. ~1,000 candidates do not cause ~1,000 CrossEncoder calls.
  4. prescore + judge does not trigger a second full N-candidate rescore.
  5. the query embedding is computed exactly once per search.
  6. an expired deadline stops post-deadline ML work.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from app.schemas import ParsedSearchQuery, SearchCriterion
from app.services import embeddings, reranker, search_profile, semantic_similarity
from app.services.deadline import Deadline
from app.services.scoring import ProfileFacts, ScoringContext, score_candidate
from app.constants import Qualification
from tests.test_search import _Exp, _Person

_FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "connections_sample.csv"


def _enriched_dataset(client) -> str:
    r = client.post("/datasets", files={"file": ("c.csv", _FIXTURE_CSV.read_bytes(), "text/csv")})
    ds_id = r.json()["dataset"]["dataset_id"]
    client.post(f"/datasets/{ds_id}/enrich")
    return ds_id


def _facts(pid: str, *, title="Barista", company="Corner Cafe", sem=None) -> ProfileFacts:
    p = _Person(current_title=title, current_company=company)
    p.id = pid
    return ProfileFacts(
        person=p, experiences=[_Exp(title, company, 2019, None, True, id=f"e-{pid}")],
        education=[], skills=[], semantic=sem or {}, embedding=None,
    )


class _SpyModel:
    """Stands in for a loaded CrossEncoder — records batch sizes."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def predict(self, pairs, **_kw):
        self.batch_sizes.append(len(pairs))
        return [0.4 + 0.001 * i for i in range(len(pairs))]


@pytest.fixture()
def spy_reranker(monkeypatch):
    spy = _SpyModel()
    monkeypatch.setattr(reranker, "_model", spy)
    monkeypatch.setattr(reranker.settings, "reranker_enabled", True)
    return spy


# ─────────────────────── 1. score_candidate is pure ───────────────────────


def test_score_candidate_makes_zero_ml_calls(monkeypatch):
    loaded: list[str] = []
    monkeypatch.setattr(reranker, "_get_model", lambda: loaded.append("cross_encoder"))
    monkeypatch.setattr(embeddings, "_get_model", lambda: loaded.append("sentence_transformer"))
    monkeypatch.setattr(reranker, "cross_encode",
                        lambda *a, **k: loaded.append("cross_encode") or [])
    monkeypatch.setattr(embeddings, "embed_text",
                        lambda *a, **k: loaded.append("embed_text") or b"")

    # a required semantic concept with no local evidence — exactly the case that
    # used to fire a per-candidate cross-encoder call inside the scorer.
    crit = SearchCriterion(id="c1", type="semantic_concept",
                           concept="career-long quantum computing research",
                           weight=100, required=True)
    parsed = ParsedSearchQuery(criteria=[crit])
    ctx = ScoringContext()
    for i in range(25):
        score_candidate(_facts(f"p{i}"), parsed, ctx)

    assert loaded == [], f"score_candidate triggered ML work: {loaded}"


def test_score_candidate_only_reads_precomputed_similarity():
    crit = SearchCriterion(id="c1", type="semantic_concept",
                           concept="renewable energy policy", weight=100, required=True)
    parsed = ParsedSearchQuery(criteria=[crit])
    ctx = ScoringContext()
    # no precomputed similarity -> concept stays UNKNOWN, candidate is POSSIBLE
    r1 = score_candidate(_facts("p1"), parsed, ctx)
    assert r1.qualification == Qualification.POSSIBLE_MATCH

    # a strong precomputed similarity is a RANKING signal only — never promotes
    # a bare concept to EXACT (STEP 13: similarity is not truth).
    ctx.semantic_similarity[("p2", "renewable energy policy")] = 0.95
    r2 = score_candidate(_facts("p2"), parsed, ctx)
    assert r2.qualification == Qualification.POSSIBLE_MATCH
    assert r2.match_score > r1.match_score


# ─────────────────────── 2 + 3. similarity is batched ───────────────────────


def test_semantic_similarity_is_one_batch_per_concept(spy_reranker):
    facts = [_facts(f"p{i}") for i in range(400)]
    parsed = ParsedSearchQuery(criteria=[
        SearchCriterion(id="a", type="semantic_concept", concept="research experience",
                        weight=50, required=True),
        SearchCriterion(id="b", type="semantic_concept", concept="industry experience",
                        weight=50, required=True),
    ])
    ctx = ScoringContext()
    semantic_similarity.compute_semantic_similarity(facts, parsed, ctx)

    # exactly one predict() per distinct concept — NOT one per candidate
    assert len(spy_reranker.batch_sizes) == 2
    assert all(n == 400 for n in spy_reranker.batch_sizes)
    assert len(ctx.semantic_similarity) == 400 * 2


def test_thousand_candidates_do_not_cause_thousand_cross_encoder_calls(spy_reranker):
    prof = search_profile.start()
    try:
        facts = [_facts(f"p{i}") for i in range(987)]
        parsed = ParsedSearchQuery(criteria=[
            SearchCriterion(id="a", type="semantic_concept", concept="research experience",
                            weight=50, required=True),
            SearchCriterion(id="b", type="industry_experience", concept="industry experience",
                            weight=50, required=True),
        ])
        ctx = ScoringContext()
        semantic_similarity.compute_semantic_similarity(facts, parsed, ctx)
        for f in facts:
            score_candidate(f, parsed, ctx)
    finally:
        search_profile.clear()

    assert prof.counters["cross_encoder_calls"] == 2          # << 987
    assert prof.counters["cross_encoder_pairs"] == 987 * 2
    assert prof.counters["score_candidate_calls"] == 987


def test_similarity_pass_respects_deadline(spy_reranker):
    facts = [_facts(f"p{i}") for i in range(50)]
    parsed = ParsedSearchQuery(criteria=[
        SearchCriterion(id="a", type="semantic_concept", concept="c-one", weight=50, required=True),
        SearchCriterion(id="b", type="semantic_concept", concept="c-two", weight=50, required=True),
    ])
    ctx = ScoringContext()
    expired = Deadline(0.0001)
    import time as _t
    _t.sleep(0.01)
    semantic_similarity.compute_semantic_similarity(facts, parsed, ctx, deadline=expired)
    assert spy_reranker.batch_sizes == []  # not a single predict once the deadline is spent


# ─────────────────────── full-pipeline regressions ───────────────────────


@pytest.fixture()
def _no_llm(monkeypatch):
    from app.services import final_auditor, semantic_judge
    monkeypatch.setattr(semantic_judge.settings, "semantic_judge_mode", "off")
    monkeypatch.setattr(final_auditor.settings, "final_result_audit_enabled", False)


def test_query_embedding_computed_once_per_search(client, _no_llm, monkeypatch):
    from app.services import search_service

    calls: list[str] = []
    real = embeddings.embed_text
    monkeypatch.setattr(embeddings, "embed_text", lambda t: calls.append(t) or real(t))
    monkeypatch.setattr(reranker.settings, "reranker_enabled", False)

    ds_id = _enriched_dataset(client)
    calls.clear()  # drop the per-profile embeds done during enrichment
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        search_service.run_connection_search(db, dataset_id=ds_id, query="senior engineers who mentor")
    finally:
        db.close()

    assert len(calls) == 1, f"query embedded {len(calls)}x (expected exactly 1)"


def test_no_second_full_rescore_after_judge(client, _no_llm, monkeypatch):
    """prescore covers every viable candidate ONCE; with the judge changing no
    verdict, the post-judge pass reuses every pre-score — never a second full
    N-candidate rescore."""
    from app.database import SessionLocal
    from app.services import search_service

    prof_holder: dict = {}
    real_start = search_profile.start
    monkeypatch.setattr(search_profile, "start",
                        lambda: prof_holder.setdefault("p", real_start()))
    monkeypatch.setattr(reranker.settings, "reranker_enabled", False)

    ds_id = _enriched_dataset(client)
    db = SessionLocal()
    try:
        search_service.run_connection_search(db, dataset_id=ds_id, query="engineers who know AWS")
    finally:
        db.close()

    c = prof_holder["p"].counters
    # one prescore per viable candidate; 0 judge-driven rescores; a small
    # near-match tail — never ~2x a full pass.
    assert c["score_candidate_calls"] <= c["prescore_candidates"] + 10
    assert c["rescore_candidates"] == 0


def test_expired_deadline_blocks_post_deadline_ml_work(client, _no_llm, monkeypatch):
    from app.database import SessionLocal
    from app.services import search_service

    monkeypatch.setattr(search_service.settings, "search_max_seconds", 0.0001)
    spy = _SpyModel()
    monkeypatch.setattr(reranker, "_model", spy)
    monkeypatch.setattr(reranker.settings, "reranker_enabled", True)

    ds_id = _enriched_dataset(client)
    db = SessionLocal()
    try:
        resp = search_service.run_connection_search(
            db, dataset_id=ds_id, query="research plus industry experience"
        )
    finally:
        db.close()

    assert spy.batch_sizes == []  # no predict() after the deadline is spent
    assert resp.llm_calls["profile"]["counters"].get("cross_encoder_calls", 0) == 0
