"""V4 PART 6 B5 — local model warmup / cache.

The models must load once per process, outside the request path, and tests must
never download real weights.
"""
from __future__ import annotations

import pytest

from app.services import model_warmup


@pytest.fixture(autouse=True)
def _reset():
    model_warmup._reset_for_tests()
    yield
    model_warmup._reset_for_tests()


def test_disabled_by_default_in_tests(monkeypatch):
    calls = []
    monkeypatch.setattr("app.services.embeddings.warm", lambda: calls.append("e"))
    monkeypatch.setattr("app.services.reranker.warm", lambda: calls.append("r"))
    # conftest sets WARM_MODELS_ON_STARTUP=false
    model_warmup.start(block=True)
    assert calls == []


def test_warms_both_models_once_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(model_warmup.settings, "warm_models_on_startup", True)
    monkeypatch.setattr("app.services.embeddings.warm", lambda: calls.append("e"))
    monkeypatch.setattr("app.services.reranker.warm", lambda: calls.append("r"))

    model_warmup.start(block=True)
    model_warmup.start(block=True)  # idempotent — second call is a no-op

    assert calls == ["e", "r"]


def test_warmup_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(model_warmup.settings, "warm_models_on_startup", True)

    def boom():
        raise RuntimeError("no torch")

    monkeypatch.setattr("app.services.embeddings.warm", boom)
    monkeypatch.setattr("app.services.reranker.warm", lambda: None)

    model_warmup.start(block=True)  # must not raise


def test_warm_is_noop_when_embeddings_disabled(monkeypatch):
    from app.services import embeddings

    loaded = []
    monkeypatch.setattr(embeddings, "_get_model", lambda: loaded.append(1))
    monkeypatch.setattr(embeddings.settings, "embeddings_enabled", False)
    embeddings.warm()
    assert loaded == []


def test_ready_reports_state(monkeypatch):
    monkeypatch.setattr("app.services.embeddings._model", None, raising=False)
    r = model_warmup.ready()
    assert r["embedding_model_ready"] is False
    assert "reranker_model_ready" in r


def test_health_endpoint_exposes_model_readiness(client):
    body = client.get("/health").json()
    assert "models" in body
    assert body["models"]["warm_on_startup"] is False
    assert "embedding_model_ready" in body["models"]
