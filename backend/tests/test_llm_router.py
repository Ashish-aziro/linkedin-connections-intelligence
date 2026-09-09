"""Provider routing / fallback / circuit-breaker tests.

The router's generic machinery must:
  * try the first available provider, stop the moment one returns validated output
  * fall through on ANY failure, fast for unretryable ones (401 / bad config)
  * cool a provider down after an unretryable or repeated-transient failure
  * return an oversized-output truncation / 413 to the caller immediately
    (never re-send the same too-large payload to the next provider)
  * return None (→ deterministic path) only when every provider is exhausted

The app's real chain is Anthropic-only (``default_chain`` returns
``[AnthropicProvider]`` when a key is set, else ``[]``). These tests drive the
generic fallback loop with FAKE providers passed via ``chain=`` — the extra
names are just opaque labels for a multi-hop chain.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.constants import LLMProviderName
from app.services.llm import circuit
from app.services.llm.base import (
    LLMAuthError,
    LLMBadOutput,
    LLMConfigError,
    LLMOutputTruncated,
    LLMProvider,
    LLMRateLimited,
    LLMRequestTooLarge,
    LLMTransport,
    LLMUnavailable,
)
from app.services.llm.providers import default_chain
from app.services.llm.router import generate_structured

ANTH = LLMProviderName.ANTHROPIC
FB1 = "fallback-a"   # opaque labels for extra hops in a fake multi-provider chain
FB2 = "fallback-b"
FB3 = "fallback-c"


class Out(BaseModel):
    answer: str
    score: int


@pytest.fixture(autouse=True)
def _clean_breakers():
    circuit.reset_all()
    yield
    circuit.reset_all()


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 1)


class FakeProvider(LLMProvider):
    def __init__(self, name, behavior):
        self.name = name
        self.model = f"model-{name}"
        self.behavior = behavior
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate_json(self, system_prompt, user_prompt, *, max_tokens=1500, timeout=None):  # noqa: ARG002
        self.calls += 1
        b = self.behavior
        if b == "ok":
            return {"answer": "hi", "score": 5}
        if b == "429":
            raise LLMRateLimited("429", retry_after=0.01)
        if b == "unavailable":
            raise LLMUnavailable("503")
        if b == "transport":
            raise LLMTransport("connection reset")
        if b == "auth":
            raise LLMAuthError("401")
        if b == "config":
            raise LLMConfigError("anthropic workspace configuration error")
        if b == "badjson":
            raise LLMBadOutput("not json")
        if b == "badschema":
            return {"answer": "hi"}  # missing score
        if b == "truncated":
            raise LLMOutputTruncated("hit max_tokens")
        if b == "413":
            raise LLMRequestTooLarge("payload too large")
        raise AssertionError(b)


def _run(chain):
    return generate_structured("s", "u", Out, chain=chain, operation="test")


# ─────────────────────────── first success stops the chain ───────────────────────────
def test_first_success_stops_chain():
    a = FakeProvider(ANTH, "ok")
    b = FakeProvider(FB1, "ok")
    c = FakeProvider(FB2, "ok")
    res = _run([a, b, c])
    assert res and res[1] == ANTH
    assert a.calls == 1 and b.calls == 0 and c.calls == 0


# ─────────────────────────── fall-through on failure ───────────────────────────
def test_rate_limited_retries_then_falls_through():
    a = FakeProvider(ANTH, "429")
    b = FakeProvider(FB1, "ok")
    res = _run([a, b])
    assert res and res[1] == FB1
    assert a.calls == 2  # initial + 1 retry, then fall through


def test_401_is_not_retried_and_cools_provider():
    a = FakeProvider(ANTH, "auth")
    b = FakeProvider(FB1, "ok")
    res = _run([a, b])
    assert res and res[1] == FB1
    assert a.calls == 1
    assert circuit.is_open(ANTH)


def test_config_error_falls_through_without_retry():
    a = FakeProvider(ANTH, "config")
    b = FakeProvider(FB1, "ok")
    res = _run([a, b])
    assert res and res[1] == FB1
    assert a.calls == 1
    assert circuit.is_open(ANTH)


def test_multi_hop_fallthrough():
    a = FakeProvider(ANTH, "unavailable")
    b = FakeProvider(FB1, "unavailable")
    c = FakeProvider(FB2, "ok")
    res = _run([a, b, c])
    assert res and res[1] == FB2


def test_last_provider_succeeds():
    chain = [
        FakeProvider(ANTH, "transport"),
        FakeProvider(FB1, "unavailable"),
        FakeProvider(FB2, "429"),
        FakeProvider(FB3, "ok"),
    ]
    res = _run(chain)
    assert res and res[1] == FB3


def test_no_provider_available_returns_none():
    class Unconfigured(FakeProvider):
        def available(self) -> bool:
            return False

    assert _run([Unconfigured("x", "ok")]) is None


# ─────────────────────────── default_chain composition ───────────────────────────
def test_default_chain_is_anthropic_only_when_key_set(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-ant-test")
    assert [p.name for p in default_chain()] == [ANTH]


def test_default_chain_empty_without_key(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "")
    assert default_chain() == []


def test_configured_key_used_even_when_enable_paid_llm_false(monkeypatch):
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr("app.config.settings.enable_paid_llm", False)
    names = [p.name for p in default_chain()]
    assert names == [ANTH]


def test_returned_provider_is_a_real_name():
    for behavior_chain, expected in [
        ([(ANTH, "ok")], ANTH),
        ([(ANTH, "auth"), (FB1, "ok")], FB1),
    ]:
        circuit.reset_all()
        chain = [FakeProvider(n, b) for n, b in behavior_chain]
        res = _run(chain)
        assert res and res[1] == expected and res[1] != "deterministic"


# ─────────────────────── circuit breaker ───────────────────────
def test_circuit_skips_cooled_provider_then_success_resets():
    _run([FakeProvider(ANTH, "auth"), FakeProvider(FB1, "ok")])
    assert circuit.is_open(ANTH)

    a2 = FakeProvider(ANTH, "ok")
    res = _run([a2, FakeProvider(FB1, "ok")])
    assert res and res[1] == FB1
    assert a2.calls == 0  # never attempted while cooling down


def test_circuit_trips_after_repeated_transient_failures(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 0)
    for _ in range(3):
        _run([FakeProvider(FB1, "unavailable"), FakeProvider(FB2, "ok")])
    assert circuit.is_open(FB1)


def test_bad_output_does_not_trip_circuit(monkeypatch):
    monkeypatch.setattr("app.services.llm.router.settings.llm_max_retries", 0)
    for _ in range(5):
        _run([FakeProvider(ANTH, "badjson"), FakeProvider(FB1, "ok")])
    assert not circuit.is_open(ANTH)  # prompt-local, not a provider fault


# ─────────────────────── meta / schema validation ───────────────────────
def test_return_meta_records_attempts():
    chain = [FakeProvider(ANTH, "429"), FakeProvider(FB1, "ok")]
    model, name, model_id, meta = generate_structured(
        "s", "u", Out, chain=chain, operation="query_interpretation", return_meta=True
    )
    assert name == FB1
    assert meta["operation"] == "query_interpretation"
    assert meta["selected_provider"] == FB1
    assert meta["attempts"][0] == {"provider": ANTH, "status": "rate_limited"}
    assert meta["attempts"][-1] == {"provider": FB1, "status": "success"}


def test_schema_validation_failure_moves_on():
    res = _run([FakeProvider("bad", "badschema"), FakeProvider("good", "ok")])
    assert res and res[1] == "good"


# ─────────────────────── truncation / 413 return to caller ───────────────────────
def test_truncation_returns_to_caller_immediately_not_the_next_provider():
    """An oversized-output truncation must NOT fall through to the next provider
    with the identical oversized request — the router returns None right away so
    the caller can split the request smaller."""
    a = FakeProvider(ANTH, "truncated")
    b = FakeProvider(FB1, "ok")
    res = _run([a, b])
    assert res is None
    assert a.calls == 1
    assert b.calls == 0


def test_truncation_does_not_trip_the_circuit():
    _run([FakeProvider(ANTH, "truncated")])
    assert not circuit.is_open(ANTH)
    res = _run([FakeProvider(ANTH, "ok")])
    assert res and res[1] == ANTH


def test_413_returns_to_caller_immediately_not_the_next_provider():
    a = FakeProvider(FB1, "413")
    b = FakeProvider(FB2, "ok")
    res = _run([a, b])
    assert res is None
    assert a.calls == 1
    assert b.calls == 0


def test_413_does_not_trip_the_circuit_and_a_smaller_request_succeeds_right_after():
    large = FakeProvider(FB1, "413")
    _run([large])
    assert large.calls == 1
    assert not circuit.is_open(FB1)

    small = FakeProvider(FB1, "ok")
    res = _run([small])
    assert res and res[1] == FB1
    assert small.calls == 1


def test_413_after_repeated_hits_still_does_not_trip_the_circuit():
    for _ in range(5):
        _run([FakeProvider(FB1, "413")])
    assert not circuit.is_open(FB1)
