"""V4 PART 6 B0 — Anthropic Messages client, Claude 5 compatibility.

The client must NOT send ``temperature`` or an assistant-turn prefill (Claude 5
rejects both with 400), must read JSON from ``text`` content blocks while
ignoring ``thinking`` blocks, and must keep the existing six-way error taxonomy.
"""
from __future__ import annotations

import httpx
import pytest

from app.services.llm import anthropic_client as ac
from app.services.llm.base import (
    LLMAuthError,
    LLMBadOutput,
    LLMConfigError,
    LLMOutputTruncated,
    LLMRateLimited,
    LLMRequestTooLarge,
    LLMTransport,
    LLMUnavailable,
)


class _Resp:
    def __init__(self, status_code=200, body=None, text="", headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text or ""
        self.headers = headers or {}

    def json(self):
        return self._body


def _ok_body(text, *, stop_reason="end_turn", extra_blocks=()):
    return {
        "stop_reason": stop_reason,
        "content": [*extra_blocks, {"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.fixture
def capture_post(monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["url"] = url
        sent["payload"] = json
        sent["headers"] = headers
        sent["timeout"] = timeout
        return sent.pop("_resp")

    monkeypatch.setattr(ac.httpx, "post", fake_post)

    def _run(resp, **kw):
        sent["_resp"] = resp
        return ac.messages_json(
            api_key="sk-ant-test", model=kw.get("model", "claude-sonnet-5"),
            system_prompt=kw.get("system", "S"), user_prompt=kw.get("user", "U"),
            max_tokens=kw.get("max_tokens", 500), timeout=kw.get("timeout", 30.0),
        )

    _run.sent = sent
    return _run


def test_claude5_text_block_parsed(capture_post):
    out = capture_post(_Resp(body=_ok_body('{"answer": "hi", "n": 3}')))
    assert out == {"answer": "hi", "n": 3}


def test_no_temperature_and_no_prefill_sent(capture_post):
    capture_post(_Resp(body=_ok_body('{"ok": true}')))
    payload = capture_post.sent["payload"]
    assert "temperature" not in payload
    assert payload["messages"] == [{"role": "user", "content": "U"}]
    assert payload["messages"][-1]["role"] == "user"  # no assistant prefill


def test_thinking_block_ignored_text_extracted(capture_post):
    body = _ok_body(
        '{"verdict": "true"}',
        extra_blocks=({"type": "thinking", "thinking": "let me reason... {\"fake\": 1}"},),
    )
    out = capture_post(_Resp(body=body))
    assert out == {"verdict": "true"}


def test_prose_prefixed_json_still_extracted(capture_post):
    out = capture_post(_Resp(body=_ok_body('Here is the JSON:\n{"x": 1}')))
    assert out == {"x": 1}


def test_fenced_json_extracted(capture_post):
    out = capture_post(_Resp(body=_ok_body('```json\n{"y": 2}\n```')))
    assert out == {"y": 2}


def test_malformed_json_is_bad_output(capture_post):
    with pytest.raises(LLMBadOutput):
        capture_post(_Resp(body=_ok_body("not json at all")))


def test_truncated_json_at_max_tokens_is_output_truncated(capture_post):
    with pytest.raises(LLMOutputTruncated):
        capture_post(_Resp(body=_ok_body('{"a": 1, "b": ', stop_reason="max_tokens")))


def test_empty_text_at_max_tokens_is_output_truncated(capture_post):
    with pytest.raises(LLMOutputTruncated):
        capture_post(_Resp(body={"stop_reason": "max_tokens", "content": []}))


def test_empty_text_otherwise_is_bad_output(capture_post):
    with pytest.raises(LLMBadOutput):
        capture_post(_Resp(body={"stop_reason": "end_turn", "content": []}))


@pytest.mark.parametrize(
    "status,exc",
    [
        (401, LLMAuthError), (403, LLMAuthError),
        (429, LLMRateLimited),
        (500, LLMUnavailable), (502, LLMUnavailable), (529, LLMUnavailable),
        (413, LLMRequestTooLarge),
        (404, LLMConfigError),
    ],
)
def test_status_code_taxonomy(capture_post, status, exc):
    with pytest.raises(exc):
        capture_post(_Resp(status_code=status, text="err"))


def test_400_workspace_is_config_error(capture_post):
    with pytest.raises(LLMConfigError):
        capture_post(_Resp(status_code=400, text="invalid workspace id"))


def test_400_credit_balance_is_config_error(capture_post):
    with pytest.raises(LLMConfigError):
        capture_post(_Resp(status_code=400, text="Your credit balance is too low"))


def test_timeout_is_transport_error(capture_post, monkeypatch):
    def boom(*a, **k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(ac.httpx, "post", boom)
    with pytest.raises(LLMTransport):
        ac.messages_json(api_key="k", model="claude-sonnet-5", system_prompt="s",
                         user_prompt="u", max_tokens=100)


def test_workspace_header_only_when_configured(capture_post):
    capture_post(_Resp(body=_ok_body('{"ok": 1}')))
    assert "anthropic-workspace-id" not in capture_post.sent["headers"]
