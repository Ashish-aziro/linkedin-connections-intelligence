"""BENCHMARK-ONLY Anthropic Messages client for Claude 5-family models.

Same role as ``app.services.llm.anthropic_client.messages_json`` and the same
error taxonomy (so the router + adaptive splitter behave identically), but:

  * sends **no** assistant-turn prefill   — Claude 5 returns 400
    ("This model does not support assistant message prefill")
  * sends **no** ``temperature``          — Claude 5 returns 400
    ("`temperature` is deprecated for this model")
  * reads only ``text`` content blocks    — ignores any ``thinking`` block

It deliberately reuses ``anthropic_client.httpx`` so the benchmark's metering /
budget-guard proxy (installed on that attribute) still sees every call.

Confined to ``eval/pilot/``. Production ``anthropic_client.py`` is NOT modified
in Phase A. The API key is taken from ``settings`` and never logged.
"""
from __future__ import annotations

import logging

from app.services.llm import anthropic_client as _ac
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
from app.services.llm.openai_compatible import _extract_json

log = logging.getLogger("app.llm.bench_v5")

_URL = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"


def messages_json_v5(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    workspace_id: str = "",
    timeout: float = 60.0,
) -> dict:
    httpx = _ac.httpx  # proxied by the benchmark metering context when active
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt + "\n\nRespond with a single JSON object and nothing else.",
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _VERSION,
        "content-type": "application/json",
    }
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id

    try:
        resp = httpx.post(_URL, json=payload, headers=headers, timeout=timeout)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise LLMTransport(f"transport error: {type(e).__name__}") from e

    if resp.status_code == 429:
        ra = resp.headers.get("retry-after")
        raise LLMRateLimited(
            "429 from Anthropic",
            retry_after=float(ra) if ra and ra.replace(".", "").isdigit() else None,
        )
    if resp.status_code in (500, 502, 503, 504, 529):
        raise LLMUnavailable(f"anthropic {resp.status_code}")
    if resp.status_code in (401, 403):
        raise LLMAuthError(f"anthropic auth {resp.status_code}")
    if resp.status_code == 413:
        raise LLMRequestTooLarge(f"anthropic {resp.status_code}: request too large")
    if resp.status_code == 400 and "workspace" in resp.text.lower():
        raise LLMConfigError("anthropic workspace configuration error")
    if resp.status_code in (400, 404):
        raise LLMConfigError(f"anthropic request rejected ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code >= 400:
        raise LLMConfigError(f"anthropic error {resp.status_code}")

    body = resp.json()
    stop_reason = body.get("stop_reason")
    try:
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
    except (AttributeError, TypeError) as e:
        raise LLMBadOutput(f"unexpected Anthropic response shape: {e}") from e

    if not text.strip():
        if stop_reason == "max_tokens":
            raise LLMOutputTruncated(
                f"Anthropic returned no text and stopped at max_tokens={max_tokens}"
            )
        raise LLMBadOutput("empty Anthropic response")

    try:
        return _extract_json(text)
    except LLMBadOutput as e:
        if stop_reason == "max_tokens":
            raise LLMOutputTruncated(
                f"Anthropic hit max_tokens={max_tokens} before completing valid JSON "
                f"({len(text)} chars produced): {e}"
            ) from e
        raise
