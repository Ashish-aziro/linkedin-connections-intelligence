"""Anthropic Messages API client (paid — opt-in only).

Not OpenAI-compatible: different endpoint, headers, and body shape. JSON output
is coaxed by prefilling the assistant turn with ``{`` so the reply always starts
as a JSON object; the leading brace is stitched back on before parsing.
"""
from __future__ import annotations

import logging

import httpx

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

log = logging.getLogger("app.llm")

_URL = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"


def messages_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    workspace_id: str = "",
    timeout: float = 60.0,
) -> dict:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "system": system_prompt + "\n\nRespond with a single JSON object and nothing else.",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "{"},
        ],
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
        # identity-linked key without / with a wrong ANTHROPIC_WORKSPACE_ID (V4 §7)
        raise LLMConfigError("anthropic workspace configuration error")
    if resp.status_code in (400, 404):
        # unknown model, malformed request — identical on retry
        raise LLMConfigError(f"anthropic request rejected ({resp.status_code})")
    if resp.status_code >= 400:
        raise LLMConfigError(f"anthropic error {resp.status_code}")

    body = resp.json()
    stop_reason = body.get("stop_reason")
    try:
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
    except (AttributeError, TypeError) as e:
        raise LLMBadOutput(f"unexpected Anthropic response shape: {e}") from e

    if not text.strip():
        if stop_reason == "max_tokens":
            raise LLMOutputTruncated(f"Anthropic returned no text and stopped at max_tokens={max_tokens}")
        raise LLMBadOutput("empty Anthropic response")

    # the assistant turn was prefilled with "{", so stitch it back on
    full = "{" + text if not text.lstrip().startswith("{") else text
    try:
        return _extract_json(full)
    except LLMBadOutput as e:
        # a genuinely malformed response stays LLMBadOutput (retry-same-provider
        # can still help); one that hit the token ceiling gets its own category
        # so the caller SPLITS the batch instead of resubmitting the same request.
        if stop_reason == "max_tokens":
            raise LLMOutputTruncated(
                f"Anthropic hit max_tokens={max_tokens} before completing valid JSON "
                f"({len(full)} chars produced): {e}"
            ) from e
        raise
