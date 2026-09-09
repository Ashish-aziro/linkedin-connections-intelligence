"""LLM call with structured-output validation, retries and a circuit breaker.

Providers come from ``default_chain()`` — Anthropic only (a list with one
``AnthropicProvider`` when a key is configured, else empty). The first provider
that returns validated structured output wins; any failure falls through to the
next provider if there is one, and errors are classified so unretryable failures
(bad key, bad workspace, unknown model) move on immediately and cool that
provider down. When the chain is empty or exhausted the call returns ``None``
and the caller uses its deterministic path.
"""
from __future__ import annotations

import logging
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.services.llm import budget, circuit
from app.services.llm.base import (
    LLMAuthError,
    LLMBadOutput,
    LLMConfigError,
    LLMError,
    LLMOutputTruncated,
    LLMProvider,
    LLMRateLimited,
    LLMRequestTooLarge,
    LLMUnavailable,
)
from app.services.llm.providers import default_chain

log = logging.getLogger("app.llm.router")

T = TypeVar("T", bound=BaseModel)

_MAX_BACKOFF = 5.0
_MAX_RETRY_AFTER = 20.0  # cap an honoured Retry-After so one call can't stall a search


def _sleep(seconds: float) -> None:
    time.sleep(max(0.0, min(seconds, _MAX_BACKOFF)))


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    *,
    max_tokens: int = 1500,
    chain: list[LLMProvider] | None = None,
    operation: str = "unspecified",
    return_meta: bool = False,
    timeout: float | None = None,
):
    """Return ``(validated_model, provider_name, model_id)``  — or, when
    ``return_meta=True``, ``(model, provider_name, model_id, meta)`` where meta
    is ``{"operation","selected_provider","selected_model","attempts":[...]}``
    (V4 §11). Returns ``None`` (or ``(None, meta)``) when every provider is
    exhausted. ``timeout`` (hardening PART 17) overrides each provider's
    default HTTP timeout for this call — used by query interpretation so one
    slow-but-not-failing provider can't hold a whole search hostage; every
    other caller leaves it ``None`` (provider default)."""
    if not budget.try_consume():
        log.warning("%s -> SEARCH_LLM_MAX_CALLS budget exhausted — skipping (deterministic result stands)",
                   operation)
        meta = {"operation": operation, "selected_provider": None, "selected_model": None,
               "attempts": [{"provider": None, "status": "budget_exhausted"}]}
        return (None, meta) if return_meta else None

    providers = chain if chain is not None else default_chain()
    retries = max(0, settings.llm_max_retries)
    full_user = user_prompt + _SCHEMA_HINT_PREFIX + _schema_skeleton(schema)

    attempts: list[dict] = []

    for provider in providers:
        if not provider.available():
            continue
        if circuit.is_open(provider.name):
            attempts.append({"provider": provider.name, "status": "circuit_open"})
            log.info("%s -> %s skipped (circuit cooling down)", operation, provider.name)
            continue

        last_category = "error"
        for attempt in range(retries + 1):
            try:
                raw = provider.generate_json(system_prompt, full_user, max_tokens=max_tokens, timeout=timeout)
                model = schema.model_validate(raw)
            except LLMRateLimited as e:
                last_category = e.category
                wait = min(e.retry_after or 2 ** attempt, _MAX_RETRY_AFTER)
                log.warning("%s -> %s rate_limited (try %d)", operation, provider.name, attempt + 1)
                if attempt < retries:
                    _sleep(wait)
                    continue
            except (LLMAuthError, LLMConfigError) as e:
                # cannot succeed by retrying — bail this provider immediately (V4 §6)
                last_category = e.category
                log.warning("%s -> %s %s: %s (no retry)", operation, provider.name, e.category, e)
                break
            except LLMUnavailable as e:  # incl. LLMTransport
                last_category = e.category
                log.warning("%s -> %s %s (try %d)", operation, provider.name, e.category, attempt + 1)
                if attempt < retries:
                    _sleep(2 ** attempt)
                    continue
            except (LLMOutputTruncated, LLMRequestTooLarge) as e:
                # the problem is the request/expected-response SIZE, not this
                # provider — re-sending the IDENTICAL oversized payload to any
                # next provider would just fail the same way (413 / another
                # truncation) and waste a round trip. Return to the caller
                # IMMEDIATELY so the adaptive splitter can shrink the request;
                # the smaller request then goes through the chain from the top.
                attempts.append({"provider": provider.name, "status": e.category})
                log.warning("%s -> %s %s: %s (returning to caller — no same-size fallback)",
                           operation, provider.name, e.category, str(e)[:160])
                meta = {"operation": operation, "selected_provider": None,
                       "selected_model": None, "attempts": attempts}
                return (None, meta) if return_meta else None
            except (LLMBadOutput, ValidationError) as e:
                last_category = "bad_output"
                log.warning("%s -> %s bad_output: %s (try %d)", operation, provider.name, str(e)[:160], attempt + 1)
                if attempt < retries:
                    continue
            except LLMError as e:  # pragma: no cover - defensive
                last_category = getattr(e, "category", "error")
                break
            else:
                circuit.record_success(provider.name)
                attempts.append({"provider": provider.name, "status": "success"})
                log.info("%s -> %s ok (%s)", operation, provider.name, provider.model)
                meta = {
                    "operation": operation,
                    "selected_provider": provider.name,
                    "selected_model": provider.model,
                    "attempts": attempts,
                }
                if return_meta:
                    return model, provider.name, provider.model, meta
                return model, provider.name, provider.model
            break  # retries for this provider exhausted -> next provider

        # provider gave up — record for the breaker. bad_output is prompt-local,
        # not a provider fault, so it does not trip the circuit. (truncated /
        # request_too_large never reach here — they return immediately above.)
        attempts.append({"provider": provider.name, "status": last_category})
        if last_category != "bad_output":
            circuit.record_failure(provider.name, last_category)

    log.error("%s -> all LLM providers exhausted (%s)", operation, [a["status"] for a in attempts])
    meta = {"operation": operation, "selected_provider": None, "selected_model": None, "attempts": attempts}
    if return_meta:
        return None, meta
    return None


_SCHEMA_HINT_PREFIX = (
    "\n\nReturn ONLY a JSON object matching this schema (no prose, no markdown).\n"
    "IMPORTANT: the values shown below are PLACEHOLDERS that demonstrate the "
    "required JSON structure only. DO NOT copy the placeholder values. "
    "Populate every field from the task instructions and supplied data:\n"
)


def _schema_skeleton(schema: type[BaseModel]) -> str:
    import json

    try:
        js = schema.model_json_schema()
        defs = js.get("$defs", {})
        props = js.get("properties", {})
        skeleton = {k: _example_for(v, defs) for k, v in props.items()}
        return json.dumps(skeleton, indent=2)
    except Exception:  # noqa: BLE001
        return "{}"


def _resolve(prop: dict, defs: dict) -> dict:
    ref = prop.get("$ref") or (prop.get("allOf", [{}])[0].get("$ref") if prop.get("allOf") else None)
    if ref:
        return defs.get(ref.split("/")[-1], {})
    return prop


def _example_for(prop: dict, defs: dict):
    prop = _resolve(prop, defs)
    if "const" in prop:
        return prop["const"]
    t = prop.get("type")
    if "anyOf" in prop and not t:
        for opt in prop["anyOf"]:
            if opt.get("type") not in (None, "null"):
                return _example_for(opt, defs)
        return None
    if t == "array":
        items = _resolve(prop.get("items", {}), defs)
        if items.get("type") == "object" or items.get("properties"):
            return [{k: _example_for(v, defs) for k, v in items.get("properties", {}).items()}]
        return []
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    if t == "object" or prop.get("properties"):
        return {k: _example_for(v, defs) for k, v in prop.get("properties", {}).items()}
    return ""
