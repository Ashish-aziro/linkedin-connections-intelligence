"""Provider preflight — V4 PART 10.1 §9.

``inspect()``  — configuration only, ZERO network. Reports which providers are
configured and which one the router would try first.

``probe_live()`` — the ONLY live operation the pilot harness performs before the
real run: ONE tiny structured request through the normal router (tiny schema,
tiny prompt). No search, no semantic enrichment. Gated behind
``--live --i-understand-costs`` in the CLI.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings


class _Ping(BaseModel):
    ok: Literal[True] = Field(description="must be true")
    word: Literal["pong"] = Field(description="must be exactly 'pong'")


def inspect() -> dict:
    from app.services.llm.providers import default_chain

    chain = default_chain()
    return {
        "network": "none — configuration inspection only",
        "anthropic_configured": bool(settings.anthropic_api_key),
        "anthropic_workspace_id_configured": bool(settings.anthropic_workspace_id),
        "anthropic_model": settings.anthropic_model,
        "provider_chain": [p.name for p in chain],
        "expected_first_provider": chain[0].name if chain else None,
        "warning": (
            "ANTHROPIC_WORKSPACE_ID is empty — the earlier accidental attempt failed with "
            "a workspace configuration error. Confirm this is intentional before a live run."
            if settings.anthropic_api_key and not settings.anthropic_workspace_id else None
        ),
    }


def probe_live() -> dict:
    """ONE tiny structured LLM call through the normal router. Live/paid."""
    from app.services.llm.router import generate_structured

    res = generate_structured(
        "You reply with strict JSON only.",
        "Return {\"ok\": true, \"word\": \"pong\"}.",
        _Ping,
        max_tokens=64,
        operation="pilot_preflight",
    )
    if res is None:
        return {"success": False, "detail": "every configured provider was exhausted / errored",
                "config": inspect()}
    model, provider, model_id = res
    return {
        "success": True,
        "provider": provider,
        "model": model_id,
        "response": model.model_dump(),
        "config": inspect(),
    }
