from __future__ import annotations

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
        "use_fixtures": settings.use_fixtures,
        "apify_configured": bool(settings.apify_api_token),
        "embedding_model": settings.embedding_model,
        "llm": _llm_health(),
        # legacy flat keys — kept for any existing dashboards
        "groq_configured": bool(settings.groq_api_key),
        "openrouter_configured": bool(settings.openrouter_api_key),
        "anthropic_active": bool(settings.anthropic_api_key),
    }


def _llm_health() -> dict:
    """Configuration-only view of the provider chain (V4 §12). No live calls,
    no secrets."""
    from app.services.llm import circuit
    from app.services.llm.providers import default_chain

    return {
        "priority": [p.name for p in default_chain()],
        "anthropic": {
            "configured": bool(settings.anthropic_api_key),
            "model": settings.anthropic_model if settings.anthropic_api_key else None,
            "workspace_id_set": bool(settings.anthropic_workspace_id),
        },
        "groq": {
            "configured": bool(settings.groq_api_key),
            "primary_model": settings.groq_primary_model,
            "fallback_model": settings.groq_fallback_model,
        },
        "openrouter": {
            "configured": bool(settings.openrouter_api_key),
            "model": settings.openrouter_model,
        },
        "deprecated_enable_paid_llm": settings.enable_paid_llm,
        "circuit_breakers": circuit.snapshot(),
    }
