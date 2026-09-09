"""Concrete LLM providers.

The application's external LLM is Anthropic only. A configured
``ANTHROPIC_API_KEY`` is the opt-in — there is no separate enable flag.
When no key is configured the chain is empty and every LLM call returns
``None``, which the callers already treat as "use the deterministic path".
"""
from __future__ import annotations

from app.config import settings
from app.constants import LLMProviderName
from app.services.llm.anthropic_client import messages_json
from app.services.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """The one external provider. Available whenever ``ANTHROPIC_API_KEY`` is set."""

    name = LLMProviderName.ANTHROPIC

    def __init__(self, model: str):
        self.model = model

    def available(self) -> bool:
        return bool(settings.anthropic_api_key)

    def generate_json(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int = 1500, timeout: float | None = None,
    ) -> dict:
        return messages_json(
            api_key=settings.anthropic_api_key,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            workspace_id=settings.anthropic_workspace_id,
            **({"timeout": timeout} if timeout is not None else {}),
        )


def default_chain() -> list[LLMProvider]:
    """The provider chain built from configured keys. Anthropic-only: a list with
    one ``AnthropicProvider`` when a key is set, otherwise empty."""
    if settings.anthropic_api_key:
        return [AnthropicProvider(settings.anthropic_model)]
    return []
