from __future__ import annotations

from django.conf import settings

from support.services.ai.base import AIProvider, AIProviderError
from support.services.ai.openrouter import OpenRouterProvider


def get_ai_provider() -> AIProvider:
    provider_name = settings.AI_PROVIDER.strip().lower()
    if provider_name == "openrouter":
        return OpenRouterProvider.from_settings()
    raise AIProviderError(f"Unsupported AI_PROVIDER: {settings.AI_PROVIDER}")
