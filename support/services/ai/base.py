from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from support.schemas import SupportAnalysis


class AIProviderError(Exception):
    """Raised when an AI provider cannot return validated analysis."""


class AIProvider(Protocol):
    def analyze(self, ticket_context: Mapping[str, Any]) -> SupportAnalysis:
        """Return validated structured support analysis for a ticket."""
