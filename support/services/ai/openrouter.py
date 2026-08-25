from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

import requests
from django.conf import settings
from pydantic import ValidationError

from support.schemas import SupportAnalysis
from support.services.ai.base import AIProviderError

logger = logging.getLogger(__name__)
OPENROUTER_FREE_ROUTER = "openrouter/free"
SENSITIVE_VALUE_RE = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"(sk-[A-Za-z0-9_-]+)|"
    r"([A-Za-z0-9_-]{20,})"
)


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise AIProviderError("OPENROUTER_API_KEY is required.")
        if not model:
            raise AIProviderError("OPENROUTER_MODEL is required.")
        if not self._is_allowed_free_model(model):
            raise AIProviderError(
                "OPENROUTER_MODEL must be 'openrouter/free' or an OpenRouter "
                "free model containing ':free'."
            )

        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds or settings.EXTERNAL_REQUEST_TIMEOUT_SECONDS
        )
        self.session = session or requests.Session()

    @classmethod
    def from_settings(cls) -> "OpenRouterProvider":
        return cls(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
            timeout_seconds=settings.EXTERNAL_REQUEST_TIMEOUT_SECONDS,
        )

    def analyze(self, ticket_context: Mapping[str, Any]) -> SupportAnalysis:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "provider": {"require_parameters": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "support_analysis",
                    "strict": True,
                    "schema": SupportAnalysis.model_json_schema(),
                },
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a support operations classifier for an ecommerce "
                        "team. Return only valid JSON matching the requested schema. "
                        "Do not include hidden chain-of-thought. reasoning_summary "
                        "must be a short operational justification."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(ticket_context),
                },
            ],
        }

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise AIProviderError("OpenRouter request timed out.") from exc
        except requests.RequestException as exc:
            raise AIProviderError("OpenRouter request failed.") from exc

        status_code = getattr(response, "status_code", 200)
        if status_code < 200 or status_code >= 300:
            detail = self._safe_error_detail(response)
            if detail:
                raise AIProviderError(
                    f"OpenRouter returned HTTP {status_code}: {detail}"
                )
            raise AIProviderError(f"OpenRouter returned HTTP {status_code}.")

        try:
            response_payload = response.json()
            content = response_payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIProviderError("OpenRouter returned a malformed response.") from exc

        try:
            return SupportAnalysis.model_validate_json(self._extract_json(content))
        except ValidationError as exc:
            failed_fields = self._validation_error_fields(exc)
            raise AIProviderError(
                "OpenRouter analysis did not match the schema. "
                f"Failed fields: {failed_fields}."
            ) from exc
        except ValueError as exc:
            raise AIProviderError("OpenRouter analysis did not match the schema.") from exc

    def _build_prompt(self, ticket_context: Mapping[str, Any]) -> str:
        schema = {
            "intent": (
                "order_status | shipping_delay | product_defect | safety_issue | "
                "refund_request | cancellation | other"
            ),
            "sentiment": "positive | neutral | negative | very_negative",
            "urgency": "low | medium | high | critical",
            "risk": "low | medium | high | critical",
            "confidence": "number between 0 and 1",
            "summary": "brief customer issue summary",
            "suggested_reply": "draft reply for a human agent to review",
            "recommended_action": "auto_resolve | agent_review | escalate",
            "reasoning_summary": "short operational explanation only",
        }
        return (
            "Analyze the latest meaningful customer support message and available "
            "ticket/customer/order context. Return JSON only.\n\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Context:\n{json.dumps(ticket_context, default=str)[:12000]}"
        )

    @staticmethod
    def _extract_json(content: str) -> str:
        text = content.strip()
        if not text:
            raise ValueError("empty response")
        try:
            json.loads(text)
            return text
        except ValueError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            candidate = text[start : end + 1]
            json.loads(candidate)
            return candidate

    @staticmethod
    def _is_allowed_free_model(model: str) -> bool:
        normalized = model.strip().lower()
        return normalized == OPENROUTER_FREE_ROUTER or ":free" in normalized

    @staticmethod
    def _safe_error_detail(response: requests.Response) -> str:
        detail = ""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                message = error.get("message") or error.get("code")
                if message:
                    detail = str(message)
            if not detail:
                for key in ("message", "detail", "error"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        detail = value
                        break

        if not detail:
            detail = str(getattr(response, "text", "") or "")

        detail = " ".join(detail.split())
        detail = SENSITIVE_VALUE_RE.sub("[redacted]", detail)
        return detail[:240]

    @staticmethod
    def _validation_error_fields(error: ValidationError) -> str:
        fields: list[str] = []
        for item in error.errors():
            location = item.get("loc", ())
            if not location:
                continue
            field = ".".join(str(part) for part in location)
            if field not in fields:
                fields.append(field)
        return ", ".join(fields) if fields else "unknown"
