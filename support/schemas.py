from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Intent = Literal[
    "order_status",
    "shipping_delay",
    "product_defect",
    "safety_issue",
    "refund_request",
    "cancellation",
    "other",
]
Sentiment = Literal["positive", "neutral", "negative", "very_negative"]
Urgency = Literal["low", "medium", "high", "critical"]
Risk = Literal["low", "medium", "high", "critical"]
RecommendedAction = Literal["auto_resolve", "agent_review", "escalate"]


class SupportAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    customer_language: str = Field(min_length=2, max_length=16)
    intent: Intent
    sentiment: Sentiment
    urgency: Urgency
    risk: Risk
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1200)
    suggested_reply_localized: str = Field(min_length=1, max_length=4000)
    suggested_reply_en: str = Field(min_length=1, max_length=4000)
    recommended_action: RecommendedAction
    reasoning_summary: str = Field(min_length=1, max_length=1200)

    @property
    def suggested_reply(self) -> str:
        return self.suggested_reply_localized

    @field_validator("customer_language", mode="before")
    @classmethod
    def normalize_customer_language(cls, value: object) -> str:
        if value is None:
            raise ValueError("customer_language is required")
        text = str(value).strip().lower()
        if not text:
            raise ValueError("customer_language must not be blank")
        return text

    @field_validator(
        "summary",
        "suggested_reply_localized",
        "suggested_reply_en",
        "reasoning_summary",
        mode="before",
    )
    @classmethod
    def require_non_empty_text(cls, value: object) -> str:
        if value is None:
            raise ValueError("value is required")
        text = str(value).strip()
        if not text:
            raise ValueError("value must not be blank")
        return text


def fallback_analysis(reason: str) -> SupportAnalysis:
    safe_reason = reason.strip() or "AI analysis was unavailable."
    return SupportAnalysis(
        customer_language="unknown",
        intent="other",
        sentiment="neutral",
        urgency="medium",
        risk="medium",
        confidence=0.0,
        summary="AI analysis could not be completed reliably.",
        suggested_reply_localized=(
            "Please review the ticket context and respond manually. "
            "The automation could not produce a validated suggestion."
        ),
        suggested_reply_en=(
            "Please review the ticket context and respond manually. "
            "The automation could not produce a validated suggestion."
        ),
        recommended_action="agent_review",
        reasoning_summary=safe_reason[:1200],
    )
