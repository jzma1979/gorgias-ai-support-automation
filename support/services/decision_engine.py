from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from support.schemas import SupportAnalysis

SAFETY_TERMS = (
    "safety",
    "unsafe",
    "overheat",
    "overheated",
    "overheating",
    "extremely hot",
    "too hot",
    "burn",
    "burned",
    "smoke",
    "fire",
    "spark",
    "injury",
    "injured",
    "child",
    "baby",
)

PRODUCT_DEFECT_TERMS = (
    "stopped working",
    "does not work",
    "doesn't work",
    "not working",
    "broken",
    "defect",
    "defective",
    "warranty",
    "stopped heating",
    "won't heat",
)

SHIPPING_DELAY_TERMS = (
    "hasn't shipped",
    "has not shipped",
    "still not shipped",
    "not shipped",
    "unfulfilled",
    "shipping delay",
    "delayed",
    "week ago",
)


@dataclass(frozen=True)
class SupportDecision:
    priority: str
    tags: tuple[str, ...]
    recommended_action: str
    human_review_required: bool
    customer_reply_allowed: bool
    decision_reason: str


def evaluate_support_decision(
    analysis: SupportAnalysis,
    ticket_context: Mapping[str, Any],
    latest_customer_message: str,
) -> SupportDecision:
    text = " ".join(
        [
            latest_customer_message,
            analysis.summary,
            analysis.reasoning_summary,
            analysis.intent,
        ]
    ).lower()

    if analysis.intent == "safety_issue" or _contains_any(text, SAFETY_TERMS):
        return SupportDecision(
            priority="critical",
            tags=("AI_SAFETY", "AI_ESCALATED"),
            recommended_action="escalate",
            human_review_required=True,
            customer_reply_allowed=False,
            decision_reason=(
                "Safety-related language detected. Immediate human review is required."
            ),
        )

    if analysis.intent == "product_defect" or _contains_any(text, PRODUCT_DEFECT_TERMS):
        return SupportDecision(
            priority="high",
            tags=("AI_PRODUCT_DEFECT", "AI_WARRANTY_REVIEW"),
            recommended_action="agent_review",
            human_review_required=True,
            customer_reply_allowed=False,
            decision_reason=(
                "Product defect or warranty language detected. Do not promise a refund "
                "or replacement automatically."
            ),
        )

    if (
        analysis.intent == "shipping_delay"
        or _contains_any(text, SHIPPING_DELAY_TERMS)
    ) and order_is_unfulfilled(ticket_context):
        return SupportDecision(
            priority="high",
            tags=("AI_SHIPPING_DELAY", "AI_ESCALATED"),
            recommended_action="escalate"
            if analysis.risk in {"high", "critical"}
            else "agent_review",
            human_review_required=True,
            customer_reply_allowed=False,
            decision_reason=(
                "Order appears unfulfilled and the customer is asking about a delay. "
                "Do not claim the order has shipped."
            ),
        )

    if (
        analysis.intent == "order_status"
        and order_is_paid(ticket_context)
        and order_is_fulfilled(ticket_context)
        and tracking_is_available(ticket_context)
    ):
        return SupportDecision(
            priority="low",
            tags=("AI_ORDER_STATUS", "AI_LOW_RISK"),
            recommended_action="agent_review",
            human_review_required=True,
            customer_reply_allowed=False,
            decision_reason=(
                "Paid and fulfilled order with tracking available. Human review remains "
                "required for the MVP."
            ),
        )

    if analysis.risk == "critical" or analysis.urgency == "critical":
        priority = "critical"
        risk_tag = "AI_ESCALATED"
        action = "escalate"
    elif analysis.risk == "high" or analysis.urgency == "high":
        priority = "high"
        risk_tag = "AI_ESCALATED"
        action = "agent_review"
    else:
        priority = "normal"
        risk_tag = f"AI_{analysis.risk.upper()}_RISK"
        action = "agent_review"

    return SupportDecision(
        priority=priority,
        tags=("AI_REVIEW_REQUIRED", risk_tag),
        recommended_action=action,
        human_review_required=True,
        customer_reply_allowed=False,
        decision_reason="No low-risk automation rule matched. Human review is required.",
    )


def order_is_paid(context: Mapping[str, Any]) -> bool:
    return _has_value_for_keys(
        context,
        keys=("financial_status", "payment_status", "paymentStatus"),
        accepted=("paid",),
    )


def order_is_fulfilled(context: Mapping[str, Any]) -> bool:
    return _has_value_for_keys(
        context,
        keys=("fulfillment_status", "fulfillmentStatus", "shipment_status"),
        accepted=("fulfilled", "shipped", "complete", "completed"),
    )


def order_is_unfulfilled(context: Mapping[str, Any]) -> bool:
    return _has_value_for_keys(
        context,
        keys=("fulfillment_status", "fulfillmentStatus", "shipment_status"),
        accepted=(
            "unfulfilled",
            "not_fulfilled",
            "not fulfilled",
            "pending",
            "open",
            "none",
            "null",
        ),
    ) or (not order_is_fulfilled(context) and _contains_any(_all_text(context), ("unfulfilled",)))


def tracking_is_available(context: Mapping[str, Any]) -> bool:
    for value in _values_for_keys(
        context,
        (
            "tracking_number",
            "tracking_numbers",
            "tracking_url",
            "tracking_urls",
            "trackingCompany",
            "tracking_company",
        ),
    ):
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            if any(str(item).strip() for item in value):
                return True
    return False


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _has_value_for_keys(
    context: Mapping[str, Any],
    *,
    keys: Iterable[str],
    accepted: Iterable[str],
) -> bool:
    accepted_values = {value.lower() for value in accepted}
    for value in _values_for_keys(context, keys):
        if value is None:
            normalized = "null"
        else:
            normalized = str(value).strip().lower()
        if normalized in accepted_values:
            return True
    return False


def _values_for_keys(obj: Any, keys: Iterable[str]) -> Iterable[Any]:
    desired = {key.lower() for key in keys}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).lower() in desired:
                yield value
            yield from _values_for_keys(value, keys)
    elif isinstance(obj, list | tuple):
        for item in obj:
            yield from _values_for_keys(item, keys)


def _all_text(obj: Any) -> str:
    parts: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            parts.append(str(key))
            parts.append(_all_text(value))
    elif isinstance(obj, list | tuple):
        for item in obj:
            parts.append(_all_text(item))
    elif obj is not None:
        parts.append(str(obj))
    return " ".join(parts).lower()
