from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from support.schemas import SupportAnalysis
from support.services.ai.base import AIProviderError
from support.services.ai.openrouter import OpenRouterProvider
from support.services.processor import SupportTicketProcessor


class FakeAIProvider:
    def __init__(self, result: SupportAnalysis | None = None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.context: Mapping[str, Any] | None = None

    def analyze(self, ticket_context: Mapping[str, Any]) -> SupportAnalysis:
        self.context = ticket_context
        if self.exc:
            raise self.exc
        assert self.result is not None
        return self.result


class FakeGorgiasClient:
    def __init__(self, ticket: dict[str, Any], customer: dict[str, Any] | None = None):
        self.ticket = ticket
        self.customer = customer or {"id": 42, "name": "Test Customer"}
        self.priority: str | None = None
        self.tags: tuple[str, ...] = ()
        self.note: str | None = None

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        return self.ticket

    def get_customer(self, customer_id: int | str) -> dict[str, Any]:
        return self.customer

    def update_ticket_priority(self, ticket_id: str, priority: str) -> dict[str, Any]:
        self.priority = priority
        return {"id": ticket_id, "priority": priority}

    def add_tags_to_ticket(self, ticket_id: str, tag_names: tuple[str, ...]) -> dict[str, Any]:
        self.tags = tuple(tag_names)
        return {"id": ticket_id, "tags": list(tag_names)}

    def create_internal_note(self, ticket_id: str, body: str) -> dict[str, Any]:
        self.note = body
        return {"id": "note-1"}


def make_analysis(**overrides: Any) -> SupportAnalysis:
    suggested_reply = overrides.pop(
        "suggested_reply",
        "Thanks for reaching out. We will review this and follow up.",
    )
    data = {
        "customer_language": "en",
        "intent": "other",
        "sentiment": "neutral",
        "urgency": "medium",
        "risk": "medium",
        "confidence": 0.88,
        "summary": "Customer needs help.",
        "suggested_reply_localized": suggested_reply,
        "suggested_reply_en": suggested_reply,
        "recommended_action": "agent_review",
        "reasoning_summary": "Human review is appropriate.",
    }
    data.update(overrides)
    return SupportAnalysis(**data)


def make_ticket(message: str, order: dict[str, Any] | None = None) -> dict[str, Any]:
    ticket: dict[str, Any] = {
        "id": 123,
        "customer": {"id": 42},
        "messages": [
            {
                "body_text": message,
                "public": True,
                "sender": {"type": "customer"},
            }
        ],
    }
    if order is not None:
        ticket["order"] = order
    return ticket


def run_processor(ticket: dict[str, Any], analysis: SupportAnalysis | None = None, exc: Exception | None = None):
    fake_client = FakeGorgiasClient(ticket)
    fake_ai = FakeAIProvider(result=analysis, exc=exc)
    result = SupportTicketProcessor(
        gorgias_client=fake_client,
        ai_provider=fake_ai,
    ).process("123", {})
    return fake_client, result


def test_order_status_scenario_tags_low_risk_and_requires_review() -> None:
    ticket = make_ticket(
        "Hi, where is my order? Could you please send me the tracking information?",
        order={
            "financial_status": "paid",
            "fulfillment_status": "fulfilled",
            "tracking_number": "TRACK123",
        },
    )
    analysis = make_analysis(
        intent="order_status",
        urgency="low",
        risk="low",
        summary="Customer is asking for tracking on a fulfilled order.",
    )

    fake_client, result = run_processor(ticket, analysis)

    assert fake_client.priority == "low"
    assert fake_client.tags == ("AI_ORDER_STATUS", "AI_LOW_RISK")
    assert result.decision.human_review_required is True
    assert "No customer-facing message was sent" in fake_client.note


def test_shipping_delay_scenario_escalates_unfulfilled_order() -> None:
    ticket = make_ticket(
        "I placed my order over a week ago and it still hasn't shipped.",
        order={
            "financial_status": "paid",
            "fulfillment_status": "unfulfilled",
        },
    )
    analysis = make_analysis(
        intent="shipping_delay",
        urgency="high",
        risk="high",
        summary="Customer reports the order has not shipped after more than one week.",
        suggested_reply="Your order has shipped and tracking is available.",
    )

    fake_client, result = run_processor(ticket, analysis)

    assert fake_client.priority == "high"
    assert fake_client.tags == ("AI_SHIPPING_DELAY", "AI_ESCALATED")
    assert result.decision.recommended_action == "escalate"
    assert "will confirm the next step after reviewing the shipment details" in fake_client.note
    assert "Your order has shipped" not in fake_client.note


def test_product_defect_scenario_requires_warranty_review() -> None:
    ticket = make_ticket(
        "My Heatbox stopped heating after three weeks. I charged it overnight and it still does not work."
    )
    analysis = make_analysis(
        intent="product_defect",
        urgency="high",
        risk="high",
        summary="Customer reports a product stopped heating after three weeks.",
        suggested_reply="We will send a replacement immediately.",
    )

    fake_client, result = run_processor(ticket, analysis)

    assert fake_client.priority == "high"
    assert fake_client.tags == ("AI_PRODUCT_DEFECT", "AI_WARRANTY_REVIEW")
    assert result.decision.recommended_action == "agent_review"
    assert "warranty case" in fake_client.note
    assert "replacement immediately" not in fake_client.note


def test_safety_issue_scenario_overrides_model_recommendation() -> None:
    ticket = make_ticket(
        "The bottle warmer became extremely hot while my child was using it.",
        order={
            "financial_status": "paid",
            "fulfillment_status": "fulfilled",
            "tracking_number": "TRACK123",
        },
    )
    analysis = make_analysis(
        intent="order_status",
        urgency="low",
        risk="low",
        recommended_action="auto_resolve",
        summary="Customer asks about a simple order status issue.",
    )

    fake_client, result = run_processor(ticket, analysis)

    assert fake_client.priority == "critical"
    assert fake_client.tags == ("AI_SAFETY", "AI_ESCALATED")
    assert result.decision.recommended_action == "escalate"
    assert result.decision.customer_reply_allowed is False
    assert "Do not send an automated reply" in fake_client.note


def test_ai_provider_failure_falls_back_to_human_review() -> None:
    ticket = make_ticket("I need help with my order.")

    fake_client, result = run_processor(
        ticket,
        exc=AIProviderError("provider unavailable"),
    )

    assert result.analysis.confidence == 0.0
    assert result.analysis.recommended_action == "agent_review"
    assert fake_client.priority == "normal"
    assert fake_client.tags == ("AI_REVIEW_REQUIRED", "AI_MEDIUM_RISK")
    assert "AI analysis could not be completed reliably" in fake_client.note


class IncompleteOpenRouterResponse:
    status_code = 200
    content = b"{}"

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"intent":"order_status"}',
                    }
                }
            ]
        }


class IncompleteOpenRouterSession:
    def post(self, *args: Any, **kwargs: Any) -> IncompleteOpenRouterResponse:
        return IncompleteOpenRouterResponse()


class TransientOpenRouterResponse:
    status_code = 503
    content = b"{}"

    def json(self) -> dict[str, Any]:
        return {"error": {"message": "Provider returned error"}}


class RepeatedTransientOpenRouterSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *args: Any, **kwargs: Any) -> TransientOpenRouterResponse:
        self.calls += 1
        return TransientOpenRouterResponse()


def test_incomplete_openrouter_structured_json_falls_back_safely() -> None:
    ticket = make_ticket("Could you help me with my order?")
    fake_client = FakeGorgiasClient(ticket)
    openrouter_provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=IncompleteOpenRouterSession(),
    )

    result = SupportTicketProcessor(
        gorgias_client=fake_client,
        ai_provider=openrouter_provider,
    ).process("123", {})

    assert result.analysis.confidence == 0.0
    assert result.analysis.recommended_action == "agent_review"
    assert fake_client.priority == "normal"
    assert fake_client.tags == ("AI_REVIEW_REQUIRED", "AI_MEDIUM_RISK")


def test_repeated_openrouter_503_falls_back_safely_after_retries() -> None:
    ticket = make_ticket("Could you help me with my order?")
    fake_client = FakeGorgiasClient(ticket)
    session = RepeatedTransientOpenRouterSession()
    openrouter_provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
        sleep_func=lambda seconds: None,
        jitter_func=lambda start, end: 0.0,
    )

    result = SupportTicketProcessor(
        gorgias_client=fake_client,
        ai_provider=openrouter_provider,
    ).process("123", {})

    assert session.calls == 3
    assert result.analysis.confidence == 0.0
    assert result.analysis.recommended_action == "agent_review"
    assert fake_client.priority == "normal"
    assert fake_client.tags == ("AI_REVIEW_REQUIRED", "AI_MEDIUM_RISK")
