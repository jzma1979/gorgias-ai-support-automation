from __future__ import annotations

from support.schemas import SupportAnalysis, fallback_analysis
from support.services.ai.openrouter import OpenRouterProvider
from support.services.decision_engine import SupportDecision
from support.services.processor import build_internal_note


class FakeResponse:
    status_code = 200
    content = b"{}"

    def __init__(self, content: str):
        self.content_text = content

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content_text}}]}


class FakeSession:
    def __init__(self, content: str):
        self.response = FakeResponse(content)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


def make_provider(content: str) -> OpenRouterProvider:
    return OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=FakeSession(content),
    )


def make_decision() -> SupportDecision:
    return SupportDecision(
        priority="normal",
        tags=("AI_REVIEW_REQUIRED", "AI_LOW_RISK"),
        recommended_action="agent_review",
        human_review_required=True,
        customer_reply_allowed=False,
        decision_reason="Human review required.",
    )


def test_english_ticket_keeps_summary_and_reply_in_english() -> None:
    analysis = make_provider(
        '{"customer_language":"en","intent":"order_status","sentiment":"neutral",'
        '"urgency":"low","risk":"low","confidence":0.86,'
        '"summary":"Customer asks for the status of order #1001.",'
        '"suggested_reply":"Hi, we will check order #1001 and follow up shortly.",'
        '"recommended_action":"agent_review",'
        '"reasoning_summary":"Low-risk order status request."}'
    ).analyze({"latest_customer_message": "Hi, where is order #1001?"})

    assert analysis.customer_language == "en"
    assert analysis.summary == "Customer asks for the status of order #1001."
    assert analysis.suggested_reply.startswith("Hi,")


def test_serbian_ticket_uses_english_summary_and_serbian_suggested_reply() -> None:
    analysis = make_provider(
        '{"customer_language":"sr","intent":"order_status","sentiment":"neutral",'
        '"urgency":"low","risk":"low","confidence":0.86,'
        '"summary":"Customer asks for the status of order #1001.",'
        '"suggested_reply":"Zdravo, provericemo porudzbinu #1001 i javiti vam se uskoro.",'
        '"recommended_action":"agent_review",'
        '"reasoning_summary":"Low-risk order status request."}'
    ).analyze({"latest_customer_message": "Zdravo, moja porudzbina #1001 jos nije stigla."})

    assert analysis.customer_language == "sr"
    assert analysis.summary == "Customer asks for the status of order #1001."
    assert analysis.suggested_reply.startswith("Zdravo")


def test_german_ticket_uses_english_summary_and_german_suggested_reply() -> None:
    analysis = make_provider(
        '{"customer_language":"de","intent":"shipping_delay","sentiment":"negative",'
        '"urgency":"high","risk":"medium","confidence":0.9,'
        '"summary":"Customer reports that order #2002 has not shipped yet.",'
        '"suggested_reply":"Hallo, wir pruefen Bestellung #2002 und melden uns in Kuerze.",'
        '"recommended_action":"agent_review",'
        '"reasoning_summary":"Shipping-delay review is appropriate."}'
    ).analyze({"latest_customer_message": "Hallo, Bestellung #2002 wurde noch nicht versendet."})

    assert analysis.customer_language == "de"
    assert analysis.summary == "Customer reports that order #2002 has not shipped yet."
    assert analysis.suggested_reply.startswith("Hallo")


def test_unknown_language_uses_english_suggested_reply() -> None:
    analysis = make_provider(
        '{"customer_language":"unknown","intent":"other","sentiment":"neutral",'
        '"urgency":"medium","risk":"medium","confidence":0.42,'
        '"summary":"The customer message language is unclear and needs review.",'
        '"suggested_reply":"Thanks for reaching out. Our team will review this and follow up.",'
        '"recommended_action":"agent_review",'
        '"reasoning_summary":"Language is ambiguous, so English fallback is used."}'
    ).analyze({"latest_customer_message": "???"})

    assert analysis.customer_language == "unknown"
    assert analysis.summary == "The customer message language is unclear and needs review."
    assert analysis.suggested_reply.startswith("Thanks")


def test_identifiers_are_preserved_in_analysis_and_internal_note() -> None:
    analysis = SupportAnalysis(
        customer_language="sr",
        intent="order_status",
        sentiment="neutral",
        urgency="low",
        risk="low",
        confidence=0.91,
        summary=(
            "Customer asks for tracking on order #1001 for Heatbox, tracking "
            "number TRACK123, URL https://track.example/ABC."
        ),
        suggested_reply=(
            "Zdravo, provericemo porudzbinu #1001 za Heatbox. Tracking broj "
            "TRACK123 i link https://track.example/ABC ostaju nepromenjeni."
        ),
        recommended_action="agent_review",
        reasoning_summary="Identifiers were preserved for agent review.",
    )

    note = build_internal_note(analysis, make_decision())

    assert "Customer language: sr" in note
    assert "#1001" in note
    assert "TRACK123" in note
    assert "https://track.example/ABC" in note
    assert "Heatbox" in note


def test_fallback_analysis_uses_unknown_language() -> None:
    analysis = fallback_analysis("AI provider failed.")

    assert analysis.customer_language == "unknown"
    assert analysis.suggested_reply.startswith("Please review")
