from __future__ import annotations

import json

from support.schemas import SupportAnalysis, fallback_analysis
from support.services.ai.openrouter import OpenRouterProvider
from support.services.decision_engine import SupportDecision
from support.services.processor import build_internal_note, extract_latest_customer_message


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


def analysis_content(
    *,
    customer_language: str,
    summary: str,
    suggested_reply: str,
    intent: str = "order_status",
) -> str:
    return json.dumps(
        {
            "customer_language": customer_language,
            "intent": intent,
            "sentiment": "neutral",
            "urgency": "low",
            "risk": "low",
            "confidence": 0.86,
            "summary": summary,
            "suggested_reply": suggested_reply,
            "recommended_action": "agent_review",
            "reasoning_summary": "Language and support intent were classified.",
        },
        ensure_ascii=False,
    )


def language_source_section(prompt: str) -> str:
    return prompt.split("CUSTOMER MESSAGE — LANGUAGE DETECTION SOURCE:", 1)[1].split(
        "ORDER / BUSINESS CONTEXT — DO NOT USE FOR LANGUAGE DETECTION:",
        1,
    )[0]


def business_context_section(prompt: str) -> str:
    return prompt.split(
        "ORDER / BUSINESS CONTEXT — DO NOT USE FOR LANGUAGE DETECTION:",
        1,
    )[1]


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


def test_serbian_language_regression_uses_customer_message_not_english_context() -> None:
    customer_message = (
        "Zdravo, moja porudžbina još nije stigla. Prošlo je više od nedelju "
        "dana i počinjem da se nerviram. Možete li da proverite gde je paket?"
    )
    provider = make_provider(
        analysis_content(
            customer_language="sr",
            summary="Customer says order #1001 has not arrived after more than a week.",
            suggested_reply=(
                "Zdravo, proverićemo porudžbinu #1001, Heatbox i Canpar Courier "
                "tracking TRACK123 pa ćemo vam se javiti."
            ),
        )
    )

    analysis = provider.analyze(
        {
            "latest_customer_message": customer_message,
            "ticket": {
                "subject": "Order #1001 tracking information",
                "order": {
                    "name": "Order #1001",
                    "product_name": "Heatbox",
                    "fulfillment_status": "fulfilled",
                    "tracking_company": "Canpar Courier",
                    "tracking_number": "TRACK123",
                    "tracking_url": "https://track.example/TRACK123",
                },
                "messages": [
                    {
                        "body_text": "AI SUPPORT ANALYSIS\nSummary in English.",
                        "public": False,
                        "sender": {"type": "agent"},
                    }
                ],
            },
        }
    )
    prompt = provider.session.calls[0]["kwargs"]["json"]["messages"][1]["content"]

    assert analysis.customer_language == "sr"
    assert analysis.summary == "Customer says order #1001 has not arrived after more than a week."
    assert analysis.suggested_reply.startswith("Zdravo")
    assert "#1001" in analysis.suggested_reply
    assert "Heatbox" in analysis.suggested_reply
    assert "Canpar Courier" in analysis.suggested_reply
    assert "TRACK123" in analysis.suggested_reply
    assert customer_message in language_source_section(prompt)
    assert "Order #1001" in business_context_section(prompt)
    assert "Heatbox" in business_context_section(prompt)
    assert "AI SUPPORT ANALYSIS" not in business_context_section(prompt)
    assert customer_message not in business_context_section(prompt)


def test_german_message_with_english_shopify_context_detects_de() -> None:
    customer_message = "Hallo, meine Bestellung #2002 ist noch nicht angekommen."
    provider = make_provider(
        analysis_content(
            customer_language="de",
            summary="Customer says order #2002 has not arrived yet.",
            suggested_reply="Hallo, wir prüfen Bestellung #2002 und melden uns in Kürze.",
        )
    )

    analysis = provider.analyze(
        {
            "latest_customer_message": customer_message,
            "ticket": {"order": {"name": "Order #2002", "product_name": "Heatbox"}},
        }
    )

    assert analysis.customer_language == "de"
    assert analysis.summary == "Customer says order #2002 has not arrived yet."
    assert analysis.suggested_reply.startswith("Hallo")
    assert "#2002" in analysis.suggested_reply


def test_french_message_with_english_context_detects_fr() -> None:
    customer_message = "Bonjour, ma commande #3003 n'est toujours pas arrivée."
    provider = make_provider(
        analysis_content(
            customer_language="fr",
            summary="Customer says order #3003 still has not arrived.",
            suggested_reply="Bonjour, nous allons vérifier la commande #3003.",
        )
    )

    analysis = provider.analyze(
        {
            "latest_customer_message": customer_message,
            "ticket": {"order": {"name": "Order #3003", "product_name": "Heatbox"}},
        }
    )

    assert analysis.customer_language == "fr"
    assert analysis.summary == "Customer says order #3003 still has not arrived."
    assert analysis.suggested_reply.startswith("Bonjour")
    assert "#3003" in analysis.suggested_reply


def test_japanese_message_with_english_context_detects_ja() -> None:
    customer_message = "こんにちは、注文 #4004 はまだ届いていません。"
    provider = make_provider(
        analysis_content(
            customer_language="ja",
            summary="Customer says order #4004 has not arrived yet.",
            suggested_reply="こんにちは、注文 #4004 を確認して折り返しご連絡します。",
        )
    )

    analysis = provider.analyze(
        {
            "latest_customer_message": customer_message,
            "ticket": {"order": {"name": "Order #4004", "product_name": "Heatbox"}},
        }
    )

    assert analysis.customer_language == "ja"
    assert analysis.summary == "Customer says order #4004 has not arrived yet."
    assert analysis.suggested_reply.startswith("こんにちは")
    assert "#4004" in analysis.suggested_reply


def test_english_message_with_english_context_detects_en() -> None:
    customer_message = "Hi, can you check where order #5005 is?"
    provider = make_provider(
        analysis_content(
            customer_language="en",
            summary="Customer asks where order #5005 is.",
            suggested_reply="Hi, we will check order #5005 and follow up shortly.",
        )
    )

    analysis = provider.analyze(
        {
            "latest_customer_message": customer_message,
            "ticket": {"order": {"name": "Order #5005", "product_name": "Heatbox"}},
        }
    )

    assert analysis.customer_language == "en"
    assert analysis.summary == "Customer asks where order #5005 is."
    assert analysis.suggested_reply.startswith("Hi")
    assert "#5005" in analysis.suggested_reply


def test_latest_inbound_customer_message_is_selected_by_timestamp() -> None:
    serbian_message = "Zdravo, moja porudžbina #1001 još nije stigla."
    ticket = {
        "subject": "English subject should not be selected",
        "messages": [
            {
                "body_text": serbian_message,
                "public": True,
                "sender": {"type": "customer"},
                "created_datetime": "2026-08-25T10:00:00Z",
            },
            {
                "body_text": "Hallo, meine Bestellung ist noch nicht angekommen.",
                "public": True,
                "sender": {"type": "customer"},
                "created_datetime": "2026-08-24T10:00:00Z",
            },
            {
                "body_text": "AI SUPPORT ANALYSIS\nEnglish internal note.",
                "public": False,
                "sender": {"type": "agent"},
                "created_datetime": "2026-08-25T11:00:00Z",
            },
            {
                "body_text": "Agent reply in English.",
                "public": True,
                "from_agent": True,
                "sender": {"type": "agent"},
                "created_datetime": "2026-08-25T12:00:00Z",
            },
        ],
    }

    assert extract_latest_customer_message(ticket, {}) == serbian_message
