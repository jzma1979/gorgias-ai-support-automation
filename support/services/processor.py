from __future__ import annotations

import html
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from django.conf import settings
from pydantic import ValidationError

from support.schemas import SupportAnalysis, fallback_analysis
from support.services.ai.base import AIProvider, AIProviderError
from support.services.ai.factory import get_ai_provider
from support.services.decision_engine import (
    SupportDecision,
    evaluate_support_decision,
)
from support.services.gorgias import GorgiasAPIError, GorgiasClient

logger = logging.getLogger(__name__)
TAG_RE = re.compile(r"<[^>]+>")
EVENT_MESSAGE_MARKER = "_event_payload_message"
SUPPORTED_REPLY_LANGUAGES = {"en", "sr", "de", "fr", "es", "ja"}
SHIPPING_DELAY_UNSAFE_REPLY_TERMS = (
    "has shipped",
    "have shipped",
    "is shipped",
    "was shipped",
    "tracking is available",
    "out for delivery",
)
PRODUCT_DEFECT_UNSAFE_REPLY_TERMS = (
    "replacement immediately",
    "replacement right away",
    "send a replacement",
    "issue a replacement",
    "refund immediately",
    "refund right away",
    "issue a refund",
    "replacement is approved",
    "refund is approved",
    "we will replace",
    "we'll replace",
    "we will refund",
    "we'll refund",
)
ENGLISH_CANNED_REPLY_TERMS = (
    "thanks for reaching out",
    "thank you for reaching out",
    "we will",
    "we'll",
    "i am checking",
    "i'm checking",
    "please share",
    "our team",
)
LOCALIZED_GUARDRAIL_REPLIES = {
    "safety": {
        "en": (
            "Do not send an automated reply. A human agent should review immediately, "
            "acknowledge the safety concern, and follow the approved escalation process."
        ),
        "sr": (
            "Ne šaljite automatski odgovor. Ljudski agent treba odmah da pregleda "
            "slučaj, potvrdi bezbednosnu zabrinutost i prati odobreni postupak eskalacije."
        ),
        "de": (
            "Senden Sie keine automatische Antwort. Ein menschlicher Mitarbeiter sollte "
            "den Fall sofort prüfen, das Sicherheitsanliegen bestätigen und den "
            "freigegebenen Eskalationsprozess befolgen."
        ),
        "fr": (
            "N'envoyez pas de réponse automatique. Un agent humain doit examiner le "
            "dossier immédiatement, reconnaître le problème de sécurité et suivre la "
            "procédure d'escalade approuvée."
        ),
        "es": (
            "No envíe una respuesta automática. Un agente humano debe revisar el caso "
            "de inmediato, reconocer la preocupación de seguridad y seguir el proceso "
            "de escalamiento aprobado."
        ),
        "ja": (
            "自動返信は送信しないでください。担当者が直ちに内容を確認し、安全上の懸念を"
            "受け止めたうえで、承認済みのエスカレーション手順に従う必要があります。"
        ),
    },
    "shipping_delay": {
        "en": (
            "Thanks for reaching out. I am checking the fulfillment status for your "
            "order and will confirm the next step after reviewing the shipment details."
        ),
        "sr": (
            "Hvala što ste nam se javili. Proveravam status ispunjenja vaše porudžbine "
            "i potvrdiću sledeći korak nakon pregleda detalja isporuke."
        ),
        "de": (
            "Danke für Ihre Nachricht. Ich prüfe den Fulfillment-Status Ihrer Bestellung "
            "und bestätige den nächsten Schritt, sobald ich die Versanddetails geprüft habe."
        ),
        "fr": (
            "Merci de nous avoir contactés. Je vérifie le statut de préparation de votre "
            "commande et confirmerai la prochaine étape après examen des détails d'expédition."
        ),
        "es": (
            "Gracias por contactarnos. Estoy revisando el estado de preparación de su "
            "pedido y confirmaré el siguiente paso después de revisar los detalles del envío."
        ),
        "ja": (
            "お問い合わせありがとうございます。ご注文の処理状況を確認し、配送情報を確認した"
            "うえで次の対応をご案内します。"
        ),
    },
    "product_defect": {
        "en": (
            "I am sorry the product is not working as expected. Please share the order "
            "number, a brief photo or video of the issue, and whether the basic charging "
            "and reset steps have already been tried so our team can review the warranty case."
        ),
        "sr": (
            "Žao mi je što proizvod ne radi kako se očekuje. Pošaljite broj porudžbine, "
            "kratku fotografiju ili video problema i navedite da li ste već probali "
            "osnovno punjenje i resetovanje, kako bi naš tim pregledao garancijski slučaj."
        ),
        "de": (
            "Es tut mir leid, dass das Produkt nicht wie erwartet funktioniert. Bitte "
            "teilen Sie die Bestellnummer, ein kurzes Foto oder Video des Problems und "
            "ob die grundlegenden Lade- und Reset-Schritte bereits ausprobiert wurden, "
            "damit unser Team den Garantiefall prüfen kann."
        ),
        "fr": (
            "Je suis désolé que le produit ne fonctionne pas comme prévu. Veuillez "
            "indiquer le numéro de commande, envoyer une courte photo ou vidéo du "
            "problème et préciser si les étapes de charge et de réinitialisation de "
            "base ont déjà été essayées afin que notre équipe examine le dossier de garantie."
        ),
        "es": (
            "Lamento que el producto no funcione como se esperaba. Envíe el número de "
            "pedido, una foto o video breve del problema e indique si ya probó los pasos "
            "básicos de carga y reinicio para que nuestro equipo revise el caso de garantía."
        ),
        "ja": (
            "製品が想定どおりに動作していないとのことで申し訳ありません。保証確認のため、"
            "注文番号、問題が分かる短い写真または動画、基本的な充電やリセットをすでに"
            "お試しいただいたかをお知らせください。"
        ),
    },
}


@dataclass(frozen=True)
class ProcessingResult:
    ticket_id: str
    analysis: SupportAnalysis
    decision: SupportDecision
    internal_note: str


@dataclass(frozen=True)
class SuggestedReplies:
    localized: str
    english: str


class SupportTicketProcessor:
    def __init__(
        self,
        gorgias_client: GorgiasClient | None = None,
        ai_provider: AIProvider | None = None,
    ) -> None:
        self.gorgias_client = gorgias_client or GorgiasClient.from_settings()
        self.ai_provider = ai_provider

    def process(
        self,
        ticket_id: int | str,
        event_payload: Mapping[str, Any] | None = None,
    ) -> ProcessingResult:
        ticket_id_text = str(ticket_id)
        ticket = self.gorgias_client.get_ticket(ticket_id_text)
        customer = self._fetch_customer(ticket)
        latest_message = extract_latest_customer_message(ticket, event_payload or {})

        ai_context = {
            "ticket": ticket,
            "customer": customer,
            "latest_customer_message": latest_message,
        }
        analysis = self._safe_analyze(ai_context)
        decision = evaluate_support_decision(
            analysis,
            {"ticket": ticket, "customer": customer},
            latest_message,
        )
        internal_note = build_internal_note(analysis, decision)

        logger.info(
            "Applying support decision for Gorgias ticket %s: priority=%s action=%s",
            ticket_id_text,
            decision.priority,
            decision.recommended_action,
        )
        self.gorgias_client.update_ticket_priority(ticket_id_text, decision.priority)
        self.gorgias_client.add_tags_to_ticket(ticket_id_text, decision.tags)
        self.gorgias_client.create_internal_note(ticket_id_text, internal_note)

        return ProcessingResult(
            ticket_id=ticket_id_text,
            analysis=analysis,
            decision=decision,
            internal_note=internal_note,
        )

    def _fetch_customer(self, ticket: Mapping[str, Any]) -> Mapping[str, Any]:
        embedded_customer = ticket.get("customer")
        if not isinstance(embedded_customer, Mapping):
            return {}

        customer_id = embedded_customer.get("id")
        if not customer_id:
            return embedded_customer

        try:
            return self.gorgias_client.get_customer(customer_id)
        except GorgiasAPIError:
            logger.warning(
                "Could not fetch customer context for Gorgias customer id %s.",
                customer_id,
            )
            return embedded_customer

    def _safe_analyze(self, ai_context: Mapping[str, Any]) -> SupportAnalysis:
        try:
            provider = self.ai_provider or get_ai_provider()
        except AIProviderError as exc:
            logger.warning("AI provider unavailable: %s", exc)
            return fallback_analysis(str(exc))

        try:
            return provider.analyze(ai_context)
        except (AIProviderError, ValidationError) as exc:
            logger.warning("AI analysis failed validation: %s", exc)
            return fallback_analysis(str(exc))
        except Exception as exc:
            logger.warning("AI analysis failed unexpectedly: %s", exc)
            return fallback_analysis("AI provider failed unexpectedly.")


def build_internal_note(analysis: SupportAnalysis, decision: SupportDecision) -> str:
    suggested_replies = _safe_suggested_replies(analysis, decision)
    action_label = {
        "auto_resolve": "Human review required before any customer-facing action.",
        "agent_review": "Human review required.",
        "escalate": "Escalate to a human agent.",
    }.get(decision.recommended_action, "Human review required.")

    return (
        "AI SUPPORT ANALYSIS\n\n"
        f"Customer language: {analysis.customer_language}\n"
        f"Intent: {analysis.intent}\n"
        f"Sentiment: {analysis.sentiment}\n"
        f"Urgency: {analysis.urgency}\n"
        f"Risk: {analysis.risk}\n"
        f"Confidence: {analysis.confidence:.2f}\n\n"
        "Summary:\n"
        f"{analysis.summary}\n\n"
        "Recommended action:\n"
        f"{action_label}\n\n"
        "Suggested reply — customer language:\n"
        f"{suggested_replies.localized}\n\n"
        "Suggested reply — English:\n"
        f"{suggested_replies.english}\n\n"
        "Decision:\n"
        f"{decision.decision_reason}\n"
        "No customer-facing message was sent by this automation.\n\n"
        "Generated by demo AI automation."
    )


def extract_latest_customer_message(
    ticket: Mapping[str, Any],
    event_payload: Mapping[str, Any],
) -> str:
    messages = _extract_messages(ticket)
    event_message = _extract_event_message(event_payload)
    if event_message:
        messages.append({EVENT_MESSAGE_MARKER: True, **event_message})

    candidates: list[tuple[Mapping[str, Any], str, int]] = []
    for index, message in enumerate(messages):
        if _is_agent_or_internal_message(message):
            continue
        body = _extract_message_body(message)
        if body:
            candidates.append((message, body, index))

    if not candidates:
        return ""

    _, latest_body, _ = max(
        candidates,
        key=lambda item: _message_sort_key(item[0], item[2]),
    )
    return latest_body


def _extract_messages(ticket: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_messages = ticket.get("messages", [])
    if isinstance(raw_messages, Mapping):
        for key in ("data", "results", "messages"):
            nested = raw_messages.get(key)
            if isinstance(nested, list):
                raw_messages = nested
                break

    if not isinstance(raw_messages, list):
        return []
    return [message for message in raw_messages if isinstance(message, Mapping)]


def _extract_event_message(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("message", "comment", "data", "object"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            if any(body_key in value for body_key in ("body_text", "body_html", "body", "text")):
                return value
            nested = _extract_event_message(value)
            if nested:
                return nested
    return None


def _extract_message_body(message: Mapping[str, Any]) -> str:
    for key in ("body_text", "stripped_text", "text", "body", "body_html", "message"):
        value = message.get(key)
        if isinstance(value, str):
            clean = TAG_RE.sub(" ", value)
            clean = html.unescape(clean)
            clean = " ".join(clean.split())
            if clean:
                return clean
    return ""


def _message_sort_key(message: Mapping[str, Any], index: int) -> tuple[int, float, int]:
    timestamp = _message_timestamp(message)
    if message.get(EVENT_MESSAGE_MARKER) is True and timestamp is None:
        return (2, 0.0, index)
    if timestamp is not None:
        return (1, timestamp, index)
    return (0, 0.0, index)


def _message_timestamp(message: Mapping[str, Any]) -> float | None:
    for key in (
        "created_datetime",
        "created_at",
        "createdAt",
        "date",
        "sent_datetime",
        "timestamp",
    ):
        value = message.get(key)
        timestamp = _parse_timestamp(value)
        if timestamp is not None:
            return timestamp
    return None


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _is_agent_or_internal_message(message: Mapping[str, Any]) -> bool:
    if message.get("public") is False:
        return True
    if message.get("from_agent") is True or message.get("from_self") is True:
        return True
    if _extract_message_body(message).lower().find("generated by demo ai automation") != -1:
        return True

    sender = message.get("sender") or message.get("author") or {}
    if isinstance(sender, Mapping):
        sender_type = str(sender.get("type", "")).lower()
        if sender_type in {"agent", "admin", "integration", "system"}:
            return True
        configured_email = settings.GORGIAS_INTEGRATION_EMAIL.strip().lower()
        if configured_email and str(sender.get("email", "")).strip().lower() == configured_email:
            return True
        configured_user_id = settings.GORGIAS_INTEGRATION_USER_ID.strip()
        if configured_user_id and str(sender.get("id", "")).strip() == configured_user_id:
            return True
    return False


def _safe_suggested_replies(
    analysis: SupportAnalysis,
    decision: SupportDecision,
) -> SuggestedReplies:
    suggested_localized = analysis.suggested_reply_localized.strip()
    suggested_english = analysis.suggested_reply_en.strip()
    customer_language = _reply_language(analysis.customer_language)

    if "AI_SAFETY" in decision.tags:
        return _localized_guardrail_replies("safety", customer_language)
    if "AI_SHIPPING_DELAY" in decision.tags and _requires_guardrail_pair(
        suggested_localized,
        suggested_english,
        customer_language,
        SHIPPING_DELAY_UNSAFE_REPLY_TERMS,
    ):
        return _localized_guardrail_replies("shipping_delay", customer_language)
    if "AI_PRODUCT_DEFECT" in decision.tags and _requires_guardrail_pair(
        suggested_localized,
        suggested_english,
        customer_language,
        PRODUCT_DEFECT_UNSAFE_REPLY_TERMS,
    ):
        return _localized_guardrail_replies("product_defect", customer_language)
    return SuggestedReplies(localized=suggested_localized, english=suggested_english)


def _reply_language(customer_language: str) -> str:
    language = customer_language.strip().lower().replace("_", "-").split("-", 1)[0]
    if language in SUPPORTED_REPLY_LANGUAGES:
        return language
    return "en"


def _localized_guardrail_replies(
    guardrail: str,
    customer_language: str,
) -> SuggestedReplies:
    replies = LOCALIZED_GUARDRAIL_REPLIES[guardrail]
    return SuggestedReplies(
        localized=replies.get(customer_language, replies["en"]),
        english=replies["en"],
    )


def _requires_guardrail_pair(
    suggested_localized: str,
    suggested_english: str,
    customer_language: str,
    unsafe_terms: tuple[str, ...],
) -> bool:
    return _requires_localized_guardrail(
        suggested_localized,
        customer_language,
        unsafe_terms,
    ) or _requires_localized_guardrail(
        suggested_english,
        "en",
        unsafe_terms,
    )


def _requires_localized_guardrail(
    suggested_reply: str,
    customer_language: str,
    unsafe_terms: tuple[str, ...],
) -> bool:
    if _contains_any_casefold(suggested_reply, unsafe_terms):
        return True
    if customer_language != "en" and _contains_any_casefold(
        suggested_reply,
        ENGLISH_CANNED_REPLY_TERMS,
    ):
        return True
    return False


def _contains_any_casefold(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in terms)
