from __future__ import annotations

from typing import Any

from support.services.gorgias import GorgiasAPIError, GorgiasClient


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200, text: str = ""):
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.content = b"{}"

    def json(self) -> Any:
        return self.payload


class QueueSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("No fake response queued.")
        return self.responses.pop(0)


def make_client(session: QueueSession) -> GorgiasClient:
    return GorgiasClient(
        base_url="https://example.gorgias.com",
        username="user",
        api_key="key",
        timeout_seconds=1,
        session=session,
    )


def test_tag_search_uses_search_never_name() -> None:
    session = QueueSession(FakeResponse([{"id": 123, "name": "AI_ORDER_STATUS"}]))
    client = make_client(session)

    assert client.resolve_or_create_tag("AI_ORDER_STATUS") == 123

    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"] == "https://example.gorgias.com/api/tags"
    assert session.calls[0]["params"] == {"search": "AI_ORDER_STATUS"}
    assert "name" not in session.calls[0]["params"]


def test_list_style_gorgias_tag_response_is_handled() -> None:
    session = QueueSession(FakeResponse([{"id": 123, "name": "AI_LOW_RISK"}]))
    client = make_client(session)

    assert client.resolve_or_create_tag("AI_LOW_RISK") == 123
    assert len(session.calls) == 1


def test_paginated_dict_gorgias_tag_response_is_handled() -> None:
    session = QueueSession(
        FakeResponse(
            {
                "data": [{"id": 456, "name": "AI_ESCALATED"}],
                "meta": {"next_cursor": None},
            }
        )
    )
    client = make_client(session)

    assert client.resolve_or_create_tag("ai_escalated") == 456
    assert len(session.calls) == 1


def test_missing_tag_is_created() -> None:
    session = QueueSession(
        FakeResponse({"data": []}),
        FakeResponse({"id": 777, "name": "AI_NEW_TAG"}),
    )
    client = make_client(session)

    assert client.resolve_or_create_tag("AI_NEW_TAG") == 777

    assert session.calls[0]["params"] == {"search": "AI_NEW_TAG"}
    assert session.calls[1]["method"] == "POST"
    assert session.calls[1]["url"] == "https://example.gorgias.com/api/tags"
    assert session.calls[1]["json"] == {"name": "AI_NEW_TAG"}


def test_existing_tag_is_reused() -> None:
    session = QueueSession(
        FakeResponse({"data": [{"id": 888, "name": "AI_REVIEW_REQUIRED"}]})
    )
    client = make_client(session)

    assert client.resolve_or_create_tag("AI_REVIEW_REQUIRED") == 888

    assert len(session.calls) == 1
    assert session.calls[0]["method"] == "GET"


def test_ticket_tags_are_added_with_ids_payload() -> None:
    session = QueueSession(
        FakeResponse({"data": [{"id": 123, "name": "AI_ORDER_STATUS"}]}),
        FakeResponse({"data": [{"id": 456, "name": "AI_LOW_RISK"}]}),
        FakeResponse({"id": 99}),
    )
    client = make_client(session)

    client.add_tags_to_ticket("99", ["AI_ORDER_STATUS", "AI_LOW_RISK"])

    assert session.calls[2]["method"] == "POST"
    assert session.calls[2]["url"] == "https://example.gorgias.com/api/tickets/99/tags"
    assert session.calls[2]["json"] == {"ids": [123, 456]}


def test_internal_note_uses_current_gorgias_contract() -> None:
    session = QueueSession(FakeResponse({"id": "message-1"}))
    client = make_client(session)

    client.create_internal_note("99", "AI note <b>review</b>\nNext line")

    payload = session.calls[0]["json"]
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"] == "https://example.gorgias.com/api/tickets/99/messages"
    assert payload["sender"] == {"email": "user"}
    assert payload["channel"] == "internal-note"
    assert payload["from_agent"] is True
    assert payload["via"] == "api"
    assert "receiver" not in payload
    assert "source" not in payload
    assert "public" not in payload
    assert payload["body_text"] == "AI note <b>review</b>\nNext line"
    assert payload["body_html"] == "AI note &lt;b&gt;review&lt;/b&gt;<br>Next line"


def test_gorgias_400_keeps_safe_error_information() -> None:
    session = QueueSession(
        FakeResponse(
            {
                "error": {
                    "message": (
                        "Invalid internal note sender user@example.com "
                        "abcdefghijklmnopqrstuvwxyz123456"
                    )
                }
            },
            status_code=400,
        )
    )
    client = make_client(session)

    try:
        client.create_internal_note("99", "AI note")
    except GorgiasAPIError as exc:
        message = str(exc)
        assert "Gorgias POST /api/tickets/99/messages returned HTTP 400" in message
        assert "Invalid internal note sender" in message
        assert "user@example.com" not in message
        assert "abcdefghijklmnopqrstuvwxyz123456" not in message
        assert "key" not in message
    else:
        raise AssertionError("Gorgias 400 should raise an API error.")
