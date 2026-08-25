from __future__ import annotations

from typing import Any

from support.services.gorgias import GorgiasClient


class FakeResponse:
    status_code = 200
    content = b"{}"

    def __init__(self, payload: Any):
        self.payload = payload

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
