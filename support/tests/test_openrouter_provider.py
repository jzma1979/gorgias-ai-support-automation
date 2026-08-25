from __future__ import annotations

from support.services.ai.base import AIProviderError
from support.services.ai.openrouter import OpenRouterProvider


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.content = b"{}"

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


def valid_analysis_payload() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"intent":"order_status","sentiment":"neutral",'
                        '"urgency":"low","risk":"low","confidence":0.91,'
                        '"summary":"Customer asks for tracking.",'
                        '"suggested_reply":"Thanks, we will check the tracking.",'
                        '"recommended_action":"agent_review",'
                        '"reasoning_summary":"Low-risk order status request."}'
                    ),
                }
            }
        ]
    }


def test_malformed_ai_output_raises_provider_error() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"intent": "order_status"}',
                        }
                    }
                ]
            }
        )
    )
    provider = OpenRouterProvider(
        api_key="test-key",
        model="test/free-model:free",
        timeout_seconds=1,
        session=session,
    )

    try:
        provider.analyze({"latest_customer_message": "Where is my order?"})
    except AIProviderError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("Malformed AI output should fail validation.")


def test_openrouter_free_router_is_accepted() -> None:
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=FakeSession(FakeResponse(valid_analysis_payload())),
    )

    assert provider.model == "openrouter/free"


def test_openrouter_explicit_free_variant_is_accepted() -> None:
    provider = OpenRouterProvider(
        api_key="test-key",
        model="provider/model-name:free",
        timeout_seconds=1,
        session=FakeSession(FakeResponse(valid_analysis_payload())),
    )

    assert provider.model == "provider/model-name:free"


def test_openrouter_rejects_non_free_model() -> None:
    try:
        OpenRouterProvider(api_key="test-key", model="paid/model")
    except AIProviderError as exc:
        assert "free model" in str(exc)
    else:
        raise AssertionError("Paid OpenRouter model should not be accepted.")


def test_openrouter_calls_chat_completions_endpoint() -> None:
    session = FakeSession(FakeResponse(valid_analysis_payload()))
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=1,
        session=session,
    )

    provider.analyze({"latest_customer_message": "Where is my order?"})

    assert session.calls[0]["args"][0] == "https://openrouter.ai/api/v1/chat/completions"


def test_openrouter_404_keeps_safe_error_information() -> None:
    session = FakeSession(
        FakeResponse(
            {"error": {"message": "No endpoints found for openrouter/free"}},
            status_code=404,
        )
    )
    provider = OpenRouterProvider(
        api_key="test-secret-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
    )

    try:
        provider.analyze({"latest_customer_message": "Where is my order?"})
    except AIProviderError as exc:
        message = str(exc)
        assert "OpenRouter returned HTTP 404" in message
        assert "No endpoints found for openrouter/free" in message
        assert "test-secret-key" not in message
        assert "Authorization" not in message
    else:
        raise AssertionError("OpenRouter 404 should raise a provider error.")
