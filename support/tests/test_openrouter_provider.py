from __future__ import annotations

import logging

from support.schemas import SupportAnalysis
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
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("No fake OpenRouter response queued.")
        return self.responses.pop(0)


def valid_analysis_payload() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"customer_language":"en","intent":"order_status",'
                        '"sentiment":"neutral",'
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
        assert len(session.calls) == 1
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


def test_openrouter_requests_strict_json_schema_response_format() -> None:
    session = FakeSession(FakeResponse(valid_analysis_payload()))
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=1,
        session=session,
    )

    provider.analyze({"latest_customer_message": "Where is my order?"})

    request_json = session.calls[0]["kwargs"]["json"]
    response_format = request_json["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "support_analysis"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == SupportAnalysis.model_json_schema()
    assert request_json["provider"]["require_parameters"] is True


def test_openrouter_prompt_instructs_multilingual_output() -> None:
    session = FakeSession(FakeResponse(valid_analysis_payload()))
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
    )

    provider.analyze({"latest_customer_message": "Zdravo, gde je porudzbina #1001?"})

    system_prompt = session.calls[0]["kwargs"]["json"]["messages"][0]["content"]
    assert "customer_language" in system_prompt
    assert "Always write summary in English" in system_prompt
    assert "Always write suggested_reply in" in system_prompt
    assert "Do not translate or alter" in system_prompt
    assert "order IDs" in system_prompt
    assert "tracking numbers" in system_prompt
    assert "URLs" in system_prompt


def test_openrouter_valid_structured_json_parses_successfully() -> None:
    session = FakeSession(FakeResponse(valid_analysis_payload()))
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
    )

    analysis = provider.analyze({"latest_customer_message": "Where is my order?"})

    assert analysis.intent == "order_status"
    assert analysis.customer_language == "en"
    assert analysis.confidence == 0.91


def test_openrouter_validation_error_names_failed_fields_only() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"intent":"order_status","summary":"Customer asks for tracking."}'
                            ),
                        }
                    }
                ]
            }
        )
    )
    provider = OpenRouterProvider(
        api_key="test-secret-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
    )

    try:
        provider.analyze({"latest_customer_message": "customer@example.com asks about order 123"})
    except AIProviderError as exc:
        message = str(exc)
        assert "Failed fields:" in message
        assert "sentiment" in message
        assert "suggested_reply" in message
        assert "customer@example.com" not in message
        assert "test-secret-key" not in message
        assert len(session.calls) == 1
    else:
        raise AssertionError("Incomplete structured output should fail validation.")


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
        assert len(session.calls) == 1
    else:
        raise AssertionError("OpenRouter 404 should raise a provider error.")


def test_openrouter_503_then_200_succeeds(caplog) -> None:
    sleeps = []
    caplog.set_level(logging.WARNING)
    session = FakeSession(
        FakeResponse(
            {"error": {"message": "Provider returned error"}},
            status_code=503,
        ),
        FakeResponse(valid_analysis_payload()),
    )
    provider = OpenRouterProvider(
        api_key="test-secret-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
        sleep_func=sleeps.append,
        jitter_func=lambda start, end: 0.0,
    )

    analysis = provider.analyze({"latest_customer_message": "Where is my order?"})

    assert analysis.intent == "order_status"
    assert len(session.calls) == 2
    assert sleeps == [1.0]
    assert "OpenRouter transient failure 503, retry 1/2" in caplog.text
    assert "test-secret-key" not in caplog.text


def test_openrouter_502_then_200_succeeds() -> None:
    session = FakeSession(
        FakeResponse(
            {"error": {"message": "Bad gateway"}},
            status_code=502,
        ),
        FakeResponse(valid_analysis_payload()),
    )
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
        sleep_func=lambda seconds: None,
        jitter_func=lambda start, end: 0.0,
    )

    analysis = provider.analyze({"latest_customer_message": "Where is my order?"})

    assert analysis.customer_language == "en"
    assert len(session.calls) == 2


def test_openrouter_429_retries_until_success() -> None:
    sleeps = []
    session = FakeSession(
        FakeResponse({"error": {"message": "Rate limited"}}, status_code=429),
        FakeResponse({"error": {"message": "Rate limited"}}, status_code=429),
        FakeResponse(valid_analysis_payload()),
    )
    provider = OpenRouterProvider(
        api_key="test-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
        sleep_func=sleeps.append,
        jitter_func=lambda start, end: 0.25,
    )

    analysis = provider.analyze({"latest_customer_message": "Where is my order?"})

    assert analysis.intent == "order_status"
    assert len(session.calls) == 3
    assert sleeps == [1.25, 2.25]


def test_openrouter_repeated_503_raises_after_three_attempts() -> None:
    sleeps = []
    session = FakeSession(
        FakeResponse(
            {"error": {"message": "Provider returned error"}},
            status_code=503,
        ),
        FakeResponse(
            {"error": {"message": "Provider returned error"}},
            status_code=503,
        ),
        FakeResponse(
            {"error": {"message": "Provider returned error"}},
            status_code=503,
        ),
    )
    provider = OpenRouterProvider(
        api_key="test-secret-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
        sleep_func=sleeps.append,
        jitter_func=lambda start, end: 0.0,
    )

    try:
        provider.analyze({"latest_customer_message": "Where is my order?"})
    except AIProviderError as exc:
        message = str(exc)
        assert "OpenRouter returned HTTP 503" in message
        assert "Provider returned error" in message
        assert "test-secret-key" not in message
    else:
        raise AssertionError("Repeated transient failures should raise after retries.")

    assert len(session.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_openrouter_401_is_not_retried() -> None:
    sleeps = []
    session = FakeSession(
        FakeResponse(
            {"error": {"message": "Unauthorized"}},
            status_code=401,
        )
    )
    provider = OpenRouterProvider(
        api_key="test-secret-key",
        model="openrouter/free",
        timeout_seconds=1,
        session=session,
        sleep_func=sleeps.append,
    )

    try:
        provider.analyze({"latest_customer_message": "Where is my order?"})
    except AIProviderError as exc:
        assert "OpenRouter returned HTTP 401" in str(exc)
    else:
        raise AssertionError("401 should fail without retry.")

    assert len(session.calls) == 1
    assert sleeps == []
