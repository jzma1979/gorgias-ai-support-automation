from __future__ import annotations

from support.services.ai.base import AIProviderError
from support.services.ai.openrouter import OpenRouterProvider


class FakeResponse:
    status_code = 200
    content = b"{}"

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"intent": "order_status"}',
                    }
                }
            ]
        }


class FakeSession:
    def post(self, *args, **kwargs):
        return FakeResponse()


def test_malformed_ai_output_raises_provider_error() -> None:
    provider = OpenRouterProvider(
        api_key="test-key",
        model="test/free-model:free",
        timeout_seconds=1,
        session=FakeSession(),
    )

    try:
        provider.analyze({"latest_customer_message": "Where is my order?"})
    except AIProviderError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("Malformed AI output should fail validation.")


def test_openrouter_rejects_non_free_model() -> None:
    try:
        OpenRouterProvider(api_key="test-key", model="paid/model")
    except AIProviderError as exc:
        assert "free model" in str(exc)
    else:
        raise AssertionError("Paid OpenRouter model should not be accepted.")
