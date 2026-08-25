from __future__ import annotations

from concurrent.futures import Future
import json
import logging
import threading
import time
from types import SimpleNamespace


def test_health_endpoint(client) -> None:
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def completed_future() -> Future:
    future = Future()
    future.set_result(None)
    return future


def test_valid_webhook_returns_202_and_accepted_payload(client, monkeypatch) -> None:
    dispatched = []

    def fake_enqueue(**kwargs):
        dispatched.append(kwargs)
        return completed_future()

    monkeypatch.setattr("support.views.enqueue_ticket_processing", fake_enqueue)

    response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "ticket-created-82455839", "ticket_id": "82455839"}),
        content_type="application/json",
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "ticket_id": "82455839"}
    assert len(dispatched) == 1
    assert dispatched[0]["ticket_id"] == "82455839"
    assert dispatched[0]["event_id"] == "ticket-created-82455839"


def test_webhook_dispatches_processing_asynchronously(client, monkeypatch) -> None:
    class FakeExecutor:
        def __init__(self):
            self.submitted = None

        def submit(self, fn, *args):
            self.submitted = (fn, args)
            return completed_future()

    class ExplodingProcessor:
        def process(self, ticket_id, payload):
            raise AssertionError("processing must not run inline")

    fake_executor = FakeExecutor()
    monkeypatch.setattr("support.views.WEBHOOK_EXECUTOR", fake_executor)
    monkeypatch.setattr("support.views.SupportTicketProcessor", ExplodingProcessor)

    response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "evt-async", "ticket_id": 12345}),
        content_type="application/json",
    )

    assert response.status_code == 202
    submitted_fn, args = fake_executor.submitted
    assert submitted_fn.__name__ == "_process_ticket_background"
    assert args[0] == "12345"
    assert args[1]["ticket_id"] == 12345
    assert args[2] == "evt-async"


def test_slow_processing_does_not_block_webhook_response(client, monkeypatch) -> None:
    release = threading.Event()
    finished = threading.Event()

    class SlowProcessor:
        def process(self, ticket_id, payload):
            release.wait(timeout=2)
            finished.set()
            return SimpleNamespace(
                decision=SimpleNamespace(
                    priority="low",
                    recommended_action="agent_review",
                ),
            )

    monkeypatch.setattr("support.views.SupportTicketProcessor", SlowProcessor)

    started_at = time.perf_counter()
    response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "evt-slow", "ticket_id": 12345}),
        content_type="application/json",
    )
    elapsed = time.perf_counter() - started_at

    try:
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        assert elapsed < 0.5
        assert not finished.is_set()
    finally:
        release.set()

    assert finished.wait(timeout=2)


def test_malformed_webhook_returns_400(client) -> None:
    response = client.post(
        "/api/webhooks/gorgias/",
        data="{not-json",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid JSON payload."


def test_missing_ticket_id_returns_400(client, monkeypatch) -> None:
    def exploding_enqueue(**kwargs):
        raise AssertionError("missing ticket id should not dispatch processing")

    monkeypatch.setattr("support.views.enqueue_ticket_processing", exploding_enqueue)

    response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "evt-missing-ticket"}),
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Could not extract a Gorgias ticket ID."


def test_duplicate_event_does_not_dispatch_duplicate_work(client, monkeypatch) -> None:
    dispatched = []

    def fake_enqueue(**kwargs):
        dispatched.append(kwargs)
        return completed_future()

    monkeypatch.setattr("support.views.enqueue_ticket_processing", fake_enqueue)

    first_response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "evt-duplicate", "ticket_id": 987}),
        content_type="application/json",
    )
    second_response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "evt-duplicate", "ticket_id": 987}),
        content_type="application/json",
    )

    assert first_response.status_code == 202
    assert first_response.json()["status"] == "accepted"
    assert second_response.status_code == 200
    assert second_response.json()["status"] == "ignored"
    assert second_response.json()["reason"] == "duplicate_event"
    assert len(dispatched) == 1


def test_processing_exception_does_not_change_accepted_response(
    client,
    monkeypatch,
    caplog,
) -> None:
    class InlineExecutor:
        def submit(self, fn, *args):
            fn(*args)
            return completed_future()

    class FailingProcessor:
        def process(self, ticket_id, payload):
            raise RuntimeError("upstream failed with sk-secretvalue1234567890")

    monkeypatch.setattr("support.views.WEBHOOK_EXECUTOR", InlineExecutor())
    monkeypatch.setattr("support.views.SupportTicketProcessor", FailingProcessor)
    caplog.set_level(logging.WARNING)

    response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps({"event_id": "evt-failure", "ticket_id": 987}),
        content_type="application/json",
    )

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "ticket_id": "987"}
    log_text = caplog.text
    assert "Background Gorgias processing failed" in log_text
    assert "evt-failure" in log_text
    assert "987" in log_text
    assert "sk-secretvalue1234567890" not in log_text


def test_webhook_loop_prevention_ignores_integration_events(client, monkeypatch) -> None:
    class ExplodingProcessor:
        def process(self, ticket_id, payload):
            raise AssertionError("processor should not be called")

    monkeypatch.setattr("support.views.SupportTicketProcessor", ExplodingProcessor)

    response = client.post(
        "/api/webhooks/gorgias/",
        data=json.dumps(
            {
                "ticket_id": 222,
                "message": {
                    "from_agent": True,
                    "body_text": "Generated by demo AI automation.",
                },
            }
        ),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert response.json()["reason"] == "integration_loop"
