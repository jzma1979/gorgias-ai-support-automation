from __future__ import annotations

import logging
import html
from collections.abc import Mapping
from typing import Any

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class GorgiasAPIError(Exception):
    """Raised when the Gorgias API cannot complete a requested action."""


class GorgiasClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        api_key: str,
        timeout_seconds: float,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url:
            raise GorgiasAPIError("GORGIAS_BASE_URL is required.")
        if not username:
            raise GorgiasAPIError("GORGIAS_USERNAME is required.")
        if not api_key:
            raise GorgiasAPIError("GORGIAS_API_KEY is required.")

        self.base_url = base_url.rstrip("/")
        self.auth = (username, api_key)
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    @classmethod
    def from_settings(cls) -> "GorgiasClient":
        return cls(
            base_url=settings.GORGIAS_BASE_URL,
            username=settings.GORGIAS_USERNAME,
            api_key=settings.GORGIAS_API_KEY,
            timeout_seconds=settings.EXTERNAL_REQUEST_TIMEOUT_SECONDS,
        )

    def get_ticket(self, ticket_id: int | str) -> dict[str, Any]:
        return self._request("GET", f"/api/tickets/{ticket_id}")

    def get_customer(self, customer_id: int | str) -> dict[str, Any]:
        return self._request("GET", f"/api/customers/{customer_id}")

    def update_ticket_priority(self, ticket_id: int | str, priority: str) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/api/tickets/{ticket_id}",
            json_body={"priority": priority},
        )

    def resolve_or_create_tag(self, name: str) -> int | str:
        clean_name = name.strip()
        if not clean_name:
            raise GorgiasAPIError("Tag name must not be blank.")

        search_payload = self._request(
            "GET",
            "/api/tags",
            params={"name": clean_name},
        )
        for tag in self._extract_collection(search_payload):
            if str(tag.get("name", "")).lower() == clean_name.lower() and tag.get("id"):
                return tag["id"]

        created = self._request("POST", "/api/tags", json_body={"name": clean_name})
        tag_id = created.get("id")
        if tag_id:
            return tag_id

        for tag in self._extract_collection(created):
            if str(tag.get("name", "")).lower() == clean_name.lower() and tag.get("id"):
                return tag["id"]

        raise GorgiasAPIError(f"Could not resolve tag: {clean_name}")

    def add_tags_to_ticket(
        self,
        ticket_id: int | str,
        tag_names: list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        tag_ids = [self.resolve_or_create_tag(name) for name in tag_names]
        return self._request(
            "POST",
            f"/api/tickets/{ticket_id}/tags",
            json_body={"tags": [{"id": tag_id} for tag_id in tag_ids]},
        )

    def create_internal_note(self, ticket_id: int | str, body: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/tickets/{ticket_id}/messages",
            json_body={
                "channel": "internal-note",
                "source": {"type": "internal-note"},
                "body_text": body,
                "body_html": html.escape(body).replace("\n", "<br>"),
                "public": False,
                "via": "api",
            },
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.request(
                method,
                url,
                auth=self.auth,
                json=dict(json_body) if json_body is not None else None,
                params=dict(params) if params is not None else None,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise GorgiasAPIError(f"Gorgias {method} {path} timed out.") from exc
        except requests.RequestException as exc:
            raise GorgiasAPIError(f"Gorgias {method} {path} failed.") from exc

        status_code = getattr(response, "status_code", 200)
        if status_code < 200 or status_code >= 300:
            raise GorgiasAPIError(f"Gorgias {method} {path} returned HTTP {status_code}.")

        if not getattr(response, "content", b""):
            return {}

        try:
            payload = response.json()
        except ValueError as exc:
            raise GorgiasAPIError(f"Gorgias {method} {path} returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise GorgiasAPIError(f"Gorgias {method} {path} returned unexpected JSON.")
        return payload

    @staticmethod
    def _extract_collection(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in ("data", "tags", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []
