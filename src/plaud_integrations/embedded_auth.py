"""PLAUD Embedded partner and user token client (backend only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
import time
from typing import Any
from uuid import uuid4

import requests

from .call_ledger import PlaudCallLedger, default_ledger
from .errors import PlaudAuthenticationError, PlaudConfigurationError, PlaudIntegrationError
from .models import PlaudCallEvent, utc_now
from .redaction import redact


class PlaudRegion(str, Enum):
    US = "us"
    JP = "jp"

    @property
    def base_url(self) -> str:
        return f"https://platform-{self.value}.plaud.ai/developer/api"

    @classmethod
    def parse(cls, value: str | "PlaudRegion") -> "PlaudRegion":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise PlaudConfigurationError("PLAUD_EMBEDDED_REGION must be 'us' or 'jp'") from exc


@dataclass
class PlaudToken:
    access_token: str
    token_type: str
    expires_in: int
    expires_at: str
    refresh_token: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        return {
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "expires_at": self.expires_at,
            "has_refresh_token": bool(self.refresh_token),
        }


class PlaudEmbeddedAuthClient:
    PARTNER_PATH = "/oauth/partner/access-token"
    REFRESH_PATH = "/oauth/partner/access-token/refresh"
    USER_PATH = "/open/partner/users/access-token"

    def __init__(
        self,
        client_id: str | None,
        secret_key: str | None,
        *,
        region: str | PlaudRegion = PlaudRegion.US,
        session: requests.Session | None = None,
        timeout: float = 30,
        ledger: PlaudCallLedger = default_ledger,
    ):
        self.client_id = (client_id or "").strip()
        self.secret_key = (secret_key or "").strip()
        self.region = PlaudRegion.parse(region)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.ledger = ledger
        if not self.client_id or not self.secret_key:
            raise PlaudConfigurationError("PLAUD_EMBEDDED_CLIENT_ID and PLAUD_EMBEDDED_SECRET_KEY are required")

    def _token(self, payload: dict[str, Any]) -> PlaudToken:
        if not payload.get("access_token"):
            raise PlaudAuthenticationError("PLAUD token response did not contain an access_token")
        seconds = int(payload.get("expires_in", 0))
        return PlaudToken(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            token_type=payload.get("token_type", "bearer"),
            expires_in=seconds,
            expires_at=(utc_now() + timedelta(seconds=seconds)).isoformat(),
        )

    def _request(
        self,
        operation: str,
        path: str,
        *,
        auth: Any,
        data: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> PlaudToken:
        correlation_id = str(uuid4())
        started = time.perf_counter()
        status: int | str | None = None
        response_payload: Any = None
        error_name: str | None = None
        try:
            response = self.session.post(
                self.region.base_url + path,
                auth=auth if isinstance(auth, tuple) else None,
                headers=auth if isinstance(auth, dict) else None,
                data=data,
                json=json_body,
                timeout=self.timeout,
            )
            status = response.status_code
            response_payload = response.json() if response.content else {}
            if response.status_code in (401, 403):
                raise PlaudAuthenticationError(f"PLAUD Embedded authentication failed ({response.status_code})")
            response.raise_for_status()
            return self._token(response_payload)
        except PlaudIntegrationError:
            error_name = "authentication"
            raise
        except requests.RequestException as exc:
            error_name = type(exc).__name__
            raise PlaudIntegrationError(
                f"PLAUD Embedded {operation} failed: {exc}",
                code="embedded_http_error",
                retryable=isinstance(exc, (requests.Timeout, requests.ConnectionError)),
            ) from exc
        finally:
            request_payload = data if data is not None else json_body
            self.ledger.record(PlaudCallEvent(
                timestamp=utc_now().isoformat(), correlation_id=correlation_id,
                transport="plaud_embedded_rest", operation=operation, safety="auth",
                request_summary=f"POST {path}", redacted_request=redact(request_payload or {}),
                response_status=status, redacted_response=redact(response_payload),
                duration_ms=int((time.perf_counter() - started) * 1000),
                error_classification=error_name,
            ))

    def acquire_partner_token(self) -> PlaudToken:
        return self._request("createPartnerAccessToken", self.PARTNER_PATH, auth=(self.client_id, self.secret_key))

    def refresh_partner_token(self, refresh_token: str) -> PlaudToken:
        return self._request(
            "refreshPartnerAccessToken", self.REFRESH_PATH,
            auth=(self.client_id, self.secret_key), data={"refresh_token": refresh_token},
        )

    def issue_user_token(self, partner_access_token: str, user_id: str, expires_in: int = 86400) -> PlaudToken:
        if not 6 <= len(user_id) <= 120:
            raise PlaudIntegrationError("user_id must contain 6-120 characters", code="invalid_user_id", status_code=422)
        return self._request(
            "createUserAccessToken", self.USER_PATH,
            auth={"Authorization": f"Bearer {partner_access_token}", "Content-Type": "application/json"},
            json_body={"user_id": user_id, "expires_in": expires_in},
        )
