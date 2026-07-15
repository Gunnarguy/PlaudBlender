"""Typed errors shared by PLAUD adapters and API routes."""

from __future__ import annotations

from typing import Any


class PlaudIntegrationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "plaud_integration_error",
        status_code: int = 502,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self, correlation_id: str | None = None) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "retryable": self.retryable,
                "details": self.details,
            },
            "correlation_id": correlation_id,
        }


class PlaudConfigurationError(PlaudIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, code="configuration_missing", status_code=503)


class PlaudAuthenticationError(PlaudIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, code="authentication_failed", status_code=401)


class PlaudUnknownToolError(PlaudIntegrationError):
    def __init__(self, tool_name: str):
        super().__init__(
            f"PLAUD MCP tool is not on the reviewed allowlist: {tool_name}",
            code="unknown_mcp_tool",
            status_code=400,
        )
