"""Recursive secret redaction for ledgers, errors, and API responses."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_SENSITIVE_FRAGMENTS = (
    "authorization",
    "access_token",
    "refresh_token",
    "client_secret",
    "secret_key",
    "api_key",
    "x-client-api-key",
    "webhook_secret",
    "cookie",
    "code",
)
_TOKEN_PATTERN = re.compile(r"\b(?:Bearer\s+)?eyJ[A-Za-z0-9_.-]{16,}\b", re.IGNORECASE)


def is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment.replace("-", "_") in normalized for fragment in _SENSITIVE_FRAGMENTS)


def redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _TOKEN_PATTERN.sub(REDACTED, value)
    if not parsed.scheme or not parsed.netloc:
        return _TOKEN_PATTERN.sub(REDACTED, value)
    query = [
        (key, REDACTED if is_sensitive_key(key) else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))


def redact(value: Any, *, key: str | None = None) -> Any:
    if key and is_sensitive_key(key):
        return REDACTED
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        return redact_url(value) if "://" in value else _TOKEN_PATTERN.sub(REDACTED, value)
    return value
