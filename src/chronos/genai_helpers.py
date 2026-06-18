"""Helpers for the Google Gen AI SDK (google-genai).

Chronos historically used the legacy `google-generativeai` SDK. Gemini 3 preview
models (Flash/Pro) are documented for the newer `google-genai` SDK.

This module provides:
- a shared GenAI client (Gemini Developer API)
- lightweight model availability checks
- small config helpers (thinking level mapping)

Refs:
- SDK docs: https://googleapis.github.io/python-genai/
- Model listing: https://ai.google.dev/api/models
- API versions: https://ai.google.dev/gemini-api/docs/api-versions
"""


import logging
from functools import lru_cache
from typing import Optional, Set

from google import genai
from google.genai import errors, types

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_genai_client_cached(api_key: str, api_version: str) -> genai.Client:
    http_options = None
    if api_version:
        http_options = types.HttpOptions(api_version=api_version)

    return genai.Client(api_key=api_key, http_options=http_options)


def get_genai_client() -> genai.Client:
    """Create (and cache) a GenAI client for the Gemini Developer API."""
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        raise ValueError(
            "CHRONOS_GEMINI_API_KEY not set. Set a dedicated Chronos key or opt "
            "into the shared GEMINI_API_KEY with CHRONOS_ALLOW_SHARED_GEMINI_KEY=1"
        )

    api_version = (settings.gemini_api_version or "").strip()
    return _get_genai_client_cached(api_key, api_version)


@lru_cache(maxsize=4)
def _list_model_names_cached(api_key: str, api_version: str) -> frozenset[str]:
    """Return the set of model IDs available to the configured API key.

    The API returns names like "models/gemini-2.5-flash".
    We normalize them to "gemini-2.5-flash".
    """
    client = _get_genai_client_cached(api_key, api_version)
    names: Set[str] = set()

    try:
        for m in client.models.list():
            name = getattr(m, "name", None)
            if not name:
                continue
            if name.startswith("models/"):
                names.add(name.split("/", 1)[1])
            else:
                names.add(name)
    except Exception as e:
        # Non-fatal: we can still attempt calls and let the API surface errors.
        logger.warning(f"Could not list Gemini models: {e}")

    return frozenset(names)


def list_model_names() -> Set[str]:
    settings = get_settings()
    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        return set()

    api_version = (settings.gemini_api_version or "").strip()
    return set(_list_model_names_cached(api_key, api_version))


def pick_first_available(*candidates: str) -> Optional[str]:
    """Return the first candidate that exists in the model list (if available)."""
    available = list_model_names()
    for c in candidates:
        if c and c in available:
            return c
    return None


def pick_first_available_or_known(*candidates: str) -> Optional[str]:
    """Return the first available candidate, or the first non-empty one if listing fails."""
    available = list_model_names()
    for c in candidates:
        if not c:
            continue
        if not available or c in available:
            return c
    return None


def normalize_thinking_level(level: str) -> Optional[types.ThinkingLevel]:
    """Map a string to the SDK ThinkingLevel enum.

    Gemini 3 docs describe thinking levels: minimal/low/medium/high.
    We accept common variants and return None when unknown.
    """
    if not level:
        return None

    key = level.strip().lower()
    mapping = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "min": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "med": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }
    return mapping.get(key)


def is_model_not_found(err: Exception) -> bool:
    """Best-effort check for a 'model not found' API failure."""
    if isinstance(err, errors.APIError):
        return err.code == 404
    msg = str(err).lower()
    return "model" in msg and ("not found" in msg or "does not exist" in msg)


def is_permission_denied(err: Exception) -> bool:
    """Check for 403 PERMISSION_DENIED — project banned or access revoked."""
    if isinstance(err, errors.APIError) and err.code == 403:
        return True
    msg = str(err).lower()
    return "permission_denied" in msg or "project has been denied" in msg


def is_model_temporarily_unavailable(err: Exception) -> bool:
    """Best-effort check for a transient Gemini availability error."""
    if isinstance(err, errors.APIError) and err.code in (429, 503):
        return True

    msg = str(err).lower()
    return any(
        marker in msg
        for marker in (
            "429 resource_exhausted",
            "503 unavailable",
            "status': 'unavailable'",
            '"status": "unavailable"',
            "currently experiencing high demand",
            "temporarily unavailable",
            "overloaded",
            "rate limit",
        )
    )
