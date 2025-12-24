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

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Set

from google import genai
from google.genai import errors, types

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Create (and cache) a GenAI client for the Gemini Developer API."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    http_options = None
    api_version = (settings.gemini_api_version or "").strip()
    if api_version:
        http_options = types.HttpOptions(api_version=api_version)

    return genai.Client(api_key=settings.gemini_api_key, http_options=http_options)


@lru_cache(maxsize=1)
def list_model_names() -> Set[str]:
    """Return the set of model IDs available to the configured API key.

    The API returns names like "models/gemini-2.5-flash".
    We normalize them to "gemini-2.5-flash".
    """
    client = get_genai_client()
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

    return names


def pick_first_available(*candidates: str) -> Optional[str]:
    """Return the first candidate that exists in the model list (if available)."""
    available = list_model_names()
    for c in candidates:
        if c and c in available:
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
