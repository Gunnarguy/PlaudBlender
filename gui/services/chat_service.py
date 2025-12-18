"""Chat helper using OpenAI Responses API.

Keeps client creation cached and exposes a single send function
returning text output suitable for the GUI chat view.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Import only for type-checkers; importing openai can be slow on some setups.
    from openai import OpenAI as OpenAIClient


@lru_cache(maxsize=1)
def _client() -> "OpenAIClient":
    # Lazily import OpenAI so the rest of the app can boot even if openai is
    # slow to import or not installed. Chat view is optional.
    try:
        from openai import OpenAI
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(
            "OpenAI SDK is required for Chat, but failed to import. "
            "Install/repair the 'openai' package in your venv and retry. "
            f"(import error: {e})"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for chat.")
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def send_response(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    instructions: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send a chat conversation to the Responses API with optional overrides.

    Args:
        messages: List of {role, content} dicts (user/assistant).
        model: Model ID (e.g., gpt-4.1, gpt-4.1-mini).
        temperature: Sampling temperature.
        instructions: Optional system/developer message inserted via `instructions`.
        overrides: Optional dict of extra Responses parameters (max_output_tokens, top_p, tools,
            tool_choice, parallel_tool_calls, metadata, store, response_format/text config,
            previous_response_id, etc.).
    Returns:
        dict with keys: text (aggregated), raw (response object).
    """

    client = _client()
    payload: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": m.get("role"), "content": m.get("content", "")} for m in messages
        ],
        "temperature": temperature,
    }
    if instructions:
        payload["instructions"] = instructions
    if overrides:
        payload.update(overrides)

    response = client.responses.create(**payload)

    text = getattr(response, "output_text", None)
    if not text:
        # Fallback: concatenate message output_text items
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        parts.append(content.text)
        text = "\n".join(parts)

    return {"text": text or "", "raw": response}
