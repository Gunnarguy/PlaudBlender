"""Minimal MCP server exposing OpenAI Responses API for PlaudBlender.

Run with: python -m scripts.mcp_server

This uses stdin/stdout transport so it works with MCP-capable clients
like ChatGPT connectors. Tools are intentionally simple to keep latency
low and avoid pulling in PlaudBlender internals unless needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache
from typing import List

from openai import OpenAI
from openai import OpenAIError
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configure logging early so MCP clients can surface server-side issues quickly.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plaudblender.mcp")

server = Server("plaudblender-mcp")


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Return a cached OpenAI client configured for Responses API.

    Environment variables:
        OPENAI_API_KEY: required.
        OPENAI_BASE_URL: optional custom endpoint.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to start the MCP server.")

    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


@server.tool()
async def ping() -> str:
    """Lightweight health probe for MCP clients."""

    return "pong"


@server.tool()
async def list_models() -> List[str]:
    """List available model IDs for the configured OpenAI project."""

    client = get_openai_client()
    models = client.models.list()
    return [model.id for model in models.data]


@server.tool()
async def respond(prompt: str, model: str | None = None, temperature: float = 0.7) -> str:
    """Create a text response using the OpenAI Responses API.

    Args:
        prompt: User prompt to send to the model.
        model: Optional override of the model (defaults to env or gpt-4.1).
        temperature: Sampling temperature.
    """

    client = get_openai_client()
    target_model = model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1")

    try:
        response = client.responses.create(
            model=target_model,
            input=prompt,
            temperature=temperature,
        )
    except OpenAIError as exc:  # pragma: no cover - passthrough for MCP clients
        logger.exception("Failed to create response via OpenAI Responses API")
        return f"error: {exc}"

    # Prefer SDK convenience property when present.
    if getattr(response, "output_text", None):
        return response.output_text

    # Fallback: stitch any message output_text fragments.
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)

    return "\n".join(parts) if parts else str(response)


async def main() -> None:
    """Start the MCP server over stdio."""

    logger.info("Starting plaudblender MCP server (stdio transport)...")
    async with stdio_server(server) as transport:
        await transport.serve()


if __name__ == "__main__":
    asyncio.run(main())
