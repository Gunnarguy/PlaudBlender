"""Minimal MCP server exposing Gemini API for PlaudBlender.

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

import google.generativeai as genai
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Configure logging early so MCP clients can surface server-side issues quickly.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plaudblender.mcp")

server = Server("plaudblender-mcp")


@lru_cache(maxsize=1)
def get_gemini_model():
    """Return a cached Gemini model configured for generation.

    Environment variables:
        GEMINI_API_KEY: required.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to start the MCP server.")

    genai.configure(api_key=api_key)
    model_name = os.getenv("CHRONOS_CLEANING_MODEL", "gemini-3-flash-preview")
    return genai.GenerativeModel(model_name)


@server.tool()
async def ping() -> str:
    """Lightweight health probe for MCP clients."""
    return "pong"


@server.tool()
async def list_models() -> List[str]:
    """List available Gemini models."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ["error: GEMINI_API_KEY not set"]
    genai.configure(api_key=api_key)
    models = genai.list_models()
    return [m.name for m in models if "generateContent" in (m.supported_generation_methods or [])]


@server.tool()
async def respond(prompt: str, temperature: float = 0.7) -> str:
    """Create a text response using Gemini.

    Args:
        prompt: User prompt to send to the model.
        temperature: Sampling temperature.
    """
    try:
        model = get_gemini_model()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        return response.text or str(response)
    except Exception as exc:
        logger.exception("Failed to create response via Gemini")
        return f"error: {exc}"


async def main() -> None:
    """Start the MCP server over stdio."""
    logger.info("Starting plaudblender MCP server (stdio transport)...")
    async with stdio_server(server) as transport:
        await transport.serve()


if __name__ == "__main__":
    asyncio.run(main())
