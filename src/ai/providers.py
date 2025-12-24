"""AI provider registry (minimal).

Historically the project supported OpenAI and Gemini. Chronos is Gemini-first.
The test suite expects this module to exist.
"""

from __future__ import annotations

from enum import Enum


class AIProvider(str, Enum):
    gemini = "gemini"
    openai = "openai"


def get_default_provider() -> AIProvider:
    return AIProvider.gemini
