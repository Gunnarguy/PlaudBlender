"""AI providers and embedding utilities.

Most active AI work for Chronos lives in `src/chronos/`, but the test suite
expects `src.ai.embeddings` and `src.ai.providers` to exist.
"""

from .providers import AIProvider, get_default_provider

__all__ = [
    "AIProvider",
    "get_default_provider",
]
