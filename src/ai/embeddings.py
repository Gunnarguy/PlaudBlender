"""Embedding helpers.

This module is intentionally small; most embedding logic lives in
`src/chronos/embedding_service.py`.

The test suite expects `src.ai.embeddings` to be importable.
"""

from __future__ import annotations

from typing import List


def embed_text(text: str) -> List[float]:
    """Convenience wrapper around Chronos embedding service.

    This will raise if GEMINI_API_KEY is not configured.
    """

    from src.chronos.embedding_service import ChronosEmbeddingService

    return ChronosEmbeddingService().embed_text(text)
