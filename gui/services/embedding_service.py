"""Embedding service used by the GUI.

This is a thin wrapper around the Chronos embedding service.
It is intentionally lazy so tests can import/instantiate without API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingService:
    """Lazy embedding service."""

    _engine: Optional[object] = None

    def _ensure(self) -> None:
        if self._engine is None:
            # Import lazily to avoid requiring GEMINI_API_KEY during tests.
            from src.chronos.embedding_service import ChronosEmbeddingService

            self._engine = ChronosEmbeddingService()

    def embed_text(self, text: str) -> List[float]:
        self._ensure()
        return self._engine.embed_text(text)
