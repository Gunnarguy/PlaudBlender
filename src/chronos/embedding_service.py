"""Chronos embedding service using Gemini Embeddings (gemini-embedding-001)."""

import logging
from typing import List

from google.genai import types

from src.config import get_settings
from src.chronos.genai_helpers import get_genai_client

logger = logging.getLogger(__name__)


class ChronosEmbeddingService:
    """Gemini-based embedding service for Chronos events."""

    def __init__(self):
        """Initialize Gemini client."""
        self.settings = get_settings()

        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = get_genai_client()
        self.model_name = self.settings.chronos_embedding_model
        self.output_dim = int(getattr(self.settings, "chronos_embedding_dim", 768))

        logger.info(f"Initialized embedding service with model: {self.model_name}")

    def embed_text(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """Embed a single text.

        Args:
            text: Text to embed
            task_type: Gemini task type (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY)

        Returns:
            List[float]: 768-dim embedding vector
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dim,
            ),
        )

        # google-genai returns a list of embeddings; each has `.values`.
        embeddings = getattr(result, "embeddings", None) or []
        if not embeddings:
            raise ValueError("No embedding returned")
        return list(embeddings[0].values)

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            task_type: Gemini task type
            batch_size: Batch size for API calls

        Returns:
            List of embedding vectors
        """
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1} ({len(batch)} texts)")

            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dim,
                ),
            )

            batch_embeddings = getattr(result, "embeddings", None) or []
            embeddings.extend([list(e.values) for e in batch_embeddings])

        return embeddings
