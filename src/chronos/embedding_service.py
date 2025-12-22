"""Chronos embedding service using Gemini text-embedding-004."""

import logging
from typing import List

import google.generativeai as genai

from src.config import get_settings

logger = logging.getLogger(__name__)


class ChronosEmbeddingService:
    """Gemini-based embedding service for Chronos events."""

    def __init__(self):
        """Initialize Gemini client."""
        self.settings = get_settings()

        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=self.settings.gemini_api_key)
        self.model_name = self.settings.chronos_embedding_model

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
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

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
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1} ({len(batch)} texts)")

            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type=task_type,
            )

            # Handle single or batch response
            if isinstance(result["embedding"][0], list):
                embeddings.extend(result["embedding"])
            else:
                embeddings.append(result["embedding"])

        return embeddings
