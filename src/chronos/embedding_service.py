"""Chronos embedding service using OpenAI Embeddings.

Uses text-embedding-3-large (or text-embedding-3-small) via the OpenAI API.

Both models support Matryoshka Representation Learning (MRL):
- text-embedding-3-large: native 3072 dims, can request 256–3072
- text-embedding-3-small: native 1536 dims, can request 256–1536

Vectors at sub-native dimensions are already L2-normalized by the API.

Note: OpenAI embeddings are text-only. The multimodal interface
(embed_text_with_audio) falls through to text-only embedding
so that callers don't need to change.
"""

import logging
import os
import time as _time
from typing import List, Optional

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)


class ChronosEmbeddingService:
    """OpenAI-based embedding service for Chronos events."""

    def __init__(self):
        """Initialize OpenAI client."""
        self.settings = get_settings()

        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        from openai import OpenAI

        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model_name = self.settings.chronos_embedding_model
        self.output_dim = int(getattr(self.settings, "chronos_embedding_dim", 768))

        logger.info(
            f"Initialized embedding service with model: {self.model_name} "
            f"(dim={self.output_dim})"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """L2-normalize a vector."""
        arr = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_text(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """Embed a single text.

        Args:
            text: Text to embed
            task_type: Ignored (kept for API compatibility with callers).

        Returns:
            List[float]: Embedding vector (dimensionality from config)
        """
        from app_v2.services.xray import xray_log

        _t0 = _time.perf_counter()
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self.output_dim,
        )

        if not response.data:
            xray_log("embed", "error", "OpenAI returned no embedding", level="error")
            raise ValueError("No embedding returned")

        vec = response.data[0].embedding
        _ms = (_time.perf_counter() - _t0) * 1000
        xray_log(
            "embed",
            "text",
            f"Turned {len(text.split())} words into a fingerprint the computer can compare",
            duration_ms=round(_ms, 1),
        )
        from src.chronos.cost_tracker import track_usage

        tokens_used = (
            getattr(response.usage, "total_tokens", 0) if response.usage else 0
        )
        track_usage(
            self.model_name,
            "embed",
            input_tokens=tokens_used or int(len(text.split()) * 1.3),
        )
        return vec

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            task_type: Ignored (kept for API compatibility).
            batch_size: Batch size for API calls (OpenAI supports up to 2048)

        Returns:
            List of embedding vectors
        """
        embeddings: List[List[float]] = []
        from app_v2.services.xray import xray_log

        _batch_t0 = _time.perf_counter()
        _total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            _batch_num = i // batch_size + 1
            logger.debug(f"Embedding batch {_batch_num} ({len(batch)} texts)")
            xray_log(
                "embed",
                "batch",
                f"Converting group {_batch_num} of {_total_batches} ({len(batch)} texts) into fingerprints",
            )
            _bt0 = _time.perf_counter()

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.output_dim,
            )

            batch_embeddings = response.data or []
            _bt_ms = (_time.perf_counter() - _bt0) * 1000
            xray_log(
                "embed",
                "batch",
                f"Group {_batch_num} done — {len(batch_embeddings)} fingerprints created",
                duration_ms=round(_bt_ms, 1),
            )
            from src.chronos.cost_tracker import track_usage

            tokens_used = (
                getattr(response.usage, "total_tokens", 0) if response.usage else 0
            )
            track_usage(
                self.model_name,
                "embed",
                input_tokens=tokens_used
                or int(sum(len(t.split()) for t in batch) * 1.3),
            )

            # OpenAI returns embeddings sorted by index
            sorted_embeddings = sorted(batch_embeddings, key=lambda e: e.index)
            embeddings.extend([e.embedding for e in sorted_embeddings])

        _total_ms = (_time.perf_counter() - _batch_t0) * 1000
        xray_log(
            "embed",
            "done",
            f"All {len(texts)} texts are now searchable fingerprints",
            duration_ms=round(_total_ms, 1),
        )
        return embeddings

    # ------------------------------------------------------------------
    # Multimodal stubs (OpenAI embeddings are text-only)
    # ------------------------------------------------------------------

    @property
    def supports_multimodal(self) -> bool:
        """OpenAI embeddings are text-only."""
        return False

    def embed_text_with_audio(
        self,
        text: str,
        audio_path: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        """Fall through to text-only embedding.

        OpenAI embedding models don't support audio input, so we ignore
        the audio_path and embed the text alone. This keeps the interface
        compatible with callers that were using Gemini multimodal.
        """
        return self.embed_text(text, task_type=task_type)

    # ------------------------------------------------------------------
    # Batch (parallel not needed — OpenAI batch API is already fast)
    # ------------------------------------------------------------------

    def embed_batch_with_audio(
        self,
        items: List[tuple],
        task_type: str = "RETRIEVAL_DOCUMENT",
        max_workers: int = 5,
    ) -> List[List[float]]:
        """Embed multiple (text, audio_path) pairs — audio is ignored.

        Falls through to text-only batch embedding since OpenAI
        doesn't support audio in embeddings.
        """
        texts = [text for text, _audio in items]
        return self.embed_batch(texts, task_type=task_type)
