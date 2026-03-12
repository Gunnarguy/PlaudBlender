"""Chronos embedding service using Gemini Embeddings.

Supports two models:
  - gemini-embedding-2-preview  (multimodal: text + audio + image + video + PDF)
  - gemini-embedding-001        (text-only, legacy)

Both use Matryoshka Representation Learning (MRL):
- Default output is 3072 dims (already normalized).
- Sub-3072 dims (768, 1536, etc.) MUST be L2-normalized after truncation.
  See: https://ai.google.dev/gemini-api/docs/embeddings#ensuring-quality

Multimodal embeddings (embedding-2-preview):
- Audio: WAV/MP3, max 80 seconds per request
- Aggregation: text + audio parts in one Content → single fused embedding
- Input token limit: 8192 (vs 2048 for embedding-001)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from google.genai import types

from src.config import get_settings
from src.chronos.genai_helpers import get_genai_client

logger = logging.getLogger(__name__)

# The native output dimensionality. Vectors at this size are already
# L2-normalized by the model; smaller MRL truncations are not.
_NATIVE_DIM = 3072

# MIME types for audio formats supported by gemini-embedding-2-preview
_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}

# Maximum audio duration supported by embedding-2-preview (seconds)
_MAX_AUDIO_SECONDS = 80


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
        self._needs_normalization = self.output_dim < _NATIVE_DIM

        logger.info(
            f"Initialized embedding service with model: {self.model_name} "
            f"(dim={self.output_dim}, normalize={self._needs_normalization})"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """L2-normalize a vector (required for MRL dims < 3072)."""
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
            task_type: Gemini embedding task type — use RETRIEVAL_DOCUMENT when
                indexing content, RETRIEVAL_QUERY for search queries, or
                QUESTION_ANSWERING for Q&A retrieval.

        Returns:
            List[float]: Embedding vector (dimensionality from config)
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
        vec = list(embeddings[0].values)
        return self._normalize(vec) if self._needs_normalization else vec

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
                contents=batch,  # type: ignore[arg-type]
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dim,
                ),
            )

            batch_embeddings = getattr(result, "embeddings", None) or []
            if self._needs_normalization:
                embeddings.extend([self._normalize(e.values) for e in batch_embeddings])
            else:
                embeddings.extend([list(e.values) for e in batch_embeddings])

        return embeddings

    # ------------------------------------------------------------------
    # Multimodal: text + audio fused embedding (gemini-embedding-2-preview)
    # ------------------------------------------------------------------

    @property
    def supports_multimodal(self) -> bool:
        """True if the configured model supports audio/image/video input."""
        return "embedding-2" in self.model_name

    def embed_text_with_audio(
        self,
        text: str,
        audio_path: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        """Create a fused text+audio embedding (single vector).

        Packs both the clean_text and raw audio bytes into one Content
        with multiple Parts, producing a single aggregated embedding that
        captures both the semantic meaning and acoustic signal.

        Falls back to text-only if audio is unavailable or unsupported.

        Args:
            text: Clean text for the event
            audio_path: Path to a WAV/MP3 file (≤80 s)
            task_type: Gemini embedding task type

        Returns:
            Embedding vector, or None if both audio and text fail.
        """
        if not self.supports_multimodal:
            return self.embed_text(text, task_type=task_type)

        audio_part = self._load_audio_part(audio_path)
        if audio_part is None:
            # No usable audio — fall back to text-only
            return self.embed_text(text, task_type=task_type)

        # Build a single Content with text + audio parts → one fused embedding
        content = types.Content(
            parts=[
                types.Part(text=text),
                audio_part,
            ]
        )

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[content],
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dim,
                ),
            )
            embeddings = getattr(result, "embeddings", None) or []
            if not embeddings:
                logger.warning("Multimodal embed returned empty — falling back to text")
                return self.embed_text(text, task_type=task_type)

            vec = list(embeddings[0].values)
            return self._normalize(vec) if self._needs_normalization else vec

        except Exception as exc:
            logger.warning(f"Multimodal embed failed ({exc}) — falling back to text")
            return self.embed_text(text, task_type=task_type)

    def _load_audio_part(self, audio_path: str) -> Optional[types.Part]:
        """Read an audio file and return a Part, or None if unusable."""
        if not audio_path:
            return None

        path = Path(audio_path)
        if not path.is_file():
            logger.debug(f"Audio file not found: {audio_path}")
            return None

        suffix = path.suffix.lower()
        mime = _AUDIO_MIME.get(suffix)
        if mime is None:
            logger.debug(f"Unsupported audio format: {suffix}")
            return None

        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning(f"Cannot read audio file {audio_path}: {exc}")
            return None

        # Rough guard: WAV is ~176 KB/s at 44.1 kHz 16-bit mono.
        # 80s × 176 KB ≈ 14 MB. Reject obviously-too-large files.
        max_bytes = 30 * 1024 * 1024  # 30 MB generous limit
        if len(data) > max_bytes:
            logger.info(f"Audio too large ({len(data) / 1024 / 1024:.1f} MB), skipping")
            return None

        return types.Part.from_bytes(data=data, mime_type=mime)
