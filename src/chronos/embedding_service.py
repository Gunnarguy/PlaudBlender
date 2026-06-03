"""Chronos embedding service with Gemini/OpenAI backends.

Gemini embeddings are the default Chronos path:
- `gemini-embedding-2` / `gemini-embedding-2-preview` — multimodal, MRL-capable
- `gemini-embedding-001` — text-only fallback

OpenAI embedding models remain supported for paid fallback / compatibility:
- `text-embedding-3-large`
- `text-embedding-3-small`

Callers use a single interface while the service dispatches based on the
configured `CHRONOS_EMBEDDING_MODEL`.
"""

import logging
import time as _time
from pathlib import Path
from typing import List, Optional

import numpy as np
from google.genai import types

from src.chronos.genai_helpers import get_genai_client
from src.config import get_settings

logger = logging.getLogger(__name__)

_NATIVE_GEMINI_DIM = 3072
_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


class ChronosEmbeddingService:
    """Provider-aware embedding service for Chronos events."""

    def __init__(self):
        """Initialize the configured embedding client."""
        self.settings = get_settings()
        self.model_name = self.settings.chronos_embedding_model
        self.output_dim = int(getattr(self.settings, "chronos_embedding_dim", 768))
        self.local_embed_batch_size = max(
            1,
            int(getattr(self.settings, "chronos_local_embed_batch_size", 1) or 1),
        )
        self.local_embed_timeout_seconds = max(
            30.0,
            float(getattr(self.settings, "chronos_local_embed_timeout_seconds", 300.0) or 300.0),
        )
        model_lower = (self.model_name or "").strip().lower()
        if model_lower.startswith("gemini"):
            self.provider = "gemini"
        elif model_lower.startswith("text-embedding"):
            self.provider = "openai"
        else:
            self.provider = "ollama"
        self._needs_normalization = (
            self.provider in {"gemini", "ollama"} and self.output_dim > 0
        )

        if self.provider == "gemini":
            if not self.settings.gemini_api_key:
                raise ValueError(
                    "CHRONOS_GEMINI_API_KEY not set. Set a dedicated Chronos key or "
                    "opt into the shared GEMINI_API_KEY with CHRONOS_ALLOW_SHARED_GEMINI_KEY=1"
                )
            self.client = get_genai_client()
        else:
            if self.provider == "ollama":
                from src.chronos.local_llm_service import LocalLLMService

                self.client = LocalLLMService(settings=self.settings)
                status = self.client.status()
                if not status.get("ok"):
                    raise ValueError(status.get("error") or status.get("detail") or "Local embedding sidecar unavailable")
                self._needs_normalization = True
                logger.info(
                    f"Initialized embedding service with local Ollama model: {self.model_name} "
                    f"(dim={self.output_dim})"
                )
                return

            if not self.settings.openai_api_key:
                raise ValueError(
                    "OpenAI embeddings are disabled. Set CHRONOS_EMBEDDING_MODEL to a Gemini/local model, "
                    "or explicitly opt in with CHRONOS_OPENAI_ENABLED=1 and OPENAI_API_KEY."
                )

            from openai import OpenAI

            self.client = OpenAI(api_key=self.settings.openai_api_key)

        logger.info(
            f"Initialized embedding service with model: {self.model_name} "
            f"(provider={self.provider}, dim={self.output_dim}, normalize={self._needs_normalization})"
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
        """Embed a single text with the configured provider."""
        if self.provider == "gemini":
            return self._embed_text_gemini(text, task_type=task_type)
        if self.provider == "ollama":
            return self._embed_text_ollama(text)
        return self._embed_text_openai(text)

    def _coerce_vector_dim(self, vector: List[float]) -> List[float]:
        if self.output_dim <= 0:
            return vector
        if len(vector) > self.output_dim:
            vector = vector[: self.output_dim]
        elif len(vector) < self.output_dim:
            vector = vector + [0.0] * (self.output_dim - len(vector))
        return self._normalize(vector)

    def _embed_text_ollama(self, text: str) -> List[float]:
        from app_v2.services.xray import xray_log

        _t0 = _time.perf_counter()
        vec = self.client.embed(
            text,
            model=self.model_name,
            timeout=self.local_embed_timeout_seconds,
        )[0]
        _ms = (_time.perf_counter() - _t0) * 1000
        xray_log(
            "embed",
            "local-text",
            f"Turned {len(text.split())} words into a local fingerprint",
            duration_ms=round(_ms, 1),
            provider="ollama",
            model=self.model_name,
        )
        return self._coerce_vector_dim(vec)

    def _embed_text_openai(self, text: str) -> List[float]:
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

    def _embed_text_gemini(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        from app_v2.services.xray import xray_log

        _t0 = _time.perf_counter()
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dim,
            ),
        )

        embeddings = getattr(result, "embeddings", None) or []
        if not embeddings:
            xray_log("embed", "error", "Gemini returned no embedding", level="error")
            raise ValueError("No embedding returned")

        vec = list(embeddings[0].values)
        _ms = (_time.perf_counter() - _t0) * 1000
        xray_log(
            "embed",
            "text",
            f"Turned {len(text.split())} words into a fingerprint the computer can compare",
            duration_ms=round(_ms, 1),
        )
        from src.chronos.cost_tracker import track_usage

        track_usage(
            self.model_name,
            "embed",
            input_tokens=int(len(text.split()) * 1.3),
        )
        return self._normalize(vec) if self._needs_normalization else vec

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
    ) -> List[List[float]]:
        """Embed multiple texts in batches."""
        if self.provider == "gemini":
            return self._embed_batch_gemini(
                texts,
                task_type=task_type,
                batch_size=batch_size,
            )
        if self.provider == "ollama":
            return self._embed_batch_ollama(
                texts,
                batch_size=min(batch_size, self.local_embed_batch_size),
            )
        return self._embed_batch_openai(texts, batch_size=batch_size)

    def _embed_batch_ollama(
        self,
        texts: List[str],
        *,
        batch_size: int = 16,
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        from app_v2.services.xray import xray_log

        _batch_t0 = _time.perf_counter()
        _total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            _batch_num = i // batch_size + 1
            _bt0 = _time.perf_counter()
            batch_embeddings = self.client.embed(
                batch,
                model=self.model_name,
                timeout=self.local_embed_timeout_seconds,
            )
            _bt_ms = (_time.perf_counter() - _bt0) * 1000
            xray_log(
                "embed",
                "local-batch",
                f"Local embedding group {_batch_num}/{_total_batches} done — {len(batch_embeddings)} fingerprints",
                duration_ms=round(_bt_ms, 1),
                provider="ollama",
                model=self.model_name,
            )
            embeddings.extend([self._coerce_vector_dim(e) for e in batch_embeddings])

        _total_ms = (_time.perf_counter() - _batch_t0) * 1000
        xray_log(
            "embed",
            "local-done",
            f"All {len(texts)} texts are now local searchable fingerprints",
            duration_ms=round(_total_ms, 1),
            provider="ollama",
            model=self.model_name,
        )
        return embeddings

    def _embed_batch_openai(
        self,
        texts: List[str],
        *,
        batch_size: int = 100,
    ) -> List[List[float]]:
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

    def _embed_batch_gemini(
        self,
        texts: List[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
    ) -> List[List[float]]:
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

            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dim,
                ),
            )

            batch_embeddings = getattr(result, "embeddings", None) or []
            _bt_ms = (_time.perf_counter() - _bt0) * 1000
            xray_log(
                "embed",
                "batch",
                f"Group {_batch_num} done — {len(batch_embeddings)} fingerprints created",
                duration_ms=round(_bt_ms, 1),
            )

            from src.chronos.cost_tracker import track_usage

            track_usage(
                self.model_name,
                "embed",
                input_tokens=int(sum(len(t.split()) for t in batch) * 1.3),
            )

            if self._needs_normalization:
                embeddings.extend([self._normalize(list(e.values)) for e in batch_embeddings])
            else:
                embeddings.extend([list(e.values) for e in batch_embeddings])

        _total_ms = (_time.perf_counter() - _batch_t0) * 1000
        xray_log(
            "embed",
            "done",
            f"All {len(texts)} texts are now searchable fingerprints",
            duration_ms=round(_total_ms, 1),
        )
        return embeddings

    # ------------------------------------------------------------------
    # Multimodal Gemini path
    # ------------------------------------------------------------------

    @property
    def supports_multimodal(self) -> bool:
        """True if the configured model supports audio/image/video input."""
        return self.provider == "gemini" and "embedding-2" in self.model_name

    def embed_text_with_audio(
        self,
        text: str,
        audio_path: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[float]:
        """Create a fused text+audio embedding when the model supports it."""
        if not self.supports_multimodal:
            return self.embed_text(text, task_type=task_type)

        from app_v2.services.xray import xray_log

        audio_part = self._load_audio_part(audio_path)
        if audio_part is None:
            xray_log(
                "embed",
                "fallback",
                "No audio file available — just using the text",
                level="warn",
            )
            return self.embed_text(text, task_type=task_type)

        content = types.Content(
            parts=[
                types.Part(text=text),
                audio_part,
            ]
        )

        try:
            _t0 = _time.perf_counter()
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
                xray_log(
                    "embed",
                    "fallback",
                    "Audio+text combo came back empty — just using text",
                    level="warn",
                )
                return self.embed_text(text, task_type=task_type)

            vec = list(embeddings[0].values)
            _ms = (_time.perf_counter() - _t0) * 1000
            xray_log(
                "embed",
                "multimodal",
                "Made a fingerprint from both the audio and text together",
                duration_ms=round(_ms, 1),
            )
            from src.chronos.cost_tracker import track_usage

            track_usage(
                self.model_name,
                "embed",
                input_tokens=int(len(text.split()) * 1.3),
            )
            return self._normalize(vec) if self._needs_normalization else vec
        except Exception as exc:
            logger.warning(f"Multimodal embed failed ({exc}) — falling back to text")
            xray_log(
                "embed",
                "fallback",
                "Audio didn't work — just using the text",
                level="warn",
            )
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

        max_bytes = 30 * 1024 * 1024
        if len(data) > max_bytes:
            logger.info(f"Audio too large ({len(data) / 1024 / 1024:.1f} MB), skipping")
            return None

        return types.Part.from_bytes(data=data, mime_type=mime)

    # ------------------------------------------------------------------
    # Batch with audio
    # ------------------------------------------------------------------

    def embed_batch_with_audio(
        self,
        items: List[tuple],
        task_type: str = "RETRIEVAL_DOCUMENT",
        max_workers: int = 5,
    ) -> List[List[float]]:
        """Embed multiple (text, audio_path) pairs."""
        if not self.supports_multimodal:
            texts = [text for text, _audio in items]
            return self.embed_batch(texts, task_type=task_type)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from app_v2.services.xray import xray_log

        if not items:
            return []

        _t0 = _time.perf_counter()
        xray_log(
            "embed",
            "batch-multimodal",
            f"Starting parallel embedding of {len(items)} items ({max_workers} workers)",
        )

        results: List[Optional[List[float]]] = [None] * len(items)

        def _embed_one(idx: int, text: str, audio_path: str) -> tuple[int, List[float]]:
            vec = self.embed_text_with_audio(text, audio_path, task_type=task_type)
            return idx, vec

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_embed_one, i, text, audio): i
                for i, (text, audio) in enumerate(items)
            }
            for future in as_completed(futures):
                idx, vec = future.result()
                results[idx] = vec

        _ms = (_time.perf_counter() - _t0) * 1000
        xray_log(
            "embed",
            "batch-multimodal",
            f"Parallel embedding done — {len(items)} items in {_ms / 1000:.1f}s",
            duration_ms=round(_ms, 1),
        )
        return results  # type: ignore[return-value]
