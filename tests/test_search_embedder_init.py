"""Search embeds with whatever CHRONOS_EMBEDDING_MODEL names, Gemini key or not (2026-10-01).

Before: ChronosDataService only built its embedder when a Gemini key was set. Production
embeds with OpenAI text-embedding-3-large and has no Gemini key (owner rule: no paid
Gemini API), so every search and Ask fell back to whole-phrase text matching and Ask
answered "I couldn't find any relevant events" for ordinary questions.
"""

import src.chronos.embedding_service as embedding_service
import src.chronos.qdrant_client as qdrant_client
from app_v2.services.data_service import ChronosDataService


class _Embedder:
    pass


class _Qdrant:
    pass


def _service():
    svc = ChronosDataService.__new__(ChronosDataService)
    svc._qdrant = None
    svc._embedder = None
    return svc


def test_embedder_is_built_without_a_gemini_key(monkeypatch):
    monkeypatch.setattr(embedding_service, "ChronosEmbeddingService", _Embedder)
    monkeypatch.setattr(qdrant_client, "ChronosQdrantClient", _Qdrant)
    monkeypatch.delenv("CHRONOS_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    svc = _service()
    svc._init_services()
    assert isinstance(svc._embedder, _Embedder)
    assert isinstance(svc._qdrant, _Qdrant)


def test_embedder_failure_leaves_search_on_the_text_fallback(monkeypatch):
    def _missing_key():
        raise ValueError("OpenAI embeddings are disabled")

    monkeypatch.setattr(embedding_service, "ChronosEmbeddingService", _missing_key)
    monkeypatch.setattr(qdrant_client, "ChronosQdrantClient", _Qdrant)
    svc = _service()
    svc._init_services()
    assert svc._embedder is None
    assert isinstance(svc._qdrant, _Qdrant)
