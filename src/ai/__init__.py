"""AI providers and embedding utilities."""
from .embeddings import get_embedder, EmbeddingError
from .providers import Provider, DEFAULT_PROVIDER

__all__ = ["get_embedder", "EmbeddingError", "Provider", "DEFAULT_PROVIDER"]
