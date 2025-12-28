"""
Vector Store Abstraction Layer

Provides a unified interface to the Qdrant vector database.
Qdrant is used for all vector operations (local-first, granular, excellent visibility).
"""

import os
from typing import Optional, Protocol, List, Dict, Any
from enum import Enum
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class VectorDBProvider(Enum):
    """Supported vector database providers."""

    QDRANT = "qdrant"


class VectorStoreProtocol(Protocol):
    """
    Protocol defining the interface for vector store clients.

    QdrantVectorClient implements this interface.
    """

    collection_name: str

    def create_collection(
        self, name: str, dimension: int, metric: str = "cosine"
    ) -> bool: ...
    def list_collections(self) -> List[str]: ...
    def get_collection_info(self) -> Dict[str, Any]: ...
    def switch_collection(self, name: str) -> None: ...

    def upsert_vectors(self, vectors: List[Dict], namespace: str = "") -> bool: ...
    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Any]: ...
    def query_namespaces(
        self,
        query_embedding: List[float],
        namespaces: List[str],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        **kwargs,
    ) -> Any: ...
    def get_all_vectors(self, namespace: str = "", limit: int = 1000) -> List[Dict]: ...
    def fetch_vectors(self, ids: List[str], namespace: str = "") -> Dict[str, Any]: ...

    def delete_vectors(self, ids: List[str], namespace: str = "") -> bool: ...
    def delete_by_filter(self, filter_dict: Dict, namespace: str = "") -> bool: ...
    def delete_all(self, namespace: str = "") -> bool: ...

    def update_metadata(
        self, vec_id: str, metadata: Dict[str, Any], namespace: str = ""
    ) -> bool: ...
    def fetch_by_metadata(
        self, filter_dict: Dict, namespace: str = "", limit: int = 100
    ) -> List[Dict]: ...

    def list_namespaces(self) -> List[str]: ...
    def health_check(self) -> bool: ...


# Cached client instance
_vector_client: Optional[VectorStoreProtocol] = None
_current_provider: Optional[VectorDBProvider] = None


def get_vector_db_provider() -> VectorDBProvider:
    """
    Get the configured vector database provider.

    Returns Qdrant (the only supported provider).
    """
    return VectorDBProvider.QDRANT


def get_vector_client(
    collection_name: Optional[str] = None,
    force_provider: Optional[VectorDBProvider] = None,
) -> VectorStoreProtocol:
    """
    Get a vector store client instance.

    Args:
        collection_name: Override the default collection/index name
        force_provider: Force a specific provider (for testing)

    Returns:
        Vector client implementing VectorStoreProtocol

    The client is cached for performance. Call reset_vector_client()
    to force re-initialization.
    """
    global _vector_client, _current_provider

    provider = force_provider or get_vector_db_provider()

    # Return cached client if provider hasn't changed
    if _vector_client is not None and _current_provider == provider:
        if collection_name and collection_name != _vector_client.collection_name:
            _vector_client.switch_collection(collection_name)
        return _vector_client

    # Create Qdrant client (the only supported provider)
    from src.qdrant_client import QdrantVectorClient

    _vector_client = QdrantVectorClient(collection_name=collection_name)
    logger.info("Using Qdrant vector store (local-first)")

    _current_provider = provider
    return _vector_client


def reset_vector_client():
    """Reset the cached client (useful for testing or switching providers)."""
    global _vector_client, _current_provider
    _vector_client = None
    _current_provider = None


def is_qdrant() -> bool:
    """Check if current provider is Qdrant (always True)."""
    return True


def get_provider_info() -> Dict[str, Any]:
    """Get info about current vector database provider."""
    provider = get_vector_db_provider()
    client = get_vector_client()

    base_info = {
        "provider": provider.value,
        "collection": client.collection_name,
        "healthy": client.health_check(),
    }

    if provider == VectorDBProvider.QDRANT:
        base_info["dashboard_url"] = getattr(client, "dashboard_url", None)
        base_info["local"] = "localhost" in getattr(client, "url", "")

    return base_info
