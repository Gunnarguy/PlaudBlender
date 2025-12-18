import os

from src.plaud_oauth import PlaudOAuthClient
from src.plaud_client import PlaudClient
from src.vector_store import (
    get_vector_client as _get_core_vector_client,
    get_vector_db_provider,
    VectorDBProvider,
    is_qdrant,
)
from gui.state import state


def get_oauth_client() -> PlaudOAuthClient:
    if not state.plaud_oauth_client:
        state.plaud_oauth_client = PlaudOAuthClient()
    return state.plaud_oauth_client


def get_plaud_client() -> PlaudClient:
    if not state.plaud_client:
        state.plaud_client = PlaudClient(get_oauth_client())
    return state.plaud_client


def get_vector_db_client(collection_name: str = None):
    """Return the active vector database client (Qdrant by default)."""
    client = _get_core_vector_client(collection_name=collection_name)
    state.vector_client = client
    state.pinecone_client = client  # legacy alias
    return client


def get_pinecone_client(collection_name: str = None):
    """Backward-compatible wrapper returning the same vector client."""
    return get_vector_db_client(collection_name=collection_name)


def current_collection_name() -> str:
    """Get the current collection/index name for the active provider."""
    if is_qdrant():
        return os.getenv("QDRANT_COLLECTION", "transcripts")
    return os.getenv("PINECONE_INDEX_NAME", "transcripts")


def get_dashboard_url() -> str:
    """Get URL to vector DB dashboard (Qdrant only)."""
    if is_qdrant():
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        return f"{url.rstrip('/')}/dashboard"
    return None
