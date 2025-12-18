"""gui.services.pinecone_service

Compatibility shim.

The project is now vector-store agnostic (Qdrant by default). This module keeps
the legacy import path and function names working by delegating to
`gui.services.vector_service`.
"""

from __future__ import annotations

from gui.services import vector_service as _vector_service

# Re-export helpers (e.g. _format_vector) and newer provider-neutral functions.
from gui.services.vector_service import *  # noqa: F401,F403

# NOTE: Python's `from module import *` does NOT import underscore-prefixed names.
# The GUI still imports `_format_vector` from this legacy module in a few places.
from gui.services.vector_service import _format_vector  # noqa: F401


# ---- Legacy Pinecone-named API (thin aliases) ------------------------


def get_indexes_and_namespaces():
    """Legacy alias for `vector_service.get_collections_and_namespaces()`."""
    return _vector_service.get_collections_and_namespaces()


def switch_index(index_name: str):
    """Legacy alias for `vector_service.switch_collection()`."""
    return _vector_service.switch_collection(index_name)


def ensure_matching_index(target_index: str, dimension: int) -> str:
    """Legacy alias for `vector_service.ensure_matching_collection()`."""
    return _vector_service.ensure_matching_collection(target_index, dimension)


def find_matching_index(dimension: int):
    """Legacy alias for `vector_service.find_matching_collection()`."""
    return _vector_service.find_matching_collection(dimension)


def reembed_all_into_index(index_name: str, namespace: str = ""):
    """Legacy alias for `vector_service.reembed_all_into_collection()`."""
    return _vector_service.reembed_all_into_collection(index_name, namespace=namespace)
