"""Search helpers for the GUI.

These are light-weight wrappers; in production you'd call Qdrant/Pinecone.
For tests we just provide importable functions.
"""

from __future__ import annotations

from typing import Any, Dict, List


def semantic_search(query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    return []


def search_with_rerank(query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    return semantic_search(query, limit=limit)


def cross_namespace_search(
    query: str, namespaces: List[str], *, limit: int = 10
) -> List[Dict[str, Any]]:
    return []
