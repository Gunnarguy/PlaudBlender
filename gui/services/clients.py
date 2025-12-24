"""Client factories for the GUI layer."""

from __future__ import annotations

from typing import Optional

from src.plaud_client import PlaudClient


def get_plaud_client() -> PlaudClient:
    """Return a Plaud API client."""

    return PlaudClient()


def get_pinecone_client() -> Optional[object]:
    """Legacy: return a Pinecone client.

    The project is moving Qdrant-first, but some older GUI wiring (and tests)
    still expect this helper to exist.

    We return None by default to avoid requiring Pinecone credentials.
    """

    return None
