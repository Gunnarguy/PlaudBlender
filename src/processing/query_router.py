"""Query router for the legacy search pipeline.

The router classifies a user query into a coarse intent so the UI can decide
whether to run semantic/vector search or a metadata lookup.

This is intentionally heuristic (no model calls) to keep tests offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryIntent(str, Enum):
    semantic = "semantic"
    metadata = "metadata"


@dataclass(frozen=True)
class RoutedQuery:
    intent: QueryIntent
    query: str
    recording_id: Optional[str] = None


class QueryRouter:
    """Rule-based router."""

    def route(self, query: str) -> RoutedQuery:
        q = (query or "").strip()
        lower = q.lower()

        # Naive heuristics used by smoke tests.
        if "recording" in lower or "rec_" in lower or lower.startswith("rec"):
            # Try to extract a token that looks like an ID.
            tokens = q.replace(":", " ").replace(",", " ").split()
            rid = next((t for t in tokens if t.startswith("rec")), None)
            return RoutedQuery(intent=QueryIntent.metadata, query=q, recording_id=rid)

        return RoutedQuery(intent=QueryIntent.semantic, query=q)
