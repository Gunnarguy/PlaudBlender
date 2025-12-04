from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppState:
    """Central store for runtime state shared across views."""

    plaud_client: Optional[Any] = None
    plaud_oauth_client: Optional[Any] = None
    pinecone_client: Optional[Any] = None

    transcripts: List[Dict[str, Any]] = field(default_factory=list)
    filtered_transcripts: List[Dict[str, Any]] = field(default_factory=list)

    pinecone_vectors: List[Any] = field(default_factory=list)
    pinecone_indexes: List[str] = field(default_factory=list)
    pinecone_stats: Dict[str, Any] = field(default_factory=dict)

    saved_searches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    search_results: List[Dict[str, Any]] = field(default_factory=list)

    status_message: str = "Initializing..."
    is_busy: bool = False
    is_authenticated: bool = False
    pinecone_loaded: bool = False

    logs: List[str] = field(default_factory=list)

    def set_status(self, message: str, busy: bool = False):
        self.status_message = message
        self.is_busy = busy


state = AppState()
