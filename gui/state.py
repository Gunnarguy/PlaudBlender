from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppState:
    """Central store for runtime state shared across views."""

    plaud_client: Optional[Any] = None
    plaud_oauth_client: Optional[Any] = None
    vector_client: Optional[Any] = None
    pinecone_client: Optional[Any] = None  # legacy alias

    transcripts: List[Dict[str, Any]] = field(default_factory=list)
    filtered_transcripts: List[Dict[str, Any]] = field(default_factory=list)

    vector_vectors: List[Any] = field(default_factory=list)
    vector_collections: List[str] = field(default_factory=list)
    vector_stats: Dict[str, Any] = field(default_factory=dict)
    vector_selected_collection: Optional[str] = None

    # Legacy Pinecone-named fields kept for compatibility
    pinecone_vectors: List[Any] = field(default_factory=list)
    pinecone_indexes: List[str] = field(default_factory=list)
    pinecone_stats: Dict[str, Any] = field(default_factory=dict)
    pinecone_selected_index: Optional[str] = None

    saved_searches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    search_results: List[Dict[str, Any]] = field(default_factory=list)

    status_message: str = "Initializing..."
    is_busy: bool = False
    is_authenticated: bool = False
    vector_store_ready: bool = False
    pinecone_loaded: bool = False  # legacy

    logs: List[str] = field(default_factory=list)

    # Performance metrics for status bar display
    last_latency_ms: Optional[float] = None  # Last query latency in ms
    last_read_units: Optional[int] = (
        None  # Last vector store read units consumed (if tracked)
    )
    last_write_units: Optional[int] = (
        None  # Last vector store write units consumed (if tracked)
    )
    active_namespace: Optional[str] = None  # Currently active namespace
    hybrid_alpha: float = 0.7  # Current hybrid search alpha value

    def set_status(self, message: str, busy: bool = False):
        self.status_message = message
        self.is_busy = busy

    def set_metrics(
        self,
        latency_ms: float = None,
        read_units: int = None,
        write_units: int = None,
        namespace: str = None,
    ):
        """Update performance metrics for display."""
        if latency_ms is not None:
            self.last_latency_ms = latency_ms
        if read_units is not None:
            self.last_read_units = read_units
        if write_units is not None:
            self.last_write_units = write_units
        if namespace is not None:
            self.active_namespace = namespace


state = AppState()
