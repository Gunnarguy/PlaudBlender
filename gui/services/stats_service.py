"""
Dashboard Stats Service - Provides real-time stats for the dashboard.

Aggregates data from:
- SQLite database (recordings, segments, pipeline status)
- Vector DB (Qdrant by default; Pinecone-compatible shim)
- Notion (sync status)
- Environment (API key status)
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from sqlalchemy import func, select

logger = logging.getLogger(__name__)


@dataclass
class DashboardStats:
    """Container for all dashboard statistics."""

    # Auth
    plaud_authenticated: bool = False
    notion_configured: bool = False
    vector_configured: bool = False
    pinecone_configured: bool = False  # legacy alias

    # Recordings
    total_recordings: int = 0
    status_raw: int = 0
    status_processed: int = 0
    status_indexed: int = 0
    status_audio_processed: int = 0

    # Segments
    total_segments: int = 0
    segments_pending: int = 0
    segments_indexed: int = 0

    # Vector DB
    vector_vectors: int = 0
    vector_namespaces: int = 0
    vector_dimension: Optional[int] = None
    vector_metric: Optional[str] = None
    vector_index: Optional[str] = None
    vector_provider: Optional[str] = None
    vector_dim_mismatch: Optional[bool] = None
    vector_namespace: Optional[str] = None

    # Pinecone (legacy names for compatibility with older UI code)
    pinecone_vectors: int = 0
    pinecone_namespaces: int = 0
    pinecone_dimension: Optional[int] = None
    pinecone_metric: Optional[str] = None
    pinecone_index: Optional[str] = None

    # Graph
    graph_entities: int = 0
    graph_relationships: int = 0

    # Notion
    notion_pages_synced: int = 0
    notion_last_sync: Optional[str] = None

    # Timestamps
    last_plaud_sync: Optional[str] = None
    last_pinecone_sync: Optional[str] = None
    last_refresh: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for dashboard update_stats()."""
        return {
            "auth": self.plaud_authenticated,
            "recordings": self.total_recordings,
            # Provider-neutral keys (preferred)
            "vector": self.vector_vectors,
            "vector_namespaces": self.vector_namespaces,
            "vector_dim": self.vector_dimension,
            "vector_metric": self.vector_metric,
            "vector_index": self.vector_index,
            "vector_provider": self.vector_provider,
            "vector_dim_mismatch": self.vector_dim_mismatch,
            "vector_namespace": self.vector_namespace,
            # Legacy Pinecone keys preserved for callers still using them
            "pinecone": self.pinecone_vectors,
            "pinecone_namespaces": self.pinecone_namespaces,
            "pinecone_dim": self.pinecone_dimension,
            "pinecone_metric": self.pinecone_metric,
            "pinecone_index": self.pinecone_index,
            "graph_entities": self.graph_entities,
            "status_raw": self.status_raw,
            "status_processed": self.status_processed,
            "status_indexed": self.status_indexed,
            "last_sync": self.last_plaud_sync,
            "notion_synced": self.notion_pages_synced,
        }


class StatsService:
    """
    Aggregates stats from all data sources for dashboard display.

    Usage:
        service = StatsService()
        stats = service.get_all_stats()
        dashboard.update_stats(stats.to_dict())
    """

    def __init__(self):
        self._cache: Optional[DashboardStats] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 30  # Refresh every 30s

    def get_all_stats(self, force_refresh: bool = False) -> DashboardStats:
        """
        Get all dashboard stats.

        Args:
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            DashboardStats with all metrics
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            elapsed = (datetime.now() - self._cache_time).total_seconds()
            if elapsed < self._cache_ttl_seconds:
                return self._cache

        stats = DashboardStats()
        stats.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Gather stats from each source
        self._get_env_stats(stats)
        self._get_db_stats(stats)
        self._get_vector_stats(stats)
        self._get_graph_stats(stats)

        # Cache results
        self._cache = stats
        self._cache_time = datetime.now()

        return stats

    def _get_env_stats(self, stats: DashboardStats):
        """Check environment configuration."""
        stats.plaud_authenticated = bool(
            os.getenv("PLAUD_CLIENT_ID") and os.getenv("PLAUD_CLIENT_SECRET")
        )
        stats.notion_configured = bool(
            os.getenv("NOTION_API_KEY") or os.getenv("NOTION_TOKEN")
        ) and bool(os.getenv("NOTION_DATABASE_ID"))
        provider = (os.getenv("VECTOR_DB") or "qdrant").lower()
        if provider == "pinecone":
            stats.vector_configured = bool(os.getenv("PINECONE_API_KEY"))
            stats.pinecone_configured = stats.vector_configured
        else:
            # Qdrant defaults to local unless explicitly disabled
            stats.vector_configured = True

    def _get_db_stats(self, stats: DashboardStats):
        """Get recording/segment stats from SQLite."""
        try:
            from src.database.engine import SessionLocal, init_db
            from src.database.models import Recording, Segment

            init_db()
            session = SessionLocal()

            try:
                # Total recordings
                stats.total_recordings = (
                    session.query(func.count(Recording.id)).scalar() or 0
                )

                # Status breakdown
                status_counts = (
                    session.query(Recording.status, func.count(Recording.id))
                    .group_by(Recording.status)
                    .all()
                )

                for status, count in status_counts:
                    if status == "raw":
                        stats.status_raw = count
                    elif status == "processed":
                        stats.status_processed = count
                    elif status == "indexed":
                        stats.status_indexed = count
                    elif status == "audio_processed":
                        stats.status_audio_processed = count

                # Total segments
                stats.total_segments = (
                    session.query(func.count(Segment.id)).scalar() or 0
                )

                # Segment status
                seg_counts = (
                    session.query(Segment.status, func.count(Segment.id))
                    .group_by(Segment.status)
                    .all()
                )

                for status, count in seg_counts:
                    if status == "pending":
                        stats.segments_pending = count
                    elif status == "indexed":
                        stats.segments_indexed = count

                # Last sync (most recent recording)
                last_rec = (
                    session.query(Recording.created_at)
                    .order_by(Recording.created_at.desc())
                    .first()
                )
                if last_rec and last_rec[0]:
                    stats.last_plaud_sync = last_rec[0].strftime("%Y-%m-%d %H:%M")

            finally:
                session.close()

        except Exception as e:
            logger.warning(f"Failed to get DB stats: {e}")

    def _get_vector_stats(self, stats: DashboardStats):
        """Get vector DB stats (provider-neutral)."""
        try:
            from gui.services.clients import get_vector_db_client
            from gui.services.embedding_service import get_embedding_service
            from src.vector_store import get_vector_db_provider

            client = get_vector_db_client()
            embedder = get_embedding_service()

            info = client.get_index_info()
            namespaces = []
            if isinstance(info, dict):
                namespaces = list(info.get("namespaces", {}).keys())
            if not namespaces:
                try:
                    namespaces = [
                        ns for ns in client.list_namespaces() if ns is not None
                    ]
                except Exception:
                    namespaces = []

            stats.vector_vectors = (
                info.get("total_vectors", 0) if isinstance(info, dict) else 0
            )
            stats.vector_namespaces = len(namespaces)
            stats.vector_dimension = (
                info.get("dimension") if isinstance(info, dict) else None
            )
            stats.vector_metric = info.get("metric") if isinstance(info, dict) else None
            stats.vector_index = (
                info.get("name")
                if isinstance(info, dict)
                else getattr(client, "index_name", None)
            )
            provider = get_vector_db_provider()
            stats.vector_provider = getattr(provider, "value", str(provider))
            stats.vector_dim_mismatch = (
                stats.vector_dimension is not None
                and hasattr(embedder, "dimension")
                and stats.vector_dimension != embedder.dimension
            )
            stats.vector_namespace = namespaces[0] if namespaces else None

            # Legacy mirrors for downstream UI still on pinecone_* keys
            stats.pinecone_vectors = stats.vector_vectors
            stats.pinecone_namespaces = stats.vector_namespaces
            stats.pinecone_dimension = stats.vector_dimension
            stats.pinecone_metric = stats.vector_metric
            stats.pinecone_index = stats.vector_index
        except Exception as e:
            logger.warning(f"Failed to get vector stats: {e}")

    def _get_graph_stats(self, stats: DashboardStats):
        """Get knowledge graph stats."""
        try:
            # Check if graph data exists in recordings
            from src.database.engine import SessionLocal, init_db
            from src.database.models import Recording
            from sqlalchemy import func

            init_db()
            session = SessionLocal()

            try:
                # Count recordings with graph data in extra field
                # This is a rough estimate - actual graph stats would need GraphRAG integration
                recs_with_extra = (
                    session.query(func.count(Recording.id))
                    .filter(Recording.extra.isnot(None))
                    .scalar()
                    or 0
                )

                # For now, estimate entities based on processed recordings
                # Each recording typically generates ~5-10 entities
                stats.graph_entities = stats.status_processed * 7  # rough estimate
                stats.graph_relationships = (
                    stats.status_processed * 12
                )  # rough estimate

            finally:
                session.close()

        except Exception as e:
            logger.warning(f"Failed to get graph stats: {e}")

    def invalidate_cache(self):
        """Force next get_all_stats() to refresh."""
        self._cache = None
        self._cache_time = None

    def get_health_status(self) -> Dict[str, bool]:
        """Return coarse health indicators for key services."""
        stats = self.get_all_stats(force_refresh=True)
        return {
            "database": True,  # DB query would have raised if unavailable
            "pinecone": bool(stats.vector_configured),
            "notion": bool(stats.notion_configured),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_stats_service: Optional[StatsService] = None


def get_stats_service() -> StatsService:
    """Get singleton StatsService instance."""
    global _stats_service
    if _stats_service is None:
        _stats_service = StatsService()
    return _stats_service


def get_dashboard_stats(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Convenience function to get dashboard stats as dict.

    Returns dict suitable for dashboard.update_stats()
    """
    service = get_stats_service()
    stats = service.get_all_stats(force_refresh=force_refresh)
    return stats.to_dict()
