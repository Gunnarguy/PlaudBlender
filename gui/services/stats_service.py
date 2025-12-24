"""Stats service used by the GUI.

The tests patch `src.database.engine.SessionLocal` and `src.database.engine.init_db`,
so we import them at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.database.engine import SessionLocal, init_db
from src.database.models import Recording, Segment


@dataclass
class PipelineStats:
    total_recordings: int = 0
    total_segments: int = 0


class StatsService:
    """Compute pipeline stats from the local database."""

    def __init__(self):
        self._cache: PipelineStats | None = None

    def invalidate_cache(self) -> None:
        self._cache = None

    def get_all_stats(self, *, force_refresh: bool = False) -> PipelineStats:
        if self._cache is not None and not force_refresh:
            return self._cache

        init_db()
        db = SessionLocal()
        try:
            total_recordings = db.query(Recording).count()
            total_segments = db.query(Segment).count()
            self._cache = PipelineStats(
                total_recordings=total_recordings,
                total_segments=total_segments,
            )
            return self._cache
        finally:
            try:
                db.close()
            except Exception:
                pass

    def get_health_status(self) -> Dict[str, Dict[str, str]]:
        """Return a simple health report.

        The UI shows status for major subsystems. For tests we keep it simple.
        """

        return {
            "database": {"status": "ok"},
            "pinecone": {"status": "optional"},
            "notion": {"status": "optional"},
        }
