"""Statistics endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.routes._ttl_cache import TTLCache
from api.schemas.responses import StatsOut
from app_v2.services.data_service import ChronosDataService

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/stats",
    tags=["stats"],
    dependencies=[Depends(require_auth)],
)

_cache = TTLCache()
_STATS_TTL_SECONDS = 15.0
_WORKFLOW_STATS_TTL_SECONDS = 10.0


@router.get("", response_model=StatsOut)
async def get_stats(svc: ChronosDataService = Depends(get_service)):
    """Aggregate statistics across all recordings."""
    stats = _cache.get_or_compute(("stats",), _STATS_TTL_SECONDS, svc.get_stats)
    if stats is None:
        return StatsOut()
    raw_cbh = getattr(stats, "categories_by_hour", None)
    categories_by_hour = {str(k): v for k, v in raw_cbh.items()} if raw_cbh else None
    return StatsOut(
        total_recordings=getattr(stats, "total_recordings", 0),
        total_events=getattr(stats, "total_events", 0),
        total_days=getattr(stats, "total_days", 0),
        total_duration_hours=getattr(stats, "total_duration_hours", 0.0),
        categories=getattr(stats, "categories", {}),
        sentiment_avg=getattr(stats, "sentiment_avg", None),
        top_keywords=[
            {"keyword": kw, "count": ct}
            for kw, ct in (getattr(stats, "top_keywords", None) or [])
        ]
        or None,
        categories_by_hour=categories_by_hour,
        sentiment_distribution=getattr(stats, "sentiment_distribution", None),
        recent_days=getattr(stats, "recent_days", None),
    )


@router.get("/db", response_model=dict)
async def get_db_stats(svc: ChronosDataService = Depends(get_service)):
    """Database-level recording stats (status counts)."""
    return svc.get_recording_db_stats()


@router.get("/workflows", response_model=dict)
async def get_workflow_stats(svc: ChronosDataService = Depends(get_service)):
    """Plaud workflow stats (submitted, completed, failed counts)."""
    return _cache.get_or_compute(
        ("workflow-stats",),
        _WORKFLOW_STATS_TTL_SECONDS,
        svc.get_plaud_workflow_stats,
    )
