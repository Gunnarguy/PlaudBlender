"""Timeline / day-view endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth.jwt import require_auth
from api.dependencies import get_service
from api.schemas.responses import DaySummaryOut, DaysResponse, EventOut
from app_v2.services.data_service import ChronosDataService

router = APIRouter(
    prefix="/api/v1/timeline",
    tags=["timeline"],
    dependencies=[Depends(require_auth)],
)


def _day_to_out(d, *, recs=None) -> DaySummaryOut:
    return DaySummaryOut(
        date=d.date,
        date_display=getattr(d, "date_display", None),
        total_duration_seconds=d.total_duration_seconds,
        recording_count=d.recording_count,
        event_count=d.event_count,
        coverage_status=getattr(d, "coverage_status", None),
        coverage_note=getattr(d, "coverage_note", None),
        top_category=getattr(d, "top_category", None),
        category_percentages=getattr(d, "category_percentages", None),
        top_keywords=getattr(d, "top_keywords", None),
        ai_summary=getattr(d, "ai_summary", None),
        recordings=recs,
    )


@router.get("/days", response_model=DaysResponse)
async def list_days(
    limit: Optional[int] = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    svc: ChronosDataService = Depends(get_service),
):
    """All days with recording counts and category summaries.

    Supports optional pagination via limit/offset query params.
    If limit is omitted, all days are returned.
    """
    days = svc.get_days()
    total = len(days)
    # Apply pagination
    if offset:
        days = days[offset:]
    if limit is not None:
        days = days[:limit]
    out = [_day_to_out(d) for d in days]
    return DaysResponse(days=out, total=total)


@router.get("/days-filled", response_model=DaysResponse)
async def list_days_filled(
    limit: Optional[int] = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
    svc: ChronosDataService = Depends(get_service),
):
    """Days with recordings list pre-attached."""
    days = svc.get_days_filled()
    # Apply date filters
    if start_date:
        days = [d for d in days if d.date >= start_date]
    if end_date:
        days = [d for d in days if d.date <= end_date]
    total = len(days)
    if offset:
        days = days[offset:]
    if limit is not None:
        days = days[:limit]
    out = []
    for d in days:
        recs = None
        if hasattr(d, "recordings") and d.recordings:
            from api.routes.recordings import _recording_summary_to_out

            recs = [_recording_summary_to_out(r) for r in d.recordings]
        out.append(_day_to_out(d, recs=recs))
    return DaysResponse(days=out, total=total)


@router.get("/days/{date}", response_model=DaySummaryOut)
async def day_detail(date: str, svc: ChronosDataService = Depends(get_service)):
    """Single day detail (date format: YYYY-MM-DD)."""
    from fastapi import HTTPException

    d = svc.get_day_detail(date)
    if d is None:
        raise HTTPException(status_code=404, detail=f"No data for {date}")

    recs = None
    if hasattr(d, "recordings") and d.recordings:
        from api.routes.recordings import _recording_summary_to_out

        recs = [_recording_summary_to_out(r) for r in d.recordings]

    return _day_to_out(d, recs=recs)
