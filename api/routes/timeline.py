"""Timeline / day-view endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth.jwt import require_auth
from api.dependencies import get_service
from api.schemas.responses import DayStoryOut, DaySummaryOut, DaysResponse, EventOut
from app_v2.services.data_service import ChronosDataService

router = APIRouter(
    prefix="/api/v1/timeline",
    tags=["timeline"],
    dependencies=[Depends(require_auth)],
)


def _day_categories(d) -> tuple[dict, str | None, dict | None]:
    """Moment counts per category, the top one, and percentages.

    DaySummary carries only the counts (``categories``); top_category and
    category_percentages were read with getattr from attributes it never had,
    so they were always null and the app's category bars never drew (2026-10-06).
    """
    counts = {str(k): int(v) for k, v in (getattr(d, "categories", None) or {}).items() if v}
    total = sum(counts.values())
    if not total:
        return counts, None, None
    top = max(counts, key=lambda k: (counts[k], k))
    return counts, top, {k: round(100.0 * v / total, 1) for k, v in counts.items()}


def _day_sentiment(d) -> float | None:
    """Moment-weighted mean of the recordings' average sentiment; None without moments."""
    weighted = [
        (float(getattr(r, "avg_sentiment", 0.0) or 0.0), int(getattr(r, "event_count", 0) or 0))
        for r in (getattr(d, "recordings", None) or [])
    ]
    moments = sum(n for _, n in weighted)
    if not moments:
        return None
    return round(sum(s * n for s, n in weighted) / moments, 3)


def _day_to_out(d, *, recs=None) -> DaySummaryOut:
    counts, top, percentages = _day_categories(d)
    return DaySummaryOut(
        date=d.date,
        date_display=getattr(d, "date_display", None),
        total_duration_seconds=d.total_duration_seconds,
        recording_count=d.recording_count,
        event_count=d.event_count,
        coverage_status=getattr(d, "coverage_status", None),
        coverage_note=getattr(d, "coverage_note", None),
        top_category=getattr(d, "top_category", None) or top,
        category_percentages=getattr(d, "category_percentages", None) or percentages,
        categories=counts or None,
        avg_sentiment=_day_sentiment(d),
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


def _day_stories():
    from src.chronos.day_story import DayStoryService
    from src.database.engine import SessionLocal

    return DayStoryService(SessionLocal)


async def _day_story(date: str, svc: ChronosDataService, force: bool) -> DayStoryOut:
    import re

    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")
    out = await run_in_threadpool(_day_stories().get, svc, date, force=force)
    if out is None:
        raise HTTPException(status_code=404, detail=f"No data for {date}")
    return DayStoryOut(**out)


@router.get("/days/{date}/story", response_model=DayStoryOut)
async def day_story(date: str, svc: ChronosDataService = Depends(get_service)):
    """The day's written story: headline, paragraphs in time order, open threads.

    Written by Gemini through the AGY subscription from the day's moments, cached, and
    rewritten when the moments change. `pending` while being written (poll every few
    seconds; a stale story may be returned meanwhile with `stale: true`)."""
    return await _day_story(date, svc, force=False)


@router.post("/days/{date}/story", response_model=DayStoryOut)
async def rewrite_day_story(date: str, svc: ChronosDataService = Depends(get_service)):
    """Write the day's story again, now."""
    return await _day_story(date, svc, force=True)
