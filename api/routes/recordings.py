"""Recording and event detail endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.auth.jwt import require_auth
from api.dependencies import get_service
from api.schemas.responses import (
    CategoryOverrideRequest,
    EventOut,
    RecordingDetailOut,
    RecordingSummaryOut,
    SuccessResponse,
)
from app_v2.services.data_service import ChronosDataService

router = APIRouter(
    prefix="/api/v1/recordings",
    tags=["recordings"],
    dependencies=[Depends(require_auth)],
)


# ── Helpers ─────────────────────────────────────────────────


def _event_to_out(e) -> EventOut:
    """Convert a data_service Event dataclass to API model."""
    return EventOut(
        id=getattr(e, "id", "") or getattr(e, "event_id", ""),
        recording_id=e.recording_id,
        start_ts=str(e.start_ts),
        end_ts=str(e.end_ts),
        day_of_week=e.day_of_week,
        hour_of_day=e.hour_of_day,
        clean_text=e.clean_text,
        category=e.category,
        category_confidence=getattr(e, "category_confidence", None),
        sentiment=getattr(e, "sentiment", None),
        keywords=getattr(e, "keywords", None) or [],
        speaker=getattr(e, "speaker", "self_talk"),
        duration_seconds=getattr(e, "duration_seconds", 0.0),
    )


def _recording_summary_to_out(r) -> RecordingSummaryOut:
    """Convert a data_service RecordingSummary to API model."""
    return RecordingSummaryOut(
        recording_id=r.recording_id,
        start_time=str(r.start_time) if getattr(r, "start_time", None) else None,
        end_time=str(r.end_time) if getattr(r, "end_time", None) else None,
        duration_seconds=getattr(r, "duration_seconds", 0),
        duration_formatted=getattr(r, "duration_formatted", None),
        top_category=getattr(r, "top_category", "unknown"),
        event_count=getattr(r, "event_count", 0),
        time_range_formatted=getattr(r, "time_range_formatted", None),
        time_is_estimated=getattr(r, "time_is_estimated", None),
        time_estimate_reason=getattr(r, "time_estimate_reason", None),
        title=getattr(r, "title", None),
        plaud_ai_summary=getattr(r, "plaud_ai_summary", None),
        cloud_status=getattr(r, "cloud_status", None),
    )


# ── Routes ──────────────────────────────────────────────────


@router.get("/{recording_id}", response_model=RecordingDetailOut)
async def recording_detail(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Full recording detail with events, transcript, AI summary."""
    detail = svc.get_recording_detail(recording_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Recording not found")

    summary = (
        _recording_summary_to_out(detail.summary)
        if hasattr(detail, "summary")
        else RecordingSummaryOut(recording_id=recording_id)
    )
    events = (
        [_event_to_out(e) for e in detail.events] if hasattr(detail, "events") else []
    )

    return RecordingDetailOut(
        summary=summary,
        events=events,
        category_percentages=getattr(detail, "category_percentages", None),
        transcript=getattr(detail, "transcript", None),
        ai_summary=getattr(detail, "ai_summary", None),
        extracted_data=getattr(detail, "extracted_data", None),
        workflow_status=getattr(detail, "workflow_status", None),
        plaud_transcript=getattr(detail, "plaud_transcript", None),
    )


@router.get("/{recording_id}/events", response_model=list[EventOut])
async def recording_events(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """All events for a recording."""
    events = svc.get_events_for_recording(recording_id)
    return [_event_to_out(e) for e in events]


@router.get("/{recording_id}/transcript")
async def recording_transcript(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Raw transcript text."""
    text = svc.get_transcript(recording_id)
    return {"transcript": text or ""}


@router.get("/{recording_id}/ai-summary")
async def recording_ai_summary(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Plaud AI summary."""
    summary = svc.get_ai_summary(recording_id)
    return {"ai_summary": summary or ""}


@router.get("/{recording_id}/extracted-data")
async def recording_extracted_data(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Plaud extracted data (JSON)."""
    data = svc.get_extracted_data(recording_id)
    return {"extracted_data": data or {}}


@router.get("/{recording_id}/plaud-transcript")
async def recording_plaud_transcript(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Plaud workflow transcript (for comparison)."""
    text = svc.get_plaud_workflow_transcript(recording_id)
    return {"plaud_transcript": text or ""}


@router.put(
    "/{recording_id}/events/{event_id}/category",
    response_model=SuccessResponse,
)
async def set_category_override(
    recording_id: str,
    event_id: str,
    body: CategoryOverrideRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """Override event category."""
    svc.save_category_override(event_id, body.category)
    return SuccessResponse(message=f"Category set to {body.category}")


@router.get("/{recording_id}/category-overrides")
async def get_category_overrides(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """All category overrides for a recording."""
    overrides = svc.get_category_overrides(recording_id)
    return {"overrides": overrides or {}}
