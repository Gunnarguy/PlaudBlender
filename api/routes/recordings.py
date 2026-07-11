"""Recording and event detail endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.auth.jwt import require_auth
from api.dependencies import get_service
from api.schemas.responses import (
    CategoryOverrideRequest,
    EventOut,
    RecordingProcessingOut,
    RecordingDetailOut,
    RecordingSummaryOut,
    SuccessResponse,
    TraceRunOut,
    TraceSpanOut,
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


def _trace_run_to_out(run) -> TraceRunOut:
    return TraceRunOut(
        run_id=str(run.run_id),
        trigger=getattr(run, "trigger", None),
        source=getattr(run, "source", None),
        status=getattr(run, "status", "running"),
        title=getattr(run, "title", None),
        started_at=str(run.started_at) if getattr(run, "started_at", None) else None,
        ended_at=str(run.ended_at) if getattr(run, "ended_at", None) else None,
        duration_ms=getattr(run, "duration_ms", None),
        summary=getattr(run, "summary", None),
        metadata=getattr(run, "run_metadata", None),
        error_message=getattr(run, "error_message", None),
    )


def _trace_span_to_out(span) -> TraceSpanOut:
    return TraceSpanOut(
        span_id=str(span.span_id),
        run_id=getattr(span, "run_id", None),
        parent_span_id=getattr(span, "parent_span_id", None),
        recording_id=getattr(span, "recording_id", None),
        event_id=getattr(span, "event_id", None),
        stage=getattr(span, "stage", None),
        operation=getattr(span, "operation", "unknown"),
        source=getattr(span, "source", None),
        provider=getattr(span, "provider", None),
        model=getattr(span, "model", None),
        status=getattr(span, "status", "running"),
        level=getattr(span, "level", "info"),
        message=getattr(span, "message", None),
        detail=getattr(span, "detail", None),
        started_at=str(span.started_at) if getattr(span, "started_at", None) else None,
        ended_at=str(span.ended_at) if getattr(span, "ended_at", None) else None,
        duration_ms=getattr(span, "duration_ms", None),
        input_hash=getattr(span, "input_hash", None),
        output_hash=getattr(span, "output_hash", None),
        input_snippet=getattr(span, "input_snippet", None),
        output_snippet=getattr(span, "output_snippet", None),
        input_tokens=getattr(span, "input_tokens", None),
        output_tokens=getattr(span, "output_tokens", None),
        cost_usd=getattr(span, "cost_usd", None),
        request_id=getattr(span, "request_id", None),
        retry_count=int(getattr(span, "retry_count", 0) or 0),
        metadata=getattr(span, "span_metadata", None),
        error_message=getattr(span, "error_message", None),
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


@router.get("/{recording_id}/processing", response_model=RecordingProcessingOut)
async def recording_processing(recording_id: str):
    """Processing lineage and per-recording execution telemetry."""
    from src.database.engine import SessionLocal
    from src.database.chronos_repository import (
        get_chronos_recording,
        get_execution_run,
        list_execution_spans,
    )

    with SessionLocal() as session:
        recording = get_chronos_recording(session, recording_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="Recording not found")

        spans = list_execution_spans(session, recording_id=recording_id, limit=500)
        run_ids = []
        for span in spans:
            if span.run_id and span.run_id not in run_ids:
                run_ids.append(span.run_id)
        runs = [run for rid in run_ids if (run := get_execution_run(session, rid))]

        total_cost = sum(float(getattr(span, "cost_usd", 0) or 0) for span in spans)
        total_input = sum(int(getattr(span, "input_tokens", 0) or 0) for span in spans)
        total_output = sum(int(getattr(span, "output_tokens", 0) or 0) for span in spans)
        providers = sorted({str(span.provider) for span in spans if getattr(span, "provider", None)})
        models = sorted({str(span.model) for span in spans if getattr(span, "model", None)})

        return RecordingProcessingOut(
            recording_id=recording_id,
            status=getattr(recording, "processing_status", "unknown"),
            latest_error=getattr(recording, "error_message", None),
            runs=[_trace_run_to_out(run) for run in runs],
            spans=[_trace_span_to_out(span) for span in reversed(spans)],
            totals={
                "span_count": len(spans),
                "total_cost_usd": total_cost,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "providers": providers,
                "models": models,
            },
        )


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
    from fastapi import HTTPException
    res = svc.save_category_override(event_id, body.category)
    is_success = res.get("success", False) if isinstance(res, dict) else bool(res)
    if not is_success:
        detail = "Event not found"
        if isinstance(res, dict) and res.get("lock_encountered"):
            detail = "Database lock/timeout encountered"
        elif isinstance(res, dict) and res.get("errors"):
            detail = f"Failed to override category: {', '.join(res.get('errors'))}"
        raise HTTPException(status_code=400, detail=detail)
    return SuccessResponse(message=f"Category set to {body.category}")


@router.get("/{recording_id}/category-overrides")
async def get_category_overrides(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """All category overrides for a recording."""
    overrides = svc.get_category_overrides(recording_id)
    return {"overrides": overrides or {}}
