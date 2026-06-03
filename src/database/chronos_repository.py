"""Chronos-specific database repository functions.

Provides CRUD operations for ChronosRecording, ChronosEvent, and
ChronosProcessingJob tables. Keeps Chronos data access isolated from
legacy Recording/Segment logic.
"""

from datetime import datetime
from typing import Any, List, Optional
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from .models import (
    ChronosExecutionRun,
    ChronosExecutionSpan,
    ChronosRecording,
    ChronosEvent,
    ChronosProcessingJob,
)


# ═══════════════════════════════════════════════════════════════════
# ChronosRecording Operations
# ═══════════════════════════════════════════════════════════════════


def upsert_chronos_recording(
    session: Session,
    recording_id: str,
    title: Optional[str],
    created_at: datetime,
    duration_seconds: int,
    local_audio_path: str,
    source: str = "plaud",
    device_id: Optional[str] = None,
    checksum: Optional[str] = None,
    time_is_estimated: Optional[bool] = None,
    time_estimate_reason: Optional[str] = None,
) -> ChronosRecording:
    """Insert or update a Chronos recording.

    Args:
        session: SQLAlchemy session
        recording_id: Plaud API recording ID
        title: Optional human title
        created_at: Recording timestamp (UTC)
        duration_seconds: Total duration
        local_audio_path: Path to downloaded audio
        source: Source system (default: plaud)
        device_id: Hardware device identifier
        checksum: SHA256 hash for integrity

    Returns:
        ChronosRecording: The upserted recording instance
    """
    rec = session.query(ChronosRecording).filter_by(recording_id=recording_id).first()

    if rec:
        # Update existing
        rec.title = title
        rec.created_at = created_at
        rec.duration_seconds = duration_seconds
        rec.local_audio_path = local_audio_path
        rec.source = source
        rec.device_id = device_id
        rec.checksum = checksum
        rec.time_is_estimated = time_is_estimated
        rec.time_estimate_reason = time_estimate_reason
    else:
        # Insert new
        rec = ChronosRecording(
            recording_id=recording_id,
            title=title,
            created_at=created_at,
            duration_seconds=duration_seconds,
            local_audio_path=local_audio_path,
            source=source,
            device_id=device_id,
            checksum=checksum,
            time_is_estimated=time_is_estimated,
            time_estimate_reason=time_estimate_reason,
        )
        session.add(rec)

    session.commit()
    session.refresh(rec)
    return rec


def set_chronos_recording_transcript(
    session: Session,
    recording_id: str,
    transcript_text: str,
) -> None:
    """Cache transcript text on a Chronos recording.

    This enables the UI to show a "library of transcriptions" without re-calling
    Plaud on every page refresh.
    """

    rec = session.query(ChronosRecording).filter_by(recording_id=recording_id).first()
    if not rec:
        return

    rec.transcript = transcript_text
    rec.transcript_cached_at = datetime.utcnow()
    session.commit()


def get_chronos_recording(
    session: Session, recording_id: str
) -> Optional[ChronosRecording]:
    """Fetch a recording by ID."""
    return session.query(ChronosRecording).filter_by(recording_id=recording_id).first()


def get_pending_chronos_recordings(
    session: Session, limit: int = 100
) -> List[ChronosRecording]:
    """Fetch recordings that are pending processing."""
    return (
        session.query(ChronosRecording)
        .filter_by(processing_status="pending")
        .limit(limit)
        .all()
    )


def mark_chronos_recording_status(
    session: Session,
    recording_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """Update processing status for a recording.

    Args:
        session: SQLAlchemy session
        recording_id: Recording to update
        status: New status (pending | processing | completed | failed)
        error_message: Error details if status is failed
    """
    rec = session.query(ChronosRecording).filter_by(recording_id=recording_id).first()
    if rec:
        rec.processing_status = status
        rec.error_message = error_message
        if status == "completed":
            rec.processed_at = datetime.utcnow()
        session.commit()


# ═══════════════════════════════════════════════════════════════════
# ChronosEvent Operations
# ═══════════════════════════════════════════════════════════════════


def add_chronos_events(session: Session, events: List[ChronosEvent]) -> int:
    """Bulk insert events.

    Args:
        session: SQLAlchemy session
        events: List of ChronosEvent instances

    Returns:
        int: Number of events inserted
    """
    session.add_all(events)
    session.commit()
    return len(events)


def get_chronos_events_by_recording(
    session: Session,
    recording_id: str,
) -> List[ChronosEvent]:
    """Fetch all events for a given recording."""
    return (
        session.query(ChronosEvent)
        .filter_by(recording_id=recording_id)
        .order_by(ChronosEvent.start_ts)
        .all()
    )


def delete_chronos_events_by_recording(session: Session, recording_id: str) -> int:
    """Delete all Chronos events for a recording.

    Returns the number of deleted rows.
    """

    q = session.query(ChronosEvent).filter_by(recording_id=recording_id)
    count = q.count()
    q.delete(synchronize_session=False)
    session.commit()
    return int(count)


def get_chronos_events_by_day(
    session: Session,
    day_of_week: str,
    limit: int = 1000,
) -> List[ChronosEvent]:
    """Fetch events for a specific day of week (e.g., 'Monday')."""
    return (
        session.query(ChronosEvent)
        .filter_by(day_of_week=day_of_week)
        .order_by(ChronosEvent.start_ts)
        .limit(limit)
        .all()
    )


def get_chronos_events_by_date_range(
    session: Session,
    start_date: datetime,
    end_date: datetime,
    limit: int = 1000,
) -> List[ChronosEvent]:
    """Fetch events within a date range."""
    return (
        session.query(ChronosEvent)
        .filter(
            and_(
                ChronosEvent.start_ts >= start_date,
                ChronosEvent.start_ts <= end_date,
            )
        )
        .order_by(ChronosEvent.start_ts)
        .limit(limit)
        .all()
    )


def get_chronos_events_by_category(
    session: Session,
    category: str,
    limit: int = 1000,
) -> List[ChronosEvent]:
    """Fetch events by category (work, personal, meeting, etc.)."""
    return (
        session.query(ChronosEvent)
        .filter_by(category=category)
        .order_by(ChronosEvent.start_ts)
        .limit(limit)
        .all()
    )


# ═══════════════════════════════════════════════════════════════════
# ChronosProcessingJob Operations
# ═══════════════════════════════════════════════════════════════════


def enqueue_chronos_job(
    session: Session,
    recording_id: str,
    job_type: str,
    priority: int = 0,
) -> ChronosProcessingJob:
    """Create a new processing job.

    Args:
        session: SQLAlchemy session
        recording_id: Recording to process
        job_type: Job type (gemini_clean | qdrant_index | graph_extract)
        priority: Job priority (higher = more urgent)

    Returns:
        ChronosProcessingJob: The created job instance
    """
    job = ChronosProcessingJob(
        recording_id=recording_id,
        job_type=job_type,
        priority=priority,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_next_chronos_job(
    session: Session, job_type: Optional[str] = None
) -> Optional[ChronosProcessingJob]:
    """Fetch the next queued job (highest priority first).

    Args:
        session: SQLAlchemy session
        job_type: Filter by job type (optional)

    Returns:
        ChronosProcessingJob: Next job to process, or None
    """
    query = session.query(ChronosProcessingJob).filter_by(status="queued")

    if job_type:
        query = query.filter_by(job_type=job_type)

    return query.order_by(ChronosProcessingJob.priority.desc()).first()


def mark_chronos_job_status(
    session: Session,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """Update job status.

    Args:
        session: SQLAlchemy session
        job_id: Job ID
        status: New status (queued | running | completed | failed)
        error_message: Error details if status is failed
    """
    job = session.query(ChronosProcessingJob).filter_by(job_id=job_id).first()
    if job:
        job.status = status
        job.error_message = error_message

        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in ("completed", "failed"):
            job.completed_at = datetime.utcnow()

        session.commit()


def retry_failed_chronos_job(session: Session, job_id: str) -> None:
    """Reset a failed job for retry."""
    job = session.query(ChronosProcessingJob).filter_by(job_id=job_id).first()
    if job:
        job.status = "queued"
        job.retry_count += 1
        job.started_at = None
        job.completed_at = None
        session.commit()


# ═══════════════════════════════════════════════════════════════════
# ChronosExecutionRun / ChronosExecutionSpan Operations
# ═══════════════════════════════════════════════════════════════════


def create_execution_run(
    session: Session,
    *,
    run_id: Optional[str] = None,
    trigger: Optional[str] = None,
    source: Optional[str] = None,
    title: Optional[str] = None,
    host: Optional[str] = None,
    process_id: Optional[int] = None,
    entrypoint: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ChronosExecutionRun:
    """Create a top-level observable execution run."""
    run = ChronosExecutionRun(
        run_id=run_id,
        trigger=trigger,
        source=source,
        title=title,
        host=host,
        process_id=process_id,
        entrypoint=entrypoint,
        run_metadata=metadata,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def finish_execution_run(
    session: Session,
    run_id: str,
    *,
    status: str = "completed",
    summary: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """Mark an execution run complete/failed and compute duration."""
    run = session.query(ChronosExecutionRun).filter_by(run_id=run_id).first()
    if not run:
        return
    ended_at = datetime.utcnow()
    run.status = status
    run.ended_at = ended_at
    run.error_message = error_message
    if summary is not None:
        run.summary = summary
    if run.started_at:
        run.duration_ms = (ended_at - run.started_at).total_seconds() * 1000
    session.commit()


def start_execution_span(
    session: Session,
    *,
    span_id: Optional[str] = None,
    run_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    event_id: Optional[str] = None,
    stage: Optional[str] = None,
    operation: str,
    source: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    message: Optional[str] = None,
    detail: Optional[str] = None,
    input_hash: Optional[str] = None,
    input_snippet: Optional[str] = None,
    request_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ChronosExecutionSpan:
    """Start a granular execution span."""
    span = ChronosExecutionSpan(
        span_id=span_id,
        run_id=run_id,
        parent_span_id=parent_span_id,
        recording_id=recording_id,
        event_id=event_id,
        stage=stage,
        operation=operation,
        source=source,
        provider=provider,
        model=model,
        message=message,
        detail=detail,
        input_hash=input_hash,
        input_snippet=input_snippet,
        request_id=request_id,
        span_metadata=metadata,
    )
    session.add(span)
    session.commit()
    session.refresh(span)
    return span


def finish_execution_span(
    session: Session,
    span_id: str,
    *,
    status: str = "completed",
    level: str = "info",
    message: Optional[str] = None,
    detail: Optional[str] = None,
    output_hash: Optional[str] = None,
    output_snippet: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    retry_count: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    """Finish/update a granular execution span."""
    span = session.query(ChronosExecutionSpan).filter_by(span_id=span_id).first()
    if not span:
        return
    ended_at = datetime.utcnow()
    span.status = status
    span.level = level
    span.ended_at = ended_at
    span.error_message = error_message
    if message is not None:
        span.message = message
    if detail is not None:
        span.detail = detail
    if output_hash is not None:
        span.output_hash = output_hash
    if output_snippet is not None:
        span.output_snippet = output_snippet
    if input_tokens is not None:
        span.input_tokens = input_tokens
    if output_tokens is not None:
        span.output_tokens = output_tokens
    if cost_usd is not None:
        span.cost_usd = cost_usd
    if retry_count is not None:
        span.retry_count = retry_count
    if metadata is not None:
        current = dict(span.span_metadata or {})
        current.update(metadata)
        span.span_metadata = current
    if span.started_at:
        span.duration_ms = (ended_at - span.started_at).total_seconds() * 1000
    session.commit()


def list_execution_runs(session: Session, limit: int = 25) -> list[ChronosExecutionRun]:
    """Return recent execution runs newest first."""
    return (
        session.query(ChronosExecutionRun)
        .order_by(ChronosExecutionRun.started_at.desc())
        .limit(max(1, min(int(limit), 200)))
        .all()
    )


def list_execution_spans(
    session: Session,
    *,
    run_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    stage: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100,
) -> list[ChronosExecutionSpan]:
    """Return recent execution spans, optionally filtered."""
    q = session.query(ChronosExecutionSpan)
    if run_id:
        q = q.filter(ChronosExecutionSpan.run_id == run_id)
    if recording_id:
        q = q.filter(ChronosExecutionSpan.recording_id == recording_id)
    if stage:
        q = q.filter(ChronosExecutionSpan.stage == stage)
    if source:
        q = q.filter(ChronosExecutionSpan.source == source)
    return (
        q.order_by(ChronosExecutionSpan.started_at.desc())
        .limit(max(1, min(int(limit), 500)))
        .all()
    )


def get_execution_run(session: Session, run_id: str) -> Optional[ChronosExecutionRun]:
    """Fetch a single execution run by ID."""
    return session.query(ChronosExecutionRun).filter_by(run_id=run_id).first()


# ═══════════════════════════════════════════════════════════════════
# Webhook event persistence for Plaud → Chronos integration
# ═══════════════════════════════════════════════════════════════════


def add_chronos_webhook_event(
    session: Session,
    webhook_id: Optional[str],
    event_type: str,
    payload: dict,
    headers: Optional[dict] = None,
    recording_id: Optional[str] = None,
) -> str:
    """Persist an incoming webhook event. Returns generated event_id."""
    from .models import ChronosWebhookEvent

    ev = ChronosWebhookEvent(
        webhook_id=webhook_id,
        event_type=event_type,
        payload=payload,
        headers=headers,
        recording_id=recording_id,
    )
    session.add(ev)
    session.commit()
    session.refresh(ev)
    # Return a plain string for callers (SQLAlchemy attribute will be resolved at runtime)
    return str(ev.event_id)


def list_chronos_webhook_events(session: Session, limit: int = 200):
    """Return recent webhook events ordered by received_at desc."""
    from .models import ChronosWebhookEvent

    return (
        session.query(ChronosWebhookEvent)
        .order_by(ChronosWebhookEvent.received_at.desc())
        .limit(limit)
        .all()
    )


def mark_webhook_event_processed(session: Session, event_id: str, status: str = "processed") -> None:
    """Mark a webhook event as processed/failed."""
    from .models import ChronosWebhookEvent

    ev = session.query(ChronosWebhookEvent).filter_by(event_id=event_id).first()
    if not ev:
        return
    # Use setattr to avoid static type-checker complaints about SQLAlchemy Column descriptors.
    setattr(ev, "processed", status)
    setattr(ev, "processed_at", datetime.utcnow())
    session.commit()
