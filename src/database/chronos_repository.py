"""Chronos-specific database repository functions.

Provides CRUD operations for ChronosRecording, ChronosEvent, and
ChronosProcessingJob tables. Keeps Chronos data access isolated from
legacy Recording/Segment logic.
"""

from datetime import datetime, timedelta
from typing import Any, List, Optional
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
    force_time: bool = False,
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
        force_time: Overwrite a deliberately-corrected timestamp anyway.

    Note:
        Recordings imported into the Plaud app (rather than synced from the
        device) carry the *import* moment in both `created_at` and `start_at`;
        the true recording time is not stored upstream at all. Where that has
        been repaired locally -- marked by `time_is_estimated` -- the upstream
        value is wrong and will stay wrong on every future sync, so it must not
        be allowed to overwrite the correction. Pass `force_time=True` only when
        the caller genuinely knows better.

    Returns:
        ChronosRecording: The upserted recording instance
    """
    # Tombstone check: ids the janitor has deliberately deleted (shadow
    # duplicates) still exist upstream and would otherwise be re-imported as
    # "new" on every sync -- re-extracted by the LLM, then re-deleted, forever.
    from sqlalchemy import text as _sql_text
    hit = session.execute(
        _sql_text("SELECT 1 FROM janitor_tombstones WHERE recording_id = :rid"),
        {"rid": recording_id}).first()
    if hit:
        return None

    rec = session.query(ChronosRecording).filter_by(recording_id=recording_id).first()

    if rec:
        # Update existing.
        # A timestamp that was deliberately corrected locally (time_is_estimated)
        # must survive re-sync -- otherwise every pipeline run silently reverts it
        # to Plaud's import-time stamp. This is what erased the March 17 repair.
        keep_time = bool(rec.time_is_estimated) and not force_time
        rec.title = title
        if not keep_time:
            rec.created_at = created_at
            rec.time_is_estimated = time_is_estimated
            rec.time_estimate_reason = time_estimate_reason
        rec.duration_seconds = duration_seconds
        rec.local_audio_path = local_audio_path
        rec.source = source
        rec.device_id = device_id
        rec.checksum = checksum
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
    worker_id: Optional[str] = None,
) -> None:
    """Update processing status for a recording with lease support.

    Args:
        session: SQLAlchemy session
        recording_id: Recording to update
        status: New status (pending | processing | completed | failed | cancelled)
        error_message: Error details if status is failed
        worker_id: ID of the active worker node/thread acquiring the lease
    """
    rec = session.query(ChronosRecording).filter_by(recording_id=recording_id).first()
    if rec:
        rec.processing_status = status
        rec.error_message = error_message
        now = datetime.utcnow()
        if status == "processing":
            rec.processing_started_at = now
            rec.heartbeat_at = now
            rec.lease_expires_at = now + timedelta(minutes=15)
            if worker_id:
                rec.worker_id = worker_id
            rec.attempt_count = (rec.attempt_count or 0) + 1
        elif status in ("completed", "failed", "cancelled"):
            rec.processed_at = now
            rec.lease_expires_at = None
            rec.heartbeat_at = None
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


def delete_chronos_events_by_recording(session: Session, recording_id: str) -> int:
    """Delete all Chronos events for a recording.

    Returns the number of deleted rows.
    """

    q = session.query(ChronosEvent).filter_by(recording_id=recording_id)
    count = q.count()
    q.delete(synchronize_session=False)
    session.commit()
    return int(count)


# ═══════════════════════════════════════════════════════════════════
# ChronosProcessingJob Operations
# ═══════════════════════════════════════════════════════════════════


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
