"""Chronos-specific database repository functions.

Provides CRUD operations for ChronosRecording, ChronosEvent, and
ChronosProcessingJob tables. Keeps Chronos data access isolated from
legacy Recording/Segment logic.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from .models import ChronosRecording, ChronosEvent, ChronosProcessingJob


# ═══════════════════════════════════════════════════════════════════
# ChronosRecording Operations
# ═══════════════════════════════════════════════════════════════════


def upsert_chronos_recording(
    session: Session,
    recording_id: str,
    created_at: datetime,
    duration_seconds: int,
    local_audio_path: str,
    source: str = "plaud",
    device_id: Optional[str] = None,
    checksum: Optional[str] = None,
) -> ChronosRecording:
    """Insert or update a Chronos recording.

    Args:
        session: SQLAlchemy session
        recording_id: Plaud API recording ID
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
        rec.created_at = created_at
        rec.duration_seconds = duration_seconds
        rec.local_audio_path = local_audio_path
        rec.source = source
        rec.device_id = device_id
        rec.checksum = checksum
    else:
        # Insert new
        rec = ChronosRecording(
            recording_id=recording_id,
            created_at=created_at,
            duration_seconds=duration_seconds,
            local_audio_path=local_audio_path,
            source=source,
            device_id=device_id,
            checksum=checksum,
        )
        session.add(rec)

    session.commit()
    session.refresh(rec)
    return rec


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
