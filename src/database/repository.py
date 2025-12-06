"""Thin repository helpers for recordings and segments.

These functions provide a small abstraction over SQLAlchemy sessions so the
pipeline can persist Plaud ingest and processing outputs deterministically.
"""
from typing import Iterable, List, Optional

from sqlalchemy.orm import Session

from src.models.schemas import RecordingSchema, SegmentSchema

from .models import Recording, Segment


def upsert_recording(
    session: Session,
    payload: RecordingSchema,
    filename: Optional[str] = None,
    status: str = "raw",
    extra: Optional[dict] = None,
) -> Recording:
    """Insert or update a recording row.

    Returns the persisted Recording instance.
    """
    record = session.get(Recording, payload.id)
    if record is None:
        record = Recording(
            id=payload.id,
            title=payload.title,
            transcript=payload.transcript,
            duration_ms=payload.duration_ms,
            created_at=payload.created_at,
            source=payload.source,
            language=payload.language,
            status=status,
            filename=filename,
            extra=extra,
        )
        session.add(record)
    else:
        record.title = payload.title or record.title
        record.transcript = payload.transcript
        record.duration_ms = payload.duration_ms
        record.created_at = payload.created_at
        record.language = payload.language
        record.source = payload.source or record.source
        record.status = status or record.status
        if filename:
            record.filename = filename
        if extra is not None:
            record.extra = extra

    session.commit()
    session.refresh(record)
    return record


def add_segments(
    session: Session,
    recording_id: str,
    segments: Iterable[SegmentSchema],
    default_namespace: str = "full_text",
    status: str = "pending",
    pinecone_ids: Optional[dict] = None,
    embedding_model: Optional[str] = None,
) -> List[Segment]:
    """Persist segment rows for a recording.

    pinecone_ids: optional mapping of segment.id -> pinecone vector id
    """
    persisted: List[Segment] = []
    pinecone_ids = pinecone_ids or {}

    for seg in segments:
        segment = Segment(
            id=seg.id,
            recording_id=recording_id,
            text=seg.text,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            namespace=seg.namespace or default_namespace,
            status=status,
            pinecone_id=pinecone_ids.get(seg.id),
            embedding_model=embedding_model,
        )
        session.add(segment)
        persisted.append(segment)

    session.commit()
    for seg in persisted:
        session.refresh(seg)
    return persisted


def mark_recording_status(session: Session, recording_id: str, status: str) -> None:
    """Update the status of a recording if it exists."""
    record = session.get(Recording, recording_id)
    if record is None:
        return
    record.status = status
    session.commit()


def get_pending_recordings(session: Session, status: str = "raw") -> List[Recording]:
    """Return recordings with the given status (default raw)."""
    return session.query(Recording).filter(Recording.status == status).all()


def get_segments_by_status(
    session: Session,
    status: str = "pending",
    recording_id: Optional[str] = None,
) -> List[Segment]:
    """Return segments filtered by status and optional recording_id."""
    q = session.query(Segment).filter(Segment.status == status)
    if recording_id:
        q = q.filter(Segment.recording_id == recording_id)
    return q.all()


def mark_segment_status(session: Session, segment_id: str, status: str) -> None:
    seg = session.get(Segment, segment_id)
    if seg is None:
        return
    seg.status = status
    session.commit()
