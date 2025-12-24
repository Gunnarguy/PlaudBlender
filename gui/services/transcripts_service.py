"""Transcript CRUD helpers for the GUI.

These functions intentionally avoid network calls in tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select

from src.database.engine import SessionLocal, init_db
from src.database.models import Recording


def fetch_transcripts(status_filter: Optional[str] = None) -> List[Recording]:
    init_db()
    db = SessionLocal()
    try:
        q = select(Recording)
        if status_filter:
            q = q.where(Recording.status == status_filter)
        return list(db.execute(q).scalars().all())
    finally:
        try:
            db.close()
        except Exception:
            pass


def sync_recording(recording_id: str) -> Dict[str, Any]:
    """Placeholder for syncing a single recording.

    In the real app, this would call Plaud and update DB.
    """

    return {"recording_id": recording_id, "status": "noop"}


def delete_recording(recording_id: str) -> None:
    init_db()
    db = SessionLocal()
    try:
        rec = db.get(Recording, recording_id)
        if rec is None:
            return None
        db.delete(rec)
        db.commit()
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass


def export_recording(recording_id: str) -> Dict[str, Any]:
    init_db()
    db = SessionLocal()
    try:
        rec = db.get(Recording, recording_id)
        if rec is None:
            return {}
        return {
            "id": rec.id,
            "title": rec.title,
            "transcript": rec.transcript,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "status": rec.status,
            "source": rec.source,
        }
    finally:
        try:
            db.close()
        except Exception:
            pass


def export_all_recordings(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export all recordings as JSON-serializable dicts."""

    return [
        export_recording(r.id) for r in fetch_transcripts(status_filter=status_filter)
    ]
