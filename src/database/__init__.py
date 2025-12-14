"""Database models and session management."""
from .engine import SessionLocal, init_db, DB_PATH
from .models import Recording, Segment, Base
from .repository import (
    upsert_recording,
    add_segments,
    get_pending_recordings,
    get_segments_by_status,
    mark_recording_status,
    mark_segment_status,
)

__all__ = [
    "SessionLocal",
    "init_db",
    "DB_PATH",
    "Recording",
    "Segment",
    "Base",
    "upsert_recording",
    "add_segments",
    "get_pending_recordings",
    "get_segments_by_status",
    "mark_recording_status",
    "mark_segment_status",
]
