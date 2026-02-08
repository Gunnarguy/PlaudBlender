"""Database models and session management."""

from .engine import SessionLocal, init_db, DB_PATH
from .models import (
    Recording,
    Segment,
    ChronosRecording,
    ChronosEvent,
    ChronosWebhookEvent,
    ChronosProcessingJob,
    Base,
)
from .repository import (
    upsert_recording,
    add_segments,
    get_pending_recordings,
    get_segments_by_status,
    mark_recording_status,
    mark_segment_status,
)
from .chronos_repository import (
    add_chronos_webhook_event,
    list_chronos_webhook_events,
    mark_webhook_event_processed,
)

__all__ = [
    "SessionLocal",
    "init_db",
    "DB_PATH",
    "Recording",
    "Segment",
    "ChronosRecording",
    "ChronosEvent",
    "ChronosProcessingJob",
    "ChronosWebhookEvent",
    "Base",
    "upsert_recording",
    "add_segments",
    "get_pending_recordings",
    "get_segments_by_status",
    "mark_recording_status",
    "mark_segment_status",
    "add_chronos_webhook_event",
    "list_chronos_webhook_events",
    "mark_webhook_event_processed",
]
