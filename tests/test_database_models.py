from datetime import datetime, timedelta
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Recording, Segment
from src.database.repository import (
    add_segments,
    get_pending_recordings,
    mark_recording_status,
    upsert_recording,
)
from src.models.schemas import RecordingSchema, SegmentSchema


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    sess = Session()
    try:
        yield sess
    finally:
        sess.close()


def test_upsert_recording_inserts_and_updates(session):
    payload = RecordingSchema(
        id="rec_123",
        title="My Recording",
        duration_ms=120000,
        created_at=datetime.utcnow(),
        transcript="This is a sufficiently long transcript " * 3,
        language="en",
        source="plaud",
    )

    rec = upsert_recording(session, payload, filename="file.wav")
    assert rec.id == "rec_123"
    assert rec.filename == "file.wav"
    assert rec.status == "raw"
    assert session.query(Recording).count() == 1

    # Update title and transcript
    updated = upsert_recording(
        session,
        payload.copy(update={"title": "Updated", "transcript": "Updated transcript text " * 3}),
        filename="file.wav",
        status="processed",
    )
    assert updated.title == "Updated"
    assert updated.status == "processed"
    assert session.query(Recording).count() == 1


def test_add_segments_and_status(session):
    base_payload = RecordingSchema(
        id="rec_abc",
        title="Segmented Recording",
        duration_ms=60000,
        created_at=datetime.utcnow() - timedelta(days=1),
        transcript="Segment test transcript " * 4,
        language="en",
        source="plaud",
    )
    upsert_recording(session, base_payload, filename="seg.wav")

    seg_payloads = [
        SegmentSchema(
            id=str(uuid.uuid4()),
            recording_id="rec_abc",
            start_ms=0,
            end_ms=5000,
            text="First chunk of text",
            namespace="full_text",
        ),
        SegmentSchema(
            id=str(uuid.uuid4()),
            recording_id="rec_abc",
            start_ms=5000,
            end_ms=10000,
            text="Second chunk of text",
            namespace="full_text",
        ),
    ]

    segments = add_segments(session, recording_id="rec_abc", segments=seg_payloads, status="pending")
    assert len(segments) == 2
    assert session.query(Segment).count() == 2

    # Mark status and verify query helper
    mark_recording_status(session, "rec_abc", "indexed")
    pending = get_pending_recordings(session, status="indexed")
    assert len(pending) == 1
    assert pending[0].id == "rec_abc"
