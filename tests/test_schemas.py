import pytest
from datetime import datetime
from src.models.schemas import RecordingSchema, SegmentSchema


def test_recording_schema_validates_transcript_length():
    with pytest.raises(ValueError):
        RecordingSchema(
            id="1",
            title="t",
            duration_ms=1000,
            created_at=datetime.utcnow(),
            transcript="too short",
        )


def test_segment_schema_end_after_start():
    with pytest.raises(ValueError):
        SegmentSchema(
            id="s1",
            recording_id="r1",
            start_ms=100,
            end_ms=50,
            text="hello world",
        )


def test_segment_schema_text_required():
    with pytest.raises(ValueError):
        SegmentSchema(
            id="s2",
            recording_id="r1",
            start_ms=0,
            end_ms=10,
            text="  ",
        )
