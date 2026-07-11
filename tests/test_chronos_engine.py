from datetime import datetime
from src.models.chronos_schemas import ChronosEvent
from src.chronos.engine import validate_event_quality
import uuid

def test_validate_event_quality_zero_duration():
    events = [
        ChronosEvent(
            event_id=str(uuid.uuid4()),
            recording_id=str(uuid.uuid4()),
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            category="work",
            category_confidence=0.9,
            clean_text="This is a test event that has enough text to pass the empty check." * 2,
            sentiment_score=0.5,
            keywords=["test"],
            topic_summary="test summary",
            day_of_week='Monday',
            hour_of_day=12
        )
    ]
    # Edge case: zero duration
    assert validate_event_quality(events, 0) is True

def test_validate_event_quality_negative_duration():
    events = [
        ChronosEvent(
            event_id=str(uuid.uuid4()),
            recording_id=str(uuid.uuid4()),
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            category="work",
            category_confidence=0.9,
            clean_text="This is a test event that has enough text to pass the empty check." * 2,
            sentiment_score=0.5,
            keywords=["test"],
            topic_summary="test summary",
            day_of_week='Monday',
            hour_of_day=12
        )
    ]
    # Edge case: negative duration
    assert validate_event_quality(events, -10) is True

def test_validate_event_quality_happy_path():
    events = [
        ChronosEvent(
            event_id=str(uuid.uuid4()),
            recording_id=str(uuid.uuid4()),
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            category="work",
            category_confidence=0.9,
            clean_text="This is a test event that has enough text to pass the empty check." * 2,
            sentiment_score=0.5,
            keywords=["test"],
            topic_summary="test summary",
            day_of_week='Monday',
            hour_of_day=12
        )
    ]
    # Happy path: 600 seconds duration, expects at least 1 event
    assert validate_event_quality(events, 600) is True

def test_validate_event_quality_too_few_events():
    events = [
        ChronosEvent(
            event_id=str(uuid.uuid4()),
            recording_id=str(uuid.uuid4()),
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            category="work",
            category_confidence=0.9,
            clean_text="This is a test event that has enough text to pass the empty check." * 2,
            sentiment_score=0.5,
            keywords=["test"],
            topic_summary="test summary",
            day_of_week='Monday',
            hour_of_day=12
        )
    ]
    # 1200 seconds expects at least 2 events, but we only have 1
    assert validate_event_quality(events, 1200) is False

def test_validate_event_quality_too_many_empty_events():
    events = [
        ChronosEvent(
            event_id=str(uuid.uuid4()),
            recording_id=str(uuid.uuid4()),
            start_ts=datetime.now(),
            end_ts=datetime.now(),
            category="work",
            category_confidence=0.9,
            clean_text="Short text", # Will fail the length < 20 check
            sentiment_score=0.5,
            keywords=["test"],
            topic_summary="test summary",
            day_of_week='Monday',
            hour_of_day=12
        )
    ]
    # We have 1 event and it is considered "empty", so > 10% are empty -> False
    assert validate_event_quality(events, 0) is False
