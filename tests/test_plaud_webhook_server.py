import pytest
from datetime import datetime
from src.plaud_webhook import PlaudEvent, PlaudEventType
from src.plaud_webhook_server import EventLogEntry, PlaudWebhookServer


def test_event_log_entry_to_dict():
    """Test that EventLogEntry.to_dict correctly formats the output dictionary."""
    # Setup a PlaudEvent
    test_timestamp = datetime(2023, 1, 1, 12, 0, 0)
    event = PlaudEvent(
        event_type=PlaudEventType.FILE_UPLOADED,
        event_id="test_event_123",
        timestamp=test_timestamp,
        data={"file_id": "file_123", "status": "success"},
        raw_payload={"raw": "data"},
    )

    # Setup EventLogEntry
    received_at_time = datetime(2023, 1, 1, 12, 0, 5)
    log_entry = EventLogEntry(event=event, received_at=received_at_time, processed=True)

    # Test to_dict mapping
    result = log_entry.to_dict()

    # Verify the structure and values
    assert result["event_type"] == "file.uploaded"
    assert result["event_id"] == "test_event_123"
    assert result["timestamp"] == test_timestamp.isoformat()
    assert result["received_at"] == received_at_time.isoformat()
    assert result["processed"] is True
    assert result["data"] == {"file_id": "file_123", "status": "success"}

    # Make sure we don't have fields we didn't intend to expose (like raw_payload)
    assert "raw_payload" not in result


def test_event_log_entry_to_dict_unprocessed():
    """Test to_dict with an unprocessed event to ensure default values map correctly."""
    # Setup a PlaudEvent
    test_timestamp = datetime(2023, 1, 1, 12, 0, 0)
    event = PlaudEvent(
        event_type=PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED,
        event_id="test_event_456",
        timestamp=test_timestamp,
        data={"file_id": "file_456"},
    )

    # Setup EventLogEntry with default processed=False
    received_at_time = datetime(2023, 1, 1, 12, 0, 5)
    log_entry = EventLogEntry(event=event, received_at=received_at_time)

    # Test to_dict mapping
    result = log_entry.to_dict()

    assert result["event_type"] == "audio_transcribe.completed"
    assert result["processed"] is False
    assert result["data"] == {"file_id": "file_456"}
