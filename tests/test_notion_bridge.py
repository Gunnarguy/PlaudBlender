from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from src.chronos.notion_bridge import (
    _sanitize_extracted_events,
    get_import_progress,
    import_notion_recording,
)
from src.notion_service import NotionRecording


def _event(**overrides):
    base = {
        "event_id": "evt-1",
        "recording_id": "notion:page-1",
        "start_ts": datetime(2026, 5, 31, 12, 0, 0),
        "end_ts": datetime(2026, 5, 31, 12, 5, 0),
        "day_of_week": "Saturday",
        "hour_of_day": 12,
        "clean_text": "Reviewed the Notion import backlog and cleaned invalid events.",
        "category": "work",
        "category_confidence": 0.9,
        "sentiment": 0.1,
        "keywords": ["notion", "import"],
        "speaker": "self_talk",
        "raw_transcript_snippet": "Reviewed the Notion import backlog and cleaned invalid events.",
        "gemini_reasoning": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_sanitize_extracted_events_repairs_and_drops_invalid_entries():
    repaired, errors = _sanitize_extracted_events(
        [
            _event(
                clean_text="Done.",
                raw_transcript_snippet="Discussed the next rollout steps for the Notion importer.",
            ),
            _event(
                event_id="evt-2",
                clean_text="Bad",
                raw_transcript_snippet="tiny",
            ),
        ]
    )

    assert len(repaired) == 1
    assert repaired[0].clean_text == "Discussed the next rollout steps for the Notion importer."
    assert errors == ["event 2: Value error, clean_text must be at least 10 characters"]


def test_import_notion_recording_uses_processor_failure_reason(monkeypatch):
    page = NotionRecording(
        page_id="page-1",
        title="05-31 Notion note",
        created_time="2026-05-31T12:00:00Z",
        last_edited_time="2026-05-31T12:05:00Z",
        url="https://www.notion.so/page-1",
        transcript=("This transcript should be long enough for the importer to try processing. " * 6),
        summary="",
    )

    status_updates = []

    class FakeProcessor:
        def __init__(self, db_session):
            self._last_processing_error = "Transcript too short to extract meaningful events"

        def process_transcript_text(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        "src.chronos.notion_bridge.get_chronos_recording",
        lambda session, recording_id: None,
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge.upsert_chronos_recording",
        lambda **kwargs: SimpleNamespace(recording_id=kwargs["recording_id"]),
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge.set_chronos_recording_transcript",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge.mark_chronos_recording_status",
        lambda session, recording_id, status, error_message=None: status_updates.append(
            (recording_id, status, error_message)
        ),
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge.get_notion_service",
        lambda: Mock(fetch_page_content=Mock(return_value="")),
    )
    monkeypatch.setattr(
        "src.chronos.transcript_processor.TranscriptProcessor",
        FakeProcessor,
    )
    monkeypatch.setattr("app_v2.services.xray.xray_log", lambda *args, **kwargs: None)

    ok, message = import_notion_recording(
        page.page_id,
        session=Mock(),
        prefetched=page,
    )

    assert ok is False
    assert message == "Transcript too short to extract meaningful events"
    assert status_updates[-1] == (
        "notion:page-1",
        "failed",
        "Transcript too short to extract meaningful events",
    )


def test_get_import_progress_pauses_dead_running_worker(monkeypatch):
    saved = {}

    monkeypatch.setattr(
        "src.chronos.notion_bridge._load_progress",
        lambda: {"status": "running", "pid": 999999, "completed": 4, "total": 6},
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge._save_progress",
        lambda data: saved.update(data),
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )

    progress = get_import_progress()

    assert progress is not None
    assert progress["status"] == "paused"
    assert progress["pause_reason"] == "Import worker stopped before finishing this batch"
    assert saved["status"] == "paused"


def test_get_import_progress_pauses_legacy_running_file_without_pid(monkeypatch):
    saved = {}

    monkeypatch.setattr(
        "src.chronos.notion_bridge._load_progress",
        lambda: {"status": "running", "completed": 4, "total": 6},
    )
    monkeypatch.setattr(
        "src.chronos.notion_bridge._save_progress",
        lambda data: saved.update(data),
    )

    progress = get_import_progress()

    assert progress is not None
    assert progress["status"] == "paused"
    assert (
        progress["pause_reason"]
        == "Import progress came from an older worker and is no longer active"
    )
    assert saved["status"] == "paused"
