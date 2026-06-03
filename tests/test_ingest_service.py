from datetime import datetime, timedelta
from unittest.mock import MagicMock


def test_fetch_all_pages_stops_once_recent_window_is_covered(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)

    now = datetime.utcnow()

    def rec(recording_id: str, days_ago: int) -> dict:
        timestamp = (now - timedelta(days=days_ago)).replace(
            microsecond=0
        ).isoformat() + "Z"
        return {
            "id": recording_id,
            "start_at": timestamp,
            "created_at": timestamp,
            "duration": 60_000,
            "serial_number": "plaud-note",
            "name": recording_id,
        }

    page_1 = [rec(f"recent-a-{index}", 0) for index in range(20)]
    page_2 = [rec(f"recent-b-{index}", 3) for index in range(20)]
    page_3 = [rec(f"old-{index}", 10) for index in range(20)]

    plaud_client = MagicMock()
    plaud_client.oauth.is_authenticated = True
    plaud_client.list_recordings.side_effect = [page_1, page_2, page_3, []]

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    monkeypatch.setattr(service, "ingest_recording", lambda **kwargs: (True, None))

    success, failed = service.ingest_recent_recordings(
        days_back=7,
        fetch_all_pages=True,
    )

    assert success == 40
    assert failed == 0
    assert plaud_client.list_recordings.call_count == 4


def test_fetch_all_pages_keeps_going_when_page_one_is_mixed(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)

    now = datetime.utcnow()

    def rec(recording_id: str, days_ago: int) -> dict:
        timestamp = (now - timedelta(days=days_ago)).replace(
            microsecond=0
        ).isoformat() + "Z"
        return {
            "id": recording_id,
            "start_at": timestamp,
            "created_at": timestamp,
            "duration": 60_000,
            "serial_number": "plaud-note",
            "name": recording_id,
        }

    page_1 = [rec(f"recent-a-{index}", 0) for index in range(10)] + [
        rec(f"old-a-{index}", 20) for index in range(10)
    ]
    page_2 = [rec(f"recent-b-{index}", 1) for index in range(10)] + [
        rec(f"old-b-{index}", 25) for index in range(10)
    ]
    page_3 = list(page_1)

    plaud_client = MagicMock()
    plaud_client.oauth.is_authenticated = True
    plaud_client.list_recordings.side_effect = [page_1, page_2, page_3]

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    monkeypatch.setattr(service, "ingest_recording", lambda **kwargs: (True, None))

    success, failed = service.ingest_recent_recordings(
        days_back=7,
        fetch_all_pages=True,
    )

    assert success == 20
    assert failed == 1
    assert plaud_client.list_recordings.call_count == 3


def test_specific_recording_ingest_fetches_detail_by_id(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)

    record = {
        "id": "rec-direct-001",
        "start_at": "2026-05-13T14:30:16Z",
        "created_at": "2026-05-14T01:11:55Z",
        "duration": 725000,
        "serial_number": "plaud-note",
        "name": "Direct ingest",
        "presigned_url": None,
    }

    plaud_client = MagicMock()
    plaud_client.oauth.is_authenticated = True
    plaud_client.get_recording.return_value = record

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    calls = []

    def fake_ingest_recording(**kwargs):
        calls.append(kwargs)
        return (True, None)

    monkeypatch.setattr(service, "ingest_recording", fake_ingest_recording)

    success, failed = service.ingest_recent_recordings(recording_id="rec-direct-001")

    assert success == 1
    assert failed == 0
    assert plaud_client.get_recording.call_count == 1
    assert calls[0]["recording_id"] == "rec-direct-001"
    assert calls[0]["title"] == "Direct ingest"


def test_all_history_fetches_full_plaud_history(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)

    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    recordings = [
        {
            "id": f"rec-{index}",
            "start_at": now,
            "created_at": now,
            "duration": 60_000,
            "serial_number": "plaud-note",
            "name": f"rec-{index}",
        }
        for index in range(3)
    ]

    plaud_client = MagicMock()
    plaud_client.oauth.is_authenticated = True
    plaud_client.list_recordings.return_value = recordings

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    monkeypatch.setattr(service, "ingest_recording", lambda **kwargs: (True, None))

    success, failed = service.ingest_recent_recordings(all_history=True)

    assert success == 3
    assert failed == 0
    assert service.last_batch_partial_success is False
    assert service.last_batch_warnings == []
    assert plaud_client.list_recordings.call_count == 1


def test_all_history_keeps_partial_progress_when_later_page_fails(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)

    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def rec(recording_id: str) -> dict:
        return {
            "id": recording_id,
            "start_at": now,
            "created_at": now,
            "duration": 60_000,
            "serial_number": "plaud-note",
            "name": recording_id,
        }

    page_1 = [rec(f"rec-a-{index}") for index in range(20)]
    page_2 = [rec(f"rec-b-{index}") for index in range(20)]

    plaud_client = MagicMock()
    plaud_client.oauth.is_authenticated = True
    plaud_client.list_recordings.side_effect = [
        page_1,
        page_2,
        RuntimeError("429 rate limit"),
    ]

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    monkeypatch.setattr(service, "ingest_recording", lambda **kwargs: (True, None))

    success, failed = service.ingest_recent_recordings(all_history=True)

    assert success == 40
    assert failed == 1
    assert service.last_batch_partial_success is True
    assert any("keeping what we already fetched" in warning for warning in service.last_batch_warnings)
    assert plaud_client.list_recordings.call_count == 3


def test_all_history_stops_when_plaud_repeats_same_page(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)

    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def rec(recording_id: str) -> dict:
        return {
            "id": recording_id,
            "start_at": now,
            "created_at": now,
            "duration": 60_000,
            "serial_number": "plaud-note",
            "name": recording_id,
        }

    page_1 = [rec(f"rec-a-{index}") for index in range(20)]

    plaud_client = MagicMock()
    plaud_client.oauth.is_authenticated = True
    plaud_client.list_recordings.side_effect = [page_1, list(page_1)]

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    monkeypatch.setattr(service, "ingest_recording", lambda **kwargs: (True, None))

    success, failed = service.ingest_recent_recordings(all_history=True)

    assert success == 20
    assert failed == 1
    assert service.last_batch_partial_success is True
    assert any("repeating the same page" in warning for warning in service.last_batch_warnings)
    assert plaud_client.list_recordings.call_count == 2
