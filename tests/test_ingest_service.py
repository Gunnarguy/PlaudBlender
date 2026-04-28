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
    plaud_client.list_recordings.side_effect = [page_1, page_2, page_3]

    service = ChronosIngestService(db_session=MagicMock(), plaud_client=plaud_client)
    monkeypatch.setattr(service, "ingest_recording", lambda **kwargs: (True, None))

    success, failed = service.ingest_recent_recordings(
        days_back=7,
        fetch_all_pages=True,
    )

    assert success == 40
    assert failed == 0
    assert plaud_client.list_recordings.call_count == 3
