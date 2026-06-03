from datetime import date as date_cls
from zoneinfo import ZoneInfo

from app_v2.services import data_service as data_service_module


def _build_service(monkeypatch):
    monkeypatch.setattr(
        data_service_module.ChronosDataService,
        "_init_services",
        lambda self: None,
    )
    return data_service_module.ChronosDataService()


def test_recent_empty_days_are_marked_verified_or_suspected(monkeypatch):
    service = _build_service(monkeypatch)
    monkeypatch.setattr(
        service,
        "_get_recent_plaud_recording_dates",
        lambda _days_back: {"2026-04-17", "2026-04-19", "2026-04-20"},
    )

    days = [
        data_service_module.DaySummary(
            date="2026-04-20",
            date_display="Monday, Apr 20",
            total_duration_seconds=3600,
            recording_count=1,
            event_count=10,
        ),
        data_service_module.DaySummary(
            date="2026-04-19",
            date_display="Sunday, Apr 19",
            total_duration_seconds=0,
            recording_count=0,
            event_count=0,
        ),
        data_service_module.DaySummary(
            date="2026-04-18",
            date_display="Saturday, Apr 18",
            total_duration_seconds=0,
            recording_count=0,
            event_count=0,
        ),
        data_service_module.DaySummary(
            date="2026-04-17",
            date_display="Friday, Apr 17",
            total_duration_seconds=7200,
            recording_count=2,
            event_count=50,
        ),
    ]

    audited = service._apply_recent_empty_day_audit(
        days,
        start_dt=date_cls(2026, 4, 17),
        end_dt=date_cls(2026, 4, 20),
        today=date_cls(2026, 4, 20),
    )

    assert audited[1].coverage_status == "suspected_gap"
    assert audited[1].coverage_note == "Possible sync gap — Plaud shows recordings"
    assert audited[2].coverage_status == "verified_empty"
    assert audited[2].coverage_note == "Verified empty in Plaud"


def test_old_empty_days_skip_recent_plaud_audit(monkeypatch):
    service = _build_service(monkeypatch)

    def _unexpected_call(_days_back):
        raise AssertionError("Old ranges should not trigger a recent Plaud audit")

    monkeypatch.setattr(service, "_get_recent_plaud_recording_dates", _unexpected_call)

    days = [
        data_service_module.DaySummary(
            date="2026-01-10",
            date_display="Saturday, Jan 10",
            total_duration_seconds=0,
            recording_count=0,
            event_count=0,
        )
    ]

    audited = service._apply_recent_empty_day_audit(
        days,
        start_dt=date_cls(2026, 1, 10),
        end_dt=date_cls(2026, 1, 10),
        today=date_cls(2026, 4, 20),
    )

    assert audited[0].coverage_status is None
    assert audited[0].coverage_note is None


def test_recent_empty_day_audit_caches_failures(monkeypatch):
    service = _build_service(monkeypatch)

    from src import plaud_client as plaud_client_module

    calls = {"count": 0}

    class FakePlaudClient:
        def __init__(self):
            calls["count"] += 1

        def list_recordings(self, page, page_size):
            raise RuntimeError("429 rate limit")

    monkeypatch.setattr(plaud_client_module, "PlaudClient", FakePlaudClient)

    assert service._get_recent_plaud_recording_dates(7) is None
    assert calls["count"] == 1

    assert service._get_recent_plaud_recording_dates(7) is None
    assert calls["count"] == 1


def test_recent_empty_day_audit_stops_after_page_budget(monkeypatch):
    service = _build_service(monkeypatch)

    from src import plaud_client as plaud_client_module

    calls = []
    current_iso = data_service_module.datetime.utcnow().strftime("%Y-%m-%dT12:00:00Z")

    class FakePlaudClient:
        def list_recordings(self, page, page_size):
            calls.append(page)
            return [{"start_at": current_iso}] * page_size

    monkeypatch.setattr(plaud_client_module, "PlaudClient", FakePlaudClient)

    assert service._get_recent_plaud_recording_dates(7) is None
    assert calls == list(range(1, data_service_module._EMPTY_DAY_AUDIT_MAX_PAGES + 1))


def test_recent_empty_day_audit_uses_local_day_keys(monkeypatch):
    service = _build_service(monkeypatch)

    from src import plaud_client as plaud_client_module

    class FrozenDateTime(data_service_module.datetime):
        @classmethod
        def utcnow(cls):
            return cls(2026, 5, 18, 12, 0, 0)

    class FakePlaudClient:
        def list_recordings(self, page, page_size):
            if page == 1:
                return [{"start_at": "2026-05-13T05:34:51Z"}]
            return []

    monkeypatch.setattr(data_service_module, "datetime", FrozenDateTime)
    monkeypatch.setattr(
        data_service_module,
        "_LOCAL_TZ",
        ZoneInfo("America/Los_Angeles"),
    )
    monkeypatch.setattr(plaud_client_module, "PlaudClient", FakePlaudClient)

    assert service._get_recent_plaud_recording_dates(7) == {"2026-05-12"}
