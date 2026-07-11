from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app_v2.services import data_service as data_service_module
from src.database.engine import init_db
from src.database.models import ChronosEvent, ChronosRecording


class _FakeQdrantClient:
    def scroll(self, **_kwargs):
        return [], None


class _FakeQdrant:
    collection_name = "chronos_events"

    def __init__(self):
        self.client = _FakeQdrantClient()


def _build_service(monkeypatch, tmp_path):
    db_path = tmp_path / "sqlite-backfill.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    init_db(engine)
    test_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(data_service_module, "SessionLocal", test_session)
    monkeypatch.setattr(
        data_service_module.ChronosDataService,
        "_init_services",
        lambda self: None,
    )

    service = data_service_module.ChronosDataService()
    service.__dict__["_qdrant"] = _FakeQdrant()
    return service, test_session


def test_sqlite_backfill_surfaces_completed_recordings_without_qdrant(
    monkeypatch, tmp_path
):
    service, TestSession = _build_service(monkeypatch, tmp_path)
    created_at = datetime(2026, 4, 8, 14, 43, 3)

    with TestSession() as session:
        session.add(
            ChronosRecording(
                recording_id="rec_backfill",
                title="Backfill Recording",
                created_at=created_at,
                duration_seconds=480,
                local_audio_path="data/raw/test.m4a",
                transcript="Transcript",
                processing_status="completed",
            )
        )
        session.add(
            ChronosEvent(
                event_id="evt_backfill_1",
                recording_id="rec_backfill",
                start_ts=datetime(2026, 4, 8, 8, 0, 0),
                end_ts=datetime(2026, 4, 8, 8, 8, 0),
                day_of_week="Wednesday",
                hour_of_day=8,
                clean_text="Reviewed supplies and coordinated prep for the upcoming case.",
                category="work",
                keywords=["supplies", "prep"],
                sentiment=0.2,
                speaker="self_talk",
                qdrant_point_id=None,
            )
        )
        session.commit()

    days = service.get_days()

    assert len(days) == 1
    assert days[0].recordings[0].recording_id == "rec_backfill"
    assert days[0].recordings[0].event_count == 1

    detail = service.get_recording_detail("rec_backfill")
    assert detail is not None
    assert len(detail.events) == 1
    assert detail.events[0].id == "evt_backfill_1"


def test_save_category_override_works_for_sqlite_only_event_ids(monkeypatch, tmp_path):
    service, TestSession = _build_service(monkeypatch, tmp_path)
    created_at = datetime.utcnow() - timedelta(hours=1)

    with TestSession() as session:
        session.add(
            ChronosRecording(
                recording_id="rec_override",
                title="Override Recording",
                created_at=created_at,
                duration_seconds=120,
                local_audio_path="data/raw/test2.m4a",
                transcript="Transcript",
                processing_status="completed",
            )
        )
        session.add(
            ChronosEvent(
                event_id="evt_override_1",
                recording_id="rec_override",
                start_ts=created_at,
                end_ts=created_at + timedelta(minutes=2),
                day_of_week=created_at.strftime("%A"),
                hour_of_day=created_at.hour,
                clean_text="Discussed a follow-up plan for tomorrow's work.",
                category="unknown",
                keywords=["follow-up"],
                sentiment=0.0,
                speaker="self_talk",
                qdrant_point_id=None,
            )
        )
        session.commit()

    assert service.save_category_override(["evt_override_1"], "work") is True

    detail = service.get_recording_detail("rec_override")
    assert detail is not None
    assert detail.events[0].category == "work"
