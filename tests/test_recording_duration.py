"""Recordings Plaud listed with length 0 get their length later (2026-10-06)."""
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.chronos_repository import set_chronos_recording_duration
from src.database.models import Base, ChronosRecording


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def add(session, rid, seconds):
    session.add(ChronosRecording(recording_id=rid, title=rid, created_at=datetime(2026, 10, 6, 14, 24),
                                 duration_seconds=seconds, local_audio_path="", source="plaud"))
    session.commit()


def test_fills_only_a_missing_length():
    session = make_session()
    add(session, "zero", 0)
    add(session, "known", 300)

    assert set_chronos_recording_duration(session, "zero", 11_377) is True
    assert set_chronos_recording_duration(session, "known", 500) is False
    assert set_chronos_recording_duration(session, "zero", 0) is False
    assert set_chronos_recording_duration(session, "absent", 60) is False

    lengths = {r.recording_id: r.duration_seconds for r in session.query(ChronosRecording)}
    assert lengths == {"zero": 11_377, "known": 300}


def test_ingest_fills_the_length_of_a_known_recording(monkeypatch):
    from app_v2.services import xray as xray_module
    from src.chronos import ingest_service
    from src.chronos.ingest_service import ChronosIngestService

    monkeypatch.setattr(xray_module, "xray_log", lambda *args, **kwargs: None)
    created = datetime(2026, 10, 6, 14, 24)
    existing = SimpleNamespace(created_at=created, duration_seconds=0)
    monkeypatch.setattr(ingest_service, "get_chronos_recording", lambda db, rid: existing)
    calls = []
    monkeypatch.setattr(ingest_service, "set_chronos_recording_duration",
                        lambda db, rid, seconds: calls.append((rid, seconds)) or True)

    service = ChronosIngestService(db_session=object(), plaud_client=object())
    ok, error = service.ingest_recording(recording_id="f_s_x", created_at=created, duration_ms=11_377_080)

    assert (ok, error) == (True, None)
    assert calls == [("f_s_x", 11_377)]
