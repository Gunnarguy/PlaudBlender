from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Recording
from src.database.repository import upsert_recording
from src.models.schemas import RecordingSchema
from src.processing.engine import ChunkingConfig, process_pending_recordings


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return Session()


def test_process_pending_creates_segments_and_updates_status():
    session = make_session()
    try:
        payload = RecordingSchema(
            id="rec_proc",
            title="Process Me",
            duration_ms=1000,
            created_at=datetime.utcnow(),
            transcript=" ".join(["word" + str(i) for i in range(120)]),
            language="en",
            source="plaud",
        )
        upsert_recording(session, payload, filename="proc.wav", status="raw")

        summary = process_pending_recordings(session, cfg=ChunkingConfig(max_words=50, overlap_words=10))

        rec = session.get(Recording, "rec_proc")
        assert rec.status == "processed"
        assert summary["recordings_processed"] == 1
        # Expect ceil(120 / (50-10)) chunks = ~3
        assert summary["segments_created"] >= 2
    finally:
        session.close()
