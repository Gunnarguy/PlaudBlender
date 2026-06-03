from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app_v2.services import data_service as data_service_module
import scripts.chronos_pipeline as pipeline
from src.database.chronos_repository import upsert_chronos_recording
from src.database.engine import init_db
from src.database.models import ChronosRecording


def test_run_repair_recent_ingests_missing_and_resets_retryable_failures(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "repair-recent.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    init_db(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    created_at = datetime.utcnow() - timedelta(hours=4)

    with TestSession() as session:
        session.add(
            ChronosRecording(
                recording_id="rec_failed",
                title="Retry me",
                created_at=created_at,
                duration_seconds=180,
                local_audio_path="data/raw/retry-me.m4a",
                source="plaud",
                processing_status="failed",
                error_message="socket timeout",
            )
        )
        session.add(
            ChronosRecording(
                recording_id="rec_archived",
                title="No transcript",
                created_at=created_at,
                duration_seconds=120,
                local_audio_path="",
                source="plaud",
                processing_status="failed",
                error_message="No transcript available in Plaud source_list",
            )
        )
        session.commit()

    start_at = created_at.replace(microsecond=0).isoformat() + "Z"
    recent_records = [
        {"id": "rec_missing", "start_at": start_at, "created_at": start_at},
        {"id": "rec_failed", "start_at": start_at, "created_at": start_at},
        {"id": "rec_archived", "start_at": start_at, "created_at": start_at},
    ]

    class FakeIngestService:
        def __init__(self, db_session):
            self.db = db_session
            self.plaud = SimpleNamespace(
                get_recording=lambda _recording_id: {
                    "source_list": [],
                }
            )

        def _fetch_recent_recordings_window(self, **_kwargs):
            return recent_records, 1, "page 1 was the final Plaud page"

        def ingest_recording_by_id(self, recording_id):
            upsert_chronos_recording(
                session=self.db,
                recording_id=recording_id,
                title="Recovered",
                created_at=created_at,
                duration_seconds=300,
                local_audio_path="",
                source="plaud",
                device_id=None,
                checksum=None,
            )
            return True, None

    monkeypatch.setattr(pipeline, "ChronosIngestService", FakeIngestService)
    class FakeWorkflowService:
        def submit_single_recording_workflow(self, recording_id, template_id=None, model="gemini"):
            with TestSession() as session:
                rec = session.get(ChronosRecording, recording_id)
                assert rec is not None
                rec.plaud_workflow_id = "wf_recover_archived"
                rec.plaud_workflow_status = "PENDING"
                session.commit()
            return {
                "workflow_id": "wf_recover_archived",
                "status": "PENDING",
                "recording_id": recording_id,
            }

    monkeypatch.setattr(
        data_service_module,
        "ChronosDataService",
        FakeWorkflowService,
    )
    monkeypatch.setattr(
        pipeline,
        "pipeline_progress",
        SimpleNamespace(
            start_phase=lambda *args, **kwargs: None,
            update=lambda *args, **kwargs: None,
            finish_phase=lambda *args, **kwargs: None,
        ),
    )

    with TestSession() as session:
        repaired = pipeline.run_repair_recent(
            session,
            days_back=7,
            limit=10,
            stale_after_minutes=90,
        )

    with TestSession() as session:
        missing = session.get(ChronosRecording, "rec_missing")
        failed = session.get(ChronosRecording, "rec_failed")
        archived = session.get(ChronosRecording, "rec_archived")

        assert repaired == 3
        assert missing is not None
        assert failed is not None
        assert failed.processing_status == "pending"
        assert failed.error_message is None
        assert archived is not None
        assert archived.processing_status == "failed"
        assert archived.plaud_workflow_id == "wf_recover_archived"
        assert archived.plaud_workflow_status == "PENDING"


def test_run_repair_recent_requeues_no_transcript_failures_when_transcript_appears(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "repair-transcript-recovery.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    init_db(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    created_at = datetime.utcnow() - timedelta(hours=3)

    with TestSession() as session:
        session.add(
            ChronosRecording(
                recording_id="rec_now_has_transcript",
                title="Recovered transcript",
                created_at=created_at,
                duration_seconds=240,
                local_audio_path="",
                source="plaud",
                processing_status="failed",
                error_message="No transcript available in Plaud source_list",
            )
        )
        session.commit()

    start_at = created_at.replace(microsecond=0).isoformat() + "Z"
    recent_records = [
        {
            "id": "rec_now_has_transcript",
            "start_at": start_at,
            "created_at": start_at,
        }
    ]

    class FakeIngestService:
        def __init__(self, db_session):
            self.db = db_session
            self.plaud = SimpleNamespace(
                get_recording=lambda recording_id: {
                    "id": recording_id,
                    "source_list": [
                        {
                            "data_type": "transaction",
                            "data_content": '[{"content": "Recovered transcript now exists."}]',
                        }
                    ],
                }
            )

        def _fetch_recent_recordings_window(self, **_kwargs):
            return recent_records, 1, "page 1 was the final Plaud page"

        def ingest_recording_by_id(self, recording_id):
            raise AssertionError(f"Unexpected ingest for {recording_id}")

    class ForbiddenWorkflowService:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Workflow submission should not run when transcript is now available")

    monkeypatch.setattr(pipeline, "ChronosIngestService", FakeIngestService)
    monkeypatch.setattr(
        data_service_module,
        "ChronosDataService",
        ForbiddenWorkflowService,
    )
    monkeypatch.setattr(
        pipeline,
        "pipeline_progress",
        SimpleNamespace(
            start_phase=lambda *args, **kwargs: None,
            update=lambda *args, **kwargs: None,
            finish_phase=lambda *args, **kwargs: None,
        ),
    )

    with TestSession() as session:
        repaired = pipeline.run_repair_recent(
            session,
            days_back=7,
            limit=10,
            stale_after_minutes=90,
        )

    with TestSession() as session:
        recovered = session.get(ChronosRecording, "rec_now_has_transcript")

        assert repaired == 1
        assert recovered is not None
        assert recovered.processing_status == "pending"
        assert recovered.error_message is None
        assert recovered.transcript == "Recovered transcript now exists."


def test_get_recording_db_stats_only_counts_actionable_failures(monkeypatch, tmp_path):
    db_path = tmp_path / "recording-db-stats.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    init_db(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    created_at = datetime.utcnow() - timedelta(hours=2)

    with TestSession() as session:
        session.add(
            ChronosRecording(
                recording_id="rec_actionable",
                title="Retry me",
                created_at=created_at,
                duration_seconds=180,
                local_audio_path="data/raw/retry-me.m4a",
                source="plaud",
                processing_status="failed",
                error_message="socket timeout",
            )
        )
        session.add(
            ChronosRecording(
                recording_id="rec_archived",
                title="No transcript",
                created_at=created_at,
                duration_seconds=120,
                local_audio_path="",
                source="plaud",
                processing_status="failed",
                error_message="No transcript available in Plaud source_list",
            )
        )
        session.commit()

    monkeypatch.setattr("src.database.engine.SessionLocal", TestSession)
    monkeypatch.setattr(data_service_module, "SessionLocal", TestSession)
    monkeypatch.setattr(
        data_service_module.ChronosDataService,
        "_init_services",
        lambda self: None,
    )

    service = data_service_module.ChronosDataService()

    stats = service.get_recording_db_stats()

    assert stats["failed"] == 1
    assert stats["total"] == 2
    assert "archived_failed" not in stats
