from datetime import datetime, timedelta
from typing import Any, Dict, cast

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app_v2.services import data_service as data_service_module
from src.database.engine import init_db
from src.database.models import ChronosRecording, Recording
from src.plaud_workflow import WorkflowResult, WorkflowStatus


def _build_service(monkeypatch, tmp_path):
    db_path = tmp_path / "workflow-sync.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    init_db(engine)
    test_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(data_service_module, "SessionLocal", test_session)
    monkeypatch.setattr(
        data_service_module.ChronosDataService,
        "_init_services",
        lambda self: None,
    )

    return data_service_module.ChronosDataService(), test_session


def test_submit_and_refresh_plaud_workflow(monkeypatch, tmp_path):
    service, TestSession = _build_service(monkeypatch, tmp_path)
    created_at = datetime.utcnow() - timedelta(hours=2)

    with TestSession() as session:
        session.add(
            ChronosRecording(
                recording_id="rec_123",
                title="Test Recording",
                created_at=created_at,
                duration_seconds=180,
                local_audio_path="data/raw/test.m4a",
                transcript="Original transcript",
                processing_status="completed",
            )
        )
        session.add(
            Recording(
                id="rec_123",
                title="Test Recording",
                transcript="Original transcript",
                duration_ms=180000,
                created_at=created_at,
                source="plaud",
                extra={},
            )
        )
        session.commit()

    class SubmitStub:
        def submit_workflow(self, **kwargs):
            assert kwargs["file_id"] == "rec_123"
            assert kwargs["template_id"] == "tpl_test"
            return "wf_123"

    monkeypatch.setattr("src.plaud_workflow.PlaudWorkflowClient", SubmitStub)

    submit_result = service.submit_plaud_workflows(
        days_back=7,
        limit=1,
        template_id="tpl_test",
    )

    assert len(submit_result["submitted"]) == 1
    assert submit_result["submitted"][0]["workflow_id"] == "wf_123"

    stats = service.get_plaud_workflow_stats(days_back=7)
    assert stats["workflow_pending"] == 1
    assert stats["ready_for_enrichment"] == 0

    with TestSession() as session:
        legacy = session.get(Recording, "rec_123")
        assert legacy is not None
        extra = cast(Dict[str, Any], getattr(legacy, "extra", {}) or {})
        assert extra["plaud_workflow"]["status"] == "PENDING"

    class RefreshStub:
        def get_workflow_status(self, workflow_id):
            assert workflow_id == "wf_123"
            return {
                "status": "SUCCESS",
                "completed_tasks": 3,
                "total_tasks": 3,
                "current_task": None,
                "error": None,
            }

        def get_workflow_results(self, workflow_id):
            assert workflow_id == "wf_123"
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.SUCCESS,
                tasks_completed=3,
                tasks_total=3,
                transcript="Cloud transcript",
                extracted_data={"action_items": ["ship it"]},
                summary="Cloud summary",
            )

    monkeypatch.setattr("src.plaud_workflow.PlaudWorkflowClient", RefreshStub)

    refresh_result = service.refresh_plaud_workflow_statuses(days_back=7, limit=1)

    assert len(refresh_result["completed"]) == 1
    assert not refresh_result["failed"]

    with TestSession() as session:
        chronos = session.get(ChronosRecording, "rec_123")
        legacy = session.get(Recording, "rec_123")

        assert chronos is not None
        assert legacy is not None
        summary = getattr(chronos, "plaud_ai_summary", None)
        extra = cast(Dict[str, Any], getattr(legacy, "extra", {}) or {})
        assert summary == "Cloud summary"
        assert extra["plaud_summary"] == "Cloud summary"
        assert extra["plaud_extracted_data"] == {"action_items": ["ship it"]}
        assert extra["plaud_workflow"]["status"] == "SUCCESS"

    refreshed_stats = service.get_plaud_workflow_stats(days_back=7)
    assert refreshed_stats["with_ai_summary"] == 1
    assert refreshed_stats["workflow_pending"] == 0
    assert refreshed_stats["workflow_success"] == 1
