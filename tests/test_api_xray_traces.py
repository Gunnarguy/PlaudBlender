"""End-to-end tests for the X-Ray trace endpoints.

These four routes had no coverage at all. They were also awkward to test before
tests/conftest.py redirected the engine: each handler opens its own SessionLocal
rather than taking an injected session, so exercising them meant either mocking
the repository out (testing nothing) or writing into the real data/brain.db. Now
that the suite runs against a scratch database they can be driven for real --
seed a run and its spans through the same trace API the pipeline uses, then
assert the endpoints report them back.

The scratch database is shared for the whole session, so every fixture here
names a recording unique to its own run rather than a fixed id; otherwise spans
accumulate across tests and the lineage assertions drift.
"""

import os
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.chronos.trace_service import finish_trace_run, start_trace_run, trace_span
from src.database.engine import SessionLocal
from src.database.models import ChronosExecutionSpan, ChronosRecording


@pytest.fixture()
def client():
    """TestClient with auth disabled; the trace routes read the real session."""
    from api.dependencies import get_service

    get_service.cache_clear()
    with patch.dict(os.environ, {"CHRONOS_API_KEY": ""}, clear=False):
        from api.main import app

        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def seeded_run():
    """A completed run with two spans, one tied to a recording of its own.

    The recording row has to be inserted first. chronos_execution_spans
    .recording_id carries a foreign key to chronos_recordings and SQLite runs
    with foreign_keys=ON, so a span naming a recording that was never imported
    is rejected and silently discarded -- see TestSpansRequireTheirRecording.

    Returns (run_id, recording_id).
    """
    run_id = start_trace_run(
        trigger="cli", source="test-xray", title="xray endpoint fixture", emit_xray=False
    )
    recording_id = f"rec-xray-{run_id}"

    with SessionLocal() as session:
        session.add(
            ChronosRecording(
                recording_id=recording_id,
                title="X-Ray fixture recording",
                created_at=datetime(2026, 8, 29, 12, 0),
                duration_seconds=60,
                local_audio_path="",
                source="plaud",
            )
        )
        session.commit()

    with trace_span(
        operation="ingest-one",
        source="test-xray",
        stage="ingest",
        recording_id=recording_id,
        run_id=run_id,
        emit_start=False,
    ):
        pass
    with trace_span(
        operation="process-one",
        source="test-xray",
        stage="process",
        run_id=run_id,
        emit_start=False,
    ):
        pass

    finish_trace_run(run_id, emit_xray=False)
    return run_id, recording_id


class TestXRayRuns:
    def test_runs_lists_the_seeded_run(self, client, seeded_run):
        run_id, _ = seeded_run

        r = client.get("/api/v1/xray/runs", params={"limit": 100})
        assert r.status_code == 200

        runs = {row["run_id"]: row for row in r.json()}
        assert run_id in runs, "seeded run missing from /xray/runs"
        assert runs[run_id]["status"] == "completed"

    def test_runs_honors_limit(self, client, seeded_run):
        r = client.get("/api/v1/xray/runs", params={"limit": 1})
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_run_detail_returns_the_run_and_its_spans(self, client, seeded_run):
        run_id, _ = seeded_run

        r = client.get(f"/api/v1/xray/runs/{run_id}")
        assert r.status_code == 200

        body = r.json()
        assert body["run"] is not None
        assert body["run"]["run_id"] == run_id
        assert [s["operation"] for s in body["spans"]] == ["ingest-one", "process-one"], (
            "run detail should return spans oldest-first"
        )

    def test_run_detail_for_an_unknown_run_is_empty_not_an_error(self, client):
        """An unknown id is a legitimately empty DAG, not a 404."""
        r = client.get("/api/v1/xray/runs/does-not-exist")
        assert r.status_code == 200
        assert r.json() == {"run": None, "spans": []}


class TestXRaySpans:
    def test_spans_filters_by_run(self, client, seeded_run):
        run_id, _ = seeded_run

        r = client.get("/api/v1/xray/spans", params={"run_id": run_id})
        assert r.status_code == 200

        spans = r.json()
        assert len(spans) == 2
        assert {s["run_id"] for s in spans} == {run_id}

    def test_spans_filters_by_stage(self, client, seeded_run):
        run_id, _ = seeded_run

        r = client.get(
            "/api/v1/xray/spans", params={"run_id": run_id, "stage": "ingest"}
        )
        assert r.status_code == 200
        assert [s["operation"] for s in r.json()] == ["ingest-one"]

    def test_spans_filter_that_matches_nothing_returns_empty(self, client, seeded_run):
        r = client.get("/api/v1/xray/spans", params={"run_id": "no-such-run"})
        assert r.status_code == 200
        assert r.json() == []


class TestXRayLineage:
    def test_lineage_returns_only_that_recordings_spans(self, client, seeded_run):
        _, recording_id = seeded_run

        r = client.get(f"/api/v1/xray/recordings/{recording_id}/lineage")
        assert r.status_code == 200

        body = r.json()
        assert body["run"] is None, "lineage is span-scoped and carries no run"
        assert [s["operation"] for s in body["spans"]] == ["ingest-one"]
        assert {s["recording_id"] for s in body["spans"]} == {recording_id}

    def test_lineage_for_an_unknown_recording_is_empty(self, client):
        r = client.get("/api/v1/xray/recordings/rec-nonexistent/lineage")
        assert r.status_code == 200
        assert r.json() == {"run": None, "spans": []}


class TestSpansRequireTheirRecording:
    """Pins the silent-drop behavior that makes per-recording lineage look empty.

    trace_service._with_session swallows every failure so tracing can never break
    the pipeline, and the recording_id foreign key rejects spans for recordings
    that are not in the database. Together those mean a traced
    `chronos_pipeline.py --ingest --recording-id <new-id>` -- a recording being
    imported for the first time, so not yet a row -- records nothing at all. In
    production this shows up as 3 of 33,901 spans carrying a recording_id.

    Asserted rather than fixed: dropping the constraint means rebuilding
    chronos_execution_spans on a live table. Pinned here so the behavior is
    visible and a future fix has something to flip.
    """

    def test_a_span_for_an_unknown_recording_is_dropped(self):
        run_id = start_trace_run(trigger="cli", source="test-xray", emit_xray=False)
        with trace_span(
            operation="ingest-unknown",
            source="test-xray",
            stage="ingest",
            recording_id="rec-never-imported",
            run_id=run_id,
            emit_start=False,
        ):
            pass
        finish_trace_run(run_id, emit_xray=False)

        with SessionLocal() as session:
            stored = (
                session.query(ChronosExecutionSpan)
                .filter_by(operation="ingest-unknown")
                .all()
            )

        assert stored == [], (
            "span for an unimported recording unexpectedly persisted -- if the "
            "foreign key was dropped, delete this test and cover lineage for "
            "not-yet-imported recordings instead"
        )

    def test_tracing_a_dropped_span_does_not_raise(self):
        """Whatever else happens, tracing must never break the caller."""
        run_id = start_trace_run(trigger="cli", source="test-xray", emit_xray=False)

        with trace_span(
            operation="ingest-unknown-2",
            source="test-xray",
            recording_id="rec-also-never-imported",
            run_id=run_id,
            emit_start=False,
        ):
            pass

        finish_trace_run(run_id, emit_xray=False)
