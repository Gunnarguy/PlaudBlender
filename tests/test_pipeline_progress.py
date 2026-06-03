import json


def test_pipeline_progress_persists_backfill_run_context(tmp_path, monkeypatch):
    import src.chronos.pipeline_progress as pipeline_progress

    progress_file = tmp_path / "pipeline_progress.json"
    monkeypatch.setattr(pipeline_progress, "PROGRESS_FILE", progress_file)

    tracker = pipeline_progress.PipelineProgressTracker()
    tracker.start_run(phases=["backfill"], trigger="cli")
    tracker.set_run_context(
        sync_mode="backfill",
        partial_success=True,
        warning="Repeated page signature detected on page 2; stopping backfill.",
        warnings=["Partial Plaud backfill preserved earlier pages."],
    )
    tracker.start_phase("backfill")
    tracker.set_phase_warnings(
        "backfill",
        ["Stopping to avoid an infinite Plaud page loop."],
    )
    tracker.finish_phase("backfill", summary="Preserved prior pages before stopping.")
    tracker.finish_run(error="Rate limit while backfilling Plaud history.")

    data = pipeline_progress.read_progress()

    assert data is not None
    assert data["sync_mode"] == "backfill"
    assert data["partial_success"] is True
    assert data["warning"] == "Repeated page signature detected on page 2; stopping backfill."
    assert data["warnings"] == ["Partial Plaud backfill preserved earlier pages."]
    assert data["phases"][0]["warnings"] == ["Stopping to avoid an infinite Plaud page loop."]


def test_read_progress_infers_legacy_backfill_sync_mode(tmp_path, monkeypatch):
    import src.chronos.pipeline_progress as pipeline_progress

    progress_file = tmp_path / "pipeline_progress.json"
    monkeypatch.setattr(pipeline_progress, "PROGRESS_FILE", progress_file)

    progress_file.write_text(
        json.dumps(
            {
                "run_id": "legacy-run",
                "status": "running",
                "current_phase": "backfill",
                "started_at": 1.0,
                "finished_at": 0.0,
                "elapsed_seconds": 2.0,
                "trigger": "api",
                "phases": [
                    {
                        "name": "backfill",
                        "status": "running",
                        "total_items": 0,
                        "completed_items": 0,
                        "current_step": "Fetching full Plaud history from Plaud…",
                        "current_item": "",
                        "started_at": 1.0,
                        "finished_at": 0.0,
                        "elapsed_seconds": 2.0,
                        "summary": "",
                        "error": "",
                    }
                ],
            }
        )
    )

    data = pipeline_progress.read_progress()

    assert data is not None
    assert data["sync_mode"] == "backfill"
    assert data["partial_success"] is False
    assert data["warnings"] == []
    assert data["phases"][0]["warnings"] == []
