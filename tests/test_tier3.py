"""
Tier 3 Tests — MCP Server, Auto-Sync, Webhook, USB Watcher.

Tests for the production MCP server, webhook+pipeline integration,
and USB watcher integration. All tests mock external services.
"""

import asyncio
import json
import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.run(coro)


# ===========================================================================
# MCP Server Tests
# ===========================================================================


class TestMCPServerImport:
    """Tests for scripts/mcp_server.py imports and structure."""

    def test_mcp_server_imports(self):
        """Verify MCP server module can be imported."""
        import scripts.mcp_server as mcp

        assert mcp.server is not None
        assert mcp.server.name == "chronos-mcp"

    def test_mcp_server_has_tools(self):
        """Verify all expected tools are registered."""
        import scripts.mcp_server as mcp

        # The server object should exist
        assert hasattr(mcp, "server")
        # Key functions should exist
        assert callable(mcp.ping)
        assert callable(mcp.search_events)
        assert callable(mcp.get_recording)
        assert callable(mcp.list_recordings)
        assert callable(mcp.get_timeline)
        assert callable(mcp.get_stats)
        assert callable(mcp.get_topics)
        assert callable(mcp.get_graph)
        assert callable(mcp.run_pipeline)
        assert callable(mcp.system_status)
        assert callable(mcp.ask_chronos)

    def test_mcp_server_has_main(self):
        """Verify main entrypoint exists."""
        import scripts.mcp_server as mcp

        assert callable(mcp.main)


class TestMCPPing:
    """Test the ping tool."""

    def test_ping_returns_pong(self):
        """Ping should return pong."""
        import scripts.mcp_server as mcp

        result = run_async(mcp.ping())
        assert result == "pong"


class TestMCPSearchEvents:
    """Test the search_events tool."""

    def test_search_returns_json(self):
        """search_events should return valid JSON."""
        import scripts.mcp_server as mcp

        mock_event = Mock()
        mock_event.start_ts = datetime(2025, 1, 15, 10, 0)
        mock_event.end_ts = datetime(2025, 1, 15, 11, 0)
        mock_event.clean_text = "Test Event. A test summary."
        mock_event.category = "work"
        mock_event.recording_id = "abc123"

        mock_result = Mock()
        mock_result.event = mock_event
        mock_result.score = 0.95

        mock_ds = Mock()
        mock_ds.search.return_value = [mock_result]

        with patch.dict(mcp._services, {"data": mock_ds}):
            result = run_async(mcp.search_events("test query"))

        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Test Event"

    def test_search_empty_results(self):
        """search_events should handle empty results."""
        import scripts.mcp_server as mcp

        mock_ds = Mock()
        mock_ds.search.return_value = []

        with patch.dict(mcp._services, {"data": mock_ds}):
            result = run_async(mcp.search_events("nonexistent query"))

        data = json.loads(result)
        assert data["results"] == []

    def test_search_error_handling(self):
        """search_events should handle errors gracefully."""
        import scripts.mcp_server as mcp

        mock_ds = Mock()
        mock_ds.search.side_effect = Exception("Connection refused")

        with patch.dict(mcp._services, {"data": mock_ds}):
            result = run_async(mcp.search_events("test"))

        data = json.loads(result)
        assert "error" in data


class TestMCPListRecordings:
    """Test the list_recordings tool."""

    def test_list_returns_json(self):
        """list_recordings should return valid JSON with recordings list."""
        import scripts.mcp_server as mcp

        mock_row = (
            "rec_123",
            "Title",
            "2025-01-15 10:00:00",
            3600,
            "completed",
            "plaud_api",
            12,
        )
        mock_db = Mock()
        mock_db.execute.return_value.fetchall.return_value = [mock_row]

        with patch.object(mcp, "_get_db_session", return_value=mock_db):
            result = run_async(mcp.list_recordings())

        data = json.loads(result)
        assert "recordings" in data
        assert len(data["recordings"]) == 1
        assert data["recordings"][0]["recording_id"] == "rec_123"
        assert data["recordings"][0]["duration_minutes"] == 60.0


# ===========================================================================
# Auto-Sync Tests
# ===========================================================================


class TestAutoSyncImport:
    """Tests for scripts/auto_sync.py."""

    def test_auto_sync_imports(self):
        """Verify AutoSync module can be imported."""
        from scripts.auto_sync import ChronosAutoSync

        assert ChronosAutoSync is not None

    def test_auto_sync_init(self):
        """Verify AutoSync initializes properly."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )
        assert not syncer.is_running
        assert syncer.activity_log == []
        assert syncer._pending_recordings == []

    def test_auto_sync_status(self):
        """Verify status property works."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )
        status = syncer.status
        assert "running" in status
        assert "webhook" in status
        assert "usb" in status
        assert "queue" in status
        assert status["running"] is False

    def test_log_activity(self):
        """Verify activity logging works."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )
        syncer._log_activity("test", "action", "details")
        assert len(syncer.activity_log) == 1
        assert syncer.activity_log[0]["source"] == "test"
        assert syncer.activity_log[0]["action"] == "action"

    def test_queue_for_processing(self):
        """Verify recording queueing works."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )
        syncer._queue_for_processing("rec_123", "test")
        assert "rec_123" in syncer._pending_recordings

        # Duplicate should not be added
        syncer._queue_for_processing("rec_123", "test")
        assert syncer._pending_recordings.count("rec_123") == 1

    def test_process_single_runs_pipeline_stages(self, monkeypatch):
        """Single-record processing should use pipeline subprocess stages."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return True

        monkeypatch.setattr(syncer, "_run_pipeline_subprocess", fake_run)
        monkeypatch.setattr(syncer, "_defer_heavy_work_if_needed", lambda label: False)

        syncer._process_single("rec_123")

        assert [call[0] for call in calls] == [
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--ingest",
                "--recording-id",
                "rec_123",
            ],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--process",
                "--recording-id",
                "rec_123",
            ],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--index",
                "--recording-id",
                "rec_123",
            ],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--graph",
                "--recording-id",
                "rec_123",
            ],
        ]

    def test_poll_runs_graph_stage(self, monkeypatch):
        """Scheduled polling should refresh workflows, self-heal, then refresh graph cache."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return True

        monkeypatch.setattr(syncer, "_run_pipeline_subprocess", fake_run)
        monkeypatch.setattr(syncer, "_defer_heavy_work_if_needed", lambda label: False)

        syncer._poll_plaud_api()

        assert [call[0] for call in calls] == [
            [sys.executable, "scripts/chronos_pipeline.py", "--ingest"],
            [sys.executable, "scripts/chronos_pipeline.py", "--refresh-workflows"],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--repair-recent",
                "--days-back",
                str(syncer.SELF_HEAL_DAYS),
                "--limit",
                str(syncer.SELF_HEAL_LIMIT),
            ],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--process",
                "--limit",
                str(syncer.process_limit),
            ],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--index",
                "--limit",
                str(syncer.index_limit),
            ],
            [
                sys.executable,
                "scripts/chronos_pipeline.py",
                "--graph",
                "--limit",
                str(syncer.graph_limit),
            ],
        ]

    def test_poll_uses_configured_micro_batch_limits(self, monkeypatch):
        """Scheduled polling should honor explicit process, index, and graph limits."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
            process_limit=3,
            index_limit=4,
            graph_limit=2,
        )
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return True

        monkeypatch.setattr(syncer, "_run_pipeline_subprocess", fake_run)
        monkeypatch.setattr(syncer, "_defer_heavy_work_if_needed", lambda label: False)

        syncer._poll_plaud_api()

        assert [sys.executable, "scripts/chronos_pipeline.py", "--process", "--limit", "3"] in calls
        assert [sys.executable, "scripts/chronos_pipeline.py", "--index", "--limit", "4"] in calls
        assert [sys.executable, "scripts/chronos_pipeline.py", "--graph", "--limit", "2"] in calls

    def test_host_pressure_allows_stale_swap_when_ram_has_recovered(self, monkeypatch):
        """High lingering swap alone should not deadlock autosync when RAM is healthy."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=False,
        )

        monkeypatch.setattr(syncer, "MAX_SWAP_USED_MB", 256)
        monkeypatch.setattr(syncer, "MIN_AVAILABLE_MB", 700)
        monkeypatch.setattr(syncer, "_meminfo_mb", lambda: {
            "MemAvailable": 1055,
            "SwapTotal": 1873,
            "SwapFree": 1295,
        })
        monkeypatch.setattr("scripts.auto_sync.os.getloadavg", lambda: (0.4, 0.3, 0.2))
        monkeypatch.setattr(syncer, "_pipeline_already_running", lambda: False)

        assert syncer._host_pressure_reason() is None

    def test_poll_runs_safe_notion_import_batch_when_configured(self, monkeypatch):
        """Scheduled polling should import a safe Notion batch after Plaud stages."""
        from scripts.auto_sync import ChronosAutoSync

        syncer = ChronosAutoSync(
            enable_webhook=False,
            enable_usb=False,
            auto_process=False,
            enable_notion_import=True,
            notion_import_batch_size=7,
        )
        calls = []
        activity = []

        class DummySession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class DummySettings:
            notion_database_id = "db-123"

        notion_status = Mock(connected=True, error="")
        notion_service = Mock()
        notion_service.check_connection.return_value = notion_status

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return True

        monkeypatch.setattr(syncer, "_run_pipeline_subprocess", fake_run)
        monkeypatch.setattr(syncer, "_log_activity", lambda source, action, details="": activity.append((source, action, details)))
        monkeypatch.setattr(syncer, "_defer_heavy_work_if_needed", lambda label: False)
        monkeypatch.setattr("scripts.auto_sync.get_settings", lambda: DummySettings())
        monkeypatch.setattr("src.notion_service.get_notion_service", lambda: notion_service)
        monkeypatch.setattr("src.database.SessionLocal", lambda: DummySession())

        captured = {}

        def fake_import_all_unmatched(session, **kwargs):
            captured.update(kwargs)
            return 3, 1, ["rate limited"]

        monkeypatch.setattr("src.chronos.notion_bridge.import_all_unmatched", fake_import_all_unmatched)

        syncer._poll_plaud_api()

        assert captured == {"process": True, "index": True, "batch_size": 7}
        assert any(action == "notion_import_done" for _, action, _ in activity)


class TestChronosPipeline:
    """Tests for scripts/chronos_pipeline.py reliability semantics."""

    def test_host_pressure_blocks_high_swap_when_ram_is_still_tight(self, monkeypatch):
        """High swap should still defer the pipeline when RAM has not recovered."""
        import scripts.chronos_pipeline as pipeline

        class DummySettings:
            chronos_autosync_max_load_avg = 3.5
            chronos_autosync_min_available_mb = 700
            chronos_autosync_max_swap_used_mb = 256

        monkeypatch.setattr(pipeline, "get_settings", lambda: DummySettings())
        monkeypatch.setattr(pipeline, "_meminfo_mb", lambda: {
            "MemAvailable": 820,
            "SwapTotal": 1873,
            "SwapFree": 1295,
        })
        monkeypatch.setattr("scripts.chronos_pipeline.os.getloadavg", lambda: (0.4, 0.3, 0.2))
        monkeypatch.setattr(pipeline, "_pipeline_already_running", lambda: False)

        reason = pipeline._host_pressure_reason()

        assert reason is not None
        assert "swap used 578MB > 256MB" in reason

    def test_full_pipeline_exits_nonzero_when_ingest_fails(self, monkeypatch):
        """Full pipeline should not silently report success when ingest fails."""
        import scripts.chronos_pipeline as pipeline

        class DummySession:
            def close(self):
                return None

        monkeypatch.setattr(sys, "argv", ["chronos_pipeline.py", "--full"])
        monkeypatch.setattr(pipeline, "_host_pressure_reason", lambda: None)
        monkeypatch.setattr(pipeline, "init_db", lambda: None)
        monkeypatch.setattr(pipeline, "SessionLocal", lambda: DummySession())
        monkeypatch.setattr(pipeline, "pipeline_progress", Mock())
        monkeypatch.setattr(
            pipeline,
            "run_ingest",
            lambda *args, **kwargs: pipeline.PhaseResult(
                processed_count=0,
                failure_count=0,
                error_message="Plaud not authenticated",
            ),
        )
        monkeypatch.setattr(pipeline, "run_process", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_index", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_graph", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_refresh_workflows", lambda *args, **kwargs: 0
        )
        monkeypatch.setattr(pipeline, "run_repair_recent", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_backfill_summaries", lambda *args, **kwargs: 0
        )

        with pytest.raises(SystemExit) as exc_info:
            pipeline.main()

        assert exc_info.value.code == 1

    def test_ingest_all_history_flag_reaches_run_ingest(self, monkeypatch):
        """CLI should propagate --all-history into the ingest phase."""
        import scripts.chronos_pipeline as pipeline

        captured = {}

        class DummySession:
            def close(self):
                return None

        monkeypatch.setattr(
            sys,
            "argv",
            ["chronos_pipeline.py", "--ingest", "--all-history"],
        )
        monkeypatch.setattr(pipeline, "_host_pressure_reason", lambda: None)
        monkeypatch.setattr(pipeline, "init_db", lambda: None)
        monkeypatch.setattr(pipeline, "SessionLocal", lambda: DummySession())
        monkeypatch.setattr(pipeline, "pipeline_progress", Mock())

        def fake_run_ingest(*args, **kwargs):
            captured.update(kwargs)
            return pipeline.PhaseResult(processed_count=0, failure_count=0)

        monkeypatch.setattr(pipeline, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(pipeline, "run_process", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_index", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_graph", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_refresh_workflows", lambda *args, **kwargs: 0
        )
        monkeypatch.setattr(pipeline, "run_repair_recent", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_backfill_summaries", lambda *args, **kwargs: 0
        )

        pipeline.main()

        assert captured["all_history"] is True

    def test_backfill_stage_reaches_full_history_path(self, monkeypatch):
        """CLI backfill stage should route into the full-history ingest path."""
        import scripts.chronos_pipeline as pipeline

        captured = {}

        class DummySession:
            def close(self):
                return None

        progress = Mock()

        monkeypatch.setattr(
            sys,
            "argv",
            ["chronos_pipeline.py", "--backfill", "--days-back", "45"],
        )
        monkeypatch.setattr(pipeline, "_host_pressure_reason", lambda: None)
        monkeypatch.setattr(pipeline, "init_db", lambda: None)
        monkeypatch.setattr(pipeline, "SessionLocal", lambda: DummySession())
        monkeypatch.setattr(pipeline, "pipeline_progress", progress)

        def fake_run_ingest(*args, **kwargs):
            captured.update(kwargs)
            return pipeline.PhaseResult(processed_count=0, failure_count=0)

        monkeypatch.setattr(pipeline, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(pipeline, "run_process", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_index", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_graph", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_refresh_workflows", lambda *args, **kwargs: 0
        )
        monkeypatch.setattr(pipeline, "run_repair_recent", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_backfill_summaries", lambda *args, **kwargs: 0
        )

        pipeline.main()

        assert captured["all_history"] is True
        assert captured["fetch_all_pages"] is True
        assert captured["phase_name"] == "backfill"
        progress.start_run.assert_called_once_with(
            phases=[
                "backfill",
                "refresh-workflows",
                "summaries",
                "repair",
                "process",
                "index",
                "graph",
            ],
            trigger="cli",
        )
        assert any(
            call.kwargs.get("sync_mode") == "backfill"
            for call in progress.set_run_context.call_args_list
        )

    def test_full_stage_keeps_recent_sync_behavior(self, monkeypatch):
        """CLI full stage should keep the recent-window sync behavior."""
        import scripts.chronos_pipeline as pipeline

        captured = {}

        class DummySession:
            def close(self):
                return None

        progress = Mock()

        monkeypatch.setattr(
            sys,
            "argv",
            ["chronos_pipeline.py", "--full", "--days-back", "45"],
        )
        monkeypatch.setattr(pipeline, "_host_pressure_reason", lambda: None)
        monkeypatch.setattr(pipeline, "init_db", lambda: None)
        monkeypatch.setattr(pipeline, "SessionLocal", lambda: DummySession())
        monkeypatch.setattr(pipeline, "pipeline_progress", progress)

        def fake_run_ingest(*args, **kwargs):
            captured.update(kwargs)
            return pipeline.PhaseResult(processed_count=0, failure_count=0)

        monkeypatch.setattr(pipeline, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(pipeline, "run_process", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_index", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_graph", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_refresh_workflows", lambda *args, **kwargs: 0
        )
        monkeypatch.setattr(pipeline, "run_repair_recent", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_backfill_summaries", lambda *args, **kwargs: 0
        )

        pipeline.main()

        assert captured["all_history"] is False
        assert captured["fetch_all_pages"] is True
        assert captured["phase_name"] == "ingest"
        progress.start_run.assert_called_once_with(
            phases=[
                "ingest",
                "refresh-workflows",
                "summaries",
                "repair",
                "process",
                "index",
                "graph",
            ],
            trigger="cli",
        )
        assert any(
            call.kwargs.get("sync_mode") == "full"
            for call in progress.set_run_context.call_args_list
        )

    def test_ingest_recording_id_reaches_run_ingest(self, monkeypatch):
        """CLI should propagate --recording-id into the ingest phase."""
        import scripts.chronos_pipeline as pipeline

        captured = {}

        class DummySession:
            def close(self):
                return None

        monkeypatch.setattr(
            sys,
            "argv",
            ["chronos_pipeline.py", "--ingest", "--recording-id", "rec-direct-001"],
        )
        monkeypatch.setattr(pipeline, "_host_pressure_reason", lambda: None)
        monkeypatch.setattr(pipeline, "init_db", lambda: None)
        monkeypatch.setattr(pipeline, "SessionLocal", lambda: DummySession())
        monkeypatch.setattr(pipeline, "pipeline_progress", Mock())

        def fake_run_ingest(*args, **kwargs):
            captured.update(kwargs)
            return pipeline.PhaseResult(processed_count=1, failure_count=0)

        monkeypatch.setattr(pipeline, "run_ingest", fake_run_ingest)
        monkeypatch.setattr(pipeline, "run_process", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_index", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_refresh_workflows", lambda *args, **kwargs: 0
        )
        monkeypatch.setattr(pipeline, "run_repair_recent", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            pipeline, "run_backfill_summaries", lambda *args, **kwargs: 0
        )

        pipeline.main()

        assert captured["recording_id"] == "rec-direct-001"

    def test_repair_recent_flag_reaches_run_repair_recent(self, monkeypatch):
        """CLI should propagate --repair-recent into the repair phase."""
        import scripts.chronos_pipeline as pipeline

        captured = {}

        class DummySession:
            def close(self):
                return None

        monkeypatch.setattr(
            sys,
            "argv",
            ["chronos_pipeline.py", "--repair-recent", "--days-back", "45", "--limit", "7"],
        )
        monkeypatch.setattr(pipeline, "_host_pressure_reason", lambda: None)
        monkeypatch.setattr(pipeline, "init_db", lambda: None)
        monkeypatch.setattr(pipeline, "SessionLocal", lambda: DummySession())
        monkeypatch.setattr(pipeline, "pipeline_progress", Mock())
        monkeypatch.setattr(
            pipeline,
            "run_repair_recent",
            lambda *args, **kwargs: captured.update(kwargs) or 0,
        )
        monkeypatch.setattr(pipeline, "run_ingest", lambda *args, **kwargs: pipeline.PhaseResult(processed_count=0, failure_count=0))
        monkeypatch.setattr(pipeline, "run_process", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_index", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_graph", lambda *args, **kwargs: 0)
        monkeypatch.setattr(pipeline, "run_refresh_workflows", lambda *args, **kwargs: 0)

        pipeline.main()

        assert captured["days_back"] == 45
        assert captured["limit"] == 7


# ===========================================================================
# Webhook Server Tests
# ===========================================================================


class TestWebhookServer:
    """Tests for src/plaud_webhook_server.py."""

    def test_webhook_server_import(self):
        """Verify webhook server can be imported."""
        from src.plaud_webhook_server import PlaudWebhookServer

        assert PlaudWebhookServer is not None

    def test_webhook_server_init(self):
        """Verify webhook server initializes."""
        from src.plaud_webhook_server import PlaudWebhookServer

        server = PlaudWebhookServer(port=9999)
        assert server.port == 9999
        assert not server.is_running

    def test_webhook_url(self):
        """Verify webhook URL is correct."""
        from src.plaud_webhook_server import PlaudWebhookServer

        server = PlaudWebhookServer(port=8090)
        assert server.webhook_url == "http://localhost:8090/webhook/plaud"


class TestWebhookHandler:
    """Tests for src/plaud_webhook.py."""

    def test_handler_import(self):
        """Verify webhook handler can be imported."""
        from src.plaud_webhook import PlaudWebhookHandler

        assert PlaudWebhookHandler is not None

    def test_parse_event(self):
        """Verify event parsing works."""
        from src.plaud_webhook import PlaudWebhookHandler, PlaudEventType

        handler = PlaudWebhookHandler(webhook_secret="test")
        event = handler.parse_event(
            {
                "event_type": "audio_transcribe.completed",
                "event_id": "evt_123",
                "timestamp": "2025-01-15T10:00:00Z",
                "data": {"file_id": "file_abc"},
            }
        )
        assert event.event_type == PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED
        assert event.event_id == "evt_123"
        assert event.file_id == "file_abc"

    def test_unknown_event_type(self):
        """Verify unknown event types are handled."""
        from src.plaud_webhook import PlaudWebhookHandler, PlaudEventType

        handler = PlaudWebhookHandler(webhook_secret="test")
        event = handler.parse_event(
            {
                "event_type": "totally.unknown",
                "event_id": "evt_456",
                "data": {},
            }
        )
        assert event.event_type == PlaudEventType.UNKNOWN

    def test_signature_verification(self):
        """Verify HMAC signature verification."""
        import hmac
        import hashlib

        from src.plaud_webhook import PlaudWebhookHandler

        secret = "test_secret_key"
        handler = PlaudWebhookHandler(webhook_secret=secret)

        payload = b'{"event_type": "test"}'
        expected_sig = hmac.new(
            secret.encode(), msg=payload, digestmod=hashlib.sha256
        ).hexdigest()

        assert handler.verify_signature(payload, expected_sig)
        assert not handler.verify_signature(payload, "invalid_signature")


# ===========================================================================
# USB Watcher Tests
# ===========================================================================


class TestUSBWatcher:
    """Tests for src/plaud_usb_watcher.py."""

    def test_watcher_import(self):
        """Verify USB watcher can be imported."""
        from src.plaud_usb_watcher import PlaudUSBWatcher

        assert PlaudUSBWatcher is not None

    def test_watcher_init(self):
        """Verify watcher initializes."""
        from src.plaud_usb_watcher import PlaudUSBWatcher

        watcher = PlaudUSBWatcher(volumes_path="/nonexistent")
        assert not watcher.is_running
        assert watcher.connected_devices == {}

    def test_device_type_enum(self):
        """Verify device type enum values."""
        from src.plaud_usb_watcher import PlaudDeviceType

        assert PlaudDeviceType.NOTE_PIN.value == "NotePin"
        assert PlaudDeviceType.NOTE.value == "Note"
        assert PlaudDeviceType.NOTE_PRO.value == "NotePro"

    def test_is_plaud_device(self):
        """Verify Plaud device detection logic."""
        from src.plaud_usb_watcher import PlaudUSBWatcher
        from pathlib import Path
        from unittest.mock import PropertyMock

        watcher = PlaudUSBWatcher(volumes_path="/nonexistent")

        # Test name-based detection
        mock_path = Mock(spec=Path)
        mock_path.name = "PLAUD_NOTE"
        mock_path.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=False)))
        assert watcher._is_plaud_device(mock_path)

    def test_callback_registration(self):
        """Verify callback registration."""
        from src.plaud_usb_watcher import PlaudUSBWatcher

        watcher = PlaudUSBWatcher(volumes_path="/nonexistent")
        callback = Mock()
        watcher.on_device_connected(callback)
        assert callback in watcher._on_connect_callbacks
