"""
MCP Server Tests.
Tests for scripts/mcp_server.py — timeout decorator, tool functions, and JSON responses.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===========================================================================
# Import Tests
# ===========================================================================


class TestMCPImports:
    """Verify mcp_server module can be imported and has expected attributes."""

    def test_module_imports(self):
        from scripts import mcp_server

        assert mcp_server is not None

    def test_server_object_exists(self):
        from scripts.mcp_server import server

        assert server is not None

    def test_timeout_decorator_exists(self):
        from scripts.mcp_server import _with_timeout

        assert callable(_with_timeout)

    def test_lazy_initializers_exist(self):
        from scripts.mcp_server import _get_db_session, _get_qdrant, _get_data_service

        assert callable(_get_db_session)
        assert callable(_get_qdrant)
        assert callable(_get_data_service)


# ===========================================================================
# Timeout Decorator Tests
# ===========================================================================


class TestTimeoutDecorator:
    """Tests for _with_timeout decorator."""

    def test_passes_result_through(self):
        """Normal function returns its result."""
        from scripts.mcp_server import _with_timeout

        @_with_timeout(5)
        async def fast_fn():
            return "ok"

        result = asyncio.get_event_loop().run_until_complete(fast_fn())
        assert result == "ok"

    def test_timeout_returns_json_error(self):
        """Timed-out function returns JSON error, not exception."""
        from scripts.mcp_server import _with_timeout

        @_with_timeout(0.1)
        async def slow_fn():
            await asyncio.sleep(10)
            return "never"

        result = asyncio.get_event_loop().run_until_complete(slow_fn())
        parsed = json.loads(result)
        assert "error" in parsed
        assert "timed out" in parsed["error"]
        assert "slow_fn" in parsed["error"]

    def test_default_timeout(self):
        """Default timeout uses MCP_TOOL_TIMEOUT constant."""
        from scripts.mcp_server import _with_timeout, MCP_TOOL_TIMEOUT

        @_with_timeout()
        async def fn():
            return "done"

        result = asyncio.get_event_loop().run_until_complete(fn())
        assert result == "done"

    def test_preserves_function_name(self):
        """Wrapper preserves the original function name."""
        from scripts.mcp_server import _with_timeout

        @_with_timeout(5)
        async def my_tool():
            return "x"

        assert my_tool.__name__ == "my_tool"


# ===========================================================================
# Tool Function Tests (mocked services)
# ===========================================================================


class TestPingTool:
    """Tests for the ping() MCP tool."""

    def test_ping_returns_pong(self):
        from scripts.mcp_server import ping

        result = asyncio.get_event_loop().run_until_complete(ping())
        assert result == "pong"


class TestSearchEventsTool:
    """Tests for search_events() with mocked data service."""

    @patch("scripts.mcp_server._get_data_service")
    def test_empty_results(self, mock_get_ds):
        from scripts.mcp_server import search_events

        mock_ds = MagicMock()
        mock_ds.search.return_value = []
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(
            search_events("test query")
        )
        parsed = json.loads(result)
        assert parsed["results"] == []
        assert "No events found" in parsed["message"]

    @patch("scripts.mcp_server._get_data_service")
    def test_returns_events_json(self, mock_get_ds):
        from scripts.mcp_server import search_events

        mock_event = MagicMock()
        mock_event.clean_text = "Had a meeting about the budget."
        mock_event.start_ts = datetime(2026, 3, 10, 9, 0)
        mock_event.end_ts = datetime(2026, 3, 10, 9, 30)
        mock_event.category = "meeting"
        mock_event.recording_id = "rec-001"

        mock_result = MagicMock()
        mock_result.event = mock_event
        mock_result.score = 0.85

        mock_ds = MagicMock()
        mock_ds.search.return_value = [mock_result]
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(
            search_events("budget meeting")
        )
        parsed = json.loads(result)
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["category"] == "meeting"
        assert parsed["results"][0]["score"] == 0.85
        assert parsed["total"] == 1

    @patch("scripts.mcp_server._get_data_service")
    def test_limit_clamped(self, mock_get_ds):
        """Limit is clamped to 1-50 range."""
        from scripts.mcp_server import search_events

        mock_ds = MagicMock()
        mock_ds.search.return_value = []
        mock_get_ds.return_value = mock_ds

        asyncio.get_event_loop().run_until_complete(search_events("q", limit=999))
        mock_ds.search.assert_called_once()
        # search was called with limit=50 (clamped)
        _, kwargs = mock_ds.search.call_args
        assert kwargs["limit"] == 50

    @patch("scripts.mcp_server._get_data_service")
    def test_date_filtering(self, mock_get_ds):
        """Date filters exclude out-of-range results."""
        from scripts.mcp_server import search_events

        mock_event = MagicMock()
        mock_event.clean_text = "Old event."
        mock_event.start_ts = datetime(2025, 1, 1, 10, 0)
        mock_event.end_ts = datetime(2025, 1, 1, 10, 30)
        mock_event.category = "work"
        mock_event.recording_id = "rec-old"

        mock_result = MagicMock()
        mock_result.event = mock_event
        mock_result.score = 0.9

        mock_ds = MagicMock()
        mock_ds.search.return_value = [mock_result]
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(
            search_events("q", date_from="2026-01-01")
        )
        parsed = json.loads(result)
        assert len(parsed["results"]) == 0  # filtered out

    @patch("scripts.mcp_server._get_data_service")
    def test_handles_exception(self, mock_get_ds):
        """Exception returns JSON error, not crash."""
        from scripts.mcp_server import search_events

        mock_get_ds.side_effect = RuntimeError("DB gone")

        result = asyncio.get_event_loop().run_until_complete(search_events("test"))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "DB gone" in parsed["error"]


class TestGetRecordingTool:
    """Tests for get_recording() with mocked data service."""

    @patch("scripts.mcp_server._get_data_service")
    def test_not_found(self, mock_get_ds):
        from scripts.mcp_server import get_recording

        mock_ds = MagicMock()
        mock_ds.get_recording_detail.return_value = None
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(
            get_recording("missing-id")
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"]

    @patch("scripts.mcp_server._get_data_service")
    def test_returns_recording_detail(self, mock_get_ds):
        from scripts.mcp_server import get_recording

        mock_event = MagicMock()
        mock_event.clean_text = "Discussed project timeline."
        mock_event.category = "work"
        mock_event.start_ts = datetime(2026, 3, 10, 14, 0)
        mock_event.end_ts = datetime(2026, 3, 10, 14, 30)

        mock_summary = MagicMock()
        mock_summary.recording_id = "rec-123"
        mock_summary.start_time = datetime(2026, 3, 10, 14, 0)
        mock_summary.duration_formatted = "30m"
        mock_summary.top_category = "work"

        mock_detail = MagicMock()
        mock_detail.events = [mock_event]
        mock_detail.summary = mock_summary
        mock_detail.category_percentages = {"work": 100}

        mock_ds = MagicMock()
        mock_ds.get_recording_detail.return_value = mock_detail
        mock_ds.get_transcript.return_value = "Full transcript text here."
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(get_recording("rec-123"))
        parsed = json.loads(result)
        assert parsed["recording_id"] == "rec-123"
        assert parsed["event_count"] == 1
        assert parsed["transcript_preview"] == "Full transcript text here."


class TestListRecordingsTool:
    """Tests for list_recordings() with mocked DB."""

    @patch("scripts.mcp_server._get_db_session")
    def test_returns_recordings(self, mock_get_db):
        from scripts.mcp_server import list_recordings

        mock_db = MagicMock()
        mock_row = (
            "rec-001",
            "Morning notes",
            "2026-03-10 08:00:00",
            600.0,
            "completed",
            "plaud_api",
            5,
        )
        mock_db.execute.return_value.fetchall.return_value = [mock_row]
        mock_get_db.return_value = mock_db

        result = asyncio.get_event_loop().run_until_complete(list_recordings())
        parsed = json.loads(result)
        assert len(parsed["recordings"]) == 1
        assert parsed["recordings"][0]["recording_id"] == "rec-001"
        assert parsed["recordings"][0]["duration_minutes"] == 10.0

    @patch("scripts.mcp_server._get_db_session")
    def test_empty_results(self, mock_get_db):
        from scripts.mcp_server import list_recordings

        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mock_get_db.return_value = mock_db

        result = asyncio.get_event_loop().run_until_complete(list_recordings())
        parsed = json.loads(result)
        assert parsed["recordings"] == []
        assert parsed["count"] == 0


class TestGetStatsTool:
    """Tests for get_stats() with mocked data service."""

    @patch("scripts.mcp_server._get_data_service")
    def test_returns_stats(self, mock_get_ds):
        from scripts.mcp_server import get_stats

        mock_stats = MagicMock()
        mock_stats.total_events = 42
        mock_stats.total_days = 7
        mock_stats.total_duration_hours = 10.5
        mock_stats.categories = {"work": 20, "meeting": 15, "personal": 7}
        mock_stats.most_productive_day = "Monday"

        mock_ds = MagicMock()
        mock_ds.get_stats.return_value = mock_stats
        mock_ds.get_recording_db_stats.return_value = {
            "completed": 30,
            "failed": 2,
            "pending": 1,
        }
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(get_stats())
        parsed = json.loads(result)
        assert parsed["events"]["total"] == 42
        assert parsed["recordings"]["completed"] == 30
        assert parsed["avg_events_per_day"] == 6.0


class TestGetTopicsTool:
    """Tests for get_topics() with mocked data service."""

    @patch("scripts.mcp_server._get_data_service")
    def test_returns_topics(self, mock_get_ds):
        from scripts.mcp_server import get_topics

        mock_ds = MagicMock()
        mock_ds.get_all_topics.return_value = [("work", 30), ("meeting", 20)]
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(get_topics())
        parsed = json.loads(result)
        assert len(parsed["topics"]) == 2
        assert parsed["topics"][0]["category"] == "work"
        assert parsed["topics"][0]["count"] == 30


class TestRunPipelineTool:
    """Tests for run_pipeline() with mocked subprocess."""

    @patch("scripts.mcp_server.subprocess.run")
    def test_invalid_stage(self, mock_run):
        from scripts.mcp_server import run_pipeline

        result = asyncio.get_event_loop().run_until_complete(
            run_pipeline(stage="invalid")
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Invalid stage" in parsed["error"]
        mock_run.assert_not_called()

    @patch("scripts.mcp_server.subprocess.run")
    def test_successful_run(self, mock_run):
        from scripts.mcp_server import run_pipeline

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Pipeline complete. 5 events indexed.",
            stderr="",
        )

        result = asyncio.get_event_loop().run_until_complete(run_pipeline(stage="full"))
        parsed = json.loads(result)
        assert parsed["stage"] == "full"
        assert parsed["exit_code"] == 0
        assert "5 events indexed" in parsed["output"]

    @patch("scripts.mcp_server.subprocess.run")
    def test_subprocess_timeout(self, mock_run):
        from scripts.mcp_server import run_pipeline
        import subprocess as sp

        mock_run.side_effect = sp.TimeoutExpired(cmd="pipeline", timeout=600)

        result = asyncio.get_event_loop().run_until_complete(run_pipeline(stage="full"))
        parsed = json.loads(result)
        assert "timed out" in parsed["error"]


class TestSystemStatusTool:
    """Tests for system_status() with mocked services."""

    @patch("scripts.mcp_server._get_db_session")
    def test_database_ok(self, mock_get_db):
        from scripts.mcp_server import system_status

        mock_db = MagicMock()
        mock_db.execute.return_value.scalar.return_value = 42
        mock_get_db.return_value = mock_db

        result = asyncio.get_event_loop().run_until_complete(system_status())
        parsed = json.loads(result)
        assert parsed["database"]["status"] == "ok"
        assert parsed["database"]["recordings"] == 42

    @patch("scripts.mcp_server._get_db_session")
    def test_database_error(self, mock_get_db):
        from scripts.mcp_server import system_status

        mock_get_db.side_effect = RuntimeError("Connection refused")

        result = asyncio.get_event_loop().run_until_complete(system_status())
        parsed = json.loads(result)
        assert parsed["database"]["status"] == "error"
        assert "Connection refused" in parsed["database"]["message"]


class TestGetGraphTool:
    """Tests for get_graph() with mocked data service."""

    @patch("scripts.mcp_server._get_data_service")
    def test_no_graph_data(self, mock_get_ds):
        from scripts.mcp_server import get_graph

        mock_ds = MagicMock()
        mock_ds.get_graph_data.return_value = None
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(get_graph())
        parsed = json.loads(result)
        assert "No graph data available" in parsed["message"]

    @patch("scripts.mcp_server._get_data_service")
    def test_returns_nodes_and_edges(self, mock_get_ds):
        from scripts.mcp_server import get_graph

        mock_node1 = {
            "id": "person:alice",
            "label": "Alice",
            "type": "person",
            "weight": 5,
        }
        mock_node2 = {
            "id": "project:chronos",
            "label": "Chronos",
            "type": "project",
            "weight": 3,
        }
        mock_edge = {
            "source": "person:alice",
            "target": "project:chronos",
            "type": "works_on",
            "weight": 2,
        }

        mock_graph = MagicMock()
        mock_graph.nodes = [mock_node1, mock_node2]
        mock_graph.edges = [mock_edge]

        mock_ds = MagicMock()
        mock_ds.get_graph_data.return_value = mock_graph
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(get_graph())
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 2
        assert len(parsed["edges"]) == 1
        assert parsed["edges"][0]["source"] == "person:alice"

    @patch("scripts.mcp_server._get_data_service")
    def test_entity_type_filter(self, mock_get_ds):
        from scripts.mcp_server import get_graph

        mock_person = {
            "id": "person:alice",
            "label": "Alice",
            "type": "person",
            "weight": 5,
        }
        mock_project = {
            "id": "project:chronos",
            "label": "Chronos",
            "type": "project",
            "weight": 3,
        }

        mock_graph = MagicMock()
        mock_graph.nodes = [mock_person, mock_project]
        mock_graph.edges = []

        mock_ds = MagicMock()
        mock_ds.get_graph_data.return_value = mock_graph
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(
            get_graph(entity_types="person")
        )
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 1
        assert parsed["nodes"][0]["type"] == "person"


class TestAskChronosTool:
    """Tests for ask_chronos() with mocked services."""

    @patch("scripts.mcp_server._get_data_service")
    def test_no_results(self, mock_get_ds):
        from scripts.mcp_server import ask_chronos

        mock_settings = MagicMock()
        mock_settings.openai_api_key = None
        mock_settings.gemini_api_key = None

        mock_ds = MagicMock()
        mock_ds.search.return_value = []
        mock_get_ds.return_value = mock_ds

        with patch("src.config.get_settings", return_value=mock_settings):
            result = asyncio.get_event_loop().run_until_complete(
                ask_chronos("What happened today?")
            )
        parsed = json.loads(result)
        assert "couldn't find" in parsed["answer"]

    @patch("scripts.mcp_server._get_data_service")
    def test_handles_exception(self, mock_get_ds):
        from scripts.mcp_server import ask_chronos

        mock_get_ds.side_effect = RuntimeError("Service down")

        result = asyncio.get_event_loop().run_until_complete(ask_chronos("anything"))
        parsed = json.loads(result)
        assert "error" in parsed


# ===========================================================================
# Tool JSON Contract Tests
# ===========================================================================


class TestJSONContracts:
    """Verify all tools return valid JSON strings."""

    def test_ping_is_not_json(self):
        """ping returns plain text, not JSON."""
        from scripts.mcp_server import ping

        result = asyncio.get_event_loop().run_until_complete(ping())
        assert result == "pong"
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @patch("scripts.mcp_server._get_data_service")
    def test_search_events_returns_valid_json(self, mock_get_ds):
        from scripts.mcp_server import search_events

        mock_ds = MagicMock()
        mock_ds.search.return_value = []
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(search_events("test"))
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @patch("scripts.mcp_server._get_data_service")
    def test_get_stats_returns_valid_json(self, mock_get_ds):
        from scripts.mcp_server import get_stats

        mock_stats = MagicMock()
        mock_stats.total_events = 0
        mock_stats.total_days = 0
        mock_stats.total_duration_hours = 0
        mock_stats.categories = {}
        mock_stats.most_productive_day = ""

        mock_ds = MagicMock()
        mock_ds.get_stats.return_value = mock_stats
        mock_ds.get_recording_db_stats.return_value = {}
        mock_get_ds.return_value = mock_ds

        result = asyncio.get_event_loop().run_until_complete(get_stats())
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
