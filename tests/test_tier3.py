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
    return asyncio.get_event_loop().run_until_complete(coro)


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

        mock_result = Mock()
        mock_result.title = "Test Event"
        mock_result.summary = "A test"
        mock_result.date = "2025-01-15"
        mock_result.time_range_formatted = "10:00-11:00"
        mock_result.category = "work"
        mock_result.score = 0.95
        mock_result.recording_id = "abc123"

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
        )
        syncer._queue_for_processing("rec_123", "test")
        assert "rec_123" in syncer._pending_recordings

        # Duplicate should not be added
        syncer._queue_for_processing("rec_123", "test")
        assert syncer._pending_recordings.count("rec_123") == 1


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
