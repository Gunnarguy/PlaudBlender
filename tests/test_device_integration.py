"""
Tests for Plaud device integration components.

Tests cover:
- USB device watcher functionality
- Webhook handler and server
- Auto-sync service
"""

import os
import getpass
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestPlaudUSBWatcher:
    """Tests for the USB device watcher."""

    def test_import(self):
        """Test that USB watcher can be imported."""
        from src.plaud_usb_watcher import (
            PlaudUSBWatcher,
            USBPlaudDevice,
            PlaudDeviceType,
            get_usb_watcher,
        )

        assert PlaudUSBWatcher is not None
        assert USBPlaudDevice is not None

    def test_watcher_initialization(self):
        """Test watcher initializes with correct defaults."""
        from src.plaud_usb_watcher import PlaudUSBWatcher

        watcher = PlaudUSBWatcher()
        expected_path = Path("/Volumes") if sys.platform == "darwin" else Path(f"/media/{getpass.getuser()}")
        assert watcher.volumes_path == expected_path
        assert watcher.poll_interval == 2.0
        assert not watcher.is_running
        assert len(watcher.connected_devices) == 0

    def test_watcher_start_stop(self):
        """Test starting and stopping the watcher."""
        from src.plaud_usb_watcher import PlaudUSBWatcher

        watcher = PlaudUSBWatcher(poll_interval=0.1)

        # Start
        watcher.start()
        assert watcher.is_running
        time.sleep(0.2)

        # Stop
        watcher.stop()
        assert not watcher.is_running

    def test_usb_device_dataclass(self):
        """Test USBPlaudDevice dataclass."""
        from src.plaud_usb_watcher import USBPlaudDevice, PlaudDeviceType

        # Create with minimal data
        device = USBPlaudDevice(
            volume_path=Path("/Volumes/TestDevice"),
            volume_name="TestDevice",
            device_type=PlaudDeviceType.NOTE,
        )

        assert device.volume_name == "TestDevice"
        assert device.device_type == PlaudDeviceType.NOTE
        assert isinstance(device.connected_at, datetime)

    def test_device_to_dict(self):
        """Test device serialization to dict."""
        from src.plaud_usb_watcher import USBPlaudDevice, PlaudDeviceType

        device = USBPlaudDevice(
            volume_path=Path("/Volumes/PLAUD_NOTE"),
            volume_name="PLAUD_NOTE",
            device_type=PlaudDeviceType.NOTE,
        )

        # Mock some stats to verify serialization
        device.audio_file_count = 5
        device.total_audio_size_mb = 10.554
        device.recording_folders = ["RECORD"]

        data = device.to_dict()
        assert data["volume_path"] == str(Path("/Volumes/PLAUD_NOTE"))
        assert data["volume_name"] == "PLAUD_NOTE"
        assert data["device_type"] == "Note"
        assert "connected_at" in data
        assert data["audio_file_count"] == 5
        assert data["total_audio_size_mb"] == 10.55
        assert data["recording_folders"] == ["RECORD"]
        assert data["has_recordings"] is True

    def test_plaud_volume_detection(self):
        """Test that Plaud volume patterns are detected."""
        from src.plaud_usb_watcher import PlaudUSBWatcher, PLAUD_VOLUME_PATTERNS

        watcher = PlaudUSBWatcher()

        # Test pattern matching
        assert "PLAUD" in PLAUD_VOLUME_PATTERNS
        assert "PLAUD_NOTE" in PLAUD_VOLUME_PATTERNS

    def test_callback_registration(self):
        """Test callback registration."""
        from src.plaud_usb_watcher import PlaudUSBWatcher

        watcher = PlaudUSBWatcher()
        callback_called = []

        def on_connect(device):
            callback_called.append(device)

        watcher.on_device_connected(on_connect)
        assert len(watcher._on_connect_callbacks) == 1

    def test_singleton_instance(self):
        """Test get_usb_watcher returns singleton."""
        from src.plaud_usb_watcher import get_usb_watcher

        watcher1 = get_usb_watcher()
        watcher2 = get_usb_watcher()
        assert watcher1 is watcher2


class TestPlaudWebhookHandler:
    """Tests for the webhook handler."""

    def test_import(self):
        """Test webhook handler can be imported."""
        from src.plaud_webhook import (
            PlaudWebhookHandler,
            PlaudEvent,
            PlaudEventType,
        )

        assert PlaudWebhookHandler is not None
        assert PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED is not None

    def test_handler_initialization(self):
        """Test handler initializes correctly."""
        from src.plaud_webhook import PlaudWebhookHandler

        handler = PlaudWebhookHandler(webhook_secret="test_secret")
        assert handler.webhook_secret == "test_secret"

    def test_event_parsing(self):
        """Test parsing webhook payloads."""
        from src.plaud_webhook import PlaudWebhookHandler, PlaudEventType

        handler = PlaudWebhookHandler()

        payload = {
            "event_type": "audio_transcribe.completed",
            "event_id": "evt_123",
            "timestamp": "2024-12-26T10:00:00Z",
            "data": {"file_id": "file_abc"},
        }

        event = handler.parse_event(payload)
        assert event.event_type == PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED
        assert event.event_id == "evt_123"
        assert event.file_id == "file_abc"

    def test_unknown_event_type(self):
        """Test handling unknown event types."""
        from src.plaud_webhook import PlaudWebhookHandler, PlaudEventType

        handler = PlaudWebhookHandler()

        payload = {
            "event_type": "some.unknown.event",
            "event_id": "evt_456",
            "data": {},
        }

        event = handler.parse_event(payload)
        assert event.event_type == PlaudEventType.UNKNOWN

    def test_signature_verification_no_secret(self):
        """Test signature verification when no secret is set."""
        from unittest.mock import patch
        from src.plaud_webhook import PlaudWebhookHandler

        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("PLAUD_WEBHOOK_SECRET", None)
            handler = PlaudWebhookHandler(webhook_secret=None)

        # Should return True when no secret is configured (skips verification)
        result = handler.verify_signature(b"test", "any_signature")
        assert result is True

    def test_signature_verification_missing_header(self):
        """Test signature verification with missing header."""
        from src.plaud_webhook import PlaudWebhookHandler

        handler = PlaudWebhookHandler(webhook_secret="test_secret")

        result = handler.verify_signature(b"test", None)
        assert result is False

    def test_handler_registration(self):
        """Test registering event handlers."""
        from src.plaud_webhook import PlaudWebhookHandler, PlaudEventType

        handler = PlaudWebhookHandler()
        callback_called = []

        def on_transcribe(event):
            callback_called.append(event)

        handler.register_handler(
            PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED, on_transcribe
        )

        # Parse and handle an event
        payload = {
            "event_type": "audio_transcribe.completed",
            "event_id": "evt_789",
            "data": {"file_id": "file_xyz"},
        }
        event = handler.parse_event(payload)
        handler.handle_event(event)

        assert len(callback_called) == 1
        assert callback_called[0].file_id == "file_xyz"


class TestPlaudAutoSync:
    """Tests for the auto-sync service."""

    def test_import(self):
        """Test auto-sync can be imported."""
        from src.plaud_auto_sync import (
            PlaudAutoSync,
            SyncConfig,
            SyncJob,
            SyncTrigger,
        )

        assert PlaudAutoSync is not None
        assert SyncConfig is not None

    def test_sync_config_defaults(self):
        """Test SyncConfig default values."""
        from src.plaud_auto_sync import SyncConfig

        config = SyncConfig()
        assert config.sync_on_usb_connect is True
        assert config.sync_on_webhook is True
        assert config.ingest_new_recordings is True
        assert config.process_after_ingest is True
        assert config.index_after_process is True
        assert config.refresh_workflows is True
        assert config.enable_scheduled_poll is True
        assert config.poll_interval_minutes == 15
        assert config.enable_webhook_server is True
        assert config.webhook_port == 8090

    def test_sync_job_creation(self):
        """Test SyncJob dataclass."""
        from src.plaud_auto_sync import SyncJob, SyncTrigger

        job = SyncJob(
            trigger=SyncTrigger.USB_CONNECTED,
            device_path="/Volumes/PLAUD_NOTE",
        )

        assert job.trigger == SyncTrigger.USB_CONNECTED
        assert job.device_path == "/Volumes/PLAUD_NOTE"
        assert job.status == "pending"

    def test_sync_job_to_dict(self):
        """Test SyncJob serialization."""
        from src.plaud_auto_sync import SyncJob, SyncTrigger

        job = SyncJob(
            trigger=SyncTrigger.MANUAL,
            recording_id="rec_123",
        )

        data = job.to_dict()
        assert data["trigger"] == "manual"
        assert data["recording_id"] == "rec_123"

    def test_auto_sync_initialization(self):
        """Test PlaudAutoSync initialization."""
        from src.plaud_auto_sync import PlaudAutoSync, SyncConfig

        config = SyncConfig(
            sync_on_usb_connect=False,
            process_after_ingest=True,
        )

        sync = PlaudAutoSync(config=config)
        assert sync.config.sync_on_usb_connect is False
        assert sync.config.process_after_ingest is True
        assert not sync.is_running

    def test_auto_sync_status(self):
        """Test getting auto-sync status."""
        from src.plaud_auto_sync import PlaudAutoSync

        sync = PlaudAutoSync()
        status = sync.get_status()

        assert "running" in status
        assert "pending_jobs" in status
        assert "config" in status
        assert status["running"] is False

    def test_manual_sync_trigger(self):
        """Test triggering manual sync."""
        from src.plaud_auto_sync import PlaudAutoSync, SyncTrigger

        sync = PlaudAutoSync()
        job = sync.trigger_manual_sync(recording_id="rec_test")

        assert job.trigger == SyncTrigger.MANUAL
        assert job.recording_id == "rec_test"
        assert sync.pending_jobs == 1

    def test_callback_registration(self):
        """Test on_sync callback registration."""
        from src.plaud_auto_sync import PlaudAutoSync

        sync = PlaudAutoSync()
        callback_called = []

        def on_sync(job):
            callback_called.append(job)

        sync.on_sync(on_sync)
        assert len(sync._on_sync_callbacks) == 1


class TestWebhookEventTypes:
    """Test that all webhook event types are defined."""

    def test_all_event_types_exist(self):
        """Test all expected event types are defined."""
        from src.plaud_webhook import PlaudEventType

        expected_types = [
            "FILE_UPLOADED",
            "FILE_DELETED",
            "AUDIO_TRANSCRIBE_STARTED",
            "AUDIO_TRANSCRIBE_COMPLETED",
            "AUDIO_TRANSCRIBE_FAILED",
            "WORKFLOW_STARTED",
            "WORKFLOW_COMPLETED",
            "WORKFLOW_FAILED",
            "DEVICE_CONNECTED",
            "DEVICE_DISCONNECTED",
            "RECORDING_UPLOADED",
        ]

        for type_name in expected_types:
            assert hasattr(
                PlaudEventType, type_name
            ), f"Missing event type: {type_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
