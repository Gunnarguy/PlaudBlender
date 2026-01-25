"""
Plaud Auto-Sync Service - Automatic synchronization when devices connect.

Combines USB device detection with webhook events to automatically
trigger the Chronos ingest pipeline when:
- A Plaud device is plugged in via USB
- A webhook indicates new recordings are available
- A device connects via WiFi/Bluetooth

Usage:
    from src.plaud_auto_sync import PlaudAutoSync, get_auto_sync

    sync_service = get_auto_sync()
    sync_service.start()
"""

import os
import logging
import threading
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue

from .plaud_usb_watcher import (
    PlaudUSBWatcher,
    USBPlaudDevice,
    get_usb_watcher,
    start_watcher,
)
from .plaud_webhook import PlaudEvent, PlaudEventType
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SyncTrigger(Enum):
    """What triggered the sync."""

    USB_CONNECTED = "usb_connected"
    WEBHOOK_RECORDING = "webhook_recording"
    WEBHOOK_TRANSCRIBE = "webhook_transcribe"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class SyncJob:
    """A sync job to be processed."""

    trigger: SyncTrigger
    timestamp: datetime = field(default_factory=datetime.now)
    device_path: Optional[str] = None
    recording_id: Optional[str] = None
    file_id: Optional[str] = None
    status: str = "pending"
    result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger.value,
            "timestamp": self.timestamp.isoformat(),
            "device_path": self.device_path,
            "recording_id": self.recording_id,
            "file_id": self.file_id,
            "status": self.status,
            "result": self.result,
        }


@dataclass
class SyncConfig:
    """Configuration for auto-sync behavior."""

    # Auto-sync triggers
    sync_on_usb_connect: bool = True
    sync_on_webhook: bool = True

    # What to sync
    ingest_new_recordings: bool = True
    process_after_ingest: bool = False  # Run Gemini processing
    index_after_process: bool = False  # Index to Qdrant

    # Rate limiting
    min_sync_interval_seconds: int = 60
    max_concurrent_syncs: int = 1

    # Notifications
    notify_on_sync: bool = True
    notification_callback: Optional[Callable[[SyncJob], None]] = None


class PlaudAutoSync:
    """
    Automatic sync service for Plaud devices.

    Monitors for USB connections and webhook events, then triggers
    the appropriate Chronos pipeline steps.
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        """
        Initialize auto-sync service.

        Args:
            config: Sync configuration options
        """
        self.config = config or SyncConfig()

        # Components
        self._usb_watcher: Optional[PlaudUSBWatcher] = None

        # State
        self._running = False
        self._last_sync_time: Optional[datetime] = None
        self._sync_queue: Queue[SyncJob] = Queue()
        self._sync_history: List[SyncJob] = []
        self._worker_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_sync_callbacks: List[Callable[[SyncJob], None]] = []

    def _handle_usb_connect(self, device: USBPlaudDevice) -> None:
        """Handle USB device connection."""
        if not self.config.sync_on_usb_connect:
            return

        logger.info(f"USB device connected: {device.volume_name}")

        # Check rate limiting
        if self._last_sync_time:
            elapsed = (datetime.now() - self._last_sync_time).total_seconds()
            if elapsed < self.config.min_sync_interval_seconds:
                logger.info(
                    f"Rate limited: {elapsed:.0f}s since last sync "
                    f"(min {self.config.min_sync_interval_seconds}s)"
                )
                return

        # Queue sync job
        job = SyncJob(
            trigger=SyncTrigger.USB_CONNECTED,
            device_path=str(device.volume_path),
        )
        self._sync_queue.put(job)
        logger.info(f"Queued sync job for USB device: {device.volume_name}")

    def _handle_usb_disconnect(self, path: str) -> None:
        """Handle USB device disconnection."""
        logger.info(f"USB device disconnected: {path}")

    def handle_webhook_event(self, event: PlaudEvent) -> None:
        """
        Handle a webhook event that might trigger sync.

        Args:
            event: PlaudEvent from webhook
        """
        if not self.config.sync_on_webhook:
            return

        # Determine if this event should trigger sync
        if event.event_type == PlaudEventType.RECORDING_UPLOADED:
            job = SyncJob(
                trigger=SyncTrigger.WEBHOOK_RECORDING,
                recording_id=event.recording_id,
                file_id=event.file_id,
            )
            self._sync_queue.put(job)
            logger.info(f"Queued sync for recording: {event.recording_id}")

        elif event.event_type == PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED:
            job = SyncJob(
                trigger=SyncTrigger.WEBHOOK_TRANSCRIBE,
                file_id=event.file_id,
            )
            self._sync_queue.put(job)
            logger.info(f"Queued sync for transcription: {event.file_id}")

    def _process_sync_job(self, job: SyncJob) -> None:
        """Process a single sync job."""
        try:
            job.status = "running"
            self._last_sync_time = datetime.now()

            logger.info(f"Processing sync job: {job.trigger.value}")

            # Build pipeline command
            cmd = [sys.executable, "scripts/chronos_pipeline.py"]

            if self.config.ingest_new_recordings:
                cmd.append("--ingest")

            if self.config.process_after_ingest:
                cmd.append("--process")

            if self.config.index_after_process:
                cmd.append("--index")

            # Add specific recording if we have one
            if job.recording_id:
                cmd.extend(["--recording-id", job.recording_id])
                cmd.extend(["--limit", "1"])
            elif job.file_id:
                cmd.extend(["--file-id", job.file_id])
                cmd.extend(["--limit", "1"])

            logger.info(f"Running: {' '.join(cmd)}")

            # Run pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                job.status = "completed"
                job.result = "Sync completed successfully"
                logger.info(f"Sync completed: {job.trigger.value}")
            else:
                job.status = "failed"
                job.result = result.stderr or result.stdout
                logger.error(f"Sync failed: {job.result}")

        except subprocess.TimeoutExpired:
            job.status = "timeout"
            job.result = "Sync timed out after 5 minutes"
            logger.error("Sync job timed out")

        except Exception as e:
            job.status = "error"
            job.result = str(e)
            logger.error(f"Sync error: {e}")

        # Add to history
        self._sync_history.append(job)
        if len(self._sync_history) > 50:
            self._sync_history = self._sync_history[-50:]

        # Fire callbacks
        for callback in self._on_sync_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Sync callback error: {e}")

        if self.config.notification_callback:
            try:
                self.config.notification_callback(job)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")

    def _worker_loop(self) -> None:
        """Worker thread that processes sync jobs."""
        logger.info("Auto-sync worker started")
        while self._running:
            try:
                job = self._sync_queue.get(timeout=1.0)
                self._process_sync_job(job)
            except:
                continue
        logger.info("Auto-sync worker stopped")

    # Public API

    def on_sync(self, callback: Callable[[SyncJob], None]) -> None:
        """
        Register a callback for sync events.

        Args:
            callback: Function that receives SyncJob
        """
        self._on_sync_callbacks.append(callback)

    def trigger_manual_sync(
        self,
        recording_id: Optional[str] = None,
        full: bool = False,
    ) -> SyncJob:
        """
        Trigger a manual sync.

        Args:
            recording_id: Specific recording to sync (or all if None)
            full: Run full pipeline (ingest + process + index)

        Returns:
            The created SyncJob
        """
        job = SyncJob(trigger=SyncTrigger.MANUAL, recording_id=recording_id)

        # Override config for this sync
        if full:
            old_process = self.config.process_after_ingest
            old_index = self.config.index_after_process
            self.config.process_after_ingest = True
            self.config.index_after_process = True

        self._sync_queue.put(job)

        if full:
            self.config.process_after_ingest = old_process
            self.config.index_after_process = old_index

        return job

    def start(self) -> None:
        """Start the auto-sync service."""
        if self._running:
            return

        self._running = True

        # Start USB watcher
        self._usb_watcher = get_usb_watcher()
        self._usb_watcher.on_device_connected(self._handle_usb_connect)
        self._usb_watcher.on_device_disconnected(self._handle_usb_disconnect)
        self._usb_watcher.start()

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info("Auto-sync service started")

    def stop(self) -> None:
        """Stop the auto-sync service."""
        self._running = False

        if self._usb_watcher:
            self._usb_watcher.stop()

        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)

        logger.info("Auto-sync service stopped")

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    @property
    def connected_devices(self) -> Dict[str, USBPlaudDevice]:
        """Get currently connected USB devices."""
        if self._usb_watcher:
            return self._usb_watcher.connected_devices
        return {}

    @property
    def sync_history(self) -> List[SyncJob]:
        """Get sync job history."""
        return self._sync_history.copy()

    @property
    def pending_jobs(self) -> int:
        """Get number of pending sync jobs."""
        return self._sync_queue.qsize()

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "running": self._running,
            "usb_watcher_running": (
                self._usb_watcher.is_running if self._usb_watcher else False
            ),
            "connected_devices": len(self.connected_devices),
            "pending_jobs": self.pending_jobs,
            "last_sync": (
                self._last_sync_time.isoformat() if self._last_sync_time else None
            ),
            "total_syncs": len(self._sync_history),
            "config": {
                "sync_on_usb": self.config.sync_on_usb_connect,
                "sync_on_webhook": self.config.sync_on_webhook,
                "process_after_ingest": self.config.process_after_ingest,
                "index_after_process": self.config.index_after_process,
            },
        }


# Singleton instance
_auto_sync: Optional[PlaudAutoSync] = None


def get_auto_sync() -> PlaudAutoSync:
    """Get the singleton auto-sync instance."""
    global _auto_sync
    if _auto_sync is None:
        _auto_sync = PlaudAutoSync()
    return _auto_sync


def start_auto_sync() -> PlaudAutoSync:
    """Start auto-sync if not already running."""
    sync = get_auto_sync()
    if not sync.is_running:
        sync.start()
    return sync


if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🔄 Starting Plaud Auto-Sync Service...")
    print("   Watching for USB connections and webhooks")
    print("   Press Ctrl+C to stop\n")

    sync = PlaudAutoSync()

    def on_sync(job: SyncJob):
        print(f"\n📥 Sync completed: {job.trigger.value}")
        print(f"   Status: {job.status}")
        if job.result:
            print(f"   Result: {job.result[:100]}...")

    sync.on_sync(on_sync)
    sync.start()

    # Show initial status
    print(f"Status: {sync.get_status()}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping auto-sync...")
        sync.stop()
