"""Chronos Auto-Sync — Connects Plaud webhooks and USB watcher to the pipeline.

When a new recording arrives (via webhook or USB), this integration layer
automatically triggers the relevant pipeline stages:
  1. Ingest → download/register the recording
  2. Process → extract events via Gemini
  3. Index → embed and store in Qdrant

Usage:
    # Start all listeners (webhook server + USB watcher + auto-processing)
    python -m scripts.auto_sync

    # Or import and use programmatically:
    from scripts.auto_sync import ChronosAutoSync
    syncer = ChronosAutoSync()
    syncer.start()
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.version_info >= (3, 11):
    sys.set_int_max_str_digits(0)

from dotenv import load_dotenv

load_dotenv()

from src.config import get_settings
from src.plaud_webhook_server import PlaudWebhookServer, start_webhook_server
from src.plaud_webhook import PlaudEvent, PlaudEventType
from src.plaud_usb_watcher import PlaudUSBWatcher, USBPlaudDevice, start_watcher

logger = logging.getLogger(__name__)
settings = get_settings()


class ChronosAutoSync:
    """Orchestrates automatic recording sync from webhooks and USB.

    Ties together:
    - PlaudWebhookServer (HTTP listener for Plaud push events)
    - PlaudUSBWatcher (macOS /Volumes/ polling for USB devices)
    - Chronos Pipeline (ingest → process → index)
    - Periodic Plaud API polling (scheduled sync every N minutes)
    """

    # How often to poll the Plaud API for new recordings (seconds)
    POLL_INTERVAL = int(os.environ.get("CHRONOS_POLL_INTERVAL", 1800))  # 30 min default

    def __init__(
        self,
        webhook_port: int = 8090,
        enable_webhook: bool = True,
        enable_usb: bool = True,
        auto_process: bool = True,
        enable_polling: bool = True,
    ):
        self.webhook_port = webhook_port
        self.enable_webhook = enable_webhook
        self.enable_usb = enable_usb
        self.auto_process = auto_process
        self.enable_polling = enable_polling

        # State
        self._running = False
        self._process_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._pending_recordings: List[str] = []
        self._lock = threading.Lock()
        self._last_poll: Optional[datetime] = None
        self._poll_count: int = 0

        # Activity log for UI
        self.activity_log: List[dict] = []
        self.max_log = 100

        # Services (lazily initialized)
        self._webhook_server: Optional[PlaudWebhookServer] = None
        self._usb_watcher: Optional[PlaudUSBWatcher] = None

    def _log_activity(self, source: str, action: str, details: str = ""):
        """Add to activity log for UI/debugging."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "action": action,
            "details": details,
        }
        self.activity_log.append(entry)
        if len(self.activity_log) > self.max_log:
            self.activity_log = self.activity_log[-self.max_log :]
        logger.info(f"[{source}] {action}: {details}")

    # ------------------------------------------------------------------
    # Webhook integration
    # ------------------------------------------------------------------

    def _setup_webhook(self):
        """Start webhook server and register handlers."""
        self._webhook_server = PlaudWebhookServer(port=self.webhook_port)

        @self._webhook_server.on_transcribe_complete
        def on_transcription(event: PlaudEvent):
            recording_id = event.recording_id or event.file_id
            self._log_activity(
                "webhook", "transcription_complete", recording_id or "unknown"
            )
            if recording_id:
                self._queue_for_processing(recording_id, "webhook")

        @self._webhook_server.on_recording_uploaded
        def on_upload(event: PlaudEvent):
            recording_id = event.recording_id or event.file_id
            self._log_activity(
                "webhook", "recording_uploaded", recording_id or "unknown"
            )
            if recording_id:
                self._queue_for_processing(recording_id, "webhook")

        @self._webhook_server.on_workflow_complete
        def on_workflow_done(event: PlaudEvent):
            workflow_id = event.workflow_id
            recording_id = event.recording_id
            self._log_activity(
                "webhook",
                "workflow_completed",
                f"workflow={workflow_id or 'unknown'} recording={recording_id or 'unknown'}",
            )
            # Refresh workflow statuses to pull results into DB
            try:
                from app_v2.services.data_service import get_service

                svc = get_service()
                svc.refresh_plaud_workflow_statuses()
                logger.info("Refreshed workflow statuses after webhook notification")
            except Exception as e:
                logger.error(f"Failed to refresh workflow statuses: {e}")

        @self._webhook_server.on_workflow_failed
        def on_workflow_err(event: PlaudEvent):
            workflow_id = event.workflow_id
            recording_id = event.recording_id
            self._log_activity(
                "webhook",
                "workflow_failed",
                f"workflow={workflow_id or 'unknown'} recording={recording_id or 'unknown'}",
            )

        self._webhook_server.start()
        self._log_activity("system", "webhook_started", f"port={self.webhook_port}")

    # ------------------------------------------------------------------
    # USB watcher integration
    # ------------------------------------------------------------------

    def _setup_usb(self):
        """Start USB watcher and register handlers."""
        self._usb_watcher = PlaudUSBWatcher()

        def on_device_connected(device: USBPlaudDevice):
            self._log_activity(
                "usb",
                "device_connected",
                f"{device.volume_name} ({device.device_type.value}): "
                f"{device.audio_file_count} files, {device.total_audio_size_mb:.1f}MB",
            )
            if device.has_recordings:
                self._import_from_usb(device)

        def on_device_disconnected(path: str):
            self._log_activity("usb", "device_disconnected", path)

        self._usb_watcher.on_device_connected(on_device_connected)
        self._usb_watcher.on_device_disconnected(on_device_disconnected)

        # Initial scan
        devices = self._usb_watcher.scan_now()
        if devices:
            for d in devices:
                self._log_activity(
                    "usb",
                    "found_device",
                    f"{d.volume_name}: {d.audio_file_count} files",
                )

        self._usb_watcher.start()
        self._log_activity("system", "usb_watcher_started", "monitoring /Volumes/")

    def _import_from_usb(self, device: USBPlaudDevice):
        """Import audio files from a connected USB device."""
        import shutil

        audio_dir = Path("data/audio/usb")
        audio_dir.mkdir(parents=True, exist_ok=True)

        files = device.list_audio_files()
        imported = 0

        for audio_file in files:
            dest = audio_dir / f"{device.volume_name}_{audio_file.name}"
            if dest.exists():
                continue  # Skip already imported
            try:
                shutil.copy2(str(audio_file), str(dest))
                imported += 1
                self._log_activity(
                    "usb",
                    "file_imported",
                    f"{audio_file.name} → {dest.name}",
                )
            except Exception as e:
                self._log_activity("usb", "import_error", f"{audio_file.name}: {e}")

        if imported > 0:
            self._log_activity("usb", "import_complete", f"{imported} new files")
            # Trigger full pipeline to pick up new audio
            self._trigger_pipeline("full")

    # ------------------------------------------------------------------
    # Processing queue
    # ------------------------------------------------------------------

    def _queue_for_processing(self, recording_id: str, source: str):
        """Add a recording to the processing queue."""
        with self._lock:
            if recording_id not in self._pending_recordings:
                self._pending_recordings.append(recording_id)
                self._log_activity(
                    source,
                    "queued",
                    f"{recording_id} (queue size: {len(self._pending_recordings)})",
                )

    def _trigger_pipeline(self, stage: str = "full"):
        """Trigger the pipeline in a background thread."""

        def run():
            try:
                import subprocess

                self._log_activity("pipeline", "started", stage)
                result = subprocess.run(
                    [sys.executable, "scripts/chronos_pipeline.py", f"--{stage}"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    cwd=str(Path(__file__).parent.parent),
                )
                status = "success" if result.returncode == 0 else "failed"
                self._log_activity("pipeline", status, f"exit={result.returncode}")
            except Exception as e:
                self._log_activity("pipeline", "error", str(e))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _process_loop(self):
        """Background loop that processes queued recordings."""
        while self._running:
            recording_id = None
            with self._lock:
                if self._pending_recordings:
                    recording_id = self._pending_recordings.pop(0)

            if recording_id:
                self._process_single(recording_id)
            else:
                time.sleep(5)  # Poll every 5 seconds

    def _process_single(self, recording_id: str):
        """Process a single recording through the pipeline."""
        try:
            self._log_activity("pipeline", "processing", recording_id)

            # Ingest (fetch from Plaud if not already in DB)
            from src.database import SessionLocal
            from src.chronos.ingest_service import ChronosIngestService

            db = SessionLocal()
            try:
                ingest = ChronosIngestService(db)
                ingest.sync_recordings()
                db.commit()
            finally:
                db.close()

            # Process through Gemini
            from src.chronos.transcript_processor import TranscriptProcessor

            db = SessionLocal()
            try:
                processor = TranscriptProcessor(db)
                processor.process_recording_id(recording_id)
                db.commit()
                self._log_activity("pipeline", "processed", recording_id)
            except Exception as e:
                self._log_activity("pipeline", "process_error", f"{recording_id}: {e}")
            finally:
                db.close()

            # Index to Qdrant
            try:
                import subprocess

                subprocess.run(
                    [sys.executable, "scripts/index_unindexed.py"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(Path(__file__).parent.parent),
                )
                self._log_activity("pipeline", "indexed", recording_id)
            except Exception as e:
                self._log_activity("pipeline", "index_error", str(e))

        except Exception as e:
            self._log_activity("pipeline", "error", f"{recording_id}: {e}")

    # ------------------------------------------------------------------
    # Scheduled Plaud API polling
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Periodically poll the Plaud API for new recordings and run the full pipeline."""
        # Wait 60s on startup before first poll (let services stabilize)
        for _ in range(60):
            if not self._running:
                return
            time.sleep(1)

        while self._running:
            try:
                self._poll_plaud_api()
            except Exception as e:
                self._log_activity("poll", "error", str(e))
                logger.exception("Poll cycle failed")

            # Sleep in 10s increments so we can stop quickly
            for _ in range(self.POLL_INTERVAL // 10):
                if not self._running:
                    return
                time.sleep(10)

    def _poll_plaud_api(self):
        """Run a full sync cycle: ingest → process unprocessed → index unindexed."""
        self._poll_count += 1
        self._last_poll = datetime.now()
        self._log_activity(
            "poll",
            "started",
            f"cycle #{self._poll_count} (every {self.POLL_INTERVAL}s)",
        )

        import subprocess

        project_root = str(Path(__file__).parent.parent)
        python = sys.executable

        # Phase 1: Ingest — fetch new recordings from Plaud API
        try:
            result = subprocess.run(
                [python, "scripts/chronos_pipeline.py", "--ingest"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )
            if result.returncode == 0:
                self._log_activity(
                    "poll", "ingest_done", "checked Plaud API for new recordings"
                )
            else:
                self._log_activity("poll", "ingest_warn", f"exit={result.returncode}")
                logger.warning(
                    f"Ingest returned {result.returncode}: {result.stderr[-500:]}"
                )
        except Exception as e:
            self._log_activity("poll", "ingest_error", str(e))

        # Phase 2: Process — run Gemini on unprocessed recordings
        try:
            result = subprocess.run(
                [python, "scripts/chronos_pipeline.py", "--process"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=project_root,
            )
            if result.returncode == 0:
                self._log_activity(
                    "poll", "process_done", "processed pending recordings"
                )
            else:
                self._log_activity("poll", "process_warn", f"exit={result.returncode}")
        except Exception as e:
            self._log_activity("poll", "process_error", str(e))

        # Phase 3: Index — embed and store any unindexed events
        try:
            result = subprocess.run(
                [python, "scripts/chronos_pipeline.py", "--index"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )
            if result.returncode == 0:
                self._log_activity("poll", "index_done", "indexed pending events")
            else:
                self._log_activity("poll", "index_warn", f"exit={result.returncode}")
        except Exception as e:
            self._log_activity("poll", "index_error", str(e))

        self._log_activity("poll", "complete", f"cycle #{self._poll_count} finished")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start all enabled listeners and the processing loop."""
        if self._running:
            logger.warning("AutoSync already running")
            return

        self._running = True
        self._log_activity("system", "starting", "Chronos AutoSync")

        if self.enable_webhook:
            try:
                self._setup_webhook()
            except Exception as e:
                self._log_activity("system", "webhook_error", str(e))

        if self.enable_usb:
            try:
                self._setup_usb()
            except Exception as e:
                self._log_activity("system", "usb_error", str(e))

        if self.auto_process:
            self._process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self._process_thread.start()
            self._log_activity(
                "system", "processor_started", "background processing enabled"
            )

        if self.enable_polling:
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
            self._log_activity(
                "system",
                "polling_started",
                f"Plaud API poll every {self.POLL_INTERVAL}s ({self.POLL_INTERVAL // 60} min)",
            )

        self._log_activity("system", "ready", "all listeners active")

    def stop(self):
        """Stop all listeners and processing."""
        self._running = False

        if self._usb_watcher:
            self._usb_watcher.stop()

        if self._webhook_server:
            self._webhook_server.stop()

        self._log_activity("system", "stopped", "Chronos AutoSync")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> dict:
        """Get current status for UI."""
        return {
            "running": self._running,
            "webhook": {
                "enabled": self.enable_webhook,
                "running": (
                    self._webhook_server.is_running if self._webhook_server else False
                ),
                "url": (
                    self._webhook_server.webhook_url if self._webhook_server else None
                ),
                "events_received": (
                    len(self._webhook_server.event_log) if self._webhook_server else 0
                ),
            },
            "usb": {
                "enabled": self.enable_usb,
                "running": self._usb_watcher.is_running if self._usb_watcher else False,
                "devices": (
                    len(self._usb_watcher.connected_devices) if self._usb_watcher else 0
                ),
            },
            "polling": {
                "enabled": self.enable_polling,
                "interval_sec": self.POLL_INTERVAL,
                "poll_count": self._poll_count,
                "last_poll": self._last_poll.isoformat() if self._last_poll else None,
            },
            "queue": {
                "pending": len(self._pending_recordings),
            },
            "recent_activity": self.activity_log[-10:] if self.activity_log else [],
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Chronos Auto-Sync Service")
    parser.add_argument(
        "--webhook-port", type=int, default=8090, help="Webhook server port"
    )
    parser.add_argument(
        "--no-webhook", action="store_true", help="Disable webhook listener"
    )
    parser.add_argument("--no-usb", action="store_true", help="Disable USB watcher")
    parser.add_argument(
        "--no-auto-process", action="store_true", help="Disable auto-processing"
    )
    parser.add_argument(
        "--no-polling", action="store_true", help="Disable periodic Plaud API polling"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Plaud API poll interval in seconds (default: 1800 = 30min)",
    )
    args = parser.parse_args()

    # Allow CLI override of poll interval
    if args.poll_interval is not None:
        os.environ["CHRONOS_POLL_INTERVAL"] = str(args.poll_interval)

    syncer = ChronosAutoSync(
        webhook_port=args.webhook_port,
        enable_webhook=not args.no_webhook,
        enable_usb=not args.no_usb,
        auto_process=not args.no_auto_process,
        enable_polling=not args.no_polling,
    )

    poll_min = syncer.POLL_INTERVAL // 60

    print("=" * 60)
    print("  Chronos Auto-Sync Service")
    print("=" * 60)
    print(
        f"  Webhook: {'ENABLED (port {})'.format(args.webhook_port) if not args.no_webhook else 'disabled'}"
    )
    print(
        f"  USB:     {'ENABLED (polling /Volumes/)' if not args.no_usb else 'disabled'}"
    )
    print(f"  Auto:    {'ENABLED' if not args.no_auto_process else 'disabled'}")
    print(
        f"  Polling: {'ENABLED (every {} min)'.format(poll_min) if not args.no_polling else 'disabled'}"
    )
    print()
    if not args.no_webhook:
        print(f"  Webhook URL: http://localhost:{args.webhook_port}/webhook/plaud")
        print(f"  For external access: ngrok http {args.webhook_port}")
    if not args.no_polling:
        print(f"  API Poll:    every {poll_min} min → ingest → process → index")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    syncer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        syncer.stop()
        print("Done.")
