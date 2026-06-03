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
    SELF_HEAL_DAYS = int(os.environ.get("CHRONOS_SELF_HEAL_DAYS", 45))
    SELF_HEAL_LIMIT = int(os.environ.get("CHRONOS_SELF_HEAL_LIMIT", 10))
    PROCESS_LIMIT = max(1, int(os.environ.get("CHRONOS_AUTOSYNC_PROCESS_LIMIT", 10)))
    INDEX_LIMIT = max(1, int(os.environ.get("CHRONOS_AUTOSYNC_INDEX_LIMIT", 10)))
    GRAPH_LIMIT = max(1, int(os.environ.get("CHRONOS_AUTOSYNC_GRAPH_LIMIT", 10)))
    MAX_LOAD_AVG = float(os.environ.get("CHRONOS_AUTOSYNC_MAX_LOAD_AVG", 3.5))
    MIN_AVAILABLE_MB = int(os.environ.get("CHRONOS_AUTOSYNC_MIN_AVAILABLE_MB", 700))
    MAX_SWAP_USED_MB = int(os.environ.get("CHRONOS_AUTOSYNC_MAX_SWAP_USED_MB", 512))
    DEFER_SECONDS = int(os.environ.get("CHRONOS_AUTOSYNC_DEFER_SECONDS", 90))
    ENABLE_NOTION_IMPORT = os.environ.get("CHRONOS_ENABLE_NOTION_IMPORT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    NOTION_IMPORT_BATCH_SIZE = max(
        0,
        int(os.environ.get("CHRONOS_NOTION_IMPORT_BATCH_SIZE", 25)),
    )

    def __init__(
        self,
        webhook_port: int = 8090,
        enable_webhook: bool = True,
        enable_usb: bool = True,
        auto_process: bool = True,
        enable_polling: bool = True,
        enable_notion_import: Optional[bool] = None,
        notion_import_batch_size: Optional[int] = None,
        process_limit: Optional[int] = None,
        index_limit: Optional[int] = None,
        graph_limit: Optional[int] = None,
    ):
        self.webhook_port = webhook_port
        self.enable_webhook = enable_webhook
        self.enable_usb = enable_usb
        self.auto_process = auto_process
        self.enable_polling = enable_polling
        self.enable_notion_import = (
            self.ENABLE_NOTION_IMPORT
            if enable_notion_import is None
            else enable_notion_import
        )
        self.notion_import_batch_size = max(
            0,
            self.NOTION_IMPORT_BATCH_SIZE
            if notion_import_batch_size is None
            else notion_import_batch_size,
        )
        self.process_limit = max(
            1,
            self.PROCESS_LIMIT if process_limit is None else process_limit,
        )
        self.index_limit = max(
            1,
            self.INDEX_LIMIT if index_limit is None else index_limit,
        )
        self.graph_limit = max(
            1,
            self.GRAPH_LIMIT if graph_limit is None else graph_limit,
        )

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

    @staticmethod
    def _meminfo_mb() -> dict[str, int]:
        values: dict[str, int] = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, raw_value = line.split(":", 1)
                    parts = raw_value.strip().split()
                    if parts:
                        values[key] = int(parts[0]) // 1024
        except Exception:
            return {}
        return values

    @staticmethod
    def _pipeline_already_running() -> bool:
        current_pid = os.getpid()
        proc_root = Path("/proc")
        for proc_dir in proc_root.iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if pid == current_pid:
                continue
            try:
                cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ")
            except Exception:
                continue
            if b"scripts/chronos_pipeline.py" in cmdline:
                return True
        return False

    @staticmethod
    def _swap_pressure_reason(
        *,
        available_mb: int,
        swap_used_mb: int,
        swap_limit_mb: int,
        min_available_mb: int,
    ) -> Optional[str]:
        if swap_used_mb <= swap_limit_mb:
            return None

        # Swap can stay warm after memory pressure has already passed. Only
        # treat high swap as an active block when RAM headroom is still tight.
        recovery_headroom_mb = max(256, min_available_mb // 3)
        if available_mb and available_mb > (min_available_mb + recovery_headroom_mb):
            return None

        return (
            f"swap used {swap_used_mb}MB > {swap_limit_mb}MB"
            + (
                f" with only {available_mb}MB RAM available"
                if available_mb
                else ""
            )
        )

    def _host_pressure_reason(self) -> Optional[str]:
        try:
            load_1m = os.getloadavg()[0]
        except OSError:
            load_1m = 0.0
        if load_1m > self.MAX_LOAD_AVG:
            return f"load {load_1m:.2f} > {self.MAX_LOAD_AVG:.2f}"

        meminfo = self._meminfo_mb()
        available_mb = meminfo.get("MemAvailable", 0)
        if available_mb and available_mb < self.MIN_AVAILABLE_MB:
            return f"available RAM {available_mb}MB < {self.MIN_AVAILABLE_MB}MB"

        swap_total_mb = meminfo.get("SwapTotal", 0)
        swap_free_mb = meminfo.get("SwapFree", 0)
        swap_used_mb = max(0, swap_total_mb - swap_free_mb)
        swap_reason = self._swap_pressure_reason(
            available_mb=available_mb,
            swap_used_mb=swap_used_mb,
            swap_limit_mb=self.MAX_SWAP_USED_MB,
            min_available_mb=self.MIN_AVAILABLE_MB,
        )
        if swap_reason:
            return swap_reason

        if self._pipeline_already_running():
            return "another Chronos pipeline is already running"

        return None

    def _defer_heavy_work_if_needed(self, label: str) -> bool:
        reason = self._host_pressure_reason()
        if not reason:
            return False
        self._log_activity("pipeline", "deferred", f"{label}: {reason}")
        return True

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
                from app_v2.services.data_service import get_data_service

                svc = get_data_service()
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
                if self._defer_heavy_work_if_needed(f"background {stage}"):
                    return
                self._log_activity("pipeline", "started", stage)
                timeout = 900 if stage == "full" else 600
                ok = self._run_pipeline_subprocess(
                    [sys.executable, "scripts/chronos_pipeline.py", f"--{stage}"],
                    timeout=timeout,
                    source="pipeline",
                    success_action="success",
                    success_details=stage,
                    failure_action="failed",
                    failure_prefix=stage,
                )
                if not ok:
                    logger.warning("Background pipeline stage failed: %s", stage)
            except Exception as e:
                self._log_activity("pipeline", "error", str(e))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    @staticmethod
    def _format_subprocess_tail(output: str, *, max_lines: int = 6) -> str:
        """Format the tail of subprocess output for activity logs."""
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return ""
        return " | ".join(lines[-max_lines:])[:500]

    def _run_pipeline_subprocess(
        self,
        args: List[str],
        *,
        timeout: int,
        source: str,
        success_action: str,
        success_details: str,
        failure_action: str,
        failure_prefix: str,
    ) -> bool:
        """Run a pipeline subprocess and log a concise result summary."""
        import subprocess

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent.parent),
        )
        if result.returncode == 0:
            self._log_activity(source, success_action, success_details)
            return True

        combined = "\n".join(
            part for part in [result.stdout or "", result.stderr or ""] if part
        )
        tail = self._format_subprocess_tail(combined)
        details = f"{failure_prefix} exit={result.returncode}"
        if tail:
            details = f"{details} — {tail}"
        self._log_activity(source, failure_action, details)
        return False

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
            if self._defer_heavy_work_if_needed(f"queued recording {recording_id}"):
                with self._lock:
                    if recording_id not in self._pending_recordings:
                        self._pending_recordings.append(recording_id)
                time.sleep(max(self.DEFER_SECONDS, 10))
                return

            self._log_activity("pipeline", "processing", recording_id)

            if not self._run_pipeline_subprocess(
                [
                    sys.executable,
                    "scripts/chronos_pipeline.py",
                    "--ingest",
                    "--recording-id",
                    recording_id,
                ],
                timeout=300,
                source="pipeline",
                success_action="ingest_done",
                success_details=recording_id,
                failure_action="ingest_warn",
                failure_prefix=f"{recording_id} ingest",
            ):
                return

            if not self._run_pipeline_subprocess(
                [
                    sys.executable,
                    "scripts/chronos_pipeline.py",
                    "--process",
                    "--recording-id",
                    recording_id,
                ],
                timeout=600,
                source="pipeline",
                success_action="processed",
                success_details=recording_id,
                failure_action="process_error",
                failure_prefix=f"{recording_id} process",
            ):
                return

            if not self._run_pipeline_subprocess(
                [
                    sys.executable,
                    "scripts/chronos_pipeline.py",
                    "--index",
                    "--recording-id",
                    recording_id,
                ],
                timeout=300,
                source="pipeline",
                success_action="indexed",
                success_details=recording_id,
                failure_action="index_error",
                failure_prefix=f"{recording_id} index",
            ):
                return

            self._run_pipeline_subprocess(
                [
                    sys.executable,
                    "scripts/chronos_pipeline.py",
                    "--graph",
                    "--recording-id",
                    recording_id,
                ],
                timeout=300,
                source="pipeline",
                success_action="graph_done",
                success_details=recording_id,
                failure_action="graph_error",
                failure_prefix=f"{recording_id} graph",
            )

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
            remaining = max(self.POLL_INTERVAL, 1)
            while remaining > 0:
                if not self._running:
                    return
                sleep_for = min(10, remaining)
                time.sleep(sleep_for)
                remaining -= sleep_for

    def _poll_plaud_api(self):
        """Run a full sync cycle: ingest → process unprocessed → index unindexed."""
        self._poll_count += 1
        self._last_poll = datetime.now()
        self._log_activity(
            "poll",
            "started",
            f"cycle #{self._poll_count} (every {self.POLL_INTERVAL}s)",
        )

        python = sys.executable

        # Phase 1: Ingest — fetch new recordings from Plaud API
        try:
            ok = self._run_pipeline_subprocess(
                [python, "scripts/chronos_pipeline.py", "--ingest"],
                timeout=300,
                source="poll",
                success_action="ingest_done",
                success_details="checked Plaud API for new recordings",
                failure_action="ingest_warn",
                failure_prefix="ingest",
            )
            if not ok:
                logger.warning("Auto-sync poll ingest stage failed")
        except Exception as e:
            self._log_activity("poll", "ingest_error", str(e))

        if self._defer_heavy_work_if_needed("poll post-ingest stages"):
            self._log_activity(
                "poll",
                "post_ingest_deferred",
                "refresh/repair/process/index/graph/Notion import will retry on the next poll",
            )
            return

        # Phase 2: Refresh Plaud workflow statuses — heals missed webhook notifications.
        try:
            ok = self._run_pipeline_subprocess(
                [python, "scripts/chronos_pipeline.py", "--refresh-workflows"],
                timeout=300,
                source="poll",
                success_action="workflow_refresh_done",
                success_details="refreshed Plaud workflow statuses",
                failure_action="workflow_refresh_warn",
                failure_prefix="refresh-workflows",
            )
            if not ok:
                logger.warning("Auto-sync poll workflow refresh stage failed")
        except Exception as e:
            self._log_activity("poll", "workflow_refresh_error", str(e))

        # Phase 3: Self-heal — re-ingest missing recent recordings and re-queue retryable failures.
        try:
            ok = self._run_pipeline_subprocess(
                [
                    python,
                    "scripts/chronos_pipeline.py",
                    "--repair-recent",
                    "--days-back",
                    str(self.SELF_HEAL_DAYS),
                    "--limit",
                    str(self.SELF_HEAL_LIMIT),
                ],
                timeout=300,
                source="poll",
                success_action="repair_done",
                success_details=(
                    f"audited and repaired recent recordings (last {self.SELF_HEAL_DAYS} days)"
                ),
                failure_action="repair_warn",
                failure_prefix="repair-recent",
            )
            if not ok:
                logger.warning("Auto-sync poll repair stage failed")
        except Exception as e:
            self._log_activity("poll", "repair_error", str(e))

        # Phase 4: Process — run Gemini on unprocessed recordings
        try:
            ok = self._run_pipeline_subprocess(
                [
                    python,
                    "scripts/chronos_pipeline.py",
                    "--process",
                    "--limit",
                    str(self.process_limit),
                ],
                timeout=600,
                source="poll",
                success_action="process_done",
                success_details="processed pending recordings",
                failure_action="process_warn",
                failure_prefix="process",
            )
            if not ok:
                logger.warning("Auto-sync poll process stage failed")
        except Exception as e:
            self._log_activity("poll", "process_error", str(e))

        # Phase 5: Index — embed and store any unindexed events
        try:
            ok = self._run_pipeline_subprocess(
                [
                    python,
                    "scripts/chronos_pipeline.py",
                    "--index",
                    "--limit",
                    str(self.index_limit),
                ],
                timeout=300,
                source="poll",
                success_action="index_done",
                success_details="indexed pending events",
                failure_action="index_warn",
                failure_prefix="index",
            )
            if not ok:
                logger.warning("Auto-sync poll index stage failed")
        except Exception as e:
            self._log_activity("poll", "index_error", str(e))

        # Phase 6: Graph — keep the knowledge graph current for UI + MCP
        try:
            ok = self._run_pipeline_subprocess(
                [
                    python,
                    "scripts/chronos_pipeline.py",
                    "--graph",
                    "--limit",
                    str(self.graph_limit),
                ],
                timeout=600,
                source="poll",
                success_action="graph_done",
                success_details="refreshed graph cache",
                failure_action="graph_warn",
                failure_prefix="graph",
            )
            if not ok:
                logger.warning("Auto-sync poll graph stage failed")
        except Exception as e:
            self._log_activity("poll", "graph_error", str(e))

        # Phase 7: Notion — gradually import unmatched Notion recordings after core Plaud work.
        try:
            self._poll_notion_import_batch()
        except Exception as e:
            self._log_activity("poll", "notion_import_error", str(e))
            logger.warning("Auto-sync poll Notion import failed: %s", e)

        self._log_activity("poll", "complete", f"cycle #{self._poll_count} finished")

    def _poll_notion_import_batch(self):
        """Import a safe batch of unmatched Notion recordings when configured."""
        if not self.enable_notion_import or self.notion_import_batch_size <= 0:
            return

        current_settings = get_settings()
        if not (getattr(current_settings, "notion_database_id", None) or "").strip():
            return

        from src.notion_service import get_notion_service

        notion_service = get_notion_service()
        notion_status = notion_service.check_connection(quick=True)
        if not getattr(notion_status, "connected", False):
            detail = getattr(notion_status, "error", "Notion not connected")
            self._log_activity("poll", "notion_import_skip", detail)
            return

        from src.chronos.notion_bridge import import_all_unmatched
        from src.database import SessionLocal

        with SessionLocal() as session:
            imported, failed, errors = import_all_unmatched(
                session,
                process=True,
                index=True,
                batch_size=self.notion_import_batch_size,
            )

        if imported == 0 and failed == 0:
            return

        detail = (
            f"imported {imported} Notion recording(s)"
            f" in batches of up to {self.notion_import_batch_size}"
        )
        if failed:
            suffix = f" ({failed} failed)"
            if errors:
                suffix += f" — {errors[0][:120]}"
            detail += suffix
        self._log_activity("poll", "notion_import_done", detail)

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
