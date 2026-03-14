"""Chronos audio ingestion service.

Handles the reliable download and local caching of Plaud recordings.
Implements the "local-first" philosophy: pull from cloud, store locally,
never rely on transient download URLs.
"""

import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import time as _time

import requests
from sqlalchemy.orm import Session

from src.config import get_settings
from src.plaud_client import PlaudClient
from src.database.chronos_repository import (
    upsert_chronos_recording,
    get_chronos_recording,
)
from src.models.chronos_schemas import ChronosRecording as ChronosRecordingSchema

logger = logging.getLogger(__name__)


class ChronosIngestService:
    """Service for ingesting Plaud recordings into Chronos pipeline.

    Responsibilities:
    - List new recordings from Plaud API
    - Download audio to local storage (hierarchical by date)
    - Compute checksums for integrity
    - Update SQLite with metadata
    - Detect duplicates
    """

    def __init__(self, db_session: Session, plaud_client: Optional[PlaudClient] = None):
        """Initialize ingestion service.

        Args:
            db_session: SQLAlchemy session for database operations
            plaud_client: PlaudClient instance (or creates one from config)
        """
        self.db = db_session
        self.plaud = plaud_client or PlaudClient()
        self.settings = get_settings()

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.settings.chronos_raw_audio_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.chronos_processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.chronos_graph_cache_dir).mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum for file integrity verification.

        Args:
            file_path: Path to audio file

        Returns:
            str: Hex digest of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in 8KB chunks to handle large files
            while chunk := f.read(8192):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def _build_local_path(
        self, recording_id: str, created_at: datetime, extension: str = "opus"
    ) -> str:
        """Build hierarchical path for local audio storage.

        Format: data/raw/YYYY/MM/DD/<recording_id>.<extension>

        Args:
            recording_id: Plaud recording ID
            created_at: Recording timestamp
            extension: Audio file extension (opus, mp3, etc.)

        Returns:
            str: Absolute path to local audio file
        """
        year = created_at.strftime("%Y")
        month = created_at.strftime("%m")
        day = created_at.strftime("%d")

        dir_path = Path(self.settings.chronos_raw_audio_dir) / year / month / day
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{recording_id}.{extension}"
        return str(dir_path / filename)

    def _download_audio_stream(
        self,
        download_url: str,
        local_path: str,
        chunk_size: int = 8192,
    ) -> bool:
        """Download audio file using chunked streaming.

        This prevents memory exhaustion for large (500MB+) files.

        Args:
            download_url: Pre-signed URL from Plaud API
            local_path: Destination path
            chunk_size: Bytes per chunk (default 8KB)

        Returns:
            bool: True if download succeeded
        """
        from app_v2.services.xray import xray_log
        try:
            # Stream download
            _t0 = _time.perf_counter()
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            _http_ms = (_time.perf_counter() - _t0) * 1000

            content_len = response.headers.get('Content-Length', '?')
            xray_log("ingest", "download", f"Downloading audio file ({content_len} bytes)",
                     duration_ms=round(_http_ms, 1))

            # Write chunks to disk
            bytes_written = 0
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            _total_ms = (_time.perf_counter() - _t0) * 1000
            mb = bytes_written / (1024 * 1024)
            xray_log("ingest", "download", f"Audio saved — {mb:.1f} MB",
                     duration_ms=round(_total_ms, 1))
            logger.info(f"Downloaded audio to {local_path}")
            return True

        except requests.RequestException as e:
            xray_log("ingest", "download", f"Audio download failed: {str(e)[:60]}", level="error")
            logger.error(f"Download failed: {e}")
            # Clean up partial file
            if os.path.exists(local_path):
                os.remove(local_path)
            return False

    def ingest_recording(
        self,
        recording_id: str,
        created_at: datetime,
        duration_ms: int,
        download_url: Optional[str] = None,
        device_id: Optional[str] = None,
        force_redownload: bool = False,
        title: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Ingest a single recording.

        Args:
            recording_id: Plaud recording ID
            created_at: Recording timestamp (UTC)
            duration_ms: Duration in milliseconds
            download_url: Pre-signed download URL from Plaud (optional)
            device_id: Device identifier (optional)
            force_redownload: Re-download even if exists

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        from app_v2.services.xray import xray_log
        # Check if already ingested
        existing = get_chronos_recording(self.db, recording_id)
        if existing and not force_redownload:
            # Even if already ingested, update the timestamp if it changed
            # (e.g. we previously used created_at but now use start_at)
            existing_ts = existing.created_at  # type: ignore[union-attr]
            if (
                existing_ts
                and created_at
                and abs((existing_ts - created_at).total_seconds()) > 3600
            ):
                logger.info(
                    f"Recording {recording_id[:16]}: updating timestamp "
                    f"{existing_ts.isoformat()[:10]} → {created_at.isoformat()[:10]}"
                )
                upsert_chronos_recording(
                    session=self.db,
                    recording_id=recording_id,
                    created_at=created_at,
                    duration_seconds=duration_ms // 1000,
                    local_audio_path=str(existing.local_audio_path or ""),
                    source=str(existing.source or "plaud"),
                    device_id=device_id
                    or (str(existing.device_id) if existing.device_id else None),
                    title=title,
                    checksum=str(existing.checksum) if existing.checksum else None,
                )
            else:
                logger.debug(f"Recording {recording_id} already ingested, skipping")
                xray_log("ingest", "skip", f"Already imported, skipping")
            return (True, None)

        # Plaud currently does not reliably provide downloadable audio URLs.
        # If we don't have a download URL, ingest metadata only and let the
        # transcript-based processor handle extraction + event creation.
        if not download_url:
            try:
                upsert_chronos_recording(
                    session=self.db,
                    recording_id=recording_id,
                    created_at=created_at,
                    duration_seconds=duration_ms // 1000,
                    local_audio_path="",  # transcript-only mode
                    source="plaud",
                    device_id=device_id,
                    title=title,
                    checksum=None,
                )
                logger.info(
                    f"Ingested recording {recording_id} (metadata only; transcript mode)"
                )
                dur_s = duration_ms // 1000
                xray_log("ingest", "store",
                         f"Saved recording ({dur_s}s) — {title[:40] if title else 'untitled'}")
                return (True, None)
            except Exception as e:
                logger.error(f"Database error: {e}")
                return (False, str(e))

        # Determine file extension from URL
        parsed_url = urlparse(download_url)
        extension = Path(parsed_url.path).suffix.lstrip(".") or "opus"

        # Build local path
        local_path = self._build_local_path(recording_id, created_at, extension)

        # Download audio
        if not self._download_audio_stream(download_url, local_path):
            return (False, "Download failed")

        # Verify file exists
        if not os.path.exists(local_path):
            return (False, f"File not found after download: {local_path}")

        # Compute checksum
        checksum = self._compute_checksum(local_path)

        # Store metadata in SQLite
        try:
            upsert_chronos_recording(
                session=self.db,
                recording_id=recording_id,
                title=None,
                created_at=created_at,
                duration_seconds=duration_ms // 1000,
                local_audio_path=local_path,
                source="plaud",
                device_id=device_id,
                checksum=checksum,
            )
            logger.info(f"Ingested recording {recording_id}")
            return (True, None)

        except Exception as e:
            logger.error(f"Database error: {e}")
            return (False, str(e))

    def ingest_recent_recordings(
        self,
        limit: int = 100,
        days_back: int = 7,
        fetch_all_pages: bool = False,
    ) -> Tuple[int, int]:
        """Ingest recent recordings from Plaud API.

        This is the main entry point for batch ingestion.

        Args:
            limit: Max recordings to fetch per batch
            days_back: Only fetch recordings from last N days
            fetch_all_pages: If True, paginate through all recordings (ignores limit for pagination)

        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        from app_v2.services.xray import xray_log
        logger.info(
            f"Fetching recordings (limit={limit}, days_back={days_back}, fetch_all_pages={fetch_all_pages})"
        )
        xray_log("ingest", "start",
                 f"Looking for new recordings from the last {days_back} days")

        # Fetch from Plaud API with optional pagination
        # Pre-check authentication so callers get actionable feedback
        if not self.plaud.oauth.is_authenticated:
            msg = "Plaud not authenticated. Run: python plaud_setup.py"
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            _t0 = _time.perf_counter()
            if fetch_all_pages:
                # Paginate through all recordings (using built-in pagination)
                logger.info("Fetching ALL recordings from Plaud...")
                xray_log("ingest", "plaud-api", "Asking Plaud for all recordings…")
                recordings = self.plaud.list_recordings(fetch_all=True)
                logger.info(f"Plaud returned {len(recordings)} total recordings")
                # Do NOT cap with limit when fetching all — we want everything
            else:
                # Single page fetch (most recent N only, max 20 per API)
                page_size = min(limit, 20) if limit else 20
                xray_log("ingest", "plaud-api", f"Asking Plaud for latest {page_size} recordings")
                recordings = self.plaud.list_recordings(page=1, page_size=page_size)
            _api_ms = (_time.perf_counter() - _t0) * 1000
            xray_log("ingest", "plaud-api",
                     f"Plaud found {len(recordings)} recordings",
                     duration_ms=round(_api_ms, 1))
        except Exception as e:
            xray_log("ingest", "plaud-api", f"Couldn't reach Plaud: {str(e)[:60]}", level="error")
            logger.error(f"Failed to fetch from Plaud API: {e}")
            raise

        success_count = 0
        failure_count = 0

        _batch_t0 = _time.perf_counter()
        for _rec_idx, rec_data in enumerate(recordings, 1):
            # Extract required fields from Plaud API list response
            recording_id = rec_data.get("id")
            # Prefer start_at (actual recording time) over created_at (cloud sync time).
            # The Plaud device batch-syncs recordings, so created_at is often days
            # after the recording was actually made.
            start_at_str = rec_data.get("start_at")
            created_at_str = rec_data.get("created_at")
            timestamp_str = start_at_str or created_at_str
            # Duration is in milliseconds from API
            duration_ms = rec_data.get("duration", 0)
            serial_number = rec_data.get("serial_number")
            title = rec_data.get("name")

            if not recording_id:
                logger.warning(f"Skipping record with no ID: {rec_data}")
                xray_log("ingest", "skip", f"Recording {_rec_idx}/{len(recordings)} has no ID", level="warn")
                failure_count += 1
                continue

            # Parse timestamp
            try:
                recording_time = datetime.fromisoformat(
                    (timestamp_str or "").replace("Z", "+00:00")
                )
            except Exception as e:
                logger.error(f"Invalid timestamp for {recording_id}: {e}")
                failure_count += 1
                continue

            if start_at_str and created_at_str and start_at_str != created_at_str:
                logger.info(
                    f"Recording {recording_id[:16]}: using start_at={start_at_str[:10]} "
                    f"(created_at was {created_at_str[:10]})"
                )

            # Store recording metadata (no audio download - we'll use transcripts)
            # Plaud doesn't provide audio downloads via API (presigned_url is null)
            # We process transcripts instead
            success, error = self.ingest_recording(
                recording_id=recording_id,
                created_at=recording_time,
                duration_ms=duration_ms,
                device_id=serial_number,
                title=title,
            )

            if success:
                success_count += 1
            else:
                failure_count += 1
                logger.error(f"Failed to ingest {recording_id}: {error}")

        _batch_ms = (_time.perf_counter() - _batch_t0) * 1000
        xray_log("ingest", "done",
                 f"Import complete — {success_count} new, {failure_count} failed out of {len(recordings)}",
                 duration_ms=round(_batch_ms, 1))
        logger.info(
            f"Ingestion complete: {success_count} success, {failure_count} failures"
        )
        return (success_count, failure_count)

    def verify_integrity(self, recording_id: str) -> bool:
        """Verify file integrity via checksum.

        Args:
            recording_id: Recording to verify

        Returns:
            bool: True if checksum matches
        """
        rec = get_chronos_recording(self.db, recording_id)
        if not rec or not rec.checksum:  # type: ignore[truthy-bool]
            return False

        audio_path = str(rec.local_audio_path)
        if not os.path.exists(audio_path):
            logger.error(f"Audio file missing: {audio_path}")
            return False

        actual_checksum = self._compute_checksum(audio_path)
        if actual_checksum != rec.checksum:
            logger.error(f"Checksum mismatch for {recording_id}")
            return False

        return True
