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
        try:
            # Stream download
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            # Write chunks to disk
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Downloaded audio to {local_path}")
            return True

        except requests.RequestException as e:
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
        download_url: str,
        device_id: Optional[str] = None,
        force_redownload: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Ingest a single recording.

        Args:
            recording_id: Plaud recording ID
            created_at: Recording timestamp (UTC)
            duration_ms: Duration in milliseconds
            download_url: Pre-signed download URL from Plaud
            device_id: Device identifier (optional)
            force_redownload: Re-download even if exists

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        # Check if already ingested
        existing = get_chronos_recording(self.db, recording_id)
        if existing and not force_redownload:
            logger.info(f"Recording {recording_id} already ingested, skipping")
            return (True, None)

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
    ) -> Tuple[int, int]:
        """Ingest recent recordings from Plaud API.

        This is the main entry point for batch ingestion.

        Args:
            limit: Max recordings to fetch per batch
            days_back: Only fetch recordings from last N days

        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        logger.info(f"Fetching recordings (limit={limit}, days_back={days_back})")

        # Fetch from Plaud API (this calls your existing PlaudClient)
        # Note: Adjust this to match your actual PlaudClient API
        try:
            recordings = self.plaud.list_recordings(limit=limit)
        except Exception as e:
            logger.error(f"Failed to fetch from Plaud API: {e}")
            return (0, 0)

        success_count = 0
        failure_count = 0

        for rec_data in recordings:
            # Extract required fields (adjust based on actual Plaud API response)
            recording_id = rec_data.get("id")
            created_at_str = rec_data.get("created_at")
            duration_ms = rec_data.get("duration_ms", 0)
            download_url = rec_data.get("download_url") or rec_data.get("audio_url")
            device_id = rec_data.get("device_id")

            if not recording_id or not download_url:
                logger.warning(f"Skipping invalid record: {rec_data}")
                failure_count += 1
                continue

            # Parse timestamp
            try:
                created_at = datetime.fromisoformat(
                    created_at_str.replace("Z", "+00:00")
                )
            except Exception as e:
                logger.error(f"Invalid timestamp for {recording_id}: {e}")
                failure_count += 1
                continue

            # Ingest
            success, error = self.ingest_recording(
                recording_id=recording_id,
                created_at=created_at,
                duration_ms=duration_ms,
                download_url=download_url,
                device_id=device_id,
            )

            if success:
                success_count += 1
            else:
                failure_count += 1
                logger.error(f"Failed to ingest {recording_id}: {error}")

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
        if not rec or not rec.checksum:
            return False

        if not os.path.exists(rec.local_audio_path):
            logger.error(f"Audio file missing: {rec.local_audio_path}")
            return False

        actual_checksum = self._compute_checksum(rec.local_audio_path)
        if actual_checksum != rec.checksum:
            logger.error(f"Checksum mismatch for {recording_id}")
            return False

        return True
