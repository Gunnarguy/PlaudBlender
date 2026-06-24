"""Chronos audio ingestion service.

Handles the reliable download and local caching of Plaud recordings.
Implements the "local-first" philosophy: pull from cloud, store locally,
never rely on transient download URLs.
"""

import os
import hashlib
import logging
from datetime import datetime, timedelta, timezone
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

logger = logging.getLogger(__name__)
_RECENT_WINDOW_MAX_PAGES = 12


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
        self.last_batch_warnings: list[str] = []
        self.last_batch_partial_success: bool = False

        # Ensure directories exist
        self._ensure_directories()

    def _reset_batch_state(self) -> None:
        self.last_batch_warnings = []
        self.last_batch_partial_success = False

    def _remember_batch_warning(
        self, message: str, *, partial_success: bool = False
    ) -> None:
        text = (message or "").strip()
        if not text:
            return
        if text not in self.last_batch_warnings:
            self.last_batch_warnings.append(text)
        if partial_success:
            self.last_batch_partial_success = True

    def _ensure_directories(self) -> None:
        """Create data directories if they don't exist."""
        Path(self.settings.chronos_raw_audio_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.chronos_processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.settings.chronos_graph_cache_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _comparison_timestamp(value: Optional[str]) -> Optional[datetime]:
        """Parse Plaud timestamps into naive UTC datetimes for comparisons."""
        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _fetch_recent_recordings_window(
        self,
        *,
        days_back: int,
        page_size: int = 20,
        max_pages: int = _RECENT_WINDOW_MAX_PAGES,
    ) -> Tuple[List[dict], int, str]:
        """Fetch Plaud pages until the recent time window is fully covered.

        Plaud returns at most 20 recordings per page. When callers ask for a
        paginated ingest pass, we only keep pulling pages while recordings still
        fall inside the requested recent window. Plaud's list ordering is not
        reliable enough to stop at the first older recording, so we continue
        across a bounded number of pages and stop early only when the API starts
        repeating a page or the feed actually ends.
        """
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        recordings: List[dict] = []
        page = 1
        pages_fetched = 0
        stop_reason = "no Plaud recordings returned"
        seen_ids: set[str] = set()
        seen_page_signatures: set[tuple[str, ...]] = set()

        while True:
            if page > max_pages:
                stop_reason = (
                    f"reached Plaud page budget ({max_pages}) before proving coverage"
                )
                logger.warning(stop_reason)
                break

            page_records = self.plaud.list_recordings(page=page, page_size=page_size)
            if not page_records:
                stop_reason = f"page {page} returned no recordings"
                break

            pages_fetched += 1
            page_signature = tuple(
                str(rec.get("id")) for rec in page_records if rec.get("id")
            )
            if page_signature and page_signature in seen_page_signatures:
                stop_reason = f"page {page} repeated an earlier Plaud page"
                logger.warning(stop_reason)
                break
            if page_signature:
                seen_page_signatures.add(page_signature)

            recent_records: List[dict] = []
            older_count = 0
            duplicate_count = 0

            for rec_data in page_records:
                recording_id = str(rec_data.get("id") or "").strip()
                if recording_id:
                    if recording_id in seen_ids:
                        duplicate_count += 1
                        continue
                    seen_ids.add(recording_id)

                record_time = self._comparison_timestamp(
                    rec_data.get("start_at") or rec_data.get("created_at")
                )
                if record_time and record_time < cutoff:
                    older_count += 1
                    continue
                recent_records.append(rec_data)

            recordings.extend(recent_records)
            logger.info(
                "Fetched Plaud page %s: kept %s recent recordings, skipped %s older and %s duplicate recordings for the last %s days",
                page,
                len(recent_records),
                older_count,
                duplicate_count,
                days_back,
            )

            if len(page_records) < page_size:
                stop_reason = f"page {page} was the final Plaud page"
                break

            page += 1

        return recordings, pages_fetched, stop_reason

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
        """Download audio file using chunked streaming."""
        from app_v2.services.xray import xray_log

        try:
            _t0 = _time.perf_counter()
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            _http_ms = (_time.perf_counter() - _t0) * 1000

            content_len = response.headers.get("Content-Length", "?")
            xray_log(
                "ingest",
                "download",
                f"Downloading the audio ({content_len} bytes)",
                duration_ms=round(_http_ms, 1),
            )

            bytes_written = 0
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

            _total_ms = (_time.perf_counter() - _t0) * 1000
            mb = bytes_written / (1024 * 1024)
            xray_log(
                "ingest",
                "download",
                f"Saved {mb:.1f} MB of audio to disk",
                duration_ms=round(_total_ms, 1),
            )
            logger.info(f"Downloaded audio to {local_path}")
            return True

        except requests.RequestException as e:
            xray_log(
                "ingest",
                "download",
                f"Download broke: {str(e)[:60]}",
                level="error",
            )
            logger.error(f"Download failed: {e}")
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
        """Ingest a single recording."""
        from app_v2.services.xray import xray_log

        existing = get_chronos_recording(self.db, recording_id)
        if existing and not force_redownload:
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
                xray_log("ingest", "skip", "Already have this one, skipping")
            return (True, None)

        if not download_url:
            try:
                upsert_chronos_recording(
                    session=self.db,
                    recording_id=recording_id,
                    created_at=created_at,
                    duration_seconds=duration_ms // 1000,
                    local_audio_path="",
                    source="plaud",
                    device_id=device_id,
                    title=title,
                    checksum=None,
                )
                logger.info(
                    f"Ingested recording {recording_id} (metadata only; transcript mode)"
                )
                dur_s = duration_ms // 1000
                xray_log(
                    "ingest",
                    "store",
                    f"New recording! '{title[:40] if title else 'untitled'}' ({dur_s} seconds long)",
                )
                return (True, None)
            except Exception as e:
                logger.error(f"Database error: {e}")
                return (False, str(e))

        parsed_url = urlparse(download_url)
        extension = Path(parsed_url.path).suffix.lstrip(".") or "opus"
        local_path = self._build_local_path(recording_id, created_at, extension)

        if not self._download_audio_stream(download_url, local_path):
            return (False, "Download failed")

        if not os.path.exists(local_path):
            return (False, f"File not found after download: {local_path}")

        checksum = self._compute_checksum(local_path)

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

    def ingest_recording_by_id(self, recording_id: str) -> Tuple[bool, Optional[str]]:
        """Fetch a specific Plaud recording by ID and ingest it directly."""
        recording_id = str(recording_id or "").strip()
        if not recording_id:
            return (False, "Missing Plaud recording ID")

        try:
            record = self.plaud.get_recording(recording_id)
        except Exception as e:
            logger.error("Failed to fetch Plaud recording %s: %s", recording_id, e)
            return (False, str(e))

        if (
            not isinstance(record, dict)
            or str(record.get("id") or "").strip() != recording_id
        ):
            return (False, f"Plaud returned invalid data for recording {recording_id}")

        created_at = (
            self._comparison_timestamp(
                record.get("start_at") or record.get("created_at")
            )
            or datetime.utcnow()
        )

        return self.ingest_recording(
            recording_id=recording_id,
            created_at=created_at,
            duration_ms=int(record.get("duration") or 0),
            download_url=(
                record.get("presigned_url")
                or record.get("download_url")
                or record.get("file_url")
            ),
            device_id=record.get("serial_number"),
            title=record.get("name"),
        )

    def ingest_recent_recordings(
        self,
        limit: int = 100,
        days_back: int = 7,
        fetch_all_pages: bool = False,
        all_history: bool = False,
        recording_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Ingest recent recordings from Plaud API.

        This is the main entry point for batch ingestion.

        Args:
            limit: Max recordings to fetch per batch
            days_back: Only fetch recordings from last N days
            fetch_all_pages: If True, paginate through Plaud pages until the
                recent time window is fully covered
            all_history: If True, fetch the entire Plaud recording history
                regardless of the recent time window
            recording_id: If provided, fetch and ingest this exact Plaud
                recording instead of relying on list pagination

        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        from app_v2.services.xray import xray_log

        self._reset_batch_state()
        logger.info(
            "Fetching recordings "
            f"(limit={limit}, days_back={days_back}, fetch_all_pages={fetch_all_pages}, all_history={all_history}, recording_id={recording_id})"
        )
        if recording_id:
            xray_log(
                "ingest",
                "start",
                f"Fetching Plaud recording {str(recording_id)[:12]} directly",
            )
        elif all_history:
            xray_log(
                "ingest",
                "start",
                "Checking your Plaud for your full recording history",
            )
        else:
            xray_log(
                "ingest",
                "start",
                f"Checking your Plaud for new recordings from the last {days_back} days",
            )

        # Fetch from Plaud API with optional pagination
        # Pre-check authentication so callers get actionable feedback
        if not self.plaud.oauth.is_authenticated:
            msg = "Plaud not authenticated. Run: python plaud_setup.py"
            logger.error(msg)
            raise RuntimeError(msg)

        success_count = 0
        failure_count = 0

        def _ingest_records(
            records: List[dict], *, start_index: int = 1
        ) -> Tuple[int, int]:
            page_success = 0
            page_failure = 0

            for _rec_idx, rec_data in enumerate(records, start_index):
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
                    xray_log(
                        "ingest",
                        "skip",
                        f"Recording #{_rec_idx} has no ID — weird, skipping it",
                        level="warn",
                    )
                    page_failure += 1
                    continue

                # Parse timestamp
                try:
                    recording_time = datetime.fromisoformat(
                        (timestamp_str or "").replace("Z", "+00:00")
                    )
                except Exception as e:
                    logger.error(f"Invalid timestamp for {recording_id}: {e}")
                    page_failure += 1
                    continue

                if start_at_str and created_at_str and start_at_str != created_at_str:
                    logger.info(
                        f"Recording {recording_id[:16]}: using start_at={start_at_str[:10]} "
                        f"(created_at was {created_at_str[:10]})"
                    )

                success, error = self.ingest_recording(
                    recording_id=recording_id,
                    created_at=recording_time,
                    duration_ms=duration_ms,
                    device_id=serial_number,
                    title=title,
                )

                if success:
                    page_success += 1
                else:
                    page_failure += 1
                    logger.error(f"Failed to ingest {recording_id}: {error}")

            return page_success, page_failure

        try:
            _t0 = _time.perf_counter()
            if recording_id:
                success, error = self.ingest_recording_by_id(str(recording_id))
                success_count += 1 if success else 0
                failure_count += 0 if success else 1
                recordings_count = 1 if success else 0
                if error:
                    logger.warning(
                        "Direct Plaud recording ingest failed for %s: %s",
                        recording_id,
                        error,
                    )
            elif all_history:
                logger.info("Fetching complete Plaud recording history")
                xray_log(
                    "ingest",
                    "plaud-api",
                    "Scanning your full Plaud history 20 at a time",
                )
                page = 1
                page_size = 20
                total_fetched = 0
                seen_page_signatures: set[tuple[str, ...]] = set()

                while True:
                    try:
                        page_records = self.plaud.list_recordings(
                            page=page,
                            page_size=page_size,
                        )
                    except Exception as e:
                        if total_fetched == 0:
                            raise

                        warning_message = f"Plaud rate-limited the deep backfill after {total_fetched} recordings — keeping what we already fetched"

                        logger.warning(
                            "Stopping full-history Plaud scan after %s pages (%s recordings fetched): %s",
                            page - 1,
                            total_fetched,
                            e,
                        )
                        xray_log(
                            "ingest",
                            "plaud-api",
                            warning_message,
                            level="warn",
                        )
                        self._remember_batch_warning(
                            warning_message,
                            partial_success=total_fetched > 0,
                        )
                        failure_count += 1
                        break

                    if not page_records:
                        break

                    page_signature = tuple(
                        str(rec.get("id")) for rec in page_records if rec.get("id")
                    )
                    if page_signature and page_signature in seen_page_signatures:
                        warning_message = "Plaud started repeating the same page, so the deep backfill stopped before looping forever"
                        logger.warning(
                            "Stopping full-history Plaud scan at page %s because the API repeated an earlier page (%s records)",
                            page,
                            len(page_records),
                        )
                        xray_log(
                            "ingest",
                            "plaud-api",
                            warning_message,
                            level="warn",
                        )
                        self._remember_batch_warning(
                            warning_message,
                            partial_success=total_fetched > 0,
                        )
                        failure_count += 1
                        break
                    if page_signature:
                        seen_page_signatures.add(page_signature)

                    total_fetched += len(page_records)
                    page_success, page_failure = _ingest_records(
                        page_records,
                        start_index=total_fetched - len(page_records) + 1,
                    )
                    success_count += page_success
                    failure_count += page_failure

                    logger.info(
                        "Plaud page %s processed: %s recordings (%s total fetched so far)",
                        page,
                        len(page_records),
                        total_fetched,
                    )

                    if len(page_records) < page_size:
                        break

                    page += 1

                recordings_count = total_fetched
            elif fetch_all_pages:
                logger.info(
                    "Fetching Plaud recordings across recent pages (days_back=%s)",
                    days_back,
                )
                xray_log(
                    "ingest",
                    "plaud-api",
                    f"Scanning Plaud 20 at a time until we leave the last {days_back} days",
                )
                recordings, pages_fetched, stop_reason = (
                    self._fetch_recent_recordings_window(
                        days_back=days_back,
                        page_size=20,
                    )
                )
                logger.info(
                    "Plaud returned %s recordings across %s page(s); stop reason: %s",
                    len(recordings),
                    pages_fetched,
                    stop_reason,
                )
                if (
                    "repeated an earlier Plaud page" in stop_reason
                    or "page budget" in stop_reason
                ):
                    warning_message = "Plaud repeated page 1 instead of giving later results, so recent days may be incomplete"
                    xray_log(
                        "ingest",
                        "plaud-api",
                        warning_message,
                        level="warn",
                    )
                    self._remember_batch_warning(
                        warning_message,
                        partial_success=len(recordings) > 0,
                    )
                    failure_count += 1
                page_success, page_failure = _ingest_records(recordings)
                success_count += page_success
                failure_count += page_failure
                recordings_count = len(recordings)
            else:
                # Single page fetch (most recent N only, max 20 per API)
                page_size = min(limit, 20) if limit else 20
                xray_log(
                    "ingest",
                    "plaud-api",
                    f"Asking Plaud for your newest {page_size} recordings",
                )
                recordings = self.plaud.list_recordings(page=1, page_size=page_size)
                page_success, page_failure = _ingest_records(recordings)
                success_count += page_success
                failure_count += page_failure
                recordings_count = len(recordings)
            _api_ms = (_time.perf_counter() - _t0) * 1000
            xray_log(
                "ingest",
                "plaud-api",
                (
                    f"Fetched Plaud recording {recording_id} directly"
                    if recording_id
                    else (
                        f"Plaud returned {recordings_count} recordings from your full history"
                        if all_history
                        else f"Plaud returned {recordings_count} recordings inside the sync window"
                    )
                ),
                duration_ms=round(_api_ms, 1),
            )
        except Exception as e:
            xray_log(
                "ingest",
                "plaud-api",
                f"Can't reach Plaud right now: {str(e)[:60]}",
                level="error",
            )
            logger.error(f"Failed to fetch from Plaud API: {e}")
            raise

        _batch_ms = (_time.perf_counter() - _t0) * 1000
        xray_log(
            "ingest",
            "done",
            f"All done — picked up {success_count} new recordings"
            + (f" ({failure_count} had issues)" if failure_count else ""),
            duration_ms=round(_batch_ms, 1),
        )
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
