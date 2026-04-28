"""
Plaud API Client - Interact with Plaud API to fetch recordings and transcripts.

Adds optional persistence to the local SQL database so recordings are stored
deterministically before being processed and indexed.

Uses bulletproof authentication that auto-refreshes and validates tokens.
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from .plaud_oauth import PlaudOAuthClient, AuthenticationRequired
from .config import get_settings
from .utils.logger import get_logger
from src.database.engine import init_db, SessionLocal
from src.database.repository import upsert_recording
from src.models.schemas import RecordingSchema

# Plaud API Configuration - matches endpoints from developer portal
PLAUD_API_BASE = "https://platform.plaud.ai/developer/api/open/third-party"

settings = get_settings()
logger = get_logger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class PlaudClient:
    """
    Client for interacting with the Plaud API.

    Provides methods to:
    - List recordings
    - Get recording details
    - Fetch transcripts
    - Get user info

    Uses bulletproof authentication that automatically handles token refresh.
    """

    def __init__(self, oauth_client: PlaudOAuthClient | None = None):
        """
        Initialize the Plaud API client.

        Args:
            oauth_client: PlaudOAuthClient instance (auto-created if not provided)
        """
        self.oauth = oauth_client or PlaudOAuthClient()

        if not self.oauth.is_authenticated:
            logger.warning("Not authenticated. Call authenticate() or oauth.authenticate_interactive()")

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> datetime:
        """Best-effort parsing of Plaud timestamps."""
        if not value:
            return datetime.utcnow()
        try:
            # Plaud timestamps are usually ISO-8601
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception as e:
            logger.debug("Could not parse Plaud timestamp %r: %s", value, e)
            return datetime.utcnow()

    def _get_headers(self) -> dict:
        """
        Get authorization headers for API requests.

        Uses ensure_valid_token() for bulletproof auth.
        """
        return {
            "Authorization": f"Bearer {self.oauth.ensure_valid_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self, method: str, endpoint: str, retries: int = MAX_RETRIES, **kwargs
    ) -> dict:
        """
        Make authenticated API request with automatic retry.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            retries: Number of retries on failure
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON data
        """
        url = f"{PLAUD_API_BASE}{endpoint}"
        last_error = None

        for attempt in range(retries):
            try:
                headers = self._get_headers()  # Fresh token each attempt
                response = requests.request(
                    method, url, headers=headers, timeout=30, **kwargs
                )

                # Handle 401 by attempting a refresh before declaring auth lost.
                if response.status_code == 401:
                    logger.warning(
                        f"Got 401 on attempt {attempt + 1}, refreshing Plaud token and retrying..."
                    )
                    try:
                        self.oauth.refresh_access_token()
                    except Exception:
                        self.oauth._clear_tokens()
                        if attempt < retries - 1:
                            time.sleep(RETRY_DELAY)
                            continue
                        raise AuthenticationRequired(
                            "Authentication expired. Run: python plaud_setup.py"
                        )
                    if attempt < retries - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    raise AuthenticationRequired(
                        "Authentication expired. Run: python plaud_setup.py"
                    )

                # Handle rate limiting with exponential backoff.
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = RETRY_DELAY * (attempt + 1)
                    try:
                        if retry_after:
                            delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                    logger.warning(
                        f"Plaud rate limit hit on attempt {attempt + 1}; retrying in {delay:.1f}s"
                    )
                    last_error = "Plaud API rate limit exceeded"
                    if attempt < retries - 1:
                        time.sleep(delay)
                        continue

                # Handle server errors with retry
                if response.status_code >= 500:
                    logger.warning(
                        f"Server error {response.status_code} on attempt {attempt + 1}"
                    )
                    if attempt < retries - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                last_error = "Request timed out"
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
                last_error = "Connection error - check internet connection"
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue

            except AuthenticationRequired:
                raise

            except Exception as e:
                logger.warning(
                    "Plaud API error on attempt %d: %s (endpoint=%s)",
                    attempt + 1,
                    e,
                    endpoint,
                )
                last_error = str(e)
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                raise

        raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")

    def verify_authentication(self) -> bool:
        """
        Verify that authentication is working.

        Returns:
            True if authenticated and can make API calls
        """
        try:
            self.get_user()
            return True
        except Exception as e:
            logger.warning(f"Authentication verification failed: {e}")
            return False

    def get_user(self) -> dict:
        """
        Get current authenticated user info.

        Returns:
            User profile data
        """
        return self._request("GET", "/users/current")

    def get_user_info(self) -> dict:
        """Compatibility alias for older UI/helpers.

        Some parts of the UI call `get_user_info()`; the canonical method is
        `get_user()`.
        """

        return self.get_user()

    def revoke_current_user(self) -> dict:
        """Revoke the currently authenticated Plaud user/session."""
        return self._request("POST", "/users/current/revoke")

    def get_recording_stats(self) -> dict:
        """Get aggregate statistics about all recordings."""
        recordings = self.list_recordings(fetch_all=True)
        total_duration = sum(r.get('duration', 0) for r in recordings) / 1000  # ms to sec

        return {
            "total_count": len(recordings),
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
            "avg_duration_minutes": (
                (total_duration / len(recordings) / 60) if recordings else 0
            ),
            "date_range": {
                "earliest": min(
                    (r["start_at"] for r in recordings if r.get("start_at")),
                    default=None,
                ),
                "latest": max(
                    (r["start_at"] for r in recordings if r.get("start_at")),
                    default=None,
                ),
            },
        }

    def list_recordings(
        self, page: int = 1, page_size: int = 20, fetch_all: bool = False
    ) -> List[dict]:
        """
        List all recordings/files.

        Args:
            page: Page number (1-indexed, default: 1)
            page_size: Items per page (min 10, max 20, default: 20)
            fetch_all: If True, fetches all pages automatically

        Returns:
            List of file/recording objects
        """
        # Plaud API: min 10, max 20
        page_size = max(10, min(page_size, 20))

        if fetch_all:
            return self._fetch_all_recordings(page_size=page_size)

        params = {"page": page, "page_size": page_size}

        result = self._request("GET", "/files/", params=params)

        # Response format: {"type": "list", "data": [...], "page": 1, "page_size": 20}
        if isinstance(result, dict):
            return result.get("data", [])
        return result if isinstance(result, list) else []

    def _fetch_all_recordings(self, page_size: int = 20) -> List[dict]:
        """
        Fetch all recordings by paginating through all pages.

        Args:
            page_size: Items per page (max 20)

        Returns:
            Complete list of all recordings
        """
        all_recordings = []
        page = 1

        while True:
            params = {"page": page, "page_size": page_size}
            result = self._request("GET", "/files/", params=params)

            if isinstance(result, dict):
                recordings = result.get("data", [])
                current_page = result.get("page", page)
                current_page_size = result.get("page_size", page_size)
            else:
                recordings = result if isinstance(result, list) else []

            if not recordings:
                break

            all_recordings.extend(recordings)
            logger.info(
                f"Fetched page {page}: {len(recordings)} recordings (total: {len(all_recordings)})"
            )

            # If we got fewer than page_size, we've reached the end
            if len(recordings) < page_size:
                break

            page += 1

        return all_recordings

    def get_recording(self, recording_id: str) -> dict:
        """
        Get a specific recording/file by ID.

        Args:
            recording_id: File UUID

        Returns:
            File object with full details
        """
        return self._request("GET", f"/files/{recording_id}")

    def get_transcript(self, recording_id: str) -> dict:
        """
        Get the transcript for a recording/file.

        Note: Transcript may be included in the file details.

        Args:
            recording_id: File UUID

        Returns:
            Transcript data including full text and segments
        """
        # Get full file details which should include transcript
        file_data = self._request("GET", f"/files/{recording_id}")
        return file_data

    def get_transcript_text(self, recording_id: str) -> str:
        """
        Get just the transcript text for a recording.

        Args:
            recording_id: File UUID

        Returns:
            Full transcript text as string
        """
        import json as json_module

        file_data = self.get_transcript(recording_id)

        # Plaud returns data in source_list with transaction type containing transcript
        if isinstance(file_data, dict) and 'source_list' in file_data:
            for source in file_data['source_list']:
                if source.get('data_type') == 'transaction':
                    content = source.get('data_content', '')
                    try:
                        # Parse the JSON transcript segments
                        segments = json_module.loads(content)
                        # Join all content from segments
                        texts = [seg.get('content', '') for seg in segments if seg.get('content')]
                        return ' '.join(texts)
                    except:
                        return content

        # Fallback: try other common field names
        if isinstance(file_data, dict):
            for field in ['transcript', 'text', 'transcription', 'content']:
                if field in file_data:
                    value = file_data[field]
                    if isinstance(value, str):
                        return value
                    if isinstance(value, dict):
                        return value.get('text', value.get('content', str(value)))

        return str(file_data)

    def _extract_text(self, data) -> str:
        """Helper to extract text from nested structures."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for field in ['transcript', 'text', 'transcription', 'content']:
                if field in data:
                    return self._extract_text(data[field])
        return str(data)

    def get_summary(self, recording_id: str) -> dict:
        """
        Get the AI-generated summary for a recording.

        Args:
            recording_id: File UUID

        Returns:
            Summary data (may be included in file details)
        """
        return self._request("GET", f"/files/{recording_id}")

    def get_new_recordings(self, minutes_ago: int = 60) -> List[dict]:
        """
        Get recordings from the last N minutes.

        Args:
            minutes_ago: How many minutes back to look

        Returns:
            List of recent recordings
        """
        # Fetch all and filter client-side (Plaud API doesn't support date filtering)
        cutoff = datetime.now() - timedelta(minutes=minutes_ago)
        all_recordings = self.list_recordings(fetch_all=True)

        recent = []
        for rec in all_recordings:
            created = rec.get("created_at") or rec.get("start_at")
            if created:
                try:
                    rec_time = self._parse_datetime(created)
                    if rec_time >= cutoff:
                        recent.append(rec)
                except Exception as e:
                    logger.debug(
                        "Could not parse timestamp for recording filter: %s", e
                    )
                    pass
        return recent

    def get_all_recordings_with_transcripts(
        self, max_recordings: Optional[int] = None
    ) -> List[dict]:
        """
        Fetch all recordings with their full transcripts.

        Args:
            max_recordings: Maximum number of recordings (None = all)

        Returns:
            List of recordings with transcript text included
        """
        recordings = self.list_recordings(fetch_all=True)
        if max_recordings:
            recordings = recordings[:max_recordings]

        results = []
        for rec in recordings:
            rec_id = rec.get('id')
            if rec_id:
                try:
                    rec['transcript_text'] = self.get_transcript_text(rec_id)
                    results.append(rec)
                    logger.info(f"✅ Fetched transcript for: {rec.get('title', rec_id)[:50]}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch transcript for {rec_id}: {e}")
                    rec['transcript_text'] = None
                    results.append(rec)

        return results

    def fetch_recordings_for_processing(
        self, since_minutes: Optional[int] = None, status: Optional[str] = None
    ) -> List[dict]:
        """
        Fetch recordings ready for processing into the knowledge graph.

        This is the main method for the processing pipeline.

        Args:
            since_minutes: Only get recordings from last N minutes (None for all)
            status: Filter by processing status if tracked

        Returns:
            List of recordings with transcripts ready for processing
        """
        if since_minutes:
            recordings = self.get_new_recordings(since_minutes)
        else:
            recordings = self.list_recordings(fetch_all=True)

        processed = []
        for rec in recordings:
            rec_data = {
                'id': rec.get('id'),
                'title': rec.get('title', 'Untitled Recording'),
                'created_at': rec.get('created_at'),
                'duration': rec.get('duration'),
                'recording_type': rec.get('type', 'unknown'),
            }

            # Fetch transcript
            try:
                rec_data['transcript'] = self.get_transcript_text(rec['id'])
                if rec_data['transcript'] and len(rec_data['transcript'].strip()) > 50:
                    processed.append(rec_data)
                    logger.info(f"📝 Loaded: {rec_data['title'][:40]}...")
                else:
                    logger.warning(f"⏭️ Skipped (no/short transcript): {rec_data['title'][:40]}")
            except Exception as e:
                logger.error(f"❌ Error fetching transcript: {e}")

        logger.info(f"📊 Loaded {len(processed)} recordings with transcripts")
        return processed

    def fetch_and_store_recordings(
        self, max_recordings: Optional[int] = None
    ) -> List[str]:
        """Fetch recordings from Plaud and persist validated rows to SQLite.

        Args:
            max_recordings: Max recordings to fetch (None = all)

        Returns a list of recording IDs that were stored.
        """
        init_db()
        session = SessionLocal()
        stored: List[str] = []

        try:
            # Fetch all recordings using pagination (handled by list_recordings)
            recordings = self.list_recordings(fetch_all=True)
            if max_recordings:
                recordings = recordings[:max_recordings]

            if not recordings:
                logger.info("No recordings found")
                return stored

            for rec in recordings:
                rec_id = rec.get("id")
                if not rec_id:
                    continue

                try:
                    transcript_text = self.get_transcript_text(rec_id)
                    payload = RecordingSchema(
                        id=rec_id,
                        title=rec.get("title")
                        or rec.get("name")
                        or "Untitled Recording",
                        duration_ms=rec.get("duration") or rec.get("duration_ms") or 0,
                        created_at=self._parse_datetime(
                            rec.get("created_at") or rec.get("start_at")
                        ),
                        transcript=transcript_text,
                        language=rec.get("language"),
                        source="plaud",
                    )
                except Exception as exc:
                    logger.warning(f"⏭️ Skipping recording {rec_id}: {exc}")
                    continue

                # Capture Plaud-provided extras (e.g., summaries/outlines/keywords) for later use
                extra_payload = {"recording_type": rec.get("type"), "raw": rec}
                plaud_summary = self._extract_summary(rec)
                if plaud_summary:
                    extra_payload["plaud_summary"] = plaud_summary
                plaud_outline = rec.get("outline") or rec.get("summary_outline")
                if plaud_outline:
                    extra_payload["plaud_outline"] = plaud_outline
                plaud_keywords = rec.get("keywords") or rec.get("tags")
                if plaud_keywords:
                    extra_payload["plaud_keywords"] = plaud_keywords

                upsert_recording(
                    session,
                    payload=payload,
                    filename=rec.get("filename") or rec.get("name"),
                    status="raw",
                    extra=extra_payload,
                )
                stored.append(rec_id)

            session.commit()
            logger.info(f"💾 Stored {len(stored)} recordings to SQLite")
        finally:
            session.close()

        return stored

    # -------------------- File Upload --------------------
    def upload_file(
        self,
        file_path: str,
        name: Optional[str] = None,
        chunk_size: int = 5 * 1024 * 1024,  # 5MB chunks per Plaud spec
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Upload a local audio file to Plaud cloud via multipart chunked upload.

        Supports Opus and MP3 formats. Large files are uploaded in 5MB chunks.

        Args:
            file_path: Path to the audio file (opus/mp3)
            name: Optional display name (defaults to filename)
            chunk_size: Upload chunk size in bytes (default 5MB, Plaud spec)
            on_progress: Optional callback(bytes_sent, total_bytes) for progress

        Returns:
            Dict with file_id and metadata from Plaud API

        Raises:
            FileNotFoundError: If file_path doesn't exist
            ValueError: If file format not supported
        """
        import mimetypes
        import hashlib

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        display_name = name or os.path.splitext(filename)[0]

        # Validate format
        ext = os.path.splitext(filename)[1].lower()
        allowed_extensions = {".opus", ".mp3", ".wav", ".m4a", ".ogg"}
        if ext not in allowed_extensions:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {', '.join(sorted(allowed_extensions))}"
            )

        content_type = mimetypes.guess_type(filename)[0] or "audio/mpeg"

        # Compute SHA256 for integrity
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        file_hash = sha256.hexdigest()

        logger.info(
            f"📤 Uploading {filename} ({file_size / 1024 / 1024:.1f} MB, {ext}) "
            f"hash={file_hash[:16]}…"
        )

        # Single-request upload for small files, chunked for large
        if file_size <= chunk_size:
            return self._upload_single(file_path, display_name, content_type, file_hash)
        else:
            return self._upload_chunked(
                file_path,
                display_name,
                content_type,
                file_hash,
                file_size,
                chunk_size,
                on_progress,
            )

    def _upload_single(
        self, file_path: str, name: str, content_type: str, file_hash: str
    ) -> Dict[str, Any]:
        """Upload a small file in a single request."""
        headers = self._get_headers()
        # Remove Content-Type for multipart — requests sets it with boundary
        headers.pop("Content-Type", None)

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, content_type)}
            data = {"name": name, "checksum": file_hash}
            response = requests.post(
                f"{PLAUD_API_BASE}/files/upload",
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

        if response.status_code in (401, 422):
            self.oauth.refresh_access_token()
            headers = self._get_headers()
            headers.pop("Content-Type", None)
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, content_type)}
                data = {"name": name, "checksum": file_hash}
                response = requests.post(
                    f"{PLAUD_API_BASE}/files/upload",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )

        response.raise_for_status()
        result = response.json()
        file_id = result.get("id") or result.get("file_id")
        logger.info(f"✅ Uploaded {name} → file_id={file_id}")
        return result

    def _upload_chunked(
        self,
        file_path: str,
        name: str,
        content_type: str,
        file_hash: str,
        file_size: int,
        chunk_size: int,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Upload a large file in chunks (resumable multipart)."""
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        # Step 1: Initiate multipart upload
        init_resp = self._request(
            "POST",
            "/files/upload/init",
            json={
                "name": name,
                "content_type": content_type,
                "file_size": file_size,
                "total_chunks": total_chunks,
                "checksum": file_hash,
            },
        )
        upload_id = init_resp.get("upload_id") or init_resp.get("id")

        # Step 2: Upload each chunk
        bytes_sent = 0
        with open(file_path, "rb") as f:
            for chunk_num in range(total_chunks):
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break

                headers = self._get_headers()
                headers.pop("Content-Type", None)

                files = {
                    "chunk": (
                        f"chunk_{chunk_num}",
                        chunk_data,
                        "application/octet-stream",
                    )
                }
                data = {
                    "upload_id": upload_id,
                    "chunk_number": chunk_num,
                    "total_chunks": total_chunks,
                }

                resp = requests.post(
                    f"{PLAUD_API_BASE}/files/upload/chunk",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )
                resp.raise_for_status()

                bytes_sent += len(chunk_data)
                logger.info(
                    f"  📦 Chunk {chunk_num + 1}/{total_chunks} "
                    f"({bytes_sent / 1024 / 1024:.1f} MB)"
                )
                if on_progress:
                    on_progress(bytes_sent, file_size)

        # Step 3: Complete the upload
        complete_resp = self._request(
            "POST",
            "/files/upload/complete",
            json={"upload_id": upload_id, "checksum": file_hash},
        )

        file_id = complete_resp.get("id") or complete_resp.get("file_id")
        logger.info(f"✅ Chunked upload complete: {name} → file_id={file_id}")
        return complete_resp

    def get_upload_candidates(
        self, data_dir: Optional[str] = None, check_cloud: bool = False
    ) -> List[Dict[str, Any]]:
        """Find local audio files that exist only locally (USB-imported, not in cloud).

        Scans the USB import directory and compares against cloud recordings.

        Args:
            data_dir: Directory to scan (defaults to data/raw/usb_import/)
            check_cloud: When True, fetch cloud recordings to mark duplicates.
                Keep False for low-latency UI/status endpoints.

        Returns:
            List of dicts with path, name, size_mb, format, and cloud_status
        """
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(project_root, "data", "raw", "usb_import")

        if not os.path.isdir(data_dir):
            return []

        cloud_names = set()
        if check_cloud:
            try:
                cloud_recs = self.list_recordings(fetch_all=True)
                for rec in cloud_recs:
                    n = rec.get("name") or rec.get("title") or ""
                    cloud_names.add(n.strip().lower())
            except Exception as e:
                logger.debug("Could not fetch cloud recordings for dedup: %s", e)
                pass

        candidates = []
        audio_extensions = {".opus", ".mp3", ".wav", ".m4a", ".ogg"}
        for entry in os.scandir(data_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in audio_extensions:
                continue

            display_name = os.path.splitext(entry.name)[0]
            in_cloud = display_name.strip().lower() in cloud_names if check_cloud else False

            candidates.append(
                {
                    "path": entry.path,
                    "name": display_name,
                    "filename": entry.name,
                    "size_mb": round(entry.stat().st_size / 1024 / 1024, 2),
                    "format": ext.lstrip("."),
                    "in_cloud": in_cloud,
                }
            )

        candidates.sort(key=lambda x: x["name"])
        return candidates

    # -------------------- Helpers --------------------
    def _extract_summary(self, rec: dict) -> Optional[str]:
        """Best-effort extraction of Plaud-provided summary text from a recording payload."""
        if not isinstance(rec, dict):
            return None
        for key in [
            "summary",
            "ai_summary",
            "summary_text",
            "overall_summary",
            "semantic_summary",
        ]:
            val = rec.get(key)
            if isinstance(val, str) and len(val.strip()) > 10:
                return val.strip()
        # Some payloads may nest summaries inside extra fields
        extra = rec.get("extra") or {}
        if isinstance(extra, dict):
            for key in extra:
                if "summary" in key.lower():
                    val = extra.get(key)
                    if isinstance(val, str) and len(val.strip()) > 10:
                        return val.strip()
        return None


def get_client() -> PlaudClient:
    """
    Convenience function to get an authenticated Plaud client.

    Returns:
        Authenticated PlaudClient instance
    """
    return PlaudClient()


if __name__ == "__main__":
    # Quick test
    client = get_client()

    if not client.oauth.is_authenticated:
        print("Not authenticated. Running OAuth flow...")
        client.oauth.authenticate_interactive()

    print("\n📱 Fetching your recordings (page 1, 5 per page)...")
    recordings = client.list_recordings(page=1, page_size=10)

    print(f"\nFound {len(recordings)} recordings:")
    for rec in recordings:
        name = rec.get("name") or rec.get("title") or "Untitled"
        print(f"  - {name} ({rec.get('id', 'unknown')[:8]}...)")
