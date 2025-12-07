"""
Plaud API Client - Interact with Plaud API to fetch recordings and transcripts.

Adds optional persistence to the local SQL database so recordings are stored
deterministically before being processed and indexed.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from .plaud_oauth import PlaudOAuthClient
from .config import get_settings
from .utils.logger import get_logger
from src.database.engine import init_db, SessionLocal
from src.database.repository import upsert_recording
from src.models.schemas import RecordingSchema

# Plaud API Configuration - matches endpoints from developer portal
PLAUD_API_BASE = "https://platform.plaud.ai/developer/api/open/third-party"

settings = get_settings()
logger = get_logger(__name__)


class PlaudClient:
    """
    Client for interacting with the Plaud API.
    
    Provides methods to:
    - List recordings
    - Get recording details
    - Fetch transcripts
    - Get user info
    """
    
    def __init__(self, oauth_client: PlaudOAuthClient = None):
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
        except Exception:
            return datetime.utcnow()
    
    def _get_headers(self) -> dict:
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.oauth.get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """
        Make authenticated API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response JSON data
        """
        url = f"{PLAUD_API_BASE}{endpoint}"
        headers = self._get_headers()
        
        response = requests.request(method, url, headers=headers, **kwargs)

        # Handle token refresh / invalid token
        if response.status_code in (401, 422):
            logger.info("Token rejected (%s), refreshing...", response.status_code)
            try:
                self.oauth.refresh_access_token()
            except Exception as exc:
                logger.error("Refresh failed: %s. Please re-authenticate via plaud_setup.py", exc)
                raise
            headers = self._get_headers()
            response = requests.request(method, url, headers=headers, **kwargs)
        
        response.raise_for_status()
        return response.json()
    
    def get_user(self) -> dict:
        """
        Get current authenticated user info.
        
        Returns:
            User profile data
        """
        return self._request("GET", "/users/current")

    def revoke_current_user(self) -> dict:
        """Revoke the currently authenticated Plaud user/session."""
        return self._request("POST", "/users/current/revoke")
    
    def get_recording_stats(self) -> dict:
        """Get aggregate statistics about all recordings."""
        recordings = self.list_recordings(limit=100)
        total_duration = sum(r.get('duration', 0) for r in recordings) / 1000  # ms to sec
        
        return {
            'total_count': len(recordings),
            'total_duration_seconds': total_duration,
            'total_duration_hours': total_duration / 3600,
            'avg_duration_minutes': (total_duration / len(recordings) / 60) if recordings else 0,
            'date_range': {
                'earliest': min((r.get('start_at') for r in recordings if r.get('start_at')), default=None),
                'latest': max((r.get('start_at') for r in recordings if r.get('start_at')), default=None)
            }
        }
    
    def list_recordings(
        self,
        limit: int = 50,
        offset: int = 0,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[dict]:
        """
        List all recordings/files.
        
        Args:
            limit: Maximum number of recordings to return
            offset: Pagination offset
            start_date: Filter recordings after this date
            end_date: Filter recordings before this date
            
        Returns:
            List of file/recording objects
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        result = self._request("GET", "/files/", params=params)
        # Handle different possible response structures
        if isinstance(result, list):
            return result
        return result.get("files", result.get("data", []))
    
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
        start_date = datetime.now() - timedelta(minutes=minutes_ago)
        return self.list_recordings(start_date=start_date, limit=100)
    
    def get_all_recordings_with_transcripts(self, limit: int = 100) -> List[dict]:
        """
        Fetch all recordings with their full transcripts.
        
        Args:
            limit: Maximum number of recordings
            
        Returns:
            List of recordings with transcript text included
        """
        recordings = self.list_recordings(limit=limit)
        
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
    
    def fetch_recordings_for_processing(self, since_minutes: int = None, status: str = None) -> List[dict]:
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
            recordings = self.list_recordings(limit=100)
        
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

    def fetch_and_store_recordings(self, limit: int = None, fetch_all: bool = True) -> List[str]:
        """Fetch recordings from Plaud and persist validated rows to SQLite.

        Args:
            limit: Max recordings to fetch (None = no limit when fetch_all=True)
            fetch_all: If True, paginate through all recordings
            
        Returns a list of recording IDs that were stored.
        """
        init_db()
        session = SessionLocal()
        stored: List[str] = []
        
        try:
            # Paginate through all recordings
            page_size = 50
            offset = 0
            total_fetched = 0
            
            while True:
                recordings = self.list_recordings(limit=page_size, offset=offset)
                if not recordings:
                    break
                    
                for rec in recordings:
                    rec_id = rec.get("id")
                    if not rec_id:
                        continue

                    try:
                        transcript_text = self.get_transcript_text(rec_id)
                        payload = RecordingSchema(
                            id=rec_id,
                            title=rec.get("title") or "Untitled Recording",
                            duration_ms=rec.get("duration") or rec.get("duration_ms") or 0,
                            created_at=self._parse_datetime(rec.get("created_at") or rec.get("start_at")),
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
                    total_fetched += 1
                    
                    # Check limit
                    if limit and total_fetched >= limit:
                        break
                
                # Pagination control - move to next page
                offset += len(recordings)
                
                # Stop conditions
                if limit and total_fetched >= limit:
                    break
                if not fetch_all:
                    break
                # Last page if we got fewer than page_size
                if len(recordings) < page_size:
                    break

            session.commit()
            logger.info(f"💾 Stored {len(stored)} recordings to SQLite (fetched {total_fetched} total)")
        finally:
            session.close()

        return stored

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
    
    print("\n📱 Fetching your recordings...")
    recordings = client.list_recordings(limit=5)
    
    print(f"\nFound {len(recordings)} recordings:")
    for rec in recordings:
        print(f"  - {rec.get('title', 'Untitled')} ({rec.get('id', 'unknown')[:8]}...)")
