"""
Notion Sync Service - Direct API integration for two-way sync.

Replaces Zapier middleman with direct Notion API calls for:
1. Push: Recordings → Notion pages (summaries, themes, audio links)
2. Pull: Notion edits → SQL/Pinecone (corrected titles, tags)
3. Idempotent upserts using recording_id as stable key

Usage:
    sync = NotionSyncService()
    
    # Push a recording to Notion
    sync.push_recording(recording)
    
    # Pull edits from Notion back to SQL
    updated = sync.pull_edits(since_hours=24)
    
    # Full sync (push new, pull edits)
    stats = sync.full_sync()
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class NotionSyncConfig:
    """Configuration for Notion sync."""
    database_id: str
    api_key: str
    
    # Property mappings (Notion property name → Recording field)
    title_property: str = "Name"
    recording_id_property: str = "Recording ID"
    transcript_property: str = "Transcript"
    summary_property: str = "Summary"
    themes_property: str = "Themes"
    audio_url_property: str = "Audio URL"
    duration_property: str = "Duration (min)"
    created_property: str = "Created"
    status_property: str = "Status"
    source_property: str = "Source"
    pinecone_synced_property: str = "Pinecone Synced"
    
    # Rate limiting
    requests_per_second: float = 3.0  # Notion rate limit is ~3 req/s
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class SyncStats:
    """Statistics from a sync operation."""
    pushed: int = 0
    pulled: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"Sync: {self.pushed} pushed, {self.pulled} pulled, "
            f"{self.errors} errors, {self.skipped} skipped "
            f"({self.duration_seconds:.1f}s)"
        )


class NotionSyncService:
    """
    Two-way sync between PlaudBlender and Notion.
    
    Features:
    - Idempotent upserts using recording_id
    - Rate limiting with exponential backoff
    - Batch operations where possible
    - Error logging with context
    """
    
    def __init__(self, config: Optional[NotionSyncConfig] = None):
        """
        Initialize Notion sync service.
        
        Args:
            config: Optional config, otherwise loads from environment
        """
        try:
            from notion_client import Client
        except ImportError:
            raise ImportError("notion-client required: pip install notion-client")
        
        if config:
            self.config = config
        else:
            # Load from environment
            api_key = os.getenv("NOTION_API_KEY") or os.getenv("NOTION_TOKEN")
            database_id = os.getenv("NOTION_DATABASE_ID")
            
            if not api_key:
                raise ValueError("NOTION_API_KEY or NOTION_TOKEN required in .env")
            if not database_id:
                raise ValueError("NOTION_DATABASE_ID required in .env")
            
            self.config = NotionSyncConfig(
                database_id=database_id,
                api_key=api_key
            )
        
        self.client = Client(auth=self.config.api_key)
        self._last_request_time = 0.0
        
        # Cache for recording_id → page_id mapping
        self._page_cache: Dict[str, str] = {}
        
        logger.info(f"✅ NotionSyncService initialized (db: {self.config.database_id[:8]}...)")
    
    # ─────────────────────────────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────────────────────────────
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        min_interval = 1.0 / self.config.requests_per_second
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    def _retry_with_backoff(self, operation: str, func, *args, **kwargs) -> Any:
        """Execute function with retry and exponential backoff."""
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check for rate limit error
                if "rate" in error_str.lower() or "429" in error_str:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited on {operation}, waiting {delay}s...")
                    time.sleep(delay)
                    continue
                
                # Other errors: log and retry with backoff
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (attempt + 1)
                    logger.warning(f"Retry {attempt + 1}/{self.config.retry_attempts} for {operation}: {e}")
                    time.sleep(delay)
                else:
                    raise
        
        raise last_error
    
    # ─────────────────────────────────────────────────────────────────
    # Page Lookup
    # ─────────────────────────────────────────────────────────────────
    
    def find_page_by_recording_id(self, recording_id: str) -> Optional[str]:
        """
        Find Notion page ID by recording_id.
        
        Args:
            recording_id: Plaud recording ID
            
        Returns:
            Notion page ID if found, None otherwise
        """
        # Check cache first
        if recording_id in self._page_cache:
            return self._page_cache[recording_id]
        
        try:
            # Use client.request() to call the database query endpoint
            response = self._retry_with_backoff(
                "find_page",
                self.client.request,
                path=f"databases/{self.config.database_id}/query",
                method="POST",
                body={
                    "filter": {
                        "property": self.config.recording_id_property,
                        "rich_text": {"equals": recording_id}
                    },
                    "page_size": 1
                }
            )
            
            results = response.get("results", [])
            if results:
                page_id = results[0]["id"]
                self._page_cache[recording_id] = page_id
                return page_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding page for recording {recording_id}: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────
    # Push: Recording → Notion
    # ─────────────────────────────────────────────────────────────────
    
    def _build_properties(self, recording: Any) -> Dict[str, Any]:
        """Build Notion properties from Recording object."""
        props = {}
        
        # Title (Name)
        title = getattr(recording, 'title', None) or getattr(recording, 'filename', 'Untitled')
        props[self.config.title_property] = {
            "title": [{"text": {"content": str(title)[:2000]}}]
        }
        
        # Recording ID (unique key)
        props[self.config.recording_id_property] = {
            "rich_text": [{"text": {"content": str(recording.id)}}]
        }
        
        # Transcript (truncated to Notion limit)
        transcript = getattr(recording, 'transcript', '')
        if transcript:
            props[self.config.transcript_property] = {
                "rich_text": [{"text": {"content": transcript[:2000]}}]
            }
        
        # Summary from extra if present
        extra = getattr(recording, 'extra', {}) or {}
        summary = extra.get('summary', '')
        if summary:
            props[self.config.summary_property] = {
                "rich_text": [{"text": {"content": summary[:2000]}}]
            }
        
        # Themes
        themes = extra.get('themes', [])
        if themes:
            theme_text = ", ".join(themes[:20])
            props[self.config.themes_property] = {
                "rich_text": [{"text": {"content": theme_text[:2000]}}]
            }
        
        # Audio URL
        audio_url = getattr(recording, 'audio_url', None)
        if audio_url:
            props[self.config.audio_url_property] = {
                "url": audio_url
            }
        
        # Duration (convert ms to minutes)
        duration_ms = getattr(recording, 'duration_ms', None)
        if duration_ms:
            props[self.config.duration_property] = {
                "number": round(duration_ms / 60000, 1)
            }
        
        # Created date
        created_at = getattr(recording, 'created_at', None)
        if created_at:
            if isinstance(created_at, datetime):
                props[self.config.created_property] = {
                    "date": {"start": created_at.isoformat()}
                }
            elif isinstance(created_at, str):
                props[self.config.created_property] = {
                    "date": {"start": created_at}
                }
        
        # Source
        source = getattr(recording, 'source', 'plaud')
        props[self.config.source_property] = {
            "select": {"name": source}
        }
        
        # Status
        status = getattr(recording, 'status', 'raw')
        props[self.config.status_property] = {
            "select": {"name": status}
        }
        
        return props
    
    def push_recording(self, recording: Any) -> Tuple[bool, Optional[str]]:
        """
        Push a recording to Notion (create or update).
        
        Uses recording.id as idempotency key.
        
        Args:
            recording: Recording object (from SQL or dict-like)
            
        Returns:
            (success, page_id or error message)
        """
        try:
            recording_id = str(recording.id)
            props = self._build_properties(recording)
            
            # Check if page exists
            existing_page_id = self.find_page_by_recording_id(recording_id)
            
            if existing_page_id:
                # Update existing page
                self._retry_with_backoff(
                    "update_page",
                    self.client.pages.update,
                    page_id=existing_page_id,
                    properties=props
                )
                logger.info(f"Updated Notion page for recording {recording_id[:8]}...")
                return True, existing_page_id
            else:
                # Create new page
                response = self._retry_with_backoff(
                    "create_page",
                    self.client.pages.create,
                    parent={"database_id": self.config.database_id},
                    properties=props
                )
                page_id = response["id"]
                self._page_cache[recording_id] = page_id
                logger.info(f"Created Notion page {page_id[:8]}... for recording {recording_id[:8]}...")
                return True, page_id
                
        except Exception as e:
            logger.error(f"Error pushing recording {recording.id}: {e}")
            return False, str(e)
    
    def push_recordings(self, recordings: List[Any]) -> SyncStats:
        """
        Push multiple recordings to Notion.
        
        Args:
            recordings: List of Recording objects
            
        Returns:
            SyncStats with push results
        """
        stats = SyncStats()
        start_time = time.time()
        
        for recording in recordings:
            success, result = self.push_recording(recording)
            if success:
                stats.pushed += 1
            else:
                stats.errors += 1
                stats.error_messages.append(f"{recording.id}: {result}")
        
        stats.duration_seconds = time.time() - start_time
        logger.info(f"Push complete: {stats}")
        return stats
    
    # ─────────────────────────────────────────────────────────────────
    # Pull: Notion → Recording updates
    # ─────────────────────────────────────────────────────────────────
    
    def pull_edits(self, since_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Pull recently edited pages from Notion.
        
        Args:
            since_hours: Look back this many hours
            
        Returns:
            List of dicts with recording_id and updated fields
        """
        since = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
        updates = []
        
        try:
            # Query recently edited pages
            has_more = True
            start_cursor = None
            
            while has_more:
                query_params = {
                    "database_id": self.config.database_id,
                    "filter": {
                        "timestamp": "last_edited_time",
                        "last_edited_time": {"after": since}
                    },
                    "page_size": 100
                }
                if start_cursor:
                    query_params["start_cursor"] = start_cursor
                
                # Use client.request() to query database
                body = {
                    "filter": query_params.get("filter", {}),
                    "page_size": query_params.get("page_size", 100)
                }
                if start_cursor:
                    body["start_cursor"] = start_cursor
                    
                response = self._retry_with_backoff(
                    "pull_edits",
                    self.client.request,
                    path=f"databases/{self.config.database_id}/query",
                    method="POST",
                    body=body
                )
                
                for page in response.get("results", []):
                    update = self._extract_edits(page)
                    if update:
                        updates.append(update)
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
            
            logger.info(f"Pulled {len(updates)} edited pages from Notion")
            return updates
            
        except Exception as e:
            logger.error(f"Error pulling edits: {e}")
            return []
    
    def _extract_edits(self, page: Dict) -> Optional[Dict[str, Any]]:
        """Extract editable fields from Notion page."""
        props = page.get("properties", {})
        
        # Get recording ID
        rec_id_prop = props.get(self.config.recording_id_property, {})
        rich_text = rec_id_prop.get("rich_text", [])
        if not rich_text:
            return None
        
        recording_id = rich_text[0].get("plain_text", "")
        if not recording_id:
            return None
        
        update = {
            "recording_id": recording_id,
            "notion_page_id": page["id"],
            "last_edited": page.get("last_edited_time"),
        }
        
        # Extract editable fields
        
        # Title
        title_prop = props.get(self.config.title_property, {})
        title_text = title_prop.get("title", [])
        if title_text:
            update["title"] = title_text[0].get("plain_text", "")
        
        # Themes (might be edited by user)
        themes_prop = props.get(self.config.themes_property, {})
        themes_text = themes_prop.get("rich_text", [])
        if themes_text:
            themes_str = themes_text[0].get("plain_text", "")
            update["themes"] = [t.strip() for t in themes_str.split(",") if t.strip()]
        
        # Status
        status_prop = props.get(self.config.status_property, {})
        status_select = status_prop.get("select", {})
        if status_select:
            update["status"] = status_select.get("name", "")
        
        return update
    
    # ─────────────────────────────────────────────────────────────────
    # Full Sync
    # ─────────────────────────────────────────────────────────────────
    
    def full_sync(
        self,
        recordings: Optional[List[Any]] = None,
        pull_edits: bool = True,
        since_hours: int = 24
    ) -> SyncStats:
        """
        Perform full two-way sync.
        
        Args:
            recordings: Recordings to push (if None, skip push)
            pull_edits: Whether to pull edits from Notion
            since_hours: Hours to look back for edits
            
        Returns:
            Combined SyncStats
        """
        stats = SyncStats()
        start_time = time.time()
        
        # Push recordings
        if recordings:
            push_stats = self.push_recordings(recordings)
            stats.pushed = push_stats.pushed
            stats.errors = push_stats.errors
            stats.error_messages.extend(push_stats.error_messages)
        
        # Pull edits
        if pull_edits:
            edits = self.pull_edits(since_hours=since_hours)
            stats.pulled = len(edits)
            # Note: Caller should apply edits to SQL database
        
        stats.duration_seconds = time.time() - start_time
        logger.info(f"Full sync complete: {stats}")
        return stats
    
    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────
    
    def get_database_schema(self) -> Dict[str, Any]:
        """
        Get the Notion database schema (properties).
        
        Useful for verifying property mappings.
        """
        try:
            db = self._retry_with_backoff(
                "get_database",
                self.client.databases.retrieve,
                database_id=self.config.database_id
            )
            return db.get("properties", {})
        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return {}
    
    def verify_schema(self) -> Tuple[bool, List[str]]:
        """
        Verify that required properties exist in Notion database.
        
        Returns:
            (all_present, list of missing properties)
        """
        schema = self.get_database_schema()
        required = [
            self.config.title_property,
            self.config.recording_id_property,
        ]
        
        missing = [prop for prop in required if prop not in schema]
        
        if missing:
            logger.warning(f"Missing Notion properties: {missing}")
            return False, missing
        
        logger.info("✅ Notion schema verified")
        return True, []
    
    def count_pages(self) -> int:
        """Count total pages in the database."""
        try:
            count = 0
            has_more = True
            start_cursor = None
            
            while has_more:
                query_params = {
                    "database_id": self.config.database_id,
                    "page_size": 100
                }
                if start_cursor:
                    query_params["start_cursor"] = start_cursor
                
                # Use client.request() to query database
                body = {
                    "filter": query_params.get("filter", {}),
                    "page_size": query_params.get("page_size", 100)
                }
                if start_cursor:
                    body["start_cursor"] = start_cursor
                    
                response = self._retry_with_backoff(
                    "count_pages",
                    self.client.request,
                    path=f"databases/{self.config.database_id}/query",
                    method="POST",
                    body=body
                )
                
                count += len(response.get("results", []))
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
            
            return count
            
        except Exception as e:
            logger.error(f"Error counting pages: {e}")
            return -1


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

_sync_instance: Optional[NotionSyncService] = None

def get_notion_sync() -> NotionSyncService:
    """Get singleton NotionSyncService instance."""
    global _sync_instance
    if _sync_instance is None:
        _sync_instance = NotionSyncService()
    return _sync_instance


def push_recording_to_notion(recording: Any) -> Tuple[bool, Optional[str]]:
    """Push a single recording to Notion."""
    return get_notion_sync().push_recording(recording)


def pull_notion_edits(since_hours: int = 24) -> List[Dict[str, Any]]:
    """Pull recent edits from Notion."""
    return get_notion_sync().pull_edits(since_hours=since_hours)
