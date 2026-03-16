"""Notion API integration service.

Fetches recordings, transcripts, and notes from Notion databases
that contain Plaud recording data. This is the missing piece —
Notion often has recordings that couldn't be pulled from the Plaud API.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class NotionRecording:
    """A recording/page pulled from Notion."""
    page_id: str
    title: str
    created_time: str  # ISO 8601
    last_edited_time: str
    url: str
    # Properties extracted from Notion page
    transcript: str = ""
    summary: str = ""
    date: str = ""  # YYYY-MM-DD
    duration: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = ""
    source: str = "notion"  # Always "notion" for these
    # Raw properties for display
    properties: Dict[str, Any] = field(default_factory=dict)
    # Content blocks (plain text extraction from page body)
    body_text: str = ""
    # Match info — does this recording exist in Chronos/Plaud?
    matched_recording_id: Optional[str] = None


@dataclass
class NotionSyncStatus:
    """Status of the Notion connection and data."""
    connected: bool = False
    database_found: bool = False
    database_title: str = ""
    total_pages: int = 0
    last_synced: Optional[str] = None
    error: str = ""
    schema: Dict[str, str] = field(default_factory=dict)  # property_name -> type


class NotionService:
    """Service for interacting with Notion API to pull recording data."""

    def __init__(self):
        self._client = None
        self._settings = get_settings()

    def _get_client(self):
        """Lazy-init the Notion client.

        Token priority:
        1. OAuth token from .notion_tokens.json (set via /auth/notion flow)
        2. Static NOTION_TOKEN from .env (internal integration fallback)
        """
        if self._client is not None:
            return self._client

        token = self._resolve_token()
        if not token:
            raise ConnectionError(
                "No Notion token available. Either connect via OAuth "
                "(Settings → Connect Notion) or set NOTION_TOKEN in .env"
            )

        from notion_client import Client
        self._client = Client(auth=token, timeout_ms=15_000)
        return self._client

    def _resolve_token(self) -> str | None:
        """Resolve the best available Notion token."""
        # Priority 1: OAuth token
        try:
            from src.notion_oauth import NotionOAuthClient
            oauth = NotionOAuthClient()
            if oauth.is_authenticated:
                return oauth.access_token
        except Exception:
            pass

        # Priority 2: Static .env token
        return self._settings.notion_token

    def invalidate_client(self):
        """Force re-creation of the Notion client (e.g. after OAuth)."""
        self._client = None

    def list_databases(self) -> list[dict]:
        """Search for all data sources accessible to this integration.

        notion-client 3.0.0 uses data_source objects (not databases).
        Returns list of {id, title, description, icon, last_edited, url, property_count}.
        """
        try:
            client = self._get_client()
            databases = []
            cursor = None

            while True:
                kwargs = {"filter": {"value": "data_source", "property": "object"}, "page_size": 100}
                if cursor:
                    kwargs["start_cursor"] = cursor

                result = client.search(**kwargs)

                for db in result.get("results", []):
                    title_parts = db.get("title", [])
                    title = "".join(t.get("plain_text", "") for t in title_parts) or "Untitled"
                    desc_parts = db.get("description", [])
                    desc = "".join(t.get("plain_text", "") for t in desc_parts)
                    icon_data = db.get("icon")
                    icon = ""
                    if icon_data:
                        if icon_data.get("type") == "emoji":
                            icon = icon_data.get("emoji", "")
                        else:
                            icon = "📁"

                    prop_count = len(db.get("properties", {}))

                    databases.append({
                        "id": db["id"],
                        "title": title,
                        "description": desc,
                        "icon": icon or "📁",
                        "last_edited": db.get("last_edited_time", ""),
                        "url": db.get("url", ""),
                        "property_count": prop_count,
                    })

                if not result.get("has_more"):
                    break
                cursor = result.get("next_cursor")

            return databases
        except Exception as e:
            logger.error(f"Error listing Notion data sources: {e}")
            return []

    def set_database_id(self, db_id: str) -> None:
        """Set the active database ID (runtime + .env persistence)."""
        self._settings.notion_database_id = db_id
        # Persist to .env
        self._save_env_key("NOTION_DATABASE_ID", db_id)
        # Invalidate the module-level singleton so the next call picks up the new ID
        global _notion_service
        _notion_service = None

    @staticmethod
    def _save_env_key(key: str, value: str) -> None:
        """Add or update a key in .env file."""
        import os
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
        if not found:
            lines.append(f"{key}={value}\n")
        with open(env_path, "w") as f:
            f.writelines(lines)

    def check_connection(self, quick: bool = False) -> NotionSyncStatus:
        """Test the Notion connection and return status.

        Args:
            quick: If True, skip the expensive page count (use when you'll
                   fetch recordings separately and can set total_pages later).
        """
        status = NotionSyncStatus()

        token = self._resolve_token()
        if not token:
            status.error = "No Notion token — set NOTION_TOKEN in .env"
            return status

        db_id = self._settings.notion_database_id
        if not db_id:
            status.error = "No data source configured — select one below or navigate to the Notion page"
            return status

        try:
            client = self._get_client()
            # Retrieve the data source (notion-client 3.0.0 uses data_sources API)
            db = client.data_sources.retrieve(data_source_id=db_id)
            status.connected = True
            status.database_found = True

            # Extract title
            title_parts = db.get("title", [])
            status.database_title = "".join(
                t.get("plain_text", "") for t in title_parts
            ) or "Untitled"

            # Extract schema (property names and types)
            props = db.get("properties", {})
            status.schema = {
                name: prop.get("type", "unknown")
                for name, prop in props.items()
            }

            # Count pages (skip when quick — caller will set from fetch results)
            if not quick:
                status.total_pages = self._count_pages(client, db_id)

        except Exception as e:
            status.connected = False
            status.error = str(e)
            logger.warning(f"Notion connection check failed: {e}")

        return status

    def _count_pages(self, client, ds_id: str) -> int:
        """Count total pages in the data source (paginated)."""
        count = 0
        cursor = None
        try:
            while True:
                kwargs = {"data_source_id": ds_id, "page_size": 100}
                if cursor:
                    kwargs["start_cursor"] = cursor

                result = client.data_sources.query(**kwargs)

                count += len(result.get("results", []))
                if not result.get("has_more"):
                    break
                cursor = result.get("next_cursor")
        except Exception as e:
            logger.warning(f"Error counting Notion pages: {e}")
        return count

    def fetch_recordings(self, limit: int = 100) -> List[NotionRecording]:
        """Fetch recordings from the Notion data source.

        Handles various Notion DB schemas — auto-detects property names.
        """
        token = self._resolve_token()
        db_id = self._settings.notion_database_id
        if not token or not db_id:
            return []

        try:
            client = self._get_client()
            ds = client.data_sources.retrieve(data_source_id=db_id)
            props_schema = ds.get("properties", {})

            # Auto-detect property mappings
            mapping = self._detect_property_mapping(props_schema)
            logger.info(f"Notion property mapping: {mapping}")

            # Fetch pages via data_sources.query (sorted by created_time descending)
            pages = []
            cursor = None
            while len(pages) < limit:
                kwargs = {
                    "data_source_id": db_id,
                    "page_size": min(100, limit - len(pages)),
                    "sorts": [{"timestamp": "created_time", "direction": "descending"}],
                }
                if cursor:
                    kwargs["start_cursor"] = cursor

                result = client.data_sources.query(**kwargs)

                for page in result.get("results", []):
                    rec = self._parse_page(page, mapping)
                    if rec:
                        pages.append(rec)

                if not result.get("has_more"):
                    break
                cursor = result.get("next_cursor")

            logger.info(f"Fetched {len(pages)} recordings from Notion")
            return pages

        except Exception as e:
            logger.error(f"Error fetching Notion recordings: {e}")
            return []

    def fetch_page_content(self, page_id: str) -> str:
        """Fetch the full text content of a Notion page (all blocks)."""
        try:
            client = self._get_client()
            blocks = []
            cursor = None

            while True:
                kwargs = {"block_id": page_id, "page_size": 100}
                if cursor:
                    kwargs["start_cursor"] = cursor

                response = client.blocks.children.list(**kwargs)
                blocks.extend(response.get("results", []))

                if not response.get("has_more"):
                    break
                cursor = response.get("next_cursor")

            # Extract plain text from blocks
            text_parts = []
            for block in blocks:
                text = self._extract_block_text(block)
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.warning(f"Error fetching page content for {page_id}: {e}")
            return ""

    def _detect_property_mapping(self, schema: dict) -> dict:
        """Auto-detect which Notion properties map to our fields.

        Looks for common property names across different Notion setups.
        """
        mapping = {
            "title": None,
            "transcript": None,
            "summary": None,
            "date": None,
            "duration": None,
            "tags": None,
            "category": None,
        }

        # Common name patterns for each field
        patterns = {
            "title": ["name", "title", "recording", "recording name"],
            "transcript": ["transcript", "transcription", "text", "content", "body"],
            "summary": ["summary", "ai summary", "notes", "description"],
            "date": ["date", "created", "recorded", "recording date", "when"],
            "duration": ["duration", "length", "time", "minutes"],
            "tags": ["tags", "labels", "keywords", "topics"],
            "category": ["category", "type", "kind", "classification"],
        }

        for prop_name, prop_info in schema.items():
            prop_lower = prop_name.lower().strip()
            prop_type = prop_info.get("type", "")

            for field_name, field_patterns in patterns.items():
                if mapping[field_name] is not None:
                    continue
                if prop_lower in field_patterns:
                    mapping[field_name] = prop_name
                    break

            # Title property is always the title type
            if prop_type == "title" and mapping["title"] is None:
                mapping["title"] = prop_name

        return mapping

    def _parse_page(self, page: dict, mapping: dict) -> Optional[NotionRecording]:
        """Parse a Notion page into a NotionRecording."""
        try:
            page_id = page.get("id", "")
            created = page.get("created_time", "")
            edited = page.get("last_edited_time", "")
            url = page.get("url", "")
            props = page.get("properties", {})

            # Extract title
            title = ""
            if mapping.get("title") and mapping["title"] in props:
                title_prop = props[mapping["title"]]
                title = self._extract_property_text(title_prop)

            # Extract other properties
            transcript = ""
            if mapping.get("transcript") and mapping["transcript"] in props:
                transcript = self._extract_property_text(props[mapping["transcript"]])

            summary = ""
            if mapping.get("summary") and mapping["summary"] in props:
                summary = self._extract_property_text(props[mapping["summary"]])

            date = ""
            if mapping.get("date") and mapping["date"] in props:
                date = self._extract_date(props[mapping["date"]])
            if not date and created:
                # Fall back to page created_time
                date = created[:10]

            duration = ""
            if mapping.get("duration") and mapping["duration"] in props:
                duration = self._extract_property_text(props[mapping["duration"]])

            tags = []
            if mapping.get("tags") and mapping["tags"] in props:
                tags = self._extract_multi_select(props[mapping["tags"]])

            category = ""
            if mapping.get("category") and mapping["category"] in props:
                category = self._extract_select(props[mapping["category"]])

            # Collect all raw properties for display
            raw_props = {}
            for prop_name, prop_val in props.items():
                raw_props[prop_name] = self._extract_property_text(prop_val)

            return NotionRecording(
                page_id=page_id,
                title=title or "Untitled",
                created_time=created,
                last_edited_time=edited,
                url=url,
                transcript=transcript,
                summary=summary,
                date=date,
                duration=duration,
                tags=tags,
                category=category,
                properties=raw_props,
            )

        except Exception as e:
            logger.warning(f"Error parsing Notion page: {e}")
            return None

    def _extract_property_text(self, prop: dict) -> str:
        """Extract plain text from any Notion property type."""
        prop_type = prop.get("type", "")

        if prop_type == "title":
            return "".join(
                t.get("plain_text", "") for t in prop.get("title", [])
            )
        elif prop_type == "rich_text":
            return "".join(
                t.get("plain_text", "") for t in prop.get("rich_text", [])
            )
        elif prop_type == "number":
            val = prop.get("number")
            return str(val) if val is not None else ""
        elif prop_type == "select":
            sel = prop.get("select")
            return sel.get("name", "") if sel else ""
        elif prop_type == "multi_select":
            return ", ".join(
                s.get("name", "") for s in prop.get("multi_select", [])
            )
        elif prop_type == "date":
            date_obj = prop.get("date")
            if date_obj:
                return date_obj.get("start", "")
            return ""
        elif prop_type == "checkbox":
            return "Yes" if prop.get("checkbox") else "No"
        elif prop_type == "url":
            return prop.get("url", "") or ""
        elif prop_type == "email":
            return prop.get("email", "") or ""
        elif prop_type == "phone_number":
            return prop.get("phone_number", "") or ""
        elif prop_type == "formula":
            formula = prop.get("formula", {})
            f_type = formula.get("type", "")
            return str(formula.get(f_type, ""))
        elif prop_type in ("created_time", "last_edited_time"):
            return prop.get(prop_type, "")
        elif prop_type in ("created_by", "last_edited_by"):
            user = prop.get(prop_type, {})
            return user.get("name", "") or user.get("id", "")
        elif prop_type == "status":
            status = prop.get("status")
            return status.get("name", "") if status else ""
        else:
            return ""

    def _extract_date(self, prop: dict) -> str:
        """Extract a date string from a Notion date property."""
        if prop.get("type") == "date":
            date_obj = prop.get("date")
            if date_obj:
                return date_obj.get("start", "")[:10]
        return ""

    def _extract_multi_select(self, prop: dict) -> List[str]:
        """Extract tags from a multi_select property."""
        if prop.get("type") == "multi_select":
            return [s.get("name", "") for s in prop.get("multi_select", [])]
        return []

    def _extract_select(self, prop: dict) -> str:
        """Extract a single select value."""
        if prop.get("type") == "select":
            sel = prop.get("select")
            return sel.get("name", "") if sel else ""
        return ""

    def _extract_block_text(self, block: dict) -> str:
        """Extract plain text from a single Notion block."""
        block_type = block.get("type", "")

        # Text-bearing block types
        text_types = [
            "paragraph", "heading_1", "heading_2", "heading_3",
            "bulleted_list_item", "numbered_list_item", "quote",
            "to_do", "toggle", "callout",
        ]

        if block_type in text_types:
            rich_text = block.get(block_type, {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich_text)

            # Add prefix for headings
            if block_type == "heading_1":
                return f"# {text}"
            elif block_type == "heading_2":
                return f"## {text}"
            elif block_type == "heading_3":
                return f"### {text}"
            elif block_type == "to_do":
                checked = block.get("to_do", {}).get("checked", False)
                return f"[{'x' if checked else ' '}] {text}"
            elif block_type == "bulleted_list_item":
                return f"• {text}"
            elif block_type == "numbered_list_item":
                return f"1. {text}"

            return text

        elif block_type == "code":
            code = block.get("code", {})
            text = "".join(t.get("plain_text", "") for t in code.get("rich_text", []))
            lang = code.get("language", "")
            return f"```{lang}\n{text}\n```"

        elif block_type == "divider":
            return "---"

        elif block_type == "table_row":
            cells = block.get("table_row", {}).get("cells", [])
            row = " | ".join(
                "".join(t.get("plain_text", "") for t in cell)
                for cell in cells
            )
            return f"| {row} |"

        return ""


# Module-level singleton
_notion_service: Optional[NotionService] = None


def get_notion_service() -> NotionService:
    """Get or create the NotionService singleton.

    Recreates the instance when the configured database ID changes
    (e.g. after the user picks a different DB in Settings).
    """
    global _notion_service
    if _notion_service is not None:
        current_db_id = get_settings().notion_database_id
        if current_db_id != _notion_service._settings.notion_database_id:
            _notion_service = None  # stale — recreate
    if _notion_service is None:
        _notion_service = NotionService()
    return _notion_service


def reset_notion_service() -> None:
    """Clear the cached singleton so the next call creates a fresh one."""
    global _notion_service
    _notion_service = None
