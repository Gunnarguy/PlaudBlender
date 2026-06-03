"""Dump the raw Notion database schema and one page's raw properties."""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings

settings = get_settings()
token = None

# Try OAuth token first
try:
    from src.notion_oauth import NotionOAuthClient

    oauth = NotionOAuthClient()
    if oauth.is_authenticated:
        token = oauth.access_token
        print("Using OAuth token")
except Exception:
    pass

if not token:
    token = settings.notion_token
    print("Using .env NOTION_TOKEN")

if not token:
    print("No Notion token available!")
    sys.exit(1)

db_id = settings.notion_database_id
print(f"Database ID: {db_id}")

# Use raw httpx instead of notion_client (which hangs on import sometimes)
import httpx

headers = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# Get database schema — try data_sources API first, then databases API
print("\n=== DATABASE SCHEMA ===")

# Try data_sources API (notion-client 3.0.0+)
resp = httpx.get(
    f"https://api.notion.com/v1/data_sources/{db_id}", headers=headers, timeout=15
)
if resp.status_code != 200:
    # Fallback to databases API
    resp = httpx.get(
        f"https://api.notion.com/v1/databases/{db_id}", headers=headers, timeout=15
    )

if resp.status_code == 200:
    db = resp.json()
    title_parts = db.get("title", [])
    title = "".join(t.get("plain_text", "") for t in title_parts)
    print(f"Title: {title}")
    props = db.get("properties", {})
    print(f"Properties ({len(props)}):")
    for name, info in sorted(props.items()):
        ptype = info.get("type", "?")
        print(f"  {name:30s}  type={ptype}")
else:
    print(f"Error: {resp.status_code} {resp.text[:200]}")
    sys.exit(1)

# Get one page to see its raw property values
print("\n=== FIRST PAGE RAW PROPERTIES ===")
resp2 = httpx.post(
    f"https://api.notion.com/v1/databases/{db_id}/query",
    headers=headers,
    json={
        "page_size": 1,
        "sorts": [{"timestamp": "created_time", "direction": "descending"}],
    },
    timeout=15,
)
if resp2.status_code == 200:
    results = resp2.json().get("results", [])
    if results:
        page = results[0]
        print(f"Page ID: {page.get('id')}")
        print(f"Created: {page.get('created_time')}")
        print(f"Edited:  {page.get('last_edited_time')}")
        props = page.get("properties", {})
        for name, val in sorted(props.items()):
            ptype = val.get("type", "?")
            # Compact display of the value
            val_str = json.dumps(val, default=str)
            if len(val_str) > 300:
                val_str = val_str[:300] + "..."
            print(f"\n  {name} ({ptype}):")
            print(f"    {val_str}")
else:
    print(f"Error: {resp2.status_code} {resp2.text[:200]}")
