"""Dump Notion database schema using raw httpx (no notion_client import)."""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

# Read token from .env
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("NOTION_TOKEN", "")
db_id = os.getenv("NOTION_DATABASE_ID", "")

print(f"Token: {token[:15]}...")
print(f"DB ID: {db_id}")

headers = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# Step 1: Search for all databases accessible to this token
print("\n=== SEARCHING FOR DATABASES ===")
resp = httpx.post(
    "https://api.notion.com/v1/search",
    headers=headers,
    json={"filter": {"value": "database", "property": "object"}, "page_size": 10},
    timeout=15,
)
print(f"Search status: {resp.status_code}")

if resp.status_code == 200:
    results = resp.json().get("results", [])
    print(f"Found {len(results)} databases")

    for db in results:
        did = db.get("id")
        title_parts = db.get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_parts)
        props = db.get("properties", {})

        print(f"\n  DB: {did}")
        print(f"  Title: {title}")
        print(f"  Properties ({len(props)}):")
        for name, info in sorted(props.items()):
            ptype = info.get("type", "?")
            print(f"    {name:35s}  type={ptype}")

        # If this is the configured DB or first one, fetch a page
        if did == db_id or len(results) == 1:
            print(f"\n  === FETCHING FIRST PAGE FROM: {title} ===")
            resp2 = httpx.post(
                f"https://api.notion.com/v1/databases/{did}/query",
                headers=headers,
                json={
                    "page_size": 1,
                    "sorts": [{"timestamp": "created_time", "direction": "descending"}],
                },
                timeout=15,
            )
            if resp2.status_code == 200:
                pages = resp2.json().get("results", [])
                if pages:
                    page = pages[0]
                    print(f"  Page ID: {page.get('id')}")
                    print(f"  Created: {page.get('created_time')}")
                    print(f"  Edited:  {page.get('last_edited_time')}")
                    pprops = page.get("properties", {})
                    for name, val in sorted(pprops.items()):
                        ptype = val.get("type", "?")
                        val_str = json.dumps(val, default=str)
                        if len(val_str) > 400:
                            val_str = val_str[:400] + "..."
                        print(f"\n    {name} ({ptype}):")
                        print(f"      {val_str}")
            else:
                print(f"  Query error: {resp2.status_code} {resp2.text[:200]}")
else:
    print(f"Error: {resp.text[:300]}")
    # Also try the specific DB ID directly
    print("\n=== TRYING DIRECT DB ACCESS ===")
    resp3 = httpx.get(
        f"https://api.notion.com/v1/databases/{db_id}",
        headers=headers,
        timeout=15,
    )
    print(f"Direct access: {resp3.status_code}")
    if resp3.status_code == 200:
        print(
            json.dumps(resp3.json().get("properties", {}), indent=2, default=str)[:2000]
        )
    else:
        print(resp3.text[:300])
