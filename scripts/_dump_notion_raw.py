"""Query the Plaud data source via raw HTTP to get schema + sample page."""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("NOTION_TOKEN", "")
db_id = os.getenv("NOTION_DATABASE_ID", "")

# Try multiple API versions — data_sources is a newer concept
for api_version in ["2022-06-28"]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": api_version,
        "Content-Type": "application/json",
    }

    # Try data_sources endpoint (Notion 3.0)
    print(f"\n=== Trying data_sources API (version {api_version}) ===")
    resp = httpx.get(
        f"https://api.notion.com/v1/data_sources/{db_id}",
        headers=headers,
        timeout=15,
    )
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        ds = resp.json()
        props = ds.get("properties", {})
        print(
            f"  Title: {''.join(t.get('plain_text', '') for t in ds.get('title', []))}"
        )
        print(f"  Properties ({len(props)}):")
        for name, info in sorted(props.items()):
            print(f"    {name:35s}  type={info.get('type', '?')}")

        # Query first page
        print("\n  === QUERYING FIRST PAGE ===")
        resp2 = httpx.post(
            f"https://api.notion.com/v1/data_sources/{db_id}/query",
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
                pprops = page.get("properties", {})
                for name, val in sorted(pprops.items()):
                    ptype = val.get("type", "?")
                    val_str = json.dumps(val, default=str)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "..."
                    print(f"\n    {name} ({ptype}):")
                    print(f"      {val_str}")
        else:
            print(f"  Query error: {resp2.status_code} {resp2.text[:200]}")
        break
    else:
        print(f"  Error: {resp.text[:200]}")

    # Also try pages API for the specific imported page
    print(f"\n=== Trying pages API for imported page ===")
    page_id = "32749a74-d54f-81df-a2b1-f3c745f64c37"
    resp3 = httpx.get(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=headers,
        timeout=15,
    )
    print(f"  Status: {resp3.status_code}")
    if resp3.status_code == 200:
        page = resp3.json()
        pprops = page.get("properties", {})
        print(f"  Created: {page.get('created_time')}")
        for name, val in sorted(pprops.items()):
            ptype = val.get("type", "?")
            val_str = json.dumps(val, default=str)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            print(f"\n    {name} ({ptype}):")
            print(f"      {val_str}")
    else:
        print(f"  Error: {resp3.text[:200]}")
