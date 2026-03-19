"""Query the actual parent database (25549a74-d54f-80aa-860e-df46277fc374) for schema + sample pages."""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("NOTION_TOKEN", "")
headers = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

# The ACTUAL parent database ID (from the page's parent field)
db_id = "25549a74-d54f-80aa-860e-df46277fc374"

print(f"=== Querying database {db_id} ===")
resp = httpx.get(
    f"https://api.notion.com/v1/databases/{db_id}", headers=headers, timeout=15
)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    db = resp.json()
    title = "".join(t.get("plain_text", "") for t in db.get("title", []))
    print(f"Title: {title}")
    props = db.get("properties", {})
    print(f"\nProperties ({len(props)}):")
    for name, info in sorted(props.items()):
        ptype = info.get("type", "?")
        detail = ""
        if ptype == "select":
            opts = info.get("select", {}).get("options", [])
            detail = f" options=[{', '.join(o.get('name','') for o in opts[:10])}]"
        elif ptype == "date":
            detail = " (DATE FIELD!)"
        elif ptype == "number":
            detail = f" format={info.get('number', {}).get('format', '?')}"
        print(f"  {name:40s}  type={ptype}{detail}")

    # Query first 3 pages to see populated properties
    print("\n=== SAMPLE PAGES ===")
    resp2 = httpx.post(
        f"https://api.notion.com/v1/databases/{db_id}/query",
        headers=headers,
        json={
            "page_size": 3,
            "sorts": [{"timestamp": "created_time", "direction": "descending"}],
        },
        timeout=15,
    )
    if resp2.status_code == 200:
        pages = resp2.json().get("results", [])
        for pi, page in enumerate(pages):
            print(f"\n--- Page {pi+1}: {page.get('id')} ---")
            print(f"  created: {page.get('created_time')}")
            pprops = page.get("properties", {})
            for name, val in sorted(pprops.items()):
                ptype = val.get("type", "?")
                if ptype == "title":
                    txt = "".join(t.get("plain_text", "") for t in val.get("title", []))
                    print(f"  {name}: {txt[:100]}")
                elif ptype == "rich_text":
                    txt = "".join(
                        t.get("plain_text", "") for t in val.get("rich_text", [])
                    )
                    print(f"  {name}: ({len(txt)} chars) {txt[:100]}...")
                elif ptype == "date":
                    print(f"  {name}: {json.dumps(val.get('date'), default=str)}")
                elif ptype == "select":
                    sel = val.get("select")
                    print(f"  {name}: {sel.get('name') if sel else 'None'}")
                elif ptype == "number":
                    print(f"  {name}: {val.get('number')}")
                else:
                    val_str = json.dumps(val, default=str)
                    print(f"  {name} ({ptype}): {val_str[:150]}")
    else:
        print(f"Query error: {resp2.status_code} {resp2.text[:200]}")
else:
    print(f"Error: {resp.text[:300]}")

# Also try the configured DB ID as a database (not data_source)
configured_id = os.getenv("NOTION_DATABASE_ID", "")
print(f"\n\n=== Also trying configured DB: {configured_id} ===")
resp3 = httpx.get(
    f"https://api.notion.com/v1/databases/{configured_id}", headers=headers, timeout=15
)
print(f"Status: {resp3.status_code}")
if resp3.status_code == 200:
    db2 = resp3.json()
    print(f"Title: {''.join(t.get('plain_text','') for t in db2.get('title', []))}")
elif resp3.status_code == 404:
    print("Not found as database — likely a data_source ID or in a different workspace")
else:
    print(resp3.text[:200])
