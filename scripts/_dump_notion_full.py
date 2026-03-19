"""Dump ALL properties of the imported Notion page in full."""

import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("NOTION_TOKEN", "")
headers = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2022-06-28",
}

page_id = "32749a74-d54f-81df-a2b1-f3c745f64c37"
resp = httpx.get(
    f"https://api.notion.com/v1/pages/{page_id}", headers=headers, timeout=15
)
page = resp.json()

print("=== TOP-LEVEL KEYS ===")
for k in page:
    if k != "properties":
        v = page[k]
        if isinstance(v, (dict, list)):
            print(f"  {k}: {json.dumps(v, default=str)[:200]}")
        else:
            print(f"  {k}: {v}")

print("\n=== ALL PROPERTIES (full) ===")
props = page.get("properties", {})
for name, val in sorted(props.items()):
    ptype = val.get("type", "?")
    # Extract plain text for rich_text fields
    if ptype == "rich_text":
        texts = val.get("rich_text", [])
        content = "".join(t.get("plain_text", "") for t in texts)
        print(f"\n--- {name} (rich_text, {len(content)} chars) ---")
        # Print first 2000 chars of content
        print(content[:2000])
        if len(content) > 2000:
            print(f"  ... +{len(content)-2000} more chars ...")
    elif ptype == "title":
        titles = val.get("title", [])
        content = "".join(t.get("plain_text", "") for t in titles)
        print(f"\n--- {name} (title) ---")
        print(content)
    else:
        print(f"\n--- {name} ({ptype}) ---")
        print(json.dumps(val, default=str)[:500])

# Also get the page CONTENT (children blocks)
print("\n\n=== PAGE BODY (children blocks) ===")
resp2 = httpx.get(
    f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=5",
    headers=headers,
    timeout=15,
)
if resp2.status_code == 200:
    blocks = resp2.json().get("results", [])
    for i, block in enumerate(blocks):
        btype = block.get("type", "?")
        block_data = block.get(btype, {})
        if "rich_text" in block_data:
            text = "".join(t.get("plain_text", "") for t in block_data["rich_text"])
            print(f"\n  Block {i} ({btype}): {text[:300]}")
        else:
            print(
                f"\n  Block {i} ({btype}): {json.dumps(block_data, default=str)[:200]}"
            )
    print(
        f"\n  Total blocks shown: {len(blocks)}, has_more: {resp2.json().get('has_more')}"
    )
else:
    print(f"  Error: {resp2.status_code} {resp2.text[:200]}")
