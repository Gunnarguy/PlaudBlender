"""Check Notion page properties for the March 17 import."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notion_service import NotionService

svc = NotionService()
pages = svc.fetch_recordings(limit=100)

TARGET = "32749a74-d54f-81df-a2b1-f3c745f64c37"
for p in pages:
    if p.page_id == TARGET:
        print(f"Title: {p.title}")
        print(f"Created: {p.created_time}")
        print(f"Edited: {p.last_edited_time}")
        print(f"Date:  {p.date}")
        print(f"Duration: {p.duration}")
        print(f"Category: {p.category}")
        print(f"Tags: {p.tags}")
        print()
        print("=== ALL RAW PROPERTIES ===")
        for k, v in p.properties.items():
            print(f"  {k}: {repr(v)[:200]}")
        break
else:
    print("Page not found — listing all page IDs:")
    for p in pages:
        print(f"  {p.page_id}: {p.title}")
