"""Inspect Plaud API responses for no-transcript recordings."""

import sys, json

sys.path.insert(0, ".")
from src.plaud_client import PlaudClient

plaud = PlaudClient()

for rid in ["db7d12e4dc2f0acc5a880deb84c909a4", "e2a6a89c9cc19ea468ec9a8c0602f93b"]:
    print(f"=== {rid[:16]} ===")
    try:
        details = plaud.get_recording(rid)
        print(f"  Title: {details.get('title')}")
        print(f"  Created: {details.get('create_time')}")
        print(f"  Duration: {details.get('duration')}")
        print(f"  File type: {details.get('file_type')}")
        source_list = details.get("source_list", [])
        print(f"  Sources ({len(source_list)}):")
        for s in source_list:
            dtype = s.get("data_type")
            status = s.get("status")
            content = s.get("data_content", "")
            url = s.get("presigned_url")
            has_url = "yes" if url else "no"
            print(
                f"    - type={dtype} status={status} content_len={len(content)} url={has_url}"
            )
            if content and len(content) > 0:
                preview = content[:300]
                print(f"      preview: {preview}")
            if url:
                print(f"      url: {url[:100]}...")
        # Print ALL keys to find anything useful
        print(f"  Response keys: {list(details.keys())}")
        # Check for any AI summary, notes, or other data fields
        for k in details.keys():
            v = details[k]
            if k not in ("source_list",) and v:
                print(f"  {k} = {str(v)[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
