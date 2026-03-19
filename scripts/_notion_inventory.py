"""One-shot: inventory Notion pages and their import status."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notion_service import NotionService
from src.database.engine import SessionLocal, init_db
from src.chronos.notion_bridge import match_notion_to_chronos
from src.database.models import ChronosRecording

init_db()
svc = NotionService()

print("Fetching Notion pages...")
pages = svc.fetch_recordings(limit=1000)
print(f"Fetched {len(pages)} Notion pages")

session = SessionLocal()
try:
    match_map = match_notion_to_chronos(pages, session)
    matched = sum(1 for v in match_map.values() if v)
    unmatched = sum(1 for v in match_map.values() if v is None)
    print(f"Matched to Chronos: {matched}")
    print(f"Unmatched (importable): {unmatched}")

    notion_recs = (
        session.query(ChronosRecording)
        .filter(ChronosRecording.recording_id.like("notion:%"))
        .all()
    )
    completed = [r for r in notion_recs if r.processing_status == "completed"]
    failed = [r for r in notion_recs if r.processing_status == "failed"]
    pending = [r for r in notion_recs if r.processing_status == "pending"]
    print(
        f"Already imported: {len(completed)} completed, {len(failed)} failed, {len(pending)} pending"
    )

    unmatched_pages = [p for p in pages if match_map.get(p.page_id) is None]
    already_done = {
        r.recording_id.replace("notion:", "")
        for r in notion_recs
        if r.processing_status == "completed"
    }
    truly_new = [p for p in unmatched_pages if p.page_id not in already_done]

    print(f"\nTruly new (never imported): {len(truly_new)}")
    print("\n--- First 20 importable pages ---")
    for i, p in enumerate(truly_new[:20]):
        has_t = bool(p.transcript and p.transcript.strip())
        has_b = bool(p.body_text and p.body_text.strip())
        content = "transcript" if has_t else ("body" if has_b else "EMPTY")
        title = (p.title or "untitled")[:55]
        print(
            f"  {i+1:3d}. [{p.page_id[:8]}] {title:<55s} date={p.date or '??':>10s} content={content}"
        )
finally:
    session.close()
