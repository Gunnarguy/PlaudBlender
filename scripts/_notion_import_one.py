"""Import a single Notion page as a test run."""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notion_service import NotionService
from src.database.engine import SessionLocal, init_db
from src.chronos.notion_bridge import match_notion_to_chronos, import_notion_recording
from src.database.models import ChronosRecording

init_db()
svc = NotionService()

print("Fetching Notion pages...")
pages = svc.fetch_recordings(limit=1000)
print(f"Fetched {len(pages)} pages")

session = SessionLocal()
try:
    match_map = match_notion_to_chronos(pages, session)

    # Get unmatched pages (not already linked to a Chronos recording)
    unmatched_pages = [p for p in pages if match_map.get(p.page_id) is None]

    # Exclude already-imported
    notion_recs = (
        session.query(ChronosRecording)
        .filter(ChronosRecording.recording_id.like("notion:%"))
        .all()
    )
    already_done = {
        r.recording_id.replace("notion:", "")
        for r in notion_recs
        if r.processing_status == "completed"
    }
    truly_new = [p for p in unmatched_pages if p.page_id not in already_done]

    if not truly_new:
        print("No new pages to import!")
        sys.exit(0)

    # Pick the first one
    page = truly_new[0]
    has_transcript = bool(page.transcript and page.transcript.strip())
    has_body = bool(page.body_text and page.body_text.strip())

    print(f"\n{'='*60}")
    print(f"TEST IMPORT: Page 1 of {len(truly_new)}")
    print(f"{'='*60}")
    print(f"  Page ID:    {page.page_id}")
    print(f"  Title:      {page.title}")
    print(f"  Date:       {page.date or page.created_time}")
    print(f"  Transcript: {'Yes' if has_transcript else 'No'}")
    print(f"  Body text:  {'Yes' if has_body else 'No'}")
    if has_transcript:
        preview = page.transcript[:200].replace("\n", " ")
        print(f"  Preview:    {preview}...")
    elif has_body:
        preview = page.body_text[:200].replace("\n", " ")
        print(f"  Preview:    {preview}...")

    print(f"\nStarting import...")
    t0 = time.perf_counter()
    success, message = import_notion_recording(
        page_id=page.page_id,
        session=session,
        process=True,
        index=True,
        prefetched=page,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"RESULT: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*60}")
    print(f"  Message: {message}")
    print(f"  Time:    {elapsed:.1f}s")

    if success:
        # Show what was created
        rec = (
            session.query(ChronosRecording)
            .filter(ChronosRecording.recording_id == f"notion:{page.page_id}")
            .first()
        )
        if rec:
            print(f"  Recording: {rec.recording_id}")
            print(f"  Status:    {rec.processing_status}")
            print(f"  Title:     {rec.title}")

finally:
    session.close()
