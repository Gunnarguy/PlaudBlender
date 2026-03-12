"""Fix recording timestamps: use start_at (actual recording time) not created_at (sync time).

This script:
1. Fetches current recordings from Plaud API
2. Updates SQLite timestamps from created_at → start_at for recordings still in the API
3. Marks corrected recordings for re-processing
4. Ingests any new recordings

Run with: venv/bin/python scripts/fix_recording_dates.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.database import init_db, SessionLocal
from src.database.chronos_repository import (
    get_chronos_recording,
    upsert_chronos_recording,
    mark_chronos_recording_status,
)
from src.plaud_client import PlaudClient


def main():
    print("=" * 60)
    print("  FIX RECORDING DATES: start_at → created_at")
    print("=" * 60)

    init_db()
    session = SessionLocal()
    client = PlaudClient()

    # Fetch all recordings from Plaud API
    print("\n📡 Fetching recordings from Plaud API...")
    api_recs = client.list_recordings(fetch_all=True)

    # Deduplicate by ID (API pagination can return dupes)
    seen_ids = set()
    unique_recs = []
    for r in api_recs:
        rid = r.get("id")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            unique_recs.append(r)

    print(f"  Got {len(unique_recs)} unique recordings from API")

    updated = 0
    new_ingested = 0
    needs_reprocess = []

    for rec_data in unique_recs:
        recording_id = rec_data.get("id")
        start_at_str = rec_data.get("start_at")
        created_at_str = rec_data.get("created_at")
        duration_ms = rec_data.get("duration", 0)
        serial_number = rec_data.get("serial_number")
        title = rec_data.get("name")

        if not recording_id or not start_at_str:
            continue

        # Parse the actual recording time
        try:
            actual_time = datetime.fromisoformat(start_at_str.replace("Z", "+00:00"))
        except Exception:
            logger.warning(f"Cannot parse start_at for {recording_id[:16]}: {start_at_str}")
            continue

        # Check existing in SQLite
        existing = get_chronos_recording(session, recording_id)

        if existing:
            old_ts = existing.created_at  # type: ignore
            old_date = str(old_ts)[:10] if old_ts else "?"
            new_date = actual_time.strftime("%Y-%m-%d")

            if old_date != new_date:
                print(f"  🔄 {recording_id[:16]}  {old_date} → {new_date}  ({title or ''})")
                upsert_chronos_recording(
                    session=session,
                    recording_id=recording_id,
                    created_at=actual_time,
                    duration_seconds=duration_ms // 1000,
                    local_audio_path=str(existing.local_audio_path or ""),
                    source=str(existing.source or "plaud"),
                    device_id=serial_number,
                    title=title,
                    checksum=str(existing.checksum) if existing.checksum else None,
                )
                updated += 1
                needs_reprocess.append(recording_id)
            else:
                print(f"  ✅ {recording_id[:16]}  {old_date} (already correct)")
        else:
            # New recording — ingest it
            new_date = actual_time.strftime("%Y-%m-%d")
            print(f"  🆕 {recording_id[:16]}  {new_date}  ({title or ''})")
            upsert_chronos_recording(
                session=session,
                recording_id=recording_id,
                created_at=actual_time,
                duration_seconds=duration_ms // 1000,
                local_audio_path="",
                source="plaud",
                device_id=serial_number,
                title=title,
                checksum=None,
            )
            new_ingested += 1
            needs_reprocess.append(recording_id)

    session.commit()

    print(f"\n📊 Results:")
    print(f"  Updated timestamps: {updated}")
    print(f"  New recordings ingested: {new_ingested}")
    print(f"  Recordings needing re-processing: {len(needs_reprocess)}")

    if needs_reprocess:
        print(f"\n🔧 Marking {len(needs_reprocess)} recordings for re-processing...")
        for rid in needs_reprocess:
            mark_chronos_recording_status(session, rid, "pending", error_message=None)
        session.commit()
        print("  Done! Run the pipeline to re-process:")
        print("  ./venv/bin/python scripts/chronos_pipeline.py --process --index --limit 50")

    # Show final state
    import sqlite3
    db = sqlite3.connect("data/brain.db")
    cur = db.cursor()
    from collections import defaultdict

    date_counts = defaultdict(int)
    rows = cur.execute("SELECT created_at FROM chronos_recordings ORDER BY created_at").fetchall()
    for (ts,) in rows:
        date_counts[ts[:10] if ts else "?"] += 1

    print(f"\n📅 Recording dates after fix:")
    for day in sorted(date_counts.keys()):
        print(f"  {day}: {date_counts[day]} recordings")


if __name__ == "__main__":
    main()
