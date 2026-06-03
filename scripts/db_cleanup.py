#!/usr/bin/env python3
"""Unified Database & Vector Store Cleanup Script.

Cleans up:
1. Legacy recordings & segments tables (drops/deletes them).
2. Failed/unusable recordings in chronos_recordings (which cascades to chronos_events).
3. Vacuums SQLite database to reclaim disk space.
4. Purges orphaned Qdrant vector points.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.database import init_db, SessionLocal
from src.database.models import ChronosRecording, ChronosEvent, Recording, Segment
from qdrant_client import QdrantClient

def cleanup_sqlite():
    print("=== SQLite Database Cleanup ===")
    
    # Initialize DB (creates tables if it was empty)
    print("Initializing database schema...")
    init_db()
    
    db = SessionLocal()
    try:
        # 1. Clean up legacy tables if they exist and contain data
        print("Checking legacy tables...")
        try:
            legacy_recs = db.query(Recording).count()
            legacy_segs = db.query(Segment).count()
            if legacy_recs > 0 or legacy_segs > 0:
                print(f"  Purging legacy data: {legacy_recs} recordings, {legacy_segs} segments...")
                db.query(Segment).delete()
                db.query(Recording).delete()
                db.commit()
                print("  ✓ Legacy tables purged.")
            else:
                print("  ✓ Legacy tables are already empty.")
        except Exception as e:
            print(f"  Note: Legacy tables check skipped or not present: {e}")
            db.rollback()

        # 2. Clean up failed chronos recordings (cascades to events)
        print("Checking failed Chronos recordings...")
        failed_recordings = db.query(ChronosRecording).filter(
            ChronosRecording.processing_status == "failed"
        ).all()
        
        if failed_recordings:
            print(f"  Found {len(failed_recordings)} failed recordings to purge:")
            for rec in failed_recordings:
                print(f"    - ID: {rec.recording_id} (Title: {rec.title or 'Untitled'})")
                # Delete the recording (cascade deletes the events)
                db.delete(rec)
            db.commit()
            print("  ✓ Failed recordings purged.")
        else:
            print("  ✓ No failed Chronos recordings found.")

        # 3. Clean up empty/stale entries
        print("Checking for orphaned/empty records...")
        empty_recs = db.query(ChronosRecording).filter(
            ChronosRecording.transcript.is_(None),
            ChronosRecording.processing_status != "pending"
        ).all()
        if empty_recs:
            print(f"  Found {len(empty_recs)} empty recordings to purge...")
            for rec in empty_recs:
                db.delete(rec)
            db.commit()
            print("  ✓ Empty records purged.")
        else:
            print("  ✓ No empty records found.")

    except Exception as e:
        print(f"❌ Error during database transaction: {e}")
        db.rollback()
    finally:
        db.close()

    # 4. Vacuum SQLite database
    print("Running SQLite VACUUM to reclaim space...")
    try:
        engine = create_engine(f"sqlite:///{ROOT}/data/brain.db")
        with engine.connect() as conn:
            # SQLite VACUUM cannot be run inside a transaction block in some drivers,
            # so we run it on raw connection.
            conn.execute(text("VACUUM"))
            print("  ✓ Database vacuumed successfully.")
    except Exception as e:
        print(f"⚠️ Vacuum warning: {e}")

def cleanup_qdrant():
    print("\n=== Qdrant Vector Store Cleanup ===")
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"Connecting to Qdrant at {qdrant_url}...")
    
    try:
        q = QdrantClient(url=qdrant_url, timeout=5)
        
        # Check if collection exists
        collections = q.get_collections().collections
        exists = any(c.name == "chronos_events" for c in collections)
        if not exists:
            print("  ✓ Collection 'chronos_events' does not exist in Qdrant. Nothing to clean.")
            return

        # Fetch active Qdrant points in SQLite
        engine = create_engine(f"sqlite:///{ROOT}/data/brain.db")
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT qdrant_point_id FROM chronos_events WHERE qdrant_point_id IS NOT NULL")).fetchall()
            db_ids = {r[0] for r in rows}
            
        print(f"SQLite has {len(db_ids)} active indexed event point IDs.")

        # Scroll through Qdrant
        all_qdrant_ids = set()
        offset = None
        while True:
            results = q.scroll(
                collection_name="chronos_events",
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = results
            for p in points:
                all_qdrant_ids.add(str(p.id))
            if next_offset is None:
                break
            offset = next_offset

        print(f"Qdrant has {len(all_qdrant_ids)} points total.")
        
        orphans = all_qdrant_ids - db_ids
        if orphans:
            print(f"  Found {len(orphans)} orphan points in Qdrant — deleting...")
            q.delete(
                collection_name="chronos_events",
                points_selector=list(orphans),
            )
            info = q.get_collection("chronos_events")
            print(f"  ✓ Cleanup finished. Current points in Qdrant: {info.points_count}")
        else:
            print("  ✓ No orphan vector points found in Qdrant.")
            
    except Exception as e:
        print(f"⚠️ Qdrant connection skipped or failed: {e}")
        print("  (Make sure Docker is running and Qdrant container is up if you want to clean vector points.)")

def main():
    cleanup_sqlite()
    cleanup_qdrant()
    print("\n🎉 Cleanup routine complete.")

if __name__ == "__main__":
    main()
