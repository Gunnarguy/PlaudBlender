#!/usr/bin/env python3
"""Analyze current data structure to inform UI redesign."""

from src.chronos.qdrant_client import ChronosQdrantClient
from src.database.engine import SessionLocal
from src.database.models import Recording
from collections import Counter, defaultdict
from datetime import datetime
import json


def main():
    # Check Qdrant events
    client = ChronosQdrantClient()
    all_pts = []
    off = None
    while True:
        r = client.client.scroll(
            collection_name=client.collection_name,
            limit=100,
            offset=off,
            with_payload=True,
        )
        pts, off = r
        all_pts.extend(pts)
        if off is None:
            break

    print(f"=== QDRANT EVENTS: {len(all_pts)} ===")
    if all_pts:
        p = all_pts[0]
        print(f"Sample event keys: {list(p.payload.keys())}")
        print(f"\nSample payload (truncated):")
        for k, v in list(p.payload.items()):
            val = str(v)[:100] + "..." if len(str(v)) > 100 else v
            print(f"  {k}: {val}")

    # Check recordings by recording_id in events
    recording_ids = Counter()
    for p in all_pts:
        rid = p.payload.get("recording_id", "no_recording_id")
        recording_ids[rid] += 1
    print(f"\n=== EVENTS PER RECORDING: {len(recording_ids)} unique recordings ===")
    for rid, count in recording_ids.most_common(10):
        short_rid = rid[:20] + "..." if len(rid) > 20 else rid
        print(f"  {short_rid}: {count} events")

    # Group events by day
    days = defaultdict(list)
    for p in all_pts:
        ts = p.payload.get("event_timestamp") or p.payload.get("timestamp")
        if ts:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromtimestamp(ts)
                day_key = dt.strftime("%Y-%m-%d")
                days[day_key].append(p)
            except:
                days["unknown"].append(p)

    print(f"\n=== EVENTS BY DAY ===")
    for day in sorted(days.keys(), reverse=True)[:10]:
        print(f"  {day}: {len(days[day])} events")

    # Check categories
    categories = Counter()
    for p in all_pts:
        cat = p.payload.get("category", "uncategorized")
        categories[cat] += 1
    print(f"\n=== CATEGORIES ===")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")

    # Check importance distribution
    importance = Counter()
    for p in all_pts:
        imp = p.payload.get("importance", 5)
        importance[imp] += 1
    print(f"\n=== IMPORTANCE DISTRIBUTION ===")
    for imp in sorted(importance.keys()):
        print(f"  {imp}: {importance[imp]}")

    # Check SQLite recordings
    print("\n=== SQLITE RECORDINGS ===")
    session = SessionLocal()
    recordings = (
        session.query(Recording).order_by(Recording.created_at.desc()).limit(20).all()
    )
    total_recordings = session.query(Recording).count()
    print(f"Total recordings in DB: {total_recordings}")

    if recordings:
        print(
            f"\nRecording table columns: {[c.name for c in Recording.__table__.columns]}"
        )
        print(f"\nRecent 10 recordings:")
        for r in recordings[:10]:
            duration_min = r.duration_seconds / 60 if r.duration_seconds else 0
            print(
                f"  {r.id[:12]}... | {duration_min:.1f}min | {r.created_at} | {r.title or 'No title'}"
            )

    session.close()

    # Check if we can link events to recordings
    print("\n=== EVENT-RECORDING LINKAGE ===")
    event_rec_ids = set(rid for rid in recording_ids.keys() if rid != "no_recording_id")
    print(f"Unique recording IDs in events: {len(event_rec_ids)}")

    session = SessionLocal()
    db_rec_ids = set(r.id for r in session.query(Recording.id).all())
    print(f"Recording IDs in SQLite: {len(db_rec_ids)}")

    matched = event_rec_ids & db_rec_ids
    print(f"Matched (events can link to recordings): {len(matched)}")

    unmatched_events = event_rec_ids - db_rec_ids
    if unmatched_events:
        print(f"Events with no matching recording: {len(unmatched_events)}")
        print(f"  Sample: {list(unmatched_events)[:3]}")

    session.close()


if __name__ == "__main__":
    main()
