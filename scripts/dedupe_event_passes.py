#!/usr/bin/env python3
"""Remove duplicate event sets left by re-processing, and orphaned Qdrant points.

    venv/bin/python scripts/dedupe_event_passes.py            # dry run: report only
    venv/bin/python scripts/dedupe_event_passes.py --apply

A recording re-processed without clearing its old events keeps both sets:
events inserted >10 min apart form separate passes, and each pass describes
the whole recording again. Measured 2026-09-28: 170 older passes in 167
recordings; 160 were processed days to weeks before the next pass, and of
the 88 with timed-transcript evidence, 85 cover the same stretch as the
newest pass and none cover a different one.

Keeps the newest pass. An older pass is deleted when it was processed at
least a day before the next pass, or its transcript-matched events cover
the same stretch as the newest pass. Anything else is kept and listed.

Also deletes Qdrant points that no event references (ghosts from earlier
re-processing: the timeline reads events from Qdrant).

--apply copies brain.db, snapshots the Qdrant collection, and archives every
deleted event row and Qdrant point (payload + vector) as JSON under
/_data/backups before deleting. chronos_execution_spans rows keep their
telemetry with event_id cleared.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402

DB = ROOT / "data" / "brain.db"
PASS_GAP = timedelta(minutes=10)
REPROCESS_GAP = timedelta(days=1)
SAME_STRETCH = 0.6


def parse(value):
    return datetime.fromisoformat(str(value)[:26]) if value else None


def split_passes(rows):
    passes = []
    for row in rows:
        if not passes or parse(row["created_at"]) - parse(passes[-1][-1]["created_at"]) > PASS_GAP:
            passes.append([])
        passes[-1].append(row)
    return passes


def transcript_span(rows, methods):
    matched = [parse(r["start_ts"]) for r in rows if methods.get(r["event_id"]) == "transcript"]
    return (min(matched), max(matched)) if len(matched) >= 2 else None


def same_stretch(a, b) -> bool:
    if not (a and b):
        return False
    overlap = (min(a[1], b[1]) - max(a[0], b[0])).total_seconds()
    return overlap / max((a[1] - a[0]).total_seconds(), 1.0) > SAME_STRETCH


def plan(conn, include_close: bool = False):
    methods = {}
    if conn.execute("select 1 from sqlite_master where name='event_time_repairs'").fetchone():
        first = conn.execute("select min(repaired_at) from event_time_repairs").fetchone()[0]
        methods = dict(conn.execute("select event_id, method from event_time_repairs where repaired_at=?", (first,)))
    by_rec = defaultdict(list)
    for row in conn.execute("select * from chronos_events where created_at is not null order by recording_id, created_at, rowid"):
        by_rec[row["recording_id"]].append(row)

    delete, kept_for_review = [], []
    for rid, rows in by_rec.items():
        passes = split_passes(rows)
        if len(passes) < 2:
            continue
        newest_span = transcript_span(passes[-1], methods)
        for older, following in zip(passes[:-1], passes[1:]):
            gap = parse(following[0]["created_at"]) - parse(older[-1]["created_at"])
            if include_close or gap >= REPROCESS_GAP or same_stretch(transcript_span(older, methods), newest_span):
                delete.extend(older)
            else:
                kept_for_review.append((rid, len(older), gap))
    return delete, kept_for_review


def qdrant_orphans(conn, client, collection):
    referenced = {r[0] for r in conn.execute("select qdrant_point_id from chronos_events where qdrant_point_id is not null")}
    orphans, offset = [], None
    while True:
        points, offset = client.scroll(collection, limit=1000, offset=offset, with_payload=False, with_vectors=False)
        orphans += [str(p.id) for p in points if str(p.id) not in referenced]
        if offset is None:
            return orphans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-close-passes", action="store_true",
                        help="also delete older passes processed <1 day apart (after checking the review list)")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    settings = get_settings()
    collection = settings.qdrant_collection_name
    client = QdrantClient(url=getattr(settings, "qdrant_url", None) or "http://localhost:6333", timeout=120)
    conn = sqlite3.connect(DB, timeout=60)
    conn.row_factory = sqlite3.Row

    delete, review = plan(conn, args.include_close_passes)
    delete_ids = {r["event_id"] for r in delete}
    event_points = [r["qdrant_point_id"] for r in delete if r["qdrant_point_id"]]
    orphans = [p for p in qdrant_orphans(conn, client, collection) if p not in set(event_points)]
    total_events = conn.execute("select count(*) from chronos_events").fetchone()[0]

    print(f"duplicate events to delete: {len(delete)} in {len({r['recording_id'] for r in delete})} recordings (of {total_events} events)")
    print(f"their qdrant points: {len(event_points)}   orphaned qdrant points: {len(orphans)}")
    print(f"older passes kept for review (processed <1 day apart, no proof of overlap): {len(review)}")
    for rid, n, gap in review:
        print(f"   {rid[:12]}  {n} events  {gap}")
    if not args.apply:
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = Path("/_data/backups")
    dst = sqlite3.connect(backups / f"brain.db.pre-dedupe-{stamp}")
    conn.backup(dst)
    dst.close()
    snap = client.create_snapshot(collection_name=collection)
    doomed_points = event_points + orphans
    archived = []
    for i in range(0, len(doomed_points), 256):
        for p in client.retrieve(collection, ids=doomed_points[i : i + 256], with_payload=True, with_vectors=True):
            archived.append({"id": str(p.id), "payload": p.payload, "vector": p.vector})
    archive = backups / f"dedupe-{stamp}.json"
    archive.write_text(json.dumps({
        "events": [dict(r) for r in delete],
        "qdrant_collection": collection,
        "qdrant_points": archived,
    }, default=str))
    print(f"backed up brain.db, qdrant snapshot {snap.name}, archive {archive} ({len(archived)} points)")

    ids = list(delete_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        marks = ",".join("?" * len(chunk))
        conn.execute(f"update chronos_execution_spans set event_id=null where event_id in ({marks})", chunk)
        conn.execute(f"delete from chronos_events where event_id in ({marks})", chunk)
    conn.commit()
    for i in range(0, len(doomed_points), 256):
        client.delete(collection, points_selector=doomed_points[i : i + 256])
    remaining = conn.execute("select count(*) from chronos_events").fetchone()[0]
    points_left = client.get_collection(collection).points_count
    print(f"deleted {len(ids)} events and {len(doomed_points)} qdrant points; now {remaining} events, {points_left} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
