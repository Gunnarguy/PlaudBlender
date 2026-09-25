#!/usr/bin/env python3
"""Re-place existing events at their real time inside each recording.

    venv/bin/python scripts/repair_event_times.py              # dry run: report only
    venv/bin/python scripts/repair_event_times.py --apply      # write DB + Qdrant
    venv/bin/python scripts/repair_event_times.py --revert     # undo the last --apply

See src/chronos/event_timing.py for why the stored times are wrong and how
placement works. Notion recordings are skipped: the data service already
re-anchors them at read time.

--apply first copies brain.db and snapshots the Qdrant collection, then
records every old value in event_time_repairs so --revert can restore both.
Each recording's events are grouped into processing passes (a recording
processed twice holds two full event sets, inserted >10 min apart); each
pass is placed across the whole recording on its own.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chronos.event_timing import EventTiming, load_transcript_lines, place_events  # noqa: E402
from src.config import get_local_timezone, get_settings  # noqa: E402

DB = ROOT / "data" / "brain.db"
ARTIFACTS = ROOT / "data" / "artifacts"
PASS_GAP = timedelta(minutes=10)
FMT = "%Y-%m-%d %H:%M:%S.%f"


def parse(value) -> datetime | None:
    return datetime.fromisoformat(str(value)[:26]) if value else None


def passes(rows):
    """Split one recording's events (sorted by insert time) into processing passes."""
    groups, last = [], None
    for row in rows:
        created = parse(row["created_at"])
        if not groups or (created and last and created - last > PASS_GAP):
            groups.append([])
        groups[-1].append(row)
        last = created or last
    return groups


def plan(conn, only: str | None, local_tz):
    recordings = {
        r["recording_id"]: r
        for r in conn.execute(
            "select recording_id, created_at, duration_seconds, source, title from chronos_recordings"
            " where created_at is not null and coalesce(source,'') != 'notion'"
        )
    }
    by_rec = defaultdict(list)
    for row in conn.execute(
        "select event_id, recording_id, start_ts, end_ts, day_of_week, hour_of_day,"
        " raw_transcript_snippet, qdrant_point_id, created_at from chronos_events order by recording_id, created_at, rowid"
    ):
        if row["recording_id"] in recordings and (not only or row["recording_id"].startswith(only)):
            by_rec[row["recording_id"]].append(row)

    changes = []
    for rid, rows in by_rec.items():
        rec = recordings[rid]
        lines = load_transcript_lines(ARTIFACTS, rid)
        for group in passes(rows):
            timings = [EventTiming(parse(r["start_ts"]), parse(r["end_ts"]) or parse(r["start_ts"]), r["raw_transcript_snippet"]) for r in group]
            for row, p in zip(group, place_events(timings, parse(rec["created_at"]), rec["duration_seconds"], local_tz, lines)):
                changes.append((row, p, rec))
    return changes


def report(changes, local_tz):
    methods = Counter(p.method for _, p, _ in changes)
    moved = sum(1 for row, p, _ in changes if abs((parse(row["start_ts"]) - p.start).total_seconds()) > 60)
    outside_after = 0
    for _, p, rec in changes:
        start = parse(rec["created_at"])
        from src.chronos.event_timing import utc_to_local_naive

        lo = utc_to_local_naive(start, local_tz)
        hi = lo + timedelta(seconds=float(rec["duration_seconds"] or 0) + 60)
        if rec["duration_seconds"] and not (lo - timedelta(seconds=1) <= p.start <= hi):
            outside_after += 1
    future_before = sum(1 for row, _, _ in changes if parse(row["start_ts"]) > datetime.now() + timedelta(days=1))
    print(f"events considered: {len(changes)} in {len({r['recording_id'] for r, _, _ in changes})} recordings")
    print(f"placement method:  " + ", ".join(f"{k} {v}" for k, v in methods.most_common()))
    print(f"events that move >1 min: {moved}")
    print(f"future-dated before: {future_before}   outside recording span after: {outside_after}")


def apply(conn, changes, local_tz):
    import shutil

    from qdrant_client import QdrantClient
    from qdrant_client.models import SetPayload, SetPayloadOperation

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = Path("/_data/backups") / f"brain.db.pre-event-times-{stamp}"
    dst = sqlite3.connect(backup)
    conn.backup(dst)
    dst.close()
    print(f"backed up brain.db -> {backup}")

    settings = get_settings()
    collection = settings.qdrant_collection_name
    client = QdrantClient(url=getattr(settings, "qdrant_url", None) or "http://localhost:6333", timeout=120)
    snap = client.create_snapshot(collection_name=collection)
    print(f"qdrant snapshot of {collection}: {snap.name}")

    conn.execute(
        "create table if not exists event_time_repairs (event_id text, recording_id text, qdrant_point_id text,"
        " old_start text, old_end text, old_day text, old_hour integer,"
        " new_start text, new_end text, method text, repaired_at text)"
    )
    now = datetime.now().isoformat(timespec="seconds")
    ops = []
    for row, p, _ in changes:
        new_start, new_end = p.start.strftime(FMT), p.end.strftime(FMT)
        day, hour = p.start.strftime("%A"), p.start.hour
        conn.execute(
            "insert into event_time_repairs values (?,?,?,?,?,?,?,?,?,?,?)",
            (row["event_id"], row["recording_id"], row["qdrant_point_id"], row["start_ts"], row["end_ts"],
             row["day_of_week"], row["hour_of_day"], new_start, new_end, p.method, now),
        )
        conn.execute(
            "update chronos_events set start_ts=?, end_ts=?, day_of_week=?, hour_of_day=? where event_id=?",
            (new_start, new_end, day, hour, row["event_id"]),
        )
        if row["qdrant_point_id"]:
            ops.append(SetPayloadOperation(set_payload=SetPayload(
                points=[row["qdrant_point_id"]],
                payload=_payload(p.start, p.end, local_tz),
            )))
    conn.commit()
    print(f"database: {len(changes)} events updated, old values in event_time_repairs")
    for i in range(0, len(ops), 256):
        client.batch_update_points(collection_name=collection, update_operations=ops[i : i + 256])
    print(f"qdrant: {len(ops)} point payloads updated")


def _payload(start: datetime, end: datetime, local_tz) -> dict:
    return {
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "timestamp": start.isoformat(),
        "start_ts_unix": start.replace(tzinfo=local_tz).timestamp(),
        "day_of_week": start.strftime("%A"),
        "hour_of_day": start.hour,
        "duration_seconds": max((end - start).total_seconds(), 0.0),
    }


def revert(conn, local_tz):
    from qdrant_client import QdrantClient
    from qdrant_client.models import SetPayload, SetPayloadOperation

    settings = get_settings()
    client = QdrantClient(url=getattr(settings, "qdrant_url", None) or "http://localhost:6333", timeout=120)
    last = conn.execute("select max(repaired_at) from event_time_repairs").fetchone()[0]
    rows = conn.execute("select * from event_time_repairs where repaired_at=?", (last,)).fetchall()
    ops = []
    for r in rows:
        conn.execute(
            "update chronos_events set start_ts=?, end_ts=?, day_of_week=?, hour_of_day=? where event_id=?",
            (r["old_start"], r["old_end"], r["old_day"], r["old_hour"], r["event_id"]),
        )
        if r["qdrant_point_id"]:
            s, e = parse(r["old_start"]), parse(r["old_end"]) or parse(r["old_start"])
            payload = _payload(s, e, local_tz)
            payload.update(day_of_week=r["old_day"], hour_of_day=r["old_hour"])
            ops.append(SetPayloadOperation(set_payload=SetPayload(points=[r["qdrant_point_id"]], payload=payload)))
    conn.execute("delete from event_time_repairs where repaired_at=?", (last,))
    conn.commit()
    for i in range(0, len(ops), 256):
        client.batch_update_points(collection_name=settings.qdrant_collection_name, update_operations=ops[i : i + 256])
    print(f"reverted {len(rows)} events from the {last} repair")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--recording-id", help="only recordings whose id starts with this")
    parser.add_argument("--show", type=int, default=0, help="print the first N placements")
    args = parser.parse_args()

    local_tz = get_local_timezone()
    # The API and UI keep brain.db open; wait for their locks instead of failing.
    conn = sqlite3.connect(DB, timeout=60)
    conn.row_factory = sqlite3.Row
    if args.revert:
        revert(conn, local_tz)
        return 0
    changes = plan(conn, args.recording_id, local_tz)
    for row, p, rec in changes[: args.show]:
        print(f"  {str(row['start_ts'])[:16]} -> {p.start:%Y-%m-%d %H:%M:%S} ({p.method:12}) {(rec['title'] or '')[:40]}")
    report(changes, local_tz)
    if args.apply:
        apply(conn, changes, local_tz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
