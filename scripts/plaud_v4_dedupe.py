#!/usr/bin/env python3
"""Retire Zapier-era Notion twins now that the 4.0 sync holds the real rows.

    venv/bin/python scripts/plaud_v4_dedupe.py --dry-run
    venv/bin/python scripts/plaud_v4_dedupe.py

Before the 4.0 API, new recordings reached this database only through
Zapier -> Notion, stored under `notion:<page>` ids with a default clock time.
Once scripts/plaud_v4_sync.py imports the same recording under its real id,
the pair shows twice on every timeline day.

A Notion row is a twin of a real row when both fall on the same calendar day
and their titles match after normalisation. That is deliberately strict: two
devices recording the same room get *different* AI titles from Plaud, so a
title match is a copy, not a second take. Two-device pairs are never touched.

Retiring a twin: anything only the Notion row had (a summary) is copied to
the real row; its events, jobs, webhook events and spans are moved to the
real row when the real row has none of that kind, otherwise dropped as
regenerable; a janitor tombstone records what was removed and why; then the
Notion row is deleted. Also normalises device ids written as the human name
(`plaud_note`) back to the bare code (`888`) so every era agrees.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.database import SessionLocal  # noqa: E402

try:
    from src.chronos.qdrant_client import ChronosQdrantClient  # noqa: E402
except Exception:  # Qdrant optional at runtime
    ChronosQdrantClient = None

CHILD_TABLES = ("chronos_events", "chronos_processing_jobs", "chronos_webhook_events", "chronos_execution_spans")
NAME_TO_CODE = {"plaud_notepin": "880", "plaud_note_pro": "881", "plaud_notepin_s": "882", "plaud_note": "888"}


def norm(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def find_twins(session) -> list[tuple[str, str, str]]:
    rows = session.execute(text("""
        select recording_id, title, date(created_at) from chronos_recordings
    """)).fetchall()
    real: dict[tuple[str, str], str] = {}
    notion: list[tuple[str, str, str]] = []
    for rid, title, day in rows:
        key = (day or "", norm(title))
        if not key[1]:
            continue
        if rid.startswith("notion:"):
            notion.append((rid, title or "", day or ""))
        else:
            real.setdefault(key, rid)
    return [(nid, real[(day, norm(title))], title) for nid, title, day in notion if (day, norm(title)) in real]


def reconcile_qdrant(session, qdrant, dead_id: str) -> tuple[int, int]:
    """Bring Qdrant in line with SQLite for a retired recording id.

    The day view builds its recording list from event payloads in Qdrant,
    so a retired id lingers there as a phantom -- title None, device None --
    until its points are dealt with. Per point: if SQLite still has that
    event (moved under the real id), re-point the vector at the row that
    owns it now; otherwise the event was dropped, so drop the vector too.
    Returns (repointed, deleted).
    """
    if qdrant is None:
        return (0, 0)
    ids = qdrant.point_ids_for_recording(dead_id)
    if not ids:
        return (0, 0)
    owners: dict[str, str] = {}
    for chunk_start in range(0, len(ids), 500):
        chunk = [str(i) for i in ids[chunk_start:chunk_start + 500]]
        placeholders = ",".join(f":p{i}" for i in range(len(chunk)))
        rows = session.execute(
            text(f"select qdrant_point_id, recording_id from chronos_events where qdrant_point_id in ({placeholders})"),
            {f"p{i}": v for i, v in enumerate(chunk)},
        ).fetchall()
        owners.update({str(pid): rid for pid, rid in rows})
    keep_by_owner: dict[str, list] = {}
    drop = []
    for pid in ids:
        owner = owners.get(str(pid))
        if owner and owner != dead_id:
            keep_by_owner.setdefault(owner, []).append(pid)
        else:
            drop.append(pid)
    repointed = 0
    for owner, pids in keep_by_owner.items():
        qdrant.client.set_payload(collection_name=qdrant.collection_name, payload={"recording_id": owner}, points=pids)
        repointed += len(pids)
    if drop:
        qdrant.client.delete(collection_name=qdrant.collection_name, points_selector=drop)
    return (repointed, len(drop))


def retire(session, notion_id: str, real_id: str, title: str, qdrant=None) -> None:
    n = session.execute(text("select plaud_ai_summary, transcript from chronos_recordings where recording_id=:i"), {"i": notion_id}).fetchone()
    r = session.execute(text("select plaud_ai_summary, transcript from chronos_recordings where recording_id=:i"), {"i": real_id}).fetchone()
    if n and r:
        if n[0] and not r[0]:
            session.execute(text("update chronos_recordings set plaud_ai_summary=:s where recording_id=:i"), {"s": n[0], "i": real_id})
        if n[1] and not r[1]:
            session.execute(text("update chronos_recordings set transcript=:t, transcript_cached_at=:now where recording_id=:i"),
                            {"t": n[1], "now": datetime.utcnow(), "i": real_id})
    for table in CHILD_TABLES:
        real_has = session.execute(text(f"select count(*) from {table} where recording_id=:i"), {"i": real_id}).scalar()
        if real_has:
            session.execute(text(f"delete from {table} where recording_id=:i"), {"i": notion_id})
        else:
            session.execute(text(f"update {table} set recording_id=:r where recording_id=:n"), {"r": real_id, "n": notion_id})
    session.execute(text("""
        insert or replace into janitor_tombstones (recording_id, deleted_at, title, reason)
        values (:i, :at, :t, :why)
    """), {"i": notion_id, "at": datetime.utcnow().isoformat(), "t": title, "why": f"notion twin of {real_id} (plaud_v4_dedupe)"})
    session.execute(text("delete from chronos_recordings where recording_id=:i"), {"i": notion_id})
    session.flush()
    reconcile_qdrant(session, qdrant, notion_id)


def normalise_devices(session, dry_run: bool) -> int:
    total = 0
    for name, code in NAME_TO_CODE.items():
        n = session.execute(text("select count(*) from chronos_recordings where device_id=:n"), {"n": name}).scalar()
        if n and not dry_run:
            session.execute(text("update chronos_recordings set device_id=:c where device_id=:n"), {"c": code, "n": name})
        total += n or 0
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repair-qdrant", action="store_true",
                        help="reconcile Qdrant for twins retired by earlier runs (reads janitor_tombstones)")
    args = parser.parse_args()

    qdrant = None
    if ChronosQdrantClient is not None and not args.dry_run:
        try:
            qdrant = ChronosQdrantClient()
        except Exception as error:
            print(f"Qdrant unavailable ({error}); SQLite only", file=sys.stderr)

    if args.repair_qdrant:
        with SessionLocal() as session:
            dead = [r[0] for r in session.execute(text(
                "select recording_id from janitor_tombstones where reason like '%plaud_v4_dedupe%'")).fetchall()]
            repointed = deleted = 0
            for dead_id in dead:
                r, d = reconcile_qdrant(session, qdrant, dead_id)
                repointed += r; deleted += d
            print(f"repaired Qdrant for {len(dead)} retired ids: {repointed} points re-pointed, {deleted} deleted")
        return 0

    with SessionLocal() as session:
        twins = find_twins(session)
        renamed = normalise_devices(session, args.dry_run)
        print(f"{len(twins)} Notion twins of real recordings · {renamed} device ids to normalise")
        for notion_id, real_id, title in twins[:12]:
            print(f"  {notion_id[:22]}  ->  {real_id[:12]}  {title[:56]}")
        if len(twins) > 12:
            print(f"  … {len(twins) - 12} more")
        if args.dry_run:
            print("dry run — nothing changed")
            return 0
        for notion_id, real_id, title in twins:
            retire(session, notion_id, real_id, title, qdrant)
        session.commit()
        print(f"retired {len(twins)} twins (tombstoned), normalised {renamed} device ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
