#!/usr/bin/env python3
"""Sync the whole Plaud library from the 4.0 API into PlaudBlender.

    venv/bin/python scripts/plaud_v4_sync.py            # everything not yet complete
    venv/bin/python scripts/plaud_v4_sync.py --limit 20 # newest 20, to see it work
    venv/bin/python scripts/plaud_v4_sync.py --dry-run  # list what would change

Walks /file-app/v4/recordings/all by cursor, newest first. For every
recording it stores title, real start time, duration, the device it was
recorded on (`scene_source`), the transcript with speaker labels, and Plaud's
summary. Ids are mapped onto the classic 32-hex id so nothing already in the
database, the iOS app or Notion is duplicated.

Complete recordings are skipped, so after the first walk this costs one list
call per hundred recordings plus a few detail calls for whatever is new.
Requires a session from scripts/plaud_v4_login.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402
from src.database import SessionLocal  # noqa: E402
from src.database.chronos_repository import (  # noqa: E402
    get_chronos_recording,
    set_chronos_recording_transcript,
    upsert_chronos_recording,
)
from src.plaud_v4 import NotLoggedIn, PlaudV4Client, PlaudV4Error, classic_id, device_code  # noqa: E402

try:
    from src.chronos.qdrant_client import ChronosQdrantClient  # noqa: E402
    from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402
except Exception:  # Qdrant optional at runtime
    ChronosQdrantClient = None

HARDWARE_DEVICES = {"888", "860", "881", "883", "880", "882"}
CLOCK_TOLERANCE_SECONDS = 60

SOURCE = "plaud_v4"


def _shift(ts, delta: timedelta):
    ts = str(ts or "")
    if len(ts) < 19:
        return None
    fmt = "%Y-%m-%d %H:%M:%S" if ts[10] == " " else "%Y-%m-%dT%H:%M:%S"
    try:
        return (datetime.strptime(ts[:19], fmt) + delta).strftime(fmt)
    except ValueError:
        return None


def redate_to_device_clock(session, qdrant, rid: str, stored: datetime, true: datetime) -> None:
    """Move a recording -- and everything hanging off it -- onto the device clock.

    A recording synced from a Plaud device carries its start time from the
    device itself. Anything that later re-dated it by inference (a title-date
    repair, a Notion default) replaced a measurement with a guess. This puts
    the measurement back and clears the estimate flag, shifting the row's
    events and their Qdrant payloads by the same delta so time-of-day fields
    and the day view stay coherent. Mirrors the janitor's own cascade.
    """
    delta = true - stored
    session.execute(text(
        "update chronos_recordings set created_at=:t, time_is_estimated=0, time_estimate_reason=NULL where recording_id=:i"
    ), {"t": true, "i": rid})
    cols = {r[1] for r in session.execute(text("pragma table_info(chronos_events)")).fetchall()}
    for row in session.execute(text("select event_id, start_ts, end_ts from chronos_events where recording_id=:i"), {"i": rid}).fetchall():
        event_id, start_ts, end_ts = row
        sets, args = [], {"e": event_id}
        ns = _shift(start_ts, delta)
        if ns:
            sets.append("start_ts=:s"); args["s"] = ns
            d = datetime.strptime(ns[:19], "%Y-%m-%d %H:%M:%S" if ns[10] == " " else "%Y-%m-%dT%H:%M:%S")
            if "day_of_week" in cols: sets.append("day_of_week=:dow"); args["dow"] = d.strftime("%A")
            if "hour_of_day" in cols: sets.append("hour_of_day=:h"); args["h"] = d.hour
            if "start_ts_unix" in cols: sets.append("start_ts_unix=:u"); args["u"] = d.timestamp()
        ne = _shift(end_ts, delta)
        if ne:
            sets.append("end_ts=:en"); args["en"] = ne
        if sets:
            session.execute(text(f"update chronos_events set {', '.join(sets)} where event_id=:e"), args)
    if qdrant is None:
        return
    selector = Filter(must=[FieldCondition(key="recording_id", match=MatchValue(value=rid))])
    offset = None
    while True:
        points, offset = qdrant.client.scroll(collection_name=qdrant.collection_name, scroll_filter=selector,
                                              with_payload=True, with_vectors=False, limit=200, offset=offset)
        for p in points:
            pl, np = p.payload or {}, {}
            for key in ("start_ts", "end_ts", "timestamp"):
                if pl.get(key):
                    ns = _shift(pl[key], delta)
                    if ns: np[key] = ns
            if np.get("start_ts"):
                s = np["start_ts"]
                d = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S" if s[10] == " " else "%Y-%m-%dT%H:%M:%S")
                np["day_of_week"] = d.strftime("%A"); np["hour_of_day"] = d.hour; np["start_ts_unix"] = d.timestamp()
            if np:
                qdrant.client.set_payload(collection_name=qdrant.collection_name, payload=np, points=[p.id])
        if offset is None:
            break
PACE_SECONDS = 0.15


def transcript_text(segments) -> str:
    """Flatten Plaud's segment array to `Speaker: text` lines.

    The speaker label is kept in the text on purpose: the flat column is what
    every downstream reader uses, and dropping the label would throw away the
    one thing the 4.0 transcript adds over the old flat export.
    """
    if not isinstance(segments, list):
        return ""
    lines = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("content") or seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker") or seg.get("original_speaker")
        lines.append(f"{speaker}: {text}" if speaker else text)
    return "\n".join(lines)


def sync(client: PlaudV4Client, *, limit: int | None, dry_run: bool, refresh_complete: bool, qdrant=None) -> int:
    created = updated = skipped = failed = reclocked = 0
    seen = 0
    started = time.monotonic()

    with SessionLocal() as session:
        for item in client.iter_recordings():
            if limit and seen >= limit:
                break
            seen += 1

            rid = classic_id(item["file_id"])
            title = item.get("name") or ""
            duration_s = int((item.get("duration_ms") or 0) // 1000)
            start_ms = item.get("created_at_show_ms")
            created_at = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).replace(tzinfo=None) if start_ms else datetime.utcnow()

            existing = get_chronos_recording(session, rid)

            # The list row's start is the device clock for hardware-synced
            # recordings. Check it here, before the completeness skip, so a
            # complete row that something re-dated by inference is still put
            # back. Detail is fetched only on a mismatch, so unchanged rows
            # cost nothing.
            if (existing and not dry_run and existing.created_at is not None
                    and str(existing.device_id or "")[:3] in HARDWARE_DEVICES
                    and abs((created_at - existing.created_at).total_seconds()) >= CLOCK_TOLERANCE_SECONDS):
                try:
                    confirm = (client.file_detail(item["file_id"]).get("meta") or {}).get("start_time")
                except PlaudV4Error:
                    confirm = None
                if confirm:
                    true = datetime.fromtimestamp(confirm / 1000, tz=timezone.utc).replace(tzinfo=None)
                    if abs((true - existing.created_at).total_seconds()) >= CLOCK_TOLERANCE_SECONDS:
                        redate_to_device_clock(session, qdrant, rid, existing.created_at, true)
                        session.commit()
                        reclocked += 1
                        print(f"  reclocked  {existing.created_at:%Y-%m-%d %H:%M} -> {true:%Y-%m-%d %H:%M}  {title[:48]}")
                        existing = get_chronos_recording(session, rid)

            complete = bool(existing and existing.transcript and existing.plaud_ai_summary and existing.device_id)
            if complete and not refresh_complete:
                skipped += 1
                continue

            if dry_run:
                print(f"  {'UPDATE' if existing else 'CREATE'}  {created_at:%Y-%m-%d %H:%M}  {duration_s // 60:4}m  {title[:60]}")
                continue

            try:
                detail = client.file_detail(item["file_id"])
                meta = detail.get("meta") or {}
                objects = client.objects_by_type(detail)

                # Real start from detail wins over the list row when present.
                if meta.get("start_time"):
                    created_at = datetime.fromtimestamp(meta["start_time"] / 1000, tz=timezone.utc).replace(tzinfo=None)

                device = device_code(meta.get("scene_source"))

                rec = upsert_chronos_recording(
                    session,
                    recording_id=rid,
                    title=title,
                    created_at=created_at,
                    duration_seconds=duration_s,
                    local_audio_path=existing.local_audio_path if existing and existing.local_audio_path else "",
                    source=existing.source if existing else SOURCE,
                    device_id=device,
                    time_is_estimated=False,
                    time_estimate_reason=None,
                )

                # The device clock is the ground truth for hardware-synced
                # rows. If anything has moved this row off it -- an inference
                # from a title date, a Notion default -- put it back, cascade,
                # and clear the estimate flag. Counted separately so it is visible.
                if "TRANSCRIPT" in objects and (not existing or not existing.transcript or refresh_complete):
                    segments = client.content_json(objects["TRANSCRIPT"]["content_id"])
                    text = transcript_text(segments)
                    if text:
                        set_chronos_recording_transcript(session, rid, text)

                if "SUMMARY" in objects and (not existing or not existing.plaud_ai_summary or refresh_complete):
                    summary = client.content(objects["SUMMARY"]["content_id"]).strip()
                    if summary:
                        rec.plaud_ai_summary = summary
                        session.commit()

                if existing:
                    updated += 1
                else:
                    created += 1
                print(f"  {'updated' if existing else 'created'}  {created_at:%Y-%m-%d %H:%M}  {duration_s // 60:4}m  {device or '?':16}  {title[:52]}")
            except NotLoggedIn:
                raise
            except PlaudV4Error as error:
                failed += 1
                print(f"  FAILED  {title[:52]}: {error}", file=sys.stderr)
            time.sleep(PACE_SECONDS)

    elapsed = time.monotonic() - started
    print(f"\n{seen} listed · {created} created · {updated} updated · {reclocked} re-clocked · {skipped} already complete · {failed} failed · {elapsed:.0f}s")
    return 1 if failed and not (created or updated) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="stop after this many recordings (newest first)")
    parser.add_argument("--dry-run", action="store_true", help="show what would be created or updated; write nothing")
    parser.add_argument("--refresh-complete", action="store_true", help="re-fetch recordings that already have transcript and summary")
    args = parser.parse_args()

    client = PlaudV4Client()
    if not client.has_session:
        print("No Plaud 4.0 session. Run: venv/bin/python scripts/plaud_v4_login.py --email <your email>", file=sys.stderr)
        return 2
    qdrant = None
    if ChronosQdrantClient is not None and not args.dry_run:
        try:
            qdrant = ChronosQdrantClient()
        except Exception as error:
            print(f"Qdrant unavailable ({error}); clock cascade will skip vectors", file=sys.stderr)
    try:
        return sync(client, limit=args.limit, dry_run=args.dry_run, refresh_complete=args.refresh_complete, qdrant=qdrant)
    except NotLoggedIn as error:
        print(f"{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
