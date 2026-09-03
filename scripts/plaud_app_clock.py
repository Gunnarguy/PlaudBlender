#!/usr/bin/env python3
"""Apply the owner's app-clock log as the true start time for recordings Plaud
never timestamped from a device.

    venv/bin/python scripts/plaud_app_clock.py /tmp/app_clock.json --dry-run
    venv/bin/python scripts/plaud_app_clock.py /tmp/app_clock.json

Two groups have no device clock upstream: the One's July recordings, which
were uploaded by hand and so carry Plaud's import moment; and 4.0 rows with
scene code 101, whose start_time is likewise the import. For both, the only
measured start is the recording list the owner photographed from each app
and logged as app_timeline.py -- date, local start, duration, neutral label.

A row is matched to a log entry on the same local date whose duration is
within tolerance and which no other row has claimed. Matched rows take the
log's start (converted to UTC), have their events and Qdrant payloads
shifted by the same delta, and are marked measured with the reason recorded.
Unmatched rows are listed and left alone; ambiguity retires nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.database import SessionLocal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plaud_v4_sync import redate_to_device_clock  # noqa: E402

try:
    from src.chronos.qdrant_client import ChronosQdrantClient  # noqa: E402
except Exception:
    ChronosQdrantClient = None

TOLERANCE_S = 180


def tolerance(duration_s: int) -> int:
    # The log rounds long takes to whole hours ("5h" for a 245-minute take),
    # so the window must grow with length; short takes stay tight.
    return max(TOLERANCE_S, int(duration_s * 0.25)) if duration_s >= 3600 else TOLERANCE_S
REASON = "app clock: owner's recording-list log"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="JSON exported from app_timeline.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log = json.loads(Path(args.log).read_text())
    tz = ZoneInfo(log.get("tz", "America/Los_Angeles"))
    entries = []
    for device, key in (("860", "one"), ("888", "ng1")):
        for e in log.get(key, []):
            local = datetime.strptime(e["start_local"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
            entries.append({"device": device, "local_date": local.date(), "utc": local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
                            "duration_s": int(e["duration_s"]), "label": e["label"], "claimed": False})

    qdrant = None
    if ChronosQdrantClient is not None and not args.dry_run:
        try: qdrant = ChronosQdrantClient()
        except Exception as error: print(f"Qdrant unavailable ({error}); vectors not shifted", file=sys.stderr)

    fixed = unmatched = 0
    with SessionLocal() as session:
        rows = session.execute(text("""
            select recording_id, title, created_at, duration_seconds, device_id, coalesce(time_estimate_reason,'')
            from chronos_recordings
            where recording_id not like 'notion:%'
              and (device_id like '________________________________-%' or device_id = '101' or coalesce(time_is_estimated,0)=1)
              and coalesce(time_estimate_reason,'') not like 'app clock%'
            order by created_at""")).fetchall()
        print(f"{len(rows)} candidate rows · {len(entries)} log entries")
        for rid, title, created_at, dur, device_id, reason in rows:
            created_at = datetime.fromisoformat(str(created_at)[:19])
            local_date = created_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz).date()
            # The hand-typed title date is the stronger date signal for July uploads.
            m = None
            t = str(title or "")
            if len(t) >= 5 and t[2] == "-" and t[:2].isdigit() and t[3:5].isdigit():
                try: local_date = datetime(created_at.year, int(t[:2]), int(t[3:5])).date()
                except ValueError: pass
            wanted_device = "860" if (device_id and "-" in str(device_id)) else None
            cands = [e for e in entries if not e["claimed"] and e["local_date"] == local_date
                     and abs(e["duration_s"] - int(dur or 0)) <= tolerance(int(dur or 0))
                     and (wanted_device is None or e["device"] == wanted_device)]
            # Two candidates that start within a minute of each other are the
            # same moment on two devices; either is the right clock. Prefer the
            # one for the row's device when known, else the first.
            if len(cands) > 1 and max(abs((a["utc"] - b["utc"]).total_seconds()) for a in cands for b in cands) <= 60:
                cands = [c for c in cands if wanted_device is None or c["device"] == wanted_device][:1] or cands[:1]
            if len(cands) != 1:
                unmatched += 1
                print(f"  {'no match' if not cands else 'ambiguous':9}  {local_date}  {int(dur or 0)//60:4}m  {t[:52]}")
                continue
            e = cands[0]; e["claimed"] = True
            delta = e["utc"] - created_at
            tag = "would fix" if args.dry_run else "fixed"
            print(f"  {tag:9}  {created_at:%Y-%m-%d %H:%M} -> {e['utc']:%Y-%m-%d %H:%M}  ({str(delta)[:9]:>9})  {e['device']}  {t[:44]}")
            if not args.dry_run:
                if abs(delta.total_seconds()) >= 60:
                    redate_to_device_clock(session, qdrant, rid, created_at, e["utc"])
                session.execute(text("update chronos_recordings set time_is_estimated=0, time_estimate_reason=:r, device_id=coalesce(nullif(device_id,'101'), :d) where recording_id=:i"),
                                {"r": REASON, "d": e["device"], "i": rid})
                session.commit()
            fixed += 1
    print(f"\n{fixed} {'matched' if args.dry_run else 'fixed'} · {unmatched} unmatched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
