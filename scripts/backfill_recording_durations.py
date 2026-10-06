#!/usr/bin/env python3
"""Fill in recordings that are stored with duration 0 (2026-10-06).

Plaud lists a new file with duration 0 while it is still processing it, and until
this fix a known recording was never revisited. This asks Plaud for each such
recording's details and stores the length it reports. Dry run unless --apply.

    scripts/backfill_recording_durations.py            # show what would change
    scripts/backfill_recording_durations.py --apply    # write the lengths
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.engine import SessionLocal  # noqa: E402
from src.database.models import ChronosRecording  # noqa: E402
from src.database.chronos_repository import set_chronos_recording_duration  # noqa: E402
from src.plaud_client import PlaudClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the lengths (default: dry run)")
    args = parser.parse_args()

    session = SessionLocal()
    plaud = PlaudClient()
    rows = (
        session.query(ChronosRecording)
        .filter(ChronosRecording.source == "plaud")
        .filter((ChronosRecording.duration_seconds == 0) | (ChronosRecording.duration_seconds.is_(None)))
        .order_by(ChronosRecording.created_at.desc())
        .all()
    )
    print(f"{len(rows)} Plaud recordings without a length")
    filled = missing = 0
    for rec in rows:
        rid = str(rec.recording_id)
        try:
            details = plaud.get_recording(rid)
        except Exception as exc:
            print(f"  {rid[:20]}  Plaud has no details yet: {str(exc)[:80]}")
            missing += 1
            continue
        seconds = int((details or {}).get("duration") or 0) // 1000
        if seconds <= 0:
            print(f"  {rid[:20]}  Plaud still reports 0")
            missing += 1
            continue
        if args.apply:
            set_chronos_recording_duration(session, rid, seconds)
        print(f"  {rid[:20]}  {seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}{'' if args.apply else '  (dry run)'}")
        filled += 1
        time.sleep(0.5)
    print(f"{'filled' if args.apply else 'would fill'} {filled}, still unknown {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
