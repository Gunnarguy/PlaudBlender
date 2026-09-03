#!/usr/bin/env python3
"""Ingest recordings that exist only as local files -- a second recorder that
never reached Plaud's cloud -- as first-class rows.

    venv/bin/python scripts/plaud_one_local_ingest.py \
        --manifest /mnt/ssd/one_local/manifest.json \
        --transcripts /mnt/ssd/one_local/transcripts \
        --audio-dir /mnt/ssd/one_local \
        --app-clock /tmp/app_clock.json --device 860 [--dry-run]

Each manifest entry is one WAV that was transcoded to Opus: stem, calendar
date from the filename, duration, sha256. A transcript JSON of the same stem
(on-device SpeechAnalyzer: full_text plus timed segments) supplies the text.
The owner's app-clock log supplies the true start: the entry on the same
date whose duration is within tolerance, claimed once. Without a log match
the row is still created, with its date at midday and flagged estimated.

Rows get a stable id derived from the stem, the given device code, source
"one_local", the transcript text, a TRANSCRIPT artifact holding the timed
segments, and the Opus file as local_audio_path. Re-running is idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.database import SessionLocal, init_db  # noqa: E402
from src.database.chronos_repository import get_chronos_recording, set_chronos_recording_transcript, upsert_chronos_recording  # noqa: E402
from src.database.models import ChronosRecordingArtifact  # noqa: E402

TZ = ZoneInfo("America/Los_Angeles")


def stable_id(stem: str) -> str:
    return "one:" + hashlib.sha1(stem.encode()).hexdigest()[:28]


def tolerance(duration_s: int) -> int:
    return max(180, int(duration_s * 0.25)) if duration_s >= 3600 else 180


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--transcripts", required=True)
    ap.add_argument("--audio-dir", required=True); ap.add_argument("--app-clock", required=True)
    ap.add_argument("--device", default="860"); ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    clock = json.loads(Path(args.app_clock).read_text())
    key = "one" if args.device == "860" else "ng1"
    entries = []
    for e in clock.get(key, []):
        local = datetime.strptime(e["start_local"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
        entries.append({"date": local.date().isoformat(), "utc": local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
                        "duration_s": int(e["duration_s"]), "claimed": False})

    init_db()
    created = updated = skipped = unclocked = 0
    root = Path(__file__).resolve().parent.parent / "data" / "artifacts"
    with SessionLocal() as session:
        # entries already claimed by rows ingested earlier must stay claimed
        for (reason,) in session.execute(text("select time_estimate_reason from chronos_recordings where source='one_local'")).fetchall():
            pass
        for stem, m in sorted(manifest.items()):
            rid = stable_id(stem)
            tpath = Path(args.transcripts) / f"{stem}.json"
            if not tpath.exists():
                skipped += 1
                print(f"  no transcript yet  {stem[:60]}")
                continue
            t = json.loads(tpath.read_text())
            segments = t.get("segments") or []
            full = (t.get("full_text") or " ".join(s.get("text", "") for s in segments)).strip()
            dur = int(m.get("duration_s") or (segments[-1]["end"] if segments else 0))
            date = m.get("date")
            cands = [e for e in entries if not e["claimed"] and e["date"] == date and abs(e["duration_s"] - dur) <= tolerance(dur)]
            if len(cands) > 1 and max(abs((a["utc"] - b["utc"]).total_seconds()) for a in cands for b in cands) <= 60:
                cands = cands[:1]
            if len(cands) == 1:
                cands[0]["claimed"] = True
                start_utc, estimated, reason = cands[0]["utc"], False, "app clock: owner's recording-list log"
            else:
                noon = datetime.fromisoformat(date + "T12:00:00").replace(tzinfo=TZ) if date else datetime.utcnow().replace(tzinfo=TZ)
                start_utc, estimated, reason = noon.astimezone(ZoneInfo("UTC")).replace(tzinfo=None), True, "local file: date from filename, no app-clock match"
                unclocked += 1
            audio = str(Path(args.audio_dir) / m["ogg"])
            existing = get_chronos_recording(session, rid)
            tag = ("update" if existing else "create") + ("" if not estimated else " (est)")
            print(f"  {tag:13} {start_utc:%Y-%m-%d %H:%M}Z {dur // 60:4}m  {len(segments):4} segs  {stem[:50]}")
            if args.dry_run:
                continue
            upsert_chronos_recording(session, recording_id=rid, title=stem, created_at=start_utc, duration_seconds=dur,
                                     local_audio_path=audio, source="one_local", device_id=args.device,
                                     checksum=m.get("ogg_sha256"), time_is_estimated=estimated, time_estimate_reason=reason, force_time=True)
            if full:
                set_chronos_recording_transcript(session, rid, full)
            apath = root / rid / "TRANSCRIPT.json"
            apath.parent.mkdir(parents=True, exist_ok=True)
            apath.write_text(json.dumps(segments))
            session.merge(ChronosRecordingArtifact(recording_id=rid, object_type="TRANSCRIPT", content_id=None, mime_type="application/json",
                                                   path=str(apath), size_bytes=apath.stat().st_size, fetched_at=datetime.utcnow()))
            session.commit()
            if existing: updated += 1
            else: created += 1
    print(f"\n{created} created · {updated} updated · {unclocked} without app-clock match · {skipped} awaiting transcript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
