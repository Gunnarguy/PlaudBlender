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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import SessionLocal  # noqa: E402
from src.database.chronos_repository import (  # noqa: E402
    get_chronos_recording,
    set_chronos_recording_transcript,
    upsert_chronos_recording,
)
from src.plaud_v4 import NotLoggedIn, PlaudV4Client, PlaudV4Error, classic_id, device_code  # noqa: E402

SOURCE = "plaud_v4"
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


def sync(client: PlaudV4Client, *, limit: int | None, dry_run: bool, refresh_complete: bool) -> int:
    created = updated = skipped = failed = 0
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
    print(f"\n{seen} listed · {created} created · {updated} updated · {skipped} already complete · {failed} failed · {elapsed:.0f}s")
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
    try:
        return sync(client, limit=args.limit, dry_run=args.dry_run, refresh_complete=args.refresh_complete)
    except NotLoggedIn as error:
        print(f"{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
