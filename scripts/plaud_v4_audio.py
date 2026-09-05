#!/usr/bin/env python3
"""Download each recording's master audio from Plaud 4.0 into the raw audio dir.

    venv/bin/python scripts/plaud_v4_audio.py --limit 20     # newest 20
    venv/bin/python scripts/plaud_v4_audio.py                # everything missing
    venv/bin/python scripts/plaud_v4_audio.py --dry-run

Plaud devices record Opus in an Ogg container (~34 kbps); a few older or
sample files are MP3. Whatever the AUDIO object's mime type says is what is
kept -- this is the master, and the web app's WAV export is a transcode of
it, so nothing is gained by converting. Each file lands in
CHRONOS_RAW_AUDIO_DIR as <recording_id>.<ext>, is checksummed, and the row's
local_audio_path and checksum are set so the existing ingest, checksum and
audio-serving paths see it. Rows that already have a present file are
skipped, so repeated runs only fetch what is new.

Presigned URLs live 24 h and are signed for GET only. Requires a session
from scripts/plaud_v4_login.py.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.database import SessionLocal  # noqa: E402
from src.database.chronos_repository import get_chronos_recording, upsert_chronos_recording  # noqa: E402
from src.plaud_v4 import NotLoggedIn, PlaudV4Client, PlaudV4Error, classic_id, device_code  # noqa: E402

EXT_BY_MIME = {"audio/ogg": ".ogg", "audio/opus": ".opus", "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/wav": ".wav", "audio/x-wav": ".wav"}
SKIP_DEVICES = {"969"}  # Plaud's onboarding samples
PACE_SECONDS = 0.25


def download(url: str, target: Path) -> str:
    """Stream to a temp file, return sha256. Atomic rename on success."""
    tmp = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    fh.write(chunk)
                    digest.update(chunk)
    tmp.replace(target)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="stop after this many downloads (newest first)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-seconds", type=int, default=30, help="skip recordings shorter than this")
    args = parser.parse_args()

    client = PlaudV4Client()
    if not client.has_session:
        print("No Plaud 4.0 session. Run scripts/plaud_v4_login.py first.", file=sys.stderr)
        return 2

    raw_dir = Path(get_settings().chronos_raw_audio_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    total_bytes = 0
    started = time.monotonic()
    with SessionLocal() as session:
        for item in client.iter_recordings():
            if args.limit and done >= args.limit:
                break
            rid = classic_id(item["file_id"])
            if (item.get("duration_ms") or 0) < args.min_seconds * 1000:
                skipped += 1
                continue
            rec = get_chronos_recording(session, rid)
            if rec and rec.local_audio_path and Path(rec.local_audio_path).exists():
                skipped += 1
                continue
            try:
                detail = client.file_detail(item["file_id"])
                meta = detail.get("meta") or {}
                if device_code(meta.get("scene_source")) in SKIP_DEVICES:
                    skipped += 1
                    continue
                audio = client.objects_by_type(detail).get("AUDIO") or {}
                url = audio.get("content_url")
                if not url:
                    urls = client._request("POST", "/file-app/v4/files/content/url", json={"content_ids": [audio.get("content_id")]})
                    url_map = (urls.get("data") or {}).get("url_map") or {}
                    entry = next(iter(url_map.values()), None)
                    url = entry.get("url") if isinstance(entry, dict) else entry
                if not url:
                    failed += 1
                    print(f"  no audio url  {item.get('name', '')[:56]}", file=sys.stderr)
                    continue
                ext = EXT_BY_MIME.get(str(audio.get("mime_type") or "").lower(), ".bin")
                target = raw_dir / f"{rid}{ext}"
                size = audio.get("object_size") or 0
                if args.dry_run:
                    print(f"  would fetch  {size / 1e6:6.1f} MB  {ext}  {item.get('name', '')[:52]}")
                    done += 1
                    continue
                checksum = download(url, target)
                total_bytes += target.stat().st_size
                if rec:
                    upsert_chronos_recording(
                        session,
                        recording_id=rid,
                        title=rec.title,
                        created_at=rec.created_at,
                        duration_seconds=rec.duration_seconds,
                        local_audio_path=str(target),
                        source=rec.source,
                        device_id=rec.device_id,
                        checksum=checksum,
                    )
                done += 1
                if ext == ".opus":
                    # The recorder uploads bare CBR Opus frames with no container; wrap them
                    # so the file is playable everywhere, losslessly.
                    try:
                        from scripts.opus_raw_to_ogg import convert as _wrap
                        wrapped = target.with_suffix(".ogg")
                        ok, why = _wrap(target, wrapped)
                        if ok:
                            target.unlink(); target = wrapped; ext = ".ogg"
                        else:
                            print(f"  (left raw: {why})")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  (wrap skipped: {type(exc).__name__})")
                print(f"  {target.stat().st_size / 1e6:6.1f} MB  {ext}  {item.get('name', '')[:52]}")
            except NotLoggedIn:
                raise
            except (PlaudV4Error, requests.RequestException, OSError) as error:
                failed += 1
                print(f"  FAILED  {item.get('name', '')[:52]}: {error}", file=sys.stderr)
            time.sleep(PACE_SECONDS)

    elapsed = time.monotonic() - started
    print(f"\n{done} downloaded ({total_bytes / 1e9:.2f} GB) · {skipped} skipped · {failed} failed · {elapsed:.0f}s")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
