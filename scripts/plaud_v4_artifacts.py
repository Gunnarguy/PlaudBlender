#!/usr/bin/env python3
"""Fetch every artifact Plaud 4.0 holds for each recording, verbatim.

    venv/bin/python scripts/plaud_v4_artifacts.py            # everything missing
    venv/bin/python scripts/plaud_v4_artifacts.py --limit 20

Per recording the 4.0 API serves up to: TRANSCRIPT (per-line timing and
speakers), POLISHED_TRANSCRIPT, SUMMARY, SUMMARY_BETA, OUTLINE (timed topic
bands), MARK_MEMO (moments flagged by the device button). The sync keeps a
flattened transcript and the summary on the row; this keeps the rest, as
files under data/artifacts/<recording_id>/<TYPE>.<ext>, with a row in
chronos_recording_artifacts saying what each is. AUDIO is the audio
script's job and is skipped here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.database import SessionLocal, init_db  # noqa: E402
from src.database.models import ChronosRecordingArtifact  # noqa: E402
from src.plaud_v4 import NotLoggedIn, PlaudV4Client, PlaudV4Error, classic_id  # noqa: E402

SKIP = {"AUDIO", "AUDIO_TRANSCODED"}
PACE = 0.1
FULL_SWEEP_SECONDS = 24 * 3600


def state_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "cache" / "v4_artifacts_seen.json"


def load_state() -> dict:
    """{"last_full": epoch, "seen": {recording_id: version_ms}} from the last runs."""
    try:
        state = json.loads(state_path().read_text())
        return {"last_full": float(state.get("last_full") or 0), "seen": dict(state.get("seen") or {})}
    except (OSError, ValueError):
        return {"last_full": 0.0, "seen": {}}


def save_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def artifact_root() -> Path:
    root = Path(__file__).resolve().parent.parent / "data" / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true", help="re-fetch artifacts already on disk")
    parser.add_argument("--full", action="store_true", help="check every recording, not just changed ones")
    args = parser.parse_args()

    client = PlaudV4Client()
    if not client.has_session:
        print("No Plaud 4.0 session.", file=sys.stderr); return 2
    init_db()
    root = artifact_root()

    fetched = skipped = failed = unchanged = 0; by_type: dict[str, int] = {}; seen = 0
    started = time.monotonic()
    # Asking Plaud for the detail of all ~700 recordings every 20 minutes cost
    # ~90 s and found nothing almost every time. Only recordings whose
    # version_ms moved since their last clean check are re-read; a full sweep
    # every 24 h catches any artifact that appears without a version bump.
    state = load_state()
    full = args.full or args.refresh or time.time() - state["last_full"] >= FULL_SWEEP_SECONDS
    with SessionLocal() as session:
        have = {(r, t) for r, t in session.execute(text("select recording_id, object_type from chronos_recording_artifacts")).fetchall()}
        for item in client.iter_recordings():
            if args.limit and seen >= args.limit:
                break
            seen += 1
            rid = classic_id(item["file_id"])
            version = item.get("version_ms")
            if not full and version is not None and state["seen"].get(rid) == version:
                unchanged += 1; continue
            failed_before = failed
            try:
                detail = client.file_detail(item["file_id"])
            except PlaudV4Error as error:
                failed += 1; print(f"  FAILED detail {rid[:12]}: {error}", file=sys.stderr); continue
            for obj in detail.get("objects") or []:
                kind = str(obj.get("object_type") or "")
                if not kind or kind in SKIP or not obj.get("content_id"):
                    continue
                if (rid, kind) in have and not args.refresh:
                    skipped += 1; continue
                try:
                    body = client.content(obj["content_id"])
                except PlaudV4Error as error:
                    failed += 1; print(f"  FAILED {kind} {rid[:12]}: {error}", file=sys.stderr); continue
                ext = ".json" if body.lstrip().startswith(("[", "{")) else ".md"
                target = root / rid / f"{kind}{ext}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body)
                session.merge(ChronosRecordingArtifact(
                    recording_id=rid, object_type=kind, content_id=obj.get("content_id"),
                    mime_type=obj.get("mime_type"), path=str(target), size_bytes=len(body.encode()),
                    fetched_at=datetime.utcnow(),
                ))
                have.add((rid, kind)); fetched += 1; by_type[kind] = by_type.get(kind, 0) + 1
                time.sleep(PACE)
            session.commit()
            if failed == failed_before and version is not None:
                state["seen"][rid] = version
    if full and not args.limit:
        state["last_full"] = time.time()
    save_state(state)
    mode = "full sweep" if full else f"{unchanged} unchanged"
    print(f"\n{seen} recordings ({mode}) · {fetched} artifacts fetched {by_type} · {skipped} already held · {failed} failed · {time.monotonic() - started:.0f}s")
    return 1 if failed and not fetched else 0


if __name__ == "__main__":
    raise SystemExit(main())
