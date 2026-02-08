#!/usr/bin/env python3
"""
Recover recordings that have audio but no transcript.

Downloads audio from Plaud's presigned S3 URL, then processes directly
through Gemini's audio understanding capabilities (bypassing Plaud's
transcription entirely).

Usage:
    python scripts/recover_audio.py
    python scripts/recover_audio.py --dry-run
"""

import sys
import os
import json
import uuid
import time
import tempfile
import argparse
import requests
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from src.config import get_settings
from src.plaud_client import PlaudClient
from src.database.engine import SessionLocal
from src.database.models import ChronosRecording, ChronosEvent as ChronosEventModel
from src.database.chronos_repository import (
    mark_chronos_recording_status,
    add_chronos_events,
    set_chronos_recording_transcript,
    delete_chronos_events_by_recording,
)
from src.chronos.engine import ChronosEngine, GeminiEventOutput


def download_audio(url: str, dest_path: str) -> bool:
    """Download audio from presigned URL."""
    print(f"      📥 Downloading audio...")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"      ✅ Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"      ❌ Download failed: {e}")
        return False


def get_recording_date_str(rec):
    """Extract date string."""
    if rec.created_at:
        try:
            if isinstance(rec.created_at, str):
                return rec.created_at[:10]
            return rec.created_at.strftime("%Y-%m-%d")
        except Exception:
            pass
    return "2025-01-01"


def main():
    parser = argparse.ArgumentParser(description="Recover audio-only recordings")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = SessionLocal()
    plaud = PlaudClient()
    engine = ChronosEngine()

    # Find recordings that failed with "No transcript" but might have audio
    recordings = (
        db.query(ChronosRecording)
        .filter(
            ChronosRecording.processing_status == "failed",
            ChronosRecording.error_message.like("%No transcript%"),
            ChronosRecording.duration_seconds > 5,  # Skip tiny ones
        )
        .all()
    )

    if not recordings:
        print("No recoverable audio recordings found.")
        db.close()
        return

    print("=" * 70)
    print(
        f"🎵 AUDIO RECOVERY — {len(recordings)} recordings with audio but no transcript"
    )
    print("=" * 70)

    for rec in recordings:
        dur = rec.duration_seconds or 0
        print(f"\n  [{rec.recording_id[:12]}...] {dur}s ({dur // 60}m)")

        # Get presigned URL from Plaud
        try:
            details = plaud.get_recording(rec.recording_id)
            presigned_url = details.get("presigned_url")
            start_at = details.get("start_at", "")

            if not presigned_url:
                print(f"      ❌ No audio URL available from Plaud")
                continue

            print(f"      🔗 Audio URL found!")
            print(f"      📅 Recording started at: {start_at}")

        except Exception as e:
            print(f"      ❌ Plaud API error: {e}")
            continue

        if args.dry_run:
            print(f"      [DRY RUN] Would download and process audio")
            continue

        # Download audio to temp file
        recording_date = get_recording_date_str(rec)

        # Use start_at from Plaud if available for more accurate dating
        if start_at:
            try:
                recording_date = start_at[:10]
            except Exception:
                pass

        # Create persistent audio directory
        audio_dir = Path("data/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{rec.recording_id}.mp3"

        if not download_audio(presigned_url, str(audio_path)):
            continue

        # Process through Gemini's audio understanding
        print(f"      🤖 Processing {dur // 60}m audio through Gemini...")
        print(f"      📅 Recording date: {recording_date}")

        mark_chronos_recording_status(
            db, rec.recording_id, "processing", error_message=None
        )

        try:
            output = engine.process_audio(
                audio_path=str(audio_path),
                recording_id=rec.recording_id,
                recording_date=recording_date,
                max_retries=3,
            )

            if output and output.events:
                # Store events
                delete_chronos_events_by_recording(db, rec.recording_id)
                db_events = [
                    ChronosEventModel(
                        event_id=str(uuid.uuid4()),
                        recording_id=e.recording_id,
                        start_ts=e.start_ts,
                        end_ts=e.end_ts,
                        day_of_week=(
                            e.day_of_week.value
                            if hasattr(e.day_of_week, "value")
                            else str(e.day_of_week)
                        ),
                        hour_of_day=e.hour_of_day,
                        clean_text=e.clean_text,
                        category=(
                            e.category.value
                            if hasattr(e.category, "value")
                            else str(e.category)
                        ),
                        sentiment=e.sentiment,
                        keywords=e.keywords,
                        speaker=(
                            e.speaker.value
                            if hasattr(e.speaker, "value")
                            else str(e.speaker)
                        ),
                        raw_transcript_snippet=getattr(
                            e, "raw_transcript_snippet", None
                        ),
                        gemini_reasoning=getattr(e, "gemini_reasoning", None),
                    )
                    for e in output.events
                ]
                add_chronos_events(db, db_events)

                # Update recording with audio path
                rec.local_audio_path = str(audio_path)
                db.commit()

                mark_chronos_recording_status(
                    db, rec.recording_id, "completed", error_message=None
                )
                print(f"      ✅ RECOVERED: {len(output.events)} events from audio!")

            else:
                print(f"      ❌ Gemini extracted no events from audio")
                mark_chronos_recording_status(
                    db,
                    rec.recording_id,
                    "failed",
                    error_message="Gemini extracted no events from audio processing",
                )

        except Exception as e:
            print(f"      ❌ Processing failed: {e}")
            mark_chronos_recording_status(
                db,
                rec.recording_id,
                "failed",
                error_message=f"Audio processing error: {str(e)[:200]}",
            )

    # Final status
    print()
    print("=" * 70)
    remaining = (
        db.query(ChronosRecording)
        .filter(ChronosRecording.processing_status.in_(["failed", "pending"]))
        .count()
    )
    total = db.query(ChronosRecording).count()
    completed = (
        db.query(ChronosRecording)
        .filter(ChronosRecording.processing_status == "completed")
        .count()
    )
    total_events = db.query(ChronosEventModel).count()
    print(
        f"📊 Status: {completed}/{total} completed | {remaining} remaining | {total_events} events"
    )
    print("=" * 70)

    db.close()


if __name__ == "__main__":
    main()
