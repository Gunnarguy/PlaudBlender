#!/usr/bin/env python3
"""Process recordings that already have cached transcripts in the DB.

Bypasses the Plaud API fetch — useful when OAuth tokens are expired but
transcripts were already ingested.
"""
import sys
import uuid
import time
from pathlib import Path

if sys.version_info >= (3, 11):
    sys.set_int_max_str_digits(0)

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.database import SessionLocal, init_db
from src.database.chronos_repository import (
    mark_chronos_recording_status,
    add_chronos_events,
    delete_chronos_events_by_recording,
)
from src.chronos.transcript_processor import TranscriptProcessor
from src.database.models import ChronosEvent as ChronosEventModel

init_db()


def process_cached_recordings():
    # Override model to avoid exhausted gemini-3-flash quota
    # Each model has its own daily free-tier limit
    import os

    fallback_model = os.environ.get("OVERRIDE_MODEL", "")
    if fallback_model:
        print(f"Using override model: {fallback_model}")
        os.environ["CHRONOS_CLEANING_MODEL"] = fallback_model

    session = SessionLocal()
    processor = TranscriptProcessor(db_session=session)

    # Find recordings that have transcripts cached but are in failed/pending status
    rows = session.execute(
        text(
            """
        SELECT recording_id, transcript, created_at, plaud_ai_summary, plaud_extracted_data
        FROM chronos_recordings
        WHERE processing_status IN ('failed', 'pending')
          AND transcript IS NOT NULL
          AND length(transcript) > 100
        ORDER BY created_at DESC
        """
        )
    ).fetchall()

    if not rows:
        print("No cached-transcript recordings to process.")
        return

    print(f"Found {len(rows)} recordings with cached transcripts\n")

    success = 0
    for i, row in enumerate(rows):
        rec_id = row[0]
        transcript = row[1]
        created_at = row[2]
        ai_summary = row[3]
        extracted_data = row[4]

        recording_date = ""
        if created_at:
            recording_date = str(created_at)[:10]

        # Build plaud context if available
        plaud_context = None
        parts = []
        if ai_summary:
            parts.append(f"AI Summary: {ai_summary}")
        if extracted_data:
            import json

            if isinstance(extracted_data, str):
                parts.append(f"Extracted Data: {extracted_data[:2000]}")
            elif isinstance(extracted_data, dict):
                parts.append(
                    f"Extracted Data: {json.dumps(extracted_data, default=str)[:2000]}"
                )
        if parts:
            plaud_context = "\n\n".join(parts)

        print(
            f"[{i+1}/{len(rows)}] {rec_id[:24]}... ({len(transcript):,} chars, date={recording_date})"
        )
        t0 = time.time()

        # Delete any existing failed events
        delete_chronos_events_by_recording(session, rec_id)

        # Process via Gemini/OpenAI (bypasses Plaud API entirely)
        output = processor.process_transcript_text(
            transcript,
            rec_id,
            recording_date=recording_date,
            plaud_context=plaud_context,
        )

        if not output or not output.events:
            err = processor._last_processing_error or "No events extracted"
            print(f"   ❌ Failed: {err}")
            mark_chronos_recording_status(session, rec_id, "failed", error_message=err)
            continue

        # Store events (same logic as process_recording_id lines 1150-1167)
        db_events = [
            ChronosEventModel(
                event_id=str(uuid.uuid4()),
                recording_id=e.recording_id,
                start_ts=e.start_ts,
                end_ts=e.end_ts,
                day_of_week=str(getattr(e.day_of_week, "value", e.day_of_week)),
                hour_of_day=e.hour_of_day,
                clean_text=e.clean_text,
                category=str(getattr(e.category, "value", e.category)),
                sentiment=e.sentiment,
                keywords=e.keywords,
                speaker=str(getattr(e.speaker, "value", e.speaker)),
                raw_transcript_snippet=e.raw_transcript_snippet,
                gemini_reasoning=e.gemini_reasoning,
            )
            for e in output.events
        ]
        add_chronos_events(session, db_events)
        mark_chronos_recording_status(session, rec_id, "completed", error_message=None)

        elapsed = time.time() - t0
        print(f"   ✅ {len(output.events)} events in {elapsed:.1f}s")
        success += 1

    session.close()
    print(f"\nDone: {success}/{len(rows)} processed successfully")


if __name__ == "__main__":
    process_cached_recordings()
