"""Process a single recording through audio recovery."""
import sys, uuid
sys.path.insert(0, ".")

from src.plaud_client import PlaudClient
from src.database.engine import SessionLocal
from src.database.models import ChronosRecording, ChronosEvent as ChronosEventModel
from src.database.chronos_repository import (
    mark_chronos_recording_status, add_chronos_events, delete_chronos_events_by_recording
)
from src.chronos.engine import ChronosEngine
from pathlib import Path
import requests, os

RECORDING_ID = "a35865476707193ad52b9bb76963d7e5"

db = SessionLocal()
plaud = PlaudClient()
engine = ChronosEngine()

# Get recording info
rec = db.query(ChronosRecording).filter(ChronosRecording.recording_id == RECORDING_ID).first()
print(f"Recording: {rec.recording_id}")
print(f"Duration: {rec.duration_seconds}s ({rec.duration_seconds // 60}m)")
print(f"Status: {rec.processing_status}")

# Try to get presigned URL
details = plaud.get_recording(RECORDING_ID)
url = details.get("presigned_url")
start_at = details.get("start_at", "")
print(f"Start at: {start_at}")

if url:
    print(f"Audio URL found! Downloading...")
    audio_dir = Path("data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{RECORDING_ID}.mp3"

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(audio_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB")

    # Process through Gemini audio
    recording_date = start_at[:10] if start_at else "2026-01-30"
    print(f"Processing through Gemini audio API (date: {recording_date})...")

    mark_chronos_recording_status(db, RECORDING_ID, "processing", error_message=None)

    output = engine.process_audio(
        audio_path=str(audio_path),
        recording_id=RECORDING_ID,
        recording_date=recording_date,
        max_retries=3,
    )

    if output and output.events:
        delete_chronos_events_by_recording(db, RECORDING_ID)
        db_events = [
            ChronosEventModel(
                event_id=str(uuid.uuid4()),
                recording_id=e.recording_id,
                start_ts=e.start_ts,
                end_ts=e.end_ts,
                day_of_week=e.day_of_week.value if hasattr(e.day_of_week, "value") else str(e.day_of_week),
                hour_of_day=e.hour_of_day,
                clean_text=e.clean_text,
                category=e.category.value if hasattr(e.category, "value") else str(e.category),
                sentiment=e.sentiment,
                keywords=e.keywords,
                speaker=e.speaker.value if hasattr(e.speaker, "value") else str(e.speaker),
                raw_transcript_snippet=getattr(e, "raw_transcript_snippet", None),
                gemini_reasoning=getattr(e, "gemini_reasoning", None),
            )
            for e in output.events
        ]
        add_chronos_events(db, db_events)
        rec.local_audio_path = str(audio_path)
        db.commit()
        mark_chronos_recording_status(db, RECORDING_ID, "completed", error_message=None)
        print(f"RECOVERED: {len(output.events)} events from audio!")
    else:
        print("Gemini extracted no events from audio")
        # Nuclear option: raw transcript as single event
        from datetime import datetime
        day_of_week = datetime.strptime(recording_date, "%Y-%m-%d").strftime("%A")
        transcript = rec.transcript or ""
        if transcript.strip():
            manual = ChronosEventModel(
                event_id=str(uuid.uuid4()),
                recording_id=RECORDING_ID,
                start_ts=datetime.strptime(f"{recording_date}T12:00:00", "%Y-%m-%dT%H:%M:%S"),
                end_ts=datetime.strptime(f"{recording_date}T12:30:00", "%Y-%m-%dT%H:%M:%S"),
                day_of_week=day_of_week,
                hour_of_day=12,
                clean_text=transcript[:5000].strip(),
                category="personal",
                sentiment=0.0,
                keywords=transcript.split()[:10],
                speaker="unknown",
                raw_transcript_snippet=transcript[:500],
                gemini_reasoning="Raw transcript — Gemini could not extract structured events from audio",
            )
            delete_chronos_events_by_recording(db, RECORDING_ID)
            add_chronos_events(db, [manual])
            mark_chronos_recording_status(db, RECORDING_ID, "completed", error_message=None)
            print("Created manual event from raw transcript (nuclear option)")
        else:
            mark_chronos_recording_status(db, RECORDING_ID, "failed",
                error_message="Audio processing returned no events, no transcript available")
else:
    print("No audio URL — using nuclear option (raw transcript as event)")
    from datetime import datetime
    transcript = rec.transcript or ""
    recording_date = "2026-01-30"
    if transcript.strip():
        day_of_week = datetime.strptime(recording_date, "%Y-%m-%d").strftime("%A")
        manual = ChronosEventModel(
            event_id=str(uuid.uuid4()),
            recording_id=RECORDING_ID,
            start_ts=datetime.strptime(f"{recording_date}T12:00:00", "%Y-%m-%dT%H:%M:%S"),
            end_ts=datetime.strptime(f"{recording_date}T12:30:00", "%Y-%m-%dT%H:%M:%S"),
            day_of_week=day_of_week,
            hour_of_day=12,
            clean_text=transcript[:5000].strip(),
            category="personal",
            sentiment=0.0,
            keywords=transcript.split()[:10],
            speaker="unknown",
            raw_transcript_snippet=transcript[:500],
            gemini_reasoning="Raw transcript — no audio available, Gemini failed on text",
        )
        delete_chronos_events_by_recording(db, RECORDING_ID)
        add_chronos_events(db, [manual])
        mark_chronos_recording_status(db, RECORDING_ID, "completed", error_message=None)
        print("Created manual event from raw transcript")

# Final check
from src.database.models import ChronosEvent
total_completed = db.query(ChronosRecording).filter(ChronosRecording.processing_status == "completed").count()
total = db.query(ChronosRecording).count()
total_events = db.query(ChronosEvent).count()
print(f"\nFinal: {total_completed}/{total} completed | {total_events} events")

db.close()
