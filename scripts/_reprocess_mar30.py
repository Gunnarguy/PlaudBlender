"""Force-reprocess March 30 and March 31 recordings through Gemini."""

import uuid
import sys

sys.path.insert(0, ".")

from src.database.engine import SessionLocal
from src.database.models import ChronosRecording, ChronosEvent as ChronosEventModel
from src.chronos.transcript_processor import TranscriptProcessor
from src.database.chronos_repository import mark_chronos_recording_status


def process_one(db, tp, partial_id, label):
    rec = (
        db.query(ChronosRecording)
        .filter(ChronosRecording.recording_id.like(f"{partial_id}%"))
        .first()
    )
    if not rec:
        print(f"[{label}] Recording not found")
        return False

    print(f"\n{'='*60}")
    print(f"[{label}] {rec.recording_id}")
    print(f"  Date: {rec.created_at}")
    print(f"  Transcript: {len(rec.transcript or '')} chars")
    print(f"  Current status: {rec.processing_status}")

    existing = (
        db.query(ChronosEventModel)
        .filter(ChronosEventModel.recording_id == rec.recording_id)
        .count()
    )
    if existing > 0:
        print(f"  Already has {existing} events — skipping")
        return True

    output = tp.process_transcript_text(
        rec.transcript,
        rec.recording_id,
        recording_date=str(rec.created_at.date()),
        verbose=True,
        max_retries=5,
    )

    if not output or not output.events:
        print(f"[{label}] FAILED — Gemini returned no events")
        mark_chronos_recording_status(
            db, rec.recording_id, "failed", error_message="Gemini returned no events"
        )
        return False

    print(f"[{label}] SUCCESS: {len(output.events)} events extracted")

    for e in output.events:
        dow = (
            e.day_of_week.value
            if hasattr(e.day_of_week, "value")
            else str(e.day_of_week)
        )
        cat = e.category.value if hasattr(e.category, "value") else str(e.category)
        spk = e.speaker.value if hasattr(e.speaker, "value") else str(e.speaker)
        kw = e.keywords if isinstance(e.keywords, str) else ",".join(e.keywords or [])
        db_ev = ChronosEventModel(
            event_id=str(uuid.uuid4()),
            recording_id=rec.recording_id,
            start_ts=e.start_ts,
            end_ts=e.end_ts,
            day_of_week=dow,
            hour_of_day=e.hour_of_day,
            clean_text=e.clean_text,
            category=cat,
            category_confidence=e.category_confidence,
            sentiment=e.sentiment,
            keywords=kw,
            speaker=spk,
            raw_transcript_snippet=getattr(e, "raw_transcript_snippet", None),
            gemini_reasoning=getattr(e, "gemini_reasoning", None),
        )
        db.add(db_ev)

    db.commit()
    mark_chronos_recording_status(db, rec.recording_id, "completed")
    print(f"[{label}] Saved {len(output.events)} events, status=completed")
    return True


def main():
    db = SessionLocal()
    tp = TranscriptProcessor(db)

    targets = [
        ("211c6d7d6b2f9eb6295b", "Mar 30"),
        ("30809412a8af7d3af1ac", "Mar 31"),
    ]

    for partial_id, label in targets:
        try:
            process_one(db, tp, partial_id, label)
        except Exception as exc:
            print(f"[{label}] ERROR: {exc}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
