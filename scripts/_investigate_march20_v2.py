import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.engine import get_db
from src.database.models import ChronosRecording, ChronosEvent
from sqlalchemy import select


def check_db():
    print("=== Checking DB ===")
    session_gen = get_db()
    session = next(session_gen)
    try:
        # Check recordings
        recordings = (
            session.execute(
                select(ChronosRecording)
                .order_by(ChronosRecording.created_at.desc())
                .limit(15)
            )
            .scalars()
            .all()
        )
        print(f"Top {len(recordings)} recent recordings:")
        for r in recordings:
            print(
                f" - {r.recording_id}: {r.created_at} - {r.title} (Status: {r.processing_status})"
            )

        # Check events
        events = (
            session.execute(
                select(ChronosEvent).order_by(ChronosEvent.start_ts.desc()).limit(15)
            )
            .scalars()
            .all()
        )
        print(f"\nTop {len(events)} recent events:")
        for e in events:
            print(f" - {e.id} (Rec: {e.recording_id}): {e.start_ts} - {e.summary[:50]}")
    finally:
        session_gen.close()


if __name__ == "__main__":
    check_db()
