from datetime import datetime
from app_v2.services.data_service import ChronosDataService
from src.database.engine import get_session
from src.database.models import ChronosRecording, ChronosEvent
from sqlalchemy import select

def check_db():
    print("=== Checking DB ===")
    with get_session() as session:
        # Check recordings
        recordings = session.execute(
            select(ChronosRecording).order_by(ChronosRecording.start_time.desc()).limit(10)
        ).scalars().all()
        print(f"Top {len(recordings)} recent recordings:")
        for r in recordings:
            print(f" - {r.id}: {r.start_time} - {r.title} (Status: {r.processing_status})")
            
        # Check events
        events = session.execute(
            select(ChronosEvent).order_by(ChronosEvent.start_time.desc()).limit(10)
        ).scalars().all()
        print(f"\nTop {len(events)} recent events:")
        for e in events:
            print(f" - {e.id} (Rec: {e.recording_id}): {e.start_time} - {e.summary[:50]}")

if __name__ == "__main__":
    check_db()
