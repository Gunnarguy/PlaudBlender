"""Check current state of the March 17 recording."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
from src.database.engine import init_db, SessionLocal
from src.database.models import ChronosRecording, ChronosEvent

init_db()
s = SessionLocal()

rec_id = "notion:32749a74-d54f-81df-a2b1-f3c745f64c37"
rec = s.query(ChronosRecording).filter_by(recording_id=rec_id).first()
if rec:
    print(f"Recording:")
    print(f"  created_at:       {rec.created_at}")
    print(f"  duration_seconds: {rec.duration_seconds}")
    print(f"  title:            {rec.title}")
    print(f"  source:           {rec.source}")

    # Check events
    events = (
        s.query(ChronosEvent)
        .filter_by(recording_id=rec_id)
        .order_by(ChronosEvent.start_ts)
        .all()
    )
    print(f"\nEvents ({len(events)}):")
    for ev in events[:5]:
        print(
            f"  start_ts={ev.start_ts}  end_ts={ev.end_ts}  hour={ev.hour_of_day}  cat={ev.category}"
        )
        print(f"    text: {str(ev.clean_text)[:80]}...")
    if len(events) > 5:
        print(f"  ... +{len(events)-5} more")
        last = events[-1]
        print(
            f"  LAST: start_ts={last.start_ts}  end_ts={last.end_ts}  hour={last.hour_of_day}"
        )
else:
    print("Recording NOT FOUND")

s.close()
