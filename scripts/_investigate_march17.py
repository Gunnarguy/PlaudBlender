"""Investigate phantom recording on March 17."""

from src.database.engine import SessionLocal, init_db
from src.database.models import ChronosRecording, ChronosEvent

init_db()
db = SessionLocal()

# Look at recordings around March 17
print("=== RECORDINGS ===")
recs = db.query(ChronosRecording).all()
for r in recs:
    created = str(r.created_at)
    if "2025-03-17" in created or "2026-03-17" in created:
        print(
            f"REC: id={str(r.recording_id)[:16]}  created={r.created_at}  title={r.title}  status={r.processing_status}"
        )

# Check events with start_ts
print("\n=== EVENTS ON MARCH 17 ===")
events = db.query(ChronosEvent).all()
march17_events = []
for e in events:
    ts = str(e.start_ts or "")
    if "2025-03-17" in ts or "2026-03-17" in ts:
        march17_events.append(e)

print(f"Total events on March 17: {len(march17_events)}")
for e in march17_events:
    print(
        f"  EVT: recording={str(e.recording_id)[:16]}  start={e.start_ts}  end={e.end_ts}  cat={e.category}"
    )

# Also check Qdrant data via data_service
print("\n=== DATA SERVICE VIEW ===")
from app_v2.services.data_service import ChronosDataService

ds = ChronosDataService()
days = ds.get_days()
for d in days:
    if "2026-03-17" in str(d.date) or "2025-03-17" in str(d.date):
        print(f"DAY: {d.date}  recordings={d.recording_count}  events={d.event_count}")
        # Get recordings for this day
        for rec in d.recordings:
            print(
                f"  REC: {rec.recording_id[:16]}  start={rec.start_time}  end={rec.end_time}  events={rec.event_count}  dur={rec.duration_seconds:.0f}s  source={rec.source}"
            )
            print(
                f"       preview={rec.preview_text[:80] if rec.preview_text else 'N/A'}"
            )

# Also look at the raw Qdrant events for this date
print("\n=== RAW QDRANT EVENTS FOR MARCH 17 ===")
all_events = ds._get_all_events()
m17_events = [e for e in all_events if "2026-03-17" in str(e.start_ts)]
print(f"Total Qdrant events on 2026-03-17: {len(m17_events)}")
for e in m17_events:
    print(
        f"  EVT: id={e.id[:12]}  rec={e.recording_id[:16]}  start={e.start_ts}  end={e.end_ts}  cat={e.category}"
    )
    print(f"       text={e.clean_text[:100] if e.clean_text else 'N/A'}")

# Check what recording_id that is
if m17_events:
    rid = m17_events[0].recording_id
    print(f"\n=== RECORDING ID: {rid} ===")
    # Check if it exists in SQLite
    from src.database.models import ChronosRecording

    cr = db.query(ChronosRecording).filter_by(recording_id=rid).first()
    print(f"  In SQLite ChronosRecording: {cr is not None}")
    if cr:
        print(
            f"  created_at={cr.created_at}  title={cr.title}  status={cr.processing_status}"
        )

db.close()
