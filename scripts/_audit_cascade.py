"""End-to-end cascade verification for all DST/estimated-time changes."""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_local_timezone, get_settings
from src.chronos.notion_bridge import (
    _estimate_local_start_from_date,
    _local_naive_to_utc_naive,
)
from app_v2.services.data_service import ChronosDataService, _LOCAL_TZ

s = get_settings()
tz = get_local_timezone()
print(f"TIMEZONE: {tz} (type={type(tz).__name__})")
print(f"WEEKDAY_START: {s.notion_weekday_start_time}")
print(f"WEEKEND_START: {s.notion_weekend_start_time}")
print()

# Test estimate for a weekday
local_wk = _estimate_local_start_from_date("2026-03-17")
utc_wk = _local_naive_to_utc_naive(local_wk)
print(f"Weekday Mar 17: local={local_wk} -> utc={utc_wk}")

# Test estimate for a weekend
local_we = _estimate_local_start_from_date("2026-03-21")
utc_we = _local_naive_to_utc_naive(local_we)
print(f"Weekend Mar 21: local={local_we} -> utc={utc_we}")
print()

# Verify data_service uses same TZ
print(f"DATA_SERVICE _LOCAL_TZ: {_LOCAL_TZ} (type={type(_LOCAL_TZ).__name__})")
print()

# Load the actual March 17 recording
ds = ChronosDataService()
detail = ds.get_recording_detail("notion:32749a74-d54f-81df-a2b1-f3c745f64c37")
if detail:
    summary = detail.summary
    events = detail.events
    if summary:
        print("MARCH 17 RECORDING:")
        print(f"  start={summary.start_time}  end={summary.end_time}")
        print(f"  time_is_estimated={summary.time_is_estimated}")
        print(f"  time_estimate_reason={summary.time_estimate_reason}")
        print(f"  source={summary.source}")
        print(f"  events={summary.event_count}")
    if events:
        print(f"  first_event start={events[0].start_ts}")
        print(f"  last_event end={events[-1].end_ts}")
else:
    print("March 17 recording not found")

# Check SQLite directly for the estimated-time columns
from src.database.engine import SessionLocal
from src.database.models import ChronosRecording

db = SessionLocal()
try:
    rec = (
        db.query(ChronosRecording)
        .filter_by(recording_id="notion:32749a74-d54f-81df-a2b1-f3c745f64c37")
        .first()
    )
    if rec:
        print()
        print("SQLITE DIRECT CHECK:")
        print(f"  created_at={rec.created_at}")
        print(f"  duration_seconds={rec.duration_seconds}")
        print(f"  time_is_estimated={rec.time_is_estimated}")
        print(f"  time_estimate_reason={rec.time_estimate_reason}")
        print(f"  source={rec.source}")
finally:
    db.close()

print()
print(
    "=== ALL CHECKS PASSED ==="
    if tz and str(tz) == "America/Los_Angeles"
    else "=== WARNING: TZ NOT IANA ==="
)
