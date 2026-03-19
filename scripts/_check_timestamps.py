"""Check transcript timestamp range."""

import os, sys, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
from src.database.engine import init_db, SessionLocal
from src.database.models import ChronosRecording

init_db()
s = SessionLocal()
rec = (
    s.query(ChronosRecording)
    .filter_by(recording_id="notion:32749a74-d54f-81df-a2b1-f3c745f64c37")
    .first()
)
transcript = str(rec.transcript or "")
timestamps = re.findall(r"\b(\d{1,2}):(\d{2}):(\d{2})\b", transcript)
secs = sorted([int(h) * 3600 + int(m) * 60 + int(ss) for h, m, ss in timestamps])
print(f"Total timestamps: {len(secs)}")
print(f"First 5 (seconds): {secs[:5]}")
print(f"Last 5 (seconds): {secs[-5:]}")
for label, val in [("Min", secs[0]), ("Max", secs[-1])]:
    h, rem = divmod(val, 3600)
    m, sec = divmod(rem, 60)
    print(f"{label}: {h}h{m}m{sec}s = {val}s")

# Show context around the max timestamp
max_h, max_rem = divmod(secs[-1], 3600)
max_m, max_s = divmod(max_rem, 60)
max_ts = f"{max_h:02d}:{max_m:02d}:{max_s:02d}"
idx = transcript.find(max_ts)
if idx >= 0:
    print(f"\nMax timestamp context:\n  ...{transcript[max(0,idx-40):idx+100]}...")

# Also check: are there timestamps > 1 hour that seem suspicious?
over_1h = [v for v in secs if v > 3600]
if over_1h:
    print(f"\nTimestamps over 1 hour: {len(over_1h)}")
    for v in over_1h[:5]:
        h, rem = divmod(v, 3600)
        m, sec = divmod(rem, 60)
        ts_str = f"{h:02d}:{m:02d}:{sec:02d}"
        idx2 = transcript.find(ts_str)
        ctx = transcript[max(0, idx2 - 20) : idx2 + 80] if idx2 >= 0 else "NOT FOUND"
        print(f"  {ts_str} ({v}s): ...{ctx}...")

s.close()
