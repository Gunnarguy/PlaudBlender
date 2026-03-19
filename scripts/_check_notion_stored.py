"""Check stored data for the Notion import without calling Notion API."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.engine import SessionLocal, init_db
from src.database.models import ChronosRecording, Recording

init_db()
db = SessionLocal()

RID = "notion:32749a74-d54f-81df-a2b1-f3c745f64c37"

# Check ChronosRecording
cr = db.query(ChronosRecording).filter_by(recording_id=RID).first()
if cr:
    print("=== ChronosRecording ===")
    print(f"  recording_id: {cr.recording_id}")
    print(f"  title: {cr.title}")
    print(f"  created_at: {cr.created_at}")
    print(f"  duration_seconds: {cr.duration_seconds}")
    print(f"  source: {cr.source}")
    print(f"  processing_status: {cr.processing_status}")
    print(f"  transcript (first 200): {(cr.transcript or '')[:200]}")

# Check Recording (legacy)
lr = db.query(Recording).filter_by(id=RID).first()
if lr:
    print("\n=== Recording (legacy) ===")
    print(f"  id: {lr.id}")
    print(f"  extra: {lr.extra}")

# Check events in Qdrant
print("\n=== Qdrant Events ===")
from src.chronos.qdrant_client import ChronosQdrantClient

qc = ChronosQdrantClient()

# Scroll for events matching this recording
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = qc.client.scroll(
    collection_name=qc.collection_name,
    scroll_filter=Filter(
        must=[FieldCondition(key="recording_id", match=MatchValue(value=RID))]
    ),
    limit=30,
    with_payload=True,
)
points = results[0]
print(f"Found {len(points)} points in Qdrant")
for i, pt in enumerate(points[:5]):
    p = pt.payload
    print(f"\n  Event {i+1}:")
    print(f"    id: {pt.id}")
    print(f"    start_ts: {p.get('start_ts')}")
    print(f"    end_ts: {p.get('end_ts')}")
    print(f"    day_of_week: {p.get('day_of_week')}")
    print(f"    hour_of_day: {p.get('hour_of_day')}")
    print(f"    category: {p.get('category')}")
    print(f"    source: {p.get('source')}")
    print(f"    recording_id: {p.get('recording_id')}")
    print(f"    text: {p.get('clean_text', '')[:100]}")

db.close()
