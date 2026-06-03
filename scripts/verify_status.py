"""Quick system status check."""
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient

e = create_engine("sqlite:///data/brain.db")
with e.connect() as c:
    recs = c.execute(text("SELECT COUNT(*) FROM chronos_recordings")).scalar()
    completed = c.execute(text("SELECT COUNT(*) FROM chronos_recordings WHERE processing_status='completed'")).scalar()
    events = c.execute(text("SELECT COUNT(*) FROM chronos_events")).scalar()
    indexed = c.execute(text("SELECT COUNT(*) FROM chronos_events WHERE qdrant_point_id IS NOT NULL")).scalar()
    unindexed = c.execute(text("SELECT COUNT(*) FROM chronos_events WHERE qdrant_point_id IS NULL")).scalar()

q = QdrantClient(url="http://localhost:6333")
info = q.get_collection("chronos_events")
qdrant_pts = info.points_count

print("=== SYSTEM STATUS ===")
print(f"Recordings: {completed}/{recs} completed")
print(f"Events in SQLite: {events}")
print(f"  - indexed: {indexed}")
print(f"  - unindexed: {unindexed}")
print(f"Qdrant points: {qdrant_pts}")
match = "YES" if indexed == qdrant_pts else "NO — MISMATCH"
print(f"Match: {match}")
