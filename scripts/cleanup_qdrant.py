"""Clean orphan Qdrant points that no longer have matching SQLite events."""
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient

e = create_engine("sqlite:///data/brain.db")
q = QdrantClient(url="http://localhost:6333")

# Get all qdrant_point_ids from SQLite
with e.connect() as c:
    rows = c.execute(text("SELECT qdrant_point_id FROM chronos_events WHERE qdrant_point_id IS NOT NULL")).fetchall()
    db_ids = {r[0] for r in rows}

print(f"SQLite has {len(db_ids)} indexed point IDs")

# Scroll through all Qdrant points
all_qdrant_ids = set()
offset = None
while True:
    results = q.scroll(
        collection_name="chronos_events",
        limit=100,
        offset=offset,
        with_payload=False,
        with_vectors=False,
    )
    points, next_offset = results
    for p in points:
        all_qdrant_ids.add(str(p.id))
    if next_offset is None:
        break
    offset = next_offset

print(f"Qdrant has {len(all_qdrant_ids)} points")

orphans = all_qdrant_ids - db_ids
if orphans:
    print(f"Found {len(orphans)} orphan points — deleting...")
    q.delete(
        collection_name="chronos_events",
        points_selector=list(orphans),
    )
    info = q.get_collection("chronos_events")
    print(f"After cleanup: {info.points_count} points")
else:
    print("No orphans found!")
