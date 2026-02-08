"""Index unindexed events from SQLite to Qdrant."""

import sys

sys.path.insert(0, ".")

from src.database.engine import SessionLocal
from src.database.models import ChronosEvent
from src.chronos.embedding_service import ChronosEmbeddingService
from src.chronos.qdrant_client import ChronosQdrantClient
from src.models.chronos_schemas import ChronosEvent as CE

db = SessionLocal()
unindexed = db.query(ChronosEvent).filter(ChronosEvent.qdrant_point_id.is_(None)).all()
print(f"Found {len(unindexed)} unindexed events")

if not unindexed:
    print("Nothing to do")
    db.close()
    sys.exit(0)

embedder = ChronosEmbeddingService()
qdrant = ChronosQdrantClient()

# Batch embed
texts = [e.clean_text for e in unindexed]
print(f"Embedding {len(texts)} texts...")
vectors = embedder.embed_batch(texts)
print(f"Got {len(vectors)} vectors")

indexed = 0
errors = 0
for event, vector in zip(unindexed, vectors):
    try:
        schema_event = CE(
            event_id=event.event_id,
            recording_id=event.recording_id,
            start_ts=event.start_ts,
            end_ts=event.end_ts,
            day_of_week=event.day_of_week,
            hour_of_day=event.hour_of_day,
            clean_text=event.clean_text,
            category=event.category,
            sentiment=event.sentiment or 0.0,
            keywords=event.keywords or [],
            speaker=event.speaker or "unknown",
            raw_transcript_snippet=event.raw_transcript_snippet,
            gemini_reasoning=event.gemini_reasoning,
        )
        point_id = qdrant.upsert_event(schema_event, vector)
        event.qdrant_point_id = point_id
        db.commit()
        indexed += 1
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"Error: {e}")

print(f"Done: {indexed} indexed, {errors} errors")
db.close()
