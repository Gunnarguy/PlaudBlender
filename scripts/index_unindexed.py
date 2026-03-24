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

# Build audio lookup if multimodal is available
audio_map: dict[str, str] = {}
if embedder.supports_multimodal:
    from src.database.models import ChronosRecording

    rec_ids = {e.recording_id for e in unindexed}
    for rec in (
        db.query(ChronosRecording)
        .filter(ChronosRecording.recording_id.in_(rec_ids))
        .all()
    ):
        if rec.local_audio_path:
            audio_map[rec.recording_id] = rec.local_audio_path
    print(f"Multimodal: {len(audio_map)} recordings have audio files")

# Generate embeddings
if audio_map:
    # Per-event multimodal (text + audio) embeddings
    print(f"Embedding {len(unindexed)} events (multimodal)...")
    vectors = []
    for e in unindexed:
        vec = embedder.embed_text_with_audio(
            text=e.clean_text,
            audio_path=audio_map.get(e.recording_id, ""),
            task_type="RETRIEVAL_DOCUMENT",
        )
        vectors.append(vec)
else:
    # Batch text-only
    texts = [e.clean_text for e in unindexed]
    print(f"Embedding {len(texts)} texts...")
    vectors = embedder.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
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
            day_of_week=str(event.day_of_week).capitalize(),
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
