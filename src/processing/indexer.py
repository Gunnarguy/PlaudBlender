"""Legacy segment indexer.

The unit tests treat the vector database as an injected dependency:
- `embed_fn(text) -> vector`
- `upsert_fn(vector, metadata, namespace) -> vector_id`

We update Segment rows with the returned vector id and mark them indexed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from src.database.models import Recording, Segment
from src.database.repository import get_segments_by_status, mark_recording_status


EmbedFn = Callable[[str], Any]
UpsertFn = Callable[[Any, Dict[str, Any], str], str]


def index_pending_segments(
    session: Session,
    *,
    embed_fn: EmbedFn,
    upsert_fn: UpsertFn,
    namespace: str = "full_text",
    embedding_model: str = "unknown",
    limit: int = 500,
    recording_id: Optional[str] = None,
) -> Dict[str, int]:
    """Embed + upsert all pending segments.

    Returns a summary dict for easy test assertions.
    """

    segments = get_segments_by_status(
        session, status="pending", recording_id=recording_id
    )
    segments = [s for s in segments if (s.namespace or "full_text") == namespace]
    if limit is not None:
        segments = segments[:limit]

    processed = 0

    for seg in segments:
        vector = embed_fn(seg.text)
        metadata = {
            "segment_id": seg.id,
            "recording_id": seg.recording_id,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "namespace": seg.namespace,
        }
        vector_id = upsert_fn(vector, metadata, namespace)

        seg.pinecone_id = vector_id  # legacy column name; used by tests
        seg.embedding_model = embedding_model
        seg.status = "indexed"
        processed += 1

    session.commit()

    # If we've fully indexed a recording, mark it indexed.
    if processed:
        rec_ids = {s.recording_id for s in segments}
        for rid in rec_ids:
            remaining = (
                session.query(Segment)
                .filter(Segment.recording_id == rid)
                .filter(Segment.namespace == namespace)
                .filter(Segment.status == "pending")
                .count()
            )
            if remaining == 0:
                mark_recording_status(session, rid, "indexed")

    return {"segments_processed": processed}
