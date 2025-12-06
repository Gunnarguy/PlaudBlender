"""Indexing pipeline: embed pending segments and record Pinecone IDs.

This module intentionally stays dependency-injected: callers provide
`embed_fn` and `upsert_fn` so we can plug in real services or stubs.
- embed_fn(text: str) -> List[float]
- upsert_fn(vector: List[float], metadata: dict, namespace: str) -> str (returns pinecone_id)
"""
from __future__ import annotations

from typing import Callable, List, Optional

from sqlalchemy.orm import Session

from src.database.repository import get_segments_by_status
from src.utils.logger import get_logger

logger = get_logger(__name__)

EmbedFn = Callable[[str], List[float]]
UpsertFn = Callable[[List[float], dict, str], str]


def index_pending_segments(
    session: Session,
    embed_fn: EmbedFn,
    upsert_fn: UpsertFn,
    namespace: str = "full_text",
    status_filter: str = "pending",
    batch_size: int = 32,
    embedding_model: Optional[str] = None,
    recording_id: Optional[str] = None,
) -> dict:
    """Embed and upsert pending segments; update statuses and pinecone_id.

    Returns a summary dict.
    """
    segments = get_segments_by_status(session, status=status_filter, recording_id=recording_id)
    processed = 0
    failures = 0

    for seg in segments:
        try:
            vec = embed_fn(seg.text or "")
            pine_id = upsert_fn(
                vector=vec,
                metadata={
                    "recording_id": seg.recording_id,
                    "segment_id": seg.id,
                    "namespace": seg.namespace or namespace,
                },
                namespace=seg.namespace or namespace,
            )
            seg.pinecone_id = pine_id
            seg.status = "indexed"
            if embedding_model:
                seg.embedding_model = embedding_model
            session.add(seg)
            processed += 1
        except Exception as exc:  # pragma: no cover - log and continue
            failures += 1
            logger.error("Indexing failed for segment %s: %s", seg.id, exc)
    session.commit()

    return {"segments_processed": processed, "failures": failures}
