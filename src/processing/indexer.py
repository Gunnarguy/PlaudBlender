"""Indexing pipeline: embed pending segments and record Pinecone IDs.

This module intentionally stays dependency-injected: callers provide
`embed_fn` and `upsert_fn` so we can plug in real services or stubs.
- embed_fn(text: str) -> List[float]
- upsert_fn(vector: List[float], metadata: dict, namespace: str) -> str (returns pinecone_id)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Any, cast

from sqlalchemy.orm import Session

from src.database.repository import get_segments_by_status
from src.utils.logger import get_logger

logger = get_logger(__name__)

EmbedFn = Callable[[str], List[float]]
UpsertFn = Callable[[List[float], dict, str], str]
BatchUpsertFn = Callable[[List[dict], str], Any]


def index_pending_segments(
    session: Session,
    embed_fn: EmbedFn,
    upsert_fn: UpsertFn,
    upsert_batch_fn: Optional[BatchUpsertFn] = None,
    namespace: str = "full_text",
    status_filter: str = "pending",
    batch_size: int = 32,
    embedding_model: Optional[str] = None,
    recording_id: Optional[str] = None,
) -> dict:
    """Embed and upsert pending segments; update statuses and pinecone_id.

    Returns a summary dict.
    """
    segments = get_segments_by_status(
        session, status=status_filter, recording_id=recording_id
    )
    processed = 0
    failures = 0

    def _mark_indexed(seg, pine_id: str) -> None:
        seg.pinecone_id = pine_id
        seg.status = "indexed"
        if embedding_model:
            seg.embedding_model = embedding_model
        session.add(seg)

    # Preferred path: batch upsert (reduces API calls and log noise)
    if upsert_batch_fn is not None and batch_size and batch_size > 1:
        # Group segments by namespace so each batch writes to a single namespace
        by_ns: dict[str, list] = {}
        for seg in segments:
            # SQLAlchemy typing stubs can make this look like Column[str] at type-check time.
            ns = cast(str, getattr(seg, "namespace", None) or namespace)
            by_ns.setdefault(ns, []).append(seg)

        for ns, segs in by_ns.items():
            for i in range(0, len(segs), batch_size):
                batch = segs[i : i + batch_size]
                vectors: List[dict] = []
                try:
                    for seg in batch:
                        text = cast(Optional[str], getattr(seg, "text", None)) or ""
                        vec = embed_fn(text)
                        meta = {
                            "recording_id": seg.recording_id,
                            "segment_id": seg.id,
                            "namespace": ns,
                            "text": text,  # Full segment text for search/display
                        }
                        vectors.append(
                            {
                                # Use segment_id as the stable vector id.
                                "id": seg.id,
                                "values": vec,
                                "metadata": meta,
                            }
                        )

                    upsert_batch_fn(vectors, ns)

                    for seg in batch:
                        _mark_indexed(seg, pine_id=seg.id)
                        processed += 1
                except Exception as exc:  # pragma: no cover - log and continue
                    failures += len(batch)
                    logger.error(
                        "Batch indexing failed for %s segments (ns=%s): %s",
                        len(batch),
                        ns,
                        exc,
                    )
        session.commit()
        return {"segments_processed": processed, "failures": failures}

    # Fallback path: one segment at a time (legacy behavior)
    for seg in segments:
        try:
            text = cast(Optional[str], getattr(seg, "text", None)) or ""
            vec = embed_fn(text)
            ns = cast(str, getattr(seg, "namespace", None) or namespace)
            pine_id = upsert_fn(
                vec,
                {
                    "recording_id": seg.recording_id,
                    "segment_id": seg.id,
                    "namespace": ns,
                    "text": text,  # Full segment text for search/display
                },
                ns,
            )
            _mark_indexed(seg, pine_id=pine_id)
            processed += 1
        except Exception as exc:  # pragma: no cover - log and continue
            failures += 1
            logger.error("Indexing failed for segment %s: %s", seg.id, exc)
    session.commit()

    return {"segments_processed": processed, "failures": failures}
