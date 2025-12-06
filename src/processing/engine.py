"""Processing engine to bridge stored recordings -> segments -> indexing.

This lightweight engine operates on the local SQL store so we have a
first-class, deterministic pipeline even before hitting Pinecone/LLM. It
creates chunked segments for pending recordings and marks status
transitions accordingly. Embedding + Pinecone upsert can be layered on
later, using the `pinecone_id` field in the segments table.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sqlalchemy.orm import Session

from src.database.repository import (
    add_segments,
    get_pending_recordings,
    mark_recording_status,
)
from src.database.models import Recording
from src.models.schemas import SegmentSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    max_words: int = 200  # soft limit per chunk
    overlap_words: int = 20  # overlap between chunks for continuity
    namespace: str = "full_text"


def _chunk_transcript(text: str, cfg: ChunkingConfig) -> Iterable[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    total = len(words)
    while start < total:
        end = min(start + cfg.max_words, total)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == total:
            break
        start = max(end - cfg.overlap_words, 0)
    return chunks


def _segments_from_chunks(
    recording_id: str, chunks: Iterable[str], cfg: ChunkingConfig
) -> List[SegmentSchema]:
    segments: List[SegmentSchema] = []
    start_word = 0
    for chunk in chunks:
        word_count = len(chunk.split())
        end_word = start_word + word_count
        segments.append(
            SegmentSchema(
                id=str(uuid.uuid4()),
                recording_id=recording_id,
                start_ms=start_word,  # approximate: word index as pseudo-ms
                end_ms=end_word,
                text=chunk,
                namespace=cfg.namespace,
            )
        )
        start_word = end_word - cfg.overlap_words if cfg.overlap_words else end_word
    return segments


def process_pending_recordings(
    session: Session,
    cfg: Optional[ChunkingConfig] = None,
    dry_run: bool = False,
    recording_id: Optional[str] = None,
) -> dict:
    """Create chunked segments for pending recordings and update status.

    If recording_id is provided, only process that recording when status is raw.
    Returns a summary dict with counts.
    """
    cfg = cfg or ChunkingConfig()
    if recording_id:
        candidate = session.get(Recording, recording_id)
        pending = [candidate] if candidate and candidate.status == "raw" else []
    else:
        pending = get_pending_recordings(session, status="raw")
    created_segments = 0

    for rec in pending:
        chunks = list(_chunk_transcript(rec.transcript or "", cfg))
        if not chunks:
            logger.warning("Skipping empty transcript for %s", rec.id)
            continue
        segments = _segments_from_chunks(rec.id, chunks, cfg)
        if dry_run:
            logger.info("[dry-run] Would create %s segments for %s", len(segments), rec.id)
            continue
        add_segments(
            session,
            recording_id=rec.id,
            segments=segments,
            default_namespace=cfg.namespace,
            status="pending",
        )
        mark_recording_status(session, rec.id, "processed")
        created_segments += len(segments)
        logger.info("Created %s segments for %s", len(segments), rec.id)

    return {"recordings_processed": len(pending), "segments_created": created_segments}
