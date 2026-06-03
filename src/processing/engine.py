"""Transcript chunking engine for legacy segment pipeline.

The test suite expects a minimal workflow:
- Find recordings with status="raw".
- Split transcript into overlapping word chunks.
- Persist those chunks as `Segment` rows (status="pending").
- Mark recordings as status="processed".

This module intentionally does NOT call any external embedding/vector DB.
Indexing happens in `src.processing.indexer`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from src.database.repository import (
    add_segments,
    get_pending_recordings,
    mark_recording_status,
)
from src.models.schemas import SegmentSchema


@dataclass(frozen=True)
class ChunkingConfig:
    """Config for transcript chunking."""

    max_words: int = 200
    overlap_words: int = 40
    namespace: str = "full_text"

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            raise ValueError("max_words must be > 0")
        if self.overlap_words < 0:
            raise ValueError("overlap_words must be >= 0")
        if self.overlap_words >= self.max_words:
            raise ValueError("overlap_words must be < max_words")


def _chunk_words(
    words: List[str], *, max_words: int, overlap_words: int
) -> List[tuple[int, int]]:
    """Return a list of (start_word_idx, end_word_idx) spans."""

    if not words:
        return []

    step = max_words - overlap_words
    spans: List[tuple[int, int]] = []

    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        spans.append((start, end))
        if end >= len(words):
            break
        start += step

    return spans


def _word_span_to_ms(
    start_word: int,
    end_word: int,
    total_words: int,
    duration_ms: Optional[int],
) -> tuple[int, int]:
    """Convert a word span to a rough (start_ms, end_ms) estimate."""

    if not duration_ms or duration_ms <= 0 or total_words <= 0:
        # Fall back to monotonic dummy values.
        start_ms = start_word
        end_ms = max(end_word, start_word)
        return (int(start_ms), int(end_ms))

    start_ms = int((start_word / total_words) * duration_ms)
    end_ms = int((end_word / total_words) * duration_ms)
    if end_ms < start_ms:
        end_ms = start_ms
    return (start_ms, end_ms)


def process_pending_recordings(
    session: Session,
    *,
    cfg: Optional[ChunkingConfig] = None,
    limit: int = 100,
) -> Dict[str, int]:
    """Chunk pending recordings and persist segments.

    Returns a small summary dict for easy test assertions.
    """

    cfg = cfg or ChunkingConfig()
    pending = get_pending_recordings(session, status="raw")
    if limit is not None:
        pending = pending[:limit]

    recordings_processed = 0
    segments_created = 0

    for rec in pending:
        transcript = (rec.transcript or "").strip()
        words = [w for w in transcript.split() if w]
        spans = _chunk_words(
            words, max_words=cfg.max_words, overlap_words=cfg.overlap_words
        )

        # Defensive: if transcript is too short, still create a single segment.
        if not spans and words:
            spans = [(0, len(words))]

        seg_payloads: List[SegmentSchema] = []
        for start_w, end_w in spans:
            start_ms, end_ms = _word_span_to_ms(
                start_w, end_w, len(words), rec.duration_ms
            )
            seg_text = " ".join(words[start_w:end_w]).strip()
            if not seg_text:
                continue

            seg_payloads.append(
                SegmentSchema(
                    id=str(uuid.uuid4()),
                    recording_id=rec.id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=seg_text,
                    namespace=cfg.namespace,
                )
            )

        if seg_payloads:
            add_segments(
                session,
                recording_id=rec.id,
                segments=seg_payloads,
                default_namespace=cfg.namespace,
                status="pending",
            )
            segments_created += len(seg_payloads)

        mark_recording_status(session, rec.id, "processed")
        recordings_processed += 1

    return {
        "recordings_processed": recordings_processed,
        "segments_created": segments_created,
    }
