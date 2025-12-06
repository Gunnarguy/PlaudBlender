from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional

from sqlalchemy import select

from gui.services.clients import get_plaud_client, get_pinecone_client
from gui.state import state
from gui.utils.logging import log
from gui.services.embedding_service import get_embedding_service

from src.database.engine import SessionLocal, init_db
from src.database.models import Recording
from src.processing.engine import process_pending_recordings
from src.processing.indexer import index_pending_segments
from src.models.schemas import RecordingSchema


def fetch_transcripts(limit: int = 100, ingest: bool = True) -> List[Dict]:
    """Fetch recordings via Plaud, persist to SQLite, then load from DB for display."""
    try:
        init_db()
        if ingest:
            client = get_plaud_client()
            stored = client.fetch_and_store_recordings(limit=limit)
            log('INFO', f"Stored {len(stored)} recordings to SQLite")
        enhanced = load_db_recordings()
        state.transcripts = enhanced
        state.filtered_transcripts = enhanced
        log('INFO', f"Loaded {len(enhanced)} recordings from SQLite")
        return enhanced
    except Exception as e:
        log('ERROR', f"Failed to load transcripts: {e}")
        state.transcripts = []
        state.filtered_transcripts = []
        return []


def filter_transcripts(query: str) -> List[Dict]:
    query_lower = query.lower()
    state.filtered_transcripts = [rec for rec in state.transcripts if query_lower in rec['display_name'].lower()]
    return state.filtered_transcripts


def get_transcript_text(recording_id: str) -> str:
    """Fetch full transcript text for a recording, with safe fallbacks."""
    # Prefer local DB copy first
    init_db()
    session = SessionLocal()
    try:
        rec = session.get(Recording, recording_id)
        if rec and rec.transcript:
            return rec.transcript
    finally:
        session.close()

    # Fallback to Plaud API
    client = get_plaud_client()
    try:
        return client.get_transcript_text(recording_id)
    except Exception as e:
        log('ERROR', f"Failed to fetch transcript text for {recording_id}: {e}")
        raise


def load_db_recordings() -> List[Dict]:
    """Load recordings from SQLite and normalize for UI."""
    init_db()
    session = SessionLocal()
    rows: List[Recording] = []
    try:
        rows = session.execute(select(Recording).order_by(Recording.created_at.desc())).scalars().all()
    finally:
        session.close()
    return [_normalize_db_recording(r) for r in rows]


def _normalize_recording(rec: Dict) -> Dict:
    name = rec.get('name') or rec.get('title') or rec.get('file_name') or 'Untitled'
    start_at = rec.get('start_at')
    if start_at:
        try:
            dt = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M')
        except ValueError:
            date_str = start_at[:10]
            time_str = start_at[11:16]
    else:
        date_str = time_str = '—'

    duration_ms = rec.get('duration', 0) or rec.get('duration_ms', 0)
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    duration_str = f"{minutes}:{seconds:02d}" if duration_ms else '—'

    rec = rec.copy()
    rec.update({
        'display_name': name,
        'display_date': date_str,
        'display_time': time_str,
        'display_duration': duration_str,
        'short_id': f"{rec.get('id','')[:10]}…" if rec.get('id') else '—',
    })
    return rec


def _normalize_db_recording(rec: Recording) -> Dict:
    duration_ms = rec.duration_ms or 0
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    duration_str = f"{minutes}:{seconds:02d}" if duration_ms else '—'
    created_at = rec.created_at or datetime.utcnow()
    date_str = created_at.strftime('%Y-%m-%d')
    time_str = created_at.strftime('%H:%M')

    return {
        "id": rec.id,
        "display_name": rec.title or "Untitled",
        "display_date": date_str,
        "display_time": time_str,
        "display_duration": duration_str,
        "short_id": f"{rec.id[:10]}…",
        "status": rec.status,
        "source": rec.source,
        "namespace": "full_text",
    }


def sync_recording(recording_id: str) -> Dict:
    """Chunk, embed, and upsert a single recording from SQLite -> Pinecone."""
    init_db()
    session = SessionLocal()
    try:
        # 1) Chunk if needed
        process_pending_recordings(session, recording_id=recording_id)

        embedder = get_embedding_service()
        client = get_pinecone_client()

        def embed_fn(text: str):
            return embedder.embed_text(text, dimension=embedder.dimension)

        def upsert_fn(vector, metadata, namespace):
            payload = [{"id": metadata.get("segment_id") or metadata.get("recording_id"), "values": vector, "metadata": metadata}]
            client.upsert_vectors(payload, namespace=namespace)
            return payload[0]["id"]

        result = index_pending_segments(
            session,
            embed_fn=embed_fn,
            upsert_fn=upsert_fn,
            namespace="full_text",
            recording_id=recording_id,
            embedding_model=getattr(embedder, "model_name", None),
        )
        return result
    finally:
        session.close()


def sync_all() -> Dict:
    """Chunk, embed, and upsert all pending recordings from SQLite -> Pinecone."""
    init_db()
    session = SessionLocal()
    try:
        # 1) Chunk all raw recordings
        chunk_summary = process_pending_recordings(session)

        embedder = get_embedding_service()
        client = get_pinecone_client()

        def embed_fn(text: str):
            return embedder.embed_text(text, dimension=embedder.dimension)

        def upsert_fn(vector, metadata, namespace):
            payload = [
                {
                    "id": metadata.get("segment_id") or metadata.get("recording_id"),
                    "values": vector,
                    "metadata": metadata,
                }
            ]
            client.upsert_vectors(payload, namespace=namespace)
            return payload[0]["id"]

        index_summary = index_pending_segments(
            session,
            embed_fn=embed_fn,
            upsert_fn=upsert_fn,
            namespace="full_text",
            embedding_model=getattr(embedder, "model_name", None),
        )
        return {"chunk": chunk_summary, "index": index_summary}
    finally:
        session.close()
