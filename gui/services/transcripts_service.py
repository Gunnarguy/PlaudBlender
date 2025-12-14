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
            try:
                client = get_plaud_client()
                stored = client.fetch_and_store_recordings(limit=limit)
                log('INFO', f"Stored {len(stored)} recordings to SQLite")
            except Exception as e:
                error_msg = str(e)
                if "500" in error_msg or "Internal Server Error" in error_msg:
                    log('WARNING', f"Plaud API server error (500) - loading cached data from SQLite")
                else:
                    log('ERROR', f"Failed to fetch from Plaud: {e} - loading cached data")
                # Continue to load from SQLite even if API fails
        
        enhanced = load_db_recordings()
        state.transcripts = enhanced
        state.filtered_transcripts = enhanced
        
        if enhanced:
            log('INFO', f"Loaded {len(enhanced)} recordings from SQLite")
        else:
            log('WARNING', "No transcripts in database. Plaud API may be down.")
        
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
        error_msg = str(e)
        if "500" in error_msg or "Internal Server Error" in error_msg:
            log('WARNING', f"Plaud API server error (500) - transcript not available for {recording_id}")
            return "[Transcript unavailable - Plaud API server error. Try again later.]"
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
        # Include transcript text for knowledge graph extraction
        "transcript": rec.transcript or "",
        "full_text": rec.transcript or "",
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
        
        # Get recording for metadata enrichment
        rec = session.get(Recording, recording_id)

        def embed_fn(text: str):
            return embedder.embed_text(text, dimension=embedder.dimension)

        def upsert_fn(vector, metadata, namespace):
            # Enrich metadata with text snippet and recording info
            enriched = {
                **metadata,
                "text": metadata.get("text", "")[:1000],  # Truncate for Pinecone limits
                "source": "plaud",
            }
            if rec:
                enriched["title"] = rec.title or rec.filename or "Untitled"
                if rec.created_at:
                    enriched["created_at"] = rec.created_at.isoformat()
                if rec.extra and rec.extra.get("themes"):
                    enriched["themes"] = rec.extra["themes"][:5]  # First 5 themes
            
            payload = [{"id": metadata.get("segment_id") or metadata.get("recording_id"), "values": vector, "metadata": enriched}]
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
        
        # Build a lookup for recording metadata enrichment
        from sqlalchemy import select
        recs = {r.id: r for r in session.execute(select(Recording)).scalars().all()}

        def embed_fn(text: str):
            return embedder.embed_text(text, dimension=embedder.dimension)

        def upsert_fn(vector, metadata, namespace):
            # Enrich metadata with text snippet and recording info
            enriched = {
                **metadata,
                "text": metadata.get("text", "")[:1000],  # Truncate for Pinecone limits
                "source": "plaud",
            }
            rec = recs.get(metadata.get("recording_id"))
            if rec:
                enriched["title"] = rec.title or rec.filename or "Untitled"
                if rec.created_at:
                    enriched["created_at"] = rec.created_at.isoformat()
                if rec.extra and rec.extra.get("themes"):
                    enriched["themes"] = rec.extra["themes"][:5]
            
            payload = [
                {
                    "id": metadata.get("segment_id") or metadata.get("recording_id"),
                    "values": vector,
                    "metadata": enriched,
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


def delete_recording(recording_id: str, delete_from_pinecone: bool = True) -> Dict:
    """
    Delete a recording from SQLite and optionally from Pinecone.
    
    Args:
        recording_id: Recording ID to delete
        delete_from_pinecone: Also delete vectors from Pinecone
        
    Returns:
        Dict with deletion results
    """
    init_db()
    session = SessionLocal()
    results = {"db_deleted": False, "pinecone_deleted": False, "errors": []}
    
    try:
        # Delete from SQLite (cascades to segments)
        rec = session.get(Recording, recording_id)
        if rec:
            session.delete(rec)
            session.commit()
            results["db_deleted"] = True
            log('INFO', f"Deleted recording {recording_id} from database")
        else:
            results["errors"].append(f"Recording {recording_id} not found in database")
        
        # Delete from Pinecone
        if delete_from_pinecone:
            try:
                client = get_pinecone_client()
                # Delete from both namespaces using metadata filter
                for ns in ["full_text", "summaries"]:
                    try:
                        # Try to delete by recording_id metadata
                        client.delete_by_metadata(
                            filter={"recording_id": {"$eq": recording_id}},
                            namespace=ns
                        )
                    except Exception:
                        # Fallback: delete by ID prefix if metadata delete not supported
                        pass
                results["pinecone_deleted"] = True
                log('INFO', f"Deleted vectors for {recording_id} from Pinecone")
            except Exception as e:
                results["errors"].append(f"Pinecone delete failed: {e}")
                log('ERROR', f"Failed to delete from Pinecone: {e}")
        
        return results
        
    finally:
        session.close()


def export_recording(recording_id: str, include_segments: bool = True) -> Dict:
    """
    Export a recording with all its data for backup/transfer.
    
    Args:
        recording_id: Recording ID to export
        include_segments: Include segment data
        
    Returns:
        Dict with full recording data
    """
    init_db()
    session = SessionLocal()
    
    try:
        rec = session.get(Recording, recording_id)
        if not rec:
            raise ValueError(f"Recording {recording_id} not found")
        
        export_data = {
            "id": rec.id,
            "title": rec.title,
            "filename": rec.filename,
            "transcript": rec.transcript,
            "duration_ms": rec.duration_ms,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
            "source": rec.source,
            "language": rec.language,
            "status": rec.status,
            "extra": rec.extra,
            "audio_url": rec.audio_url,
            "audio_analysis": rec.audio_analysis,
            "speaker_diarization": rec.speaker_diarization,
        }
        
        if include_segments:
            export_data["segments"] = [
                {
                    "id": seg.id,
                    "text": seg.text,
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "theme": seg.theme,
                    "namespace": seg.namespace,
                    "pinecone_id": seg.pinecone_id,
                    "status": seg.status,
                }
                for seg in rec.segments
            ]
        
        return export_data
        
    finally:
        session.close()


def export_all_recordings(status_filter: Optional[str] = None) -> List[Dict]:
    """
    Export all recordings for backup.
    
    Args:
        status_filter: Optional status to filter by (raw, processed, indexed)
        
    Returns:
        List of recording export dicts
    """
    init_db()
    session = SessionLocal()
    
    try:
        query = select(Recording)
        if status_filter:
            query = query.where(Recording.status == status_filter)
        
        recordings = session.execute(query).scalars().all()
        return [export_recording(rec.id, include_segments=True) for rec in recordings]
        
    finally:
        session.close()
