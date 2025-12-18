#!/usr/bin/env python
"""
Re-embed all transcripts into Qdrant.

This script reads all recordings from SQLite and embeds them into Qdrant.
Run from project root: python scripts/embed_to_qdrant.py
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.database.engine import SessionLocal
from src.database.models import Recording
from src.qdrant_client import QdrantVectorClient
from src.models.vector_metadata import build_metadata, compute_text_hash
from gui.services.embedding_service import get_embedding_service


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks for better retrieval."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def main():
    print("🚀 Re-embedding all transcripts into Qdrant...")
    print()

    # Initialize services
    embedder = get_embedding_service()
    print(
        f"📊 Embedding service: {embedder.provider.value}, dimension={embedder.dimension}"
    )

    client = QdrantVectorClient()
    print(f"🔗 Qdrant: {client.url}, collection={client.collection_name}")
    print(f"🌐 Dashboard: {client.dashboard_url}")
    print()

    # Ensure collection exists with correct dimension
    client.create_collection(client.collection_name, embedder.dimension, "cosine")

    # Get all recordings from SQLite
    session = SessionLocal()
    recordings = session.query(Recording).all()
    total = len(recordings)
    print(f"📚 Found {total} recordings in SQLite")
    print()

    embedded_count = 0
    skipped_count = 0
    error_count = 0

    for i, rec in enumerate(recordings, 1):
        try:
            # Get transcript text
            transcript = rec.transcript or ""
            if not transcript.strip():
                print(
                    f"  [{i}/{total}] ⏭️  {rec.title or rec.id[:8]}... (no transcript)"
                )
                skipped_count += 1
                continue

            # Chunk the transcript
            chunks = chunk_text(transcript, chunk_size=1000, overlap=200)

            # Prepare vectors for all chunks
            vectors = []
            for chunk_idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = embedder.embed_document(chunk)

                # Build metadata
                metadata = build_metadata(
                    recording_id=rec.id,
                    text=chunk,
                    model=(
                        str(embedder.model.value)
                        if hasattr(embedder.model, "value")
                        else str(embedder.model)
                    ),
                    dimension=embedder.dimension,
                    source="plaud",
                    provider=(
                        embedder.provider.value
                        if hasattr(embedder.provider, "value")
                        else str(embedder.provider)
                    ),
                    title=rec.title,
                    start_at=rec.created_at.isoformat() if rec.created_at else None,
                    duration_ms=rec.duration_ms,
                    segment_id=f"chunk_{chunk_idx}",
                )

                vector_id = f"{rec.id}_chunk_{chunk_idx}"
                vectors.append(
                    {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata,
                    }
                )

            # Upsert all chunks for this recording
            client.upsert_vectors(vectors, namespace="full_text")
            embedded_count += 1

            title = (rec.title or "Untitled")[:40]
            print(f"  [{i}/{total}] ✅ {title}... ({len(chunks)} chunks)")

        except Exception as e:
            error_count += 1
            print(f"  [{i}/{total}] ❌ Error: {e}")

    session.close()

    # Get final stats
    print()
    stats = client.get_stats()
    print("=" * 50)
    print(f"✅ Embedded: {embedded_count} recordings")
    print(f"⏭️  Skipped: {skipped_count} (no transcript)")
    print(f"❌ Errors: {error_count}")
    print(f"📊 Total vectors in Qdrant: {stats.get('vectors', 0)}")
    print()
    print(f"🌐 View your data: {client.dashboard_url}")
    print()


if __name__ == "__main__":
    main()
