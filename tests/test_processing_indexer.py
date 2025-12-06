from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Recording, Segment
from src.database.repository import add_segments, upsert_recording
from src.models.schemas import RecordingSchema, SegmentSchema
from src.processing.indexer import index_pending_segments


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return Session()


def test_index_pending_segments_updates_status_and_ids():
    session = make_session()
    try:
        rec_payload = RecordingSchema(
            id="rec_idx",
            title="To Index",
            duration_ms=500,
            created_at=datetime.utcnow(),
            transcript="word " * 40,
            language="en",
            source="plaud",
        )
        upsert_recording(session, rec_payload, filename="idx.wav", status="processed")

        seg_payloads = [
            SegmentSchema(
                id=f"seg-{i}",
                recording_id="rec_idx",
                start_ms=i * 10,
                end_ms=i * 10 + 5,
                text=f"chunk {i}",
                namespace="full_text",
            )
            for i in range(3)
        ]
        add_segments(session, recording_id="rec_idx", segments=seg_payloads, status="pending")

        embedded = {}

        def embed_fn(text: str):
            return [1.0, 0.0, 0.0]

        def upsert_fn(vector, metadata, namespace):
            embedded[metadata["segment_id"]] = {"vector": vector, "metadata": metadata, "namespace": namespace}
            return f"vec-{metadata['segment_id']}"

        summary = index_pending_segments(
            session, embed_fn=embed_fn, upsert_fn=upsert_fn, namespace="full_text", embedding_model="dummy"
        )

        assert summary["segments_processed"] == 3
        for i in range(3):
            seg_id = f"seg-{i}"
            seg = session.get(Segment, seg_id)
            assert seg.status == "indexed"
            assert seg.pinecone_id == f"vec-{seg_id}"
            assert seg.embedding_model == "dummy"
            assert embedded[seg_id]["metadata"]["recording_id"] == "rec_idx"
    finally:
        session.close()
