from datetime import datetime
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    ForeignKey,
    Text,
    DateTime,
    JSON,
    LargeBinary,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Recording(Base):
    """
    Recording model with full audio processing support.

    Audio Pipeline Fields:
        audio_path: Local path to cached audio file (downloaded from Plaud)
        audio_url: Remote URL to original audio (from Plaud API)
        audio_embedding: CLAP audio embedding vector (512-dim) for audio similarity search
        speaker_diarization: JSON with speaker segments from Whisper diarization
        audio_analysis: JSON with Gemini audio analysis (tone, sentiment, topics)
    """

    __tablename__ = "recordings"

    id = Column(String, primary_key=True)  # Plaud ID
    title = Column(String, nullable=True)
    filename = Column(String, nullable=True)
    transcript = Column(Text, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="plaud")
    language = Column(String, nullable=True)
    status = Column(
        String, default="raw"
    )  # raw | processed | indexed | audio_processed
    extra = Column(JSON, nullable=True)

    # ───────────────────────────────────────────────────────────────
    # Audio Processing Fields
    # ───────────────────────────────────────────────────────────────
    audio_path = Column(String, nullable=True)  # Local cached audio file path
    audio_url = Column(String, nullable=True)  # Remote Plaud audio URL
    audio_embedding = Column(JSON, nullable=True)  # CLAP 512-dim vector as list
    speaker_diarization = Column(JSON, nullable=True)  # Whisper speaker segments
    audio_analysis = Column(JSON, nullable=True)  # Gemini tone/sentiment/topics

    segments = relationship(
        "Segment", back_populates="recording", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - repr utility
        return f"Recording(id={self.id}, title={self.title}, status={self.status})"


class Segment(Base):
    __tablename__ = "segments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(String, ForeignKey("recordings.id"), nullable=False)

    text = Column(Text, nullable=False)
    start_ms = Column(Integer, nullable=True)
    end_ms = Column(Integer, nullable=True)
    theme = Column(String, nullable=True)
    namespace = Column(String, default="full_text")
    pinecone_id = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | indexed
    extra = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    recording = relationship("Recording", back_populates="segments")

    def __repr__(self) -> str:  # pragma: no cover - repr utility
        return f"Segment(id={self.id}, recording_id={self.recording_id}, namespace={self.namespace})"


# ═══════════════════════════════════════════════════════════════════
# Chronos-Specific Tables
# ═══════════════════════════════════════════════════════════════════


class ChronosRecording(Base):
    """Chronos ingestion metadata for Plaud recordings.

    Tracks the local audio cache, processing status, and integrity checks.
    Separate from legacy Recording table to avoid coupling.
    """

    __tablename__ = "chronos_recordings"

    recording_id = Column(String, primary_key=True)  # Plaud API ID
    created_at = Column(DateTime, nullable=False)  # Recording start time (UTC)
    duration_seconds = Column(Integer, nullable=False)
    local_audio_path = Column(String, nullable=False)
    source = Column(String, default="plaud", nullable=False)
    device_id = Column(String, nullable=True)
    checksum = Column(String, nullable=True)  # SHA256 for integrity

    # Processing workflow
    processing_status = Column(
        String, default="pending", nullable=False
    )  # pending | processing | completed | failed
    error_message = Column(Text, nullable=True)
    processed_at = Column(DateTime, nullable=True)

    # Provenance
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    events = relationship(
        "ChronosEvent", back_populates="recording", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"ChronosRecording(id={self.recording_id}, status={self.processing_status})"
        )


class ChronosEvent(Base):
    """Chronos reconstructed narrative events.

    These are the "clean" events produced by Gemini. The actual vector
    lives in Qdrant; this table stores the source-of-truth text and metadata.
    """

    __tablename__ = "chronos_events"

    event_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(
        String, ForeignKey("chronos_recordings.recording_id"), nullable=False
    )

    # Temporal indexing (mandatory)
    start_ts = Column(DateTime, nullable=False)
    end_ts = Column(DateTime, nullable=False)
    day_of_week = Column(String, nullable=False)  # Monday, Tuesday, etc.
    hour_of_day = Column(Integer, nullable=False)  # 0-23

    # Content
    clean_text = Column(Text, nullable=False)
    category = Column(
        String, default="unknown", nullable=False
    )  # work, personal, meeting, etc.

    # Optional enrichment
    sentiment = Column(Float, nullable=True)  # -1.0 to 1.0
    keywords = Column(JSON, nullable=True)  # List of extracted keywords
    speaker = Column(
        String, default="self_talk", nullable=True
    )  # self_talk | conversation

    # Provenance & debugging
    raw_transcript_snippet = Column(Text, nullable=True)
    gemini_reasoning = Column(Text, nullable=True)

    # Vector storage reference
    qdrant_point_id = Column(String, nullable=True)  # UUID of Qdrant point

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    recording = relationship("ChronosRecording", back_populates="events")

    def __repr__(self) -> str:
        return f"ChronosEvent(id={self.event_id}, recording={self.recording_id}, category={self.category})"


class ChronosProcessingJob(Base):
    """Queue table for Chronos processing jobs.

    Tracks the async processing pipeline: ingest → clean → index → graph.
    """

    __tablename__ = "chronos_processing_jobs"

    job_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recording_id = Column(
        String, ForeignKey("chronos_recordings.recording_id"), nullable=False
    )

    # Job metadata
    job_type = Column(
        String, nullable=False
    )  # gemini_clean | qdrant_index | graph_extract
    status = Column(
        String, default="queued", nullable=False
    )  # queued | running | completed | failed
    priority = Column(Integer, default=0, nullable=False)  # Higher = more urgent

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)

    def __repr__(self) -> str:
        return f"ChronosProcessingJob(id={self.job_id}, type={self.job_type}, status={self.status})"
