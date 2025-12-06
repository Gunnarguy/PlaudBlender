from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, DateTime, JSON, LargeBinary
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
    status = Column(String, default="raw")  # raw | processed | indexed | audio_processed
    extra = Column(JSON, nullable=True)
    
    # ───────────────────────────────────────────────────────────────
    # Audio Processing Fields
    # ───────────────────────────────────────────────────────────────
    audio_path = Column(String, nullable=True)       # Local cached audio file path
    audio_url = Column(String, nullable=True)        # Remote Plaud audio URL
    audio_embedding = Column(JSON, nullable=True)    # CLAP 512-dim vector as list
    speaker_diarization = Column(JSON, nullable=True)  # Whisper speaker segments
    audio_analysis = Column(JSON, nullable=True)     # Gemini tone/sentiment/topics

    segments = relationship("Segment", back_populates="recording", cascade="all, delete-orphan")

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
