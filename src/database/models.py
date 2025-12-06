from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(String, primary_key=True)  # Plaud ID
    title = Column(String, nullable=True)
    filename = Column(String, nullable=True)
    transcript = Column(Text, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="plaud")
    language = Column(String, nullable=True)
    status = Column(String, default="raw")  # raw | processed | indexed
    extra = Column(JSON, nullable=True)

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
