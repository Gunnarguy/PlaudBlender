"""Pydantic schemas to validate external inputs (Plaud, processors).

These schemas act as contracts at ingress points so we fail fast when
external payloads change shape. They can be expanded as we add fields.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RecordingSchema(BaseModel):
    id: str
    title: str = Field(default="Untitled")
    duration_ms: int = Field(ge=0)
    created_at: datetime
    transcript: str
    language: Optional[str] = None
    source: str = Field(default="plaud")

    @field_validator("transcript")
    @classmethod
    def transcript_nonempty(cls, v: str) -> str:
        if not v or len(v.strip()) < 50:
            raise ValueError("Transcript too short or empty")
        return v.strip()


class SegmentSchema(BaseModel):
    id: str
    recording_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str
    embedding: Optional[List[float]] = None
    namespace: str = "full_text"

    @field_validator("end_ms")
    @classmethod
    def end_after_start(cls, v: int, info):
        start = info.data.get("start_ms", 0)
        if v < start:
            raise ValueError("end_ms must be >= start_ms")
        return v

    @field_validator("text")
    @classmethod
    def text_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Segment text cannot be empty")
        return v.strip()
