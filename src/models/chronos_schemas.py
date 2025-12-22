"""Chronos-specific Pydantic schemas for temporal event reconstruction.

These schemas define the "event contract" for the Chronos pipeline:
- ChronosRecording: metadata for ingested audio files
- ChronosEvent: reconstructed narrative segments with temporal indexing
- ChronosQuery: structured query with temporal + semantic filters
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, computed_field


class DayOfWeek(str, Enum):
    """ISO 8601 day names for temporal indexing."""

    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


class EventCategory(str, Enum):
    """Semantic categories for event classification."""

    WORK = "work"
    PERSONAL = "personal"
    MEETING = "meeting"
    DEEP_WORK = "deep_work"
    BREAK = "break"
    REFLECTION = "reflection"
    IDEA = "idea"
    UNKNOWN = "unknown"


class SpeakerMode(str, Enum):
    """Speaker classification for diarization."""

    SELF_TALK = "self_talk"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


class ChronosRecording(BaseModel):
    """Metadata for a single Plaud recording ingested into Chronos.

    This represents the "raw input" before event reconstruction.
    SQLite stores this; Qdrant does not.
    """

    recording_id: str = Field(..., description="Plaud API recording ID")
    created_at: datetime = Field(..., description="Recording start time (UTC)")
    duration_seconds: int = Field(ge=0, description="Total recording duration")
    local_audio_path: str = Field(..., description="Path to downloaded audio file")
    source: str = Field(
        default="plaud", description="Source system (plaud, manual, etc)"
    )
    device_id: Optional[str] = Field(None, description="Hardware device identifier")
    checksum: Optional[str] = Field(
        None, description="SHA256 hash for integrity verification"
    )
    processing_status: str = Field(
        default="pending", description="pending | processing | completed | failed"
    )
    error_message: Optional[str] = Field(
        None, description="Error details if processing failed"
    )

    @field_validator("local_audio_path")
    @classmethod
    def path_exists_or_pending(cls, v: str) -> str:
        """Allow path validation to be deferred (file may not exist yet during ingest)."""
        return v


class ChronosEvent(BaseModel):
    """A single reconstructed narrative event from Gemini processing.

    This is the core unit of storage in Qdrant. Each event represents
    a cohesive thought, topic, or activity segment with temporal metadata.
    """

    event_id: str = Field(..., description="Unique identifier (UUID)")
    recording_id: str = Field(..., description="Parent recording ID")

    # Temporal fields (mandatory for Chronos queries)
    start_ts: datetime = Field(..., description="Event start timestamp (UTC)")
    end_ts: datetime = Field(..., description="Event end timestamp (UTC)")
    day_of_week: DayOfWeek = Field(..., description="ISO day name for filtering")
    hour_of_day: int = Field(ge=0, le=23, description="Hour of day (0-23) for patterns")

    # Content fields
    clean_text: str = Field(..., description="Clean verbatim narrative (no filler)")
    category: EventCategory = Field(
        default=EventCategory.UNKNOWN, description="Semantic category"
    )

    # Optional enrichment
    sentiment: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Extracted keywords/entities"
    )
    speaker: SpeakerMode = Field(
        default=SpeakerMode.SELF_TALK, description="Speaker classification"
    )

    # Provenance
    raw_transcript_snippet: Optional[str] = Field(
        None, description="Original messy text (for debugging)"
    )
    gemini_reasoning: Optional[str] = Field(
        None, description="Gemini's internal reasoning (if available)"
    )

    @field_validator("end_ts")
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        """Ensure event time range is valid."""
        start = info.data.get("start_ts")
        if start and v < start:
            raise ValueError("end_ts must be >= start_ts")
        return v

    @field_validator("clean_text")
    @classmethod
    def text_nonempty(cls, v: str) -> str:
        """Reject empty events (Gemini should not output these)."""
        if not v or len(v.strip()) < 10:
            raise ValueError("clean_text must be at least 10 characters")
        return v.strip()

    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Computed field: event duration in seconds."""
        return (self.end_ts - self.start_ts).total_seconds()


class TemporalFilter(BaseModel):
    """Temporal constraints for Chronos queries."""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    days_of_week: Optional[List[DayOfWeek]] = None
    hours_of_day: Optional[List[int]] = Field(None, description="List of hours (0-23)")


class ChronosQuery(BaseModel):
    """Structured query for hybrid temporal + semantic search.

    Supports:
    - Pure temporal: "all Mondays in October"
    - Pure semantic: "similar to anxiety"
    - Hybrid: "similar to anxiety on Mondays"
    """

    query_text: Optional[str] = Field(
        None, description="Semantic query for vector search"
    )
    temporal_filter: Optional[TemporalFilter] = Field(
        None, description="Time-based constraints"
    )
    categories: Optional[List[EventCategory]] = Field(
        None, description="Filter by event category"
    )
    limit: int = Field(default=10, ge=1, le=1000, description="Max results to return")
    include_graph_expansion: bool = Field(
        default=False, description="Use GraphRAG for query expansion"
    )

    @field_validator("query_text")
    @classmethod
    def query_or_filter_required(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure at least one search criterion is provided."""
        temporal_filter = info.data.get("temporal_filter")
        if not v and not temporal_filter:
            raise ValueError("Must provide either query_text or temporal_filter")
        return v


class GeminiEventOutput(BaseModel):
    """Expected JSON structure from Gemini's event reconstruction.

    This is the schema we instruct Gemini to output via prompt.
    """

    events: List[ChronosEvent] = Field(..., description="Array of reconstructed events")
    processing_metadata: Optional[dict] = Field(
        None, description="Gemini's internal metadata"
    )
    total_events: int = Field(..., ge=0, description="Total events extracted")

    @field_validator("events")
    @classmethod
    def events_nonempty(cls, v: List[ChronosEvent]) -> List[ChronosEvent]:
        """Require at least one event for long recordings."""
        if not v:
            raise ValueError("Gemini must output at least one event")
        return v
