from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    String,
    Integer,
    Float,
    ForeignKey,
    Text,
    DateTime,
    JSON,
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
    vector_id = Column(String, nullable=True)  # Qdrant vector ID
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
    # Optional human metadata (Plaud sometimes provides a title)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)  # Recording start time (UTC)
    duration_seconds = Column(Integer, nullable=False)
    local_audio_path = Column(String, nullable=False)
    source = Column(String, default="plaud", nullable=False)
    device_id = Column(String, nullable=True)
    checksum = Column(String, nullable=True)  # SHA256 for integrity

    # Transcript cache (Chronos is currently transcript-first because Plaud audio URLs
    # are not reliably available via API).
    transcript = Column(Text, nullable=True)
    transcript_cached_at = Column(DateTime, nullable=True)

    # Processing workflow
    processing_status = Column(
        String, default="pending", nullable=False
    )  # raw | pending | processing | completed | failed | deferred | cancelled
    error_message = Column(Text, nullable=True)
    processed_at = Column(DateTime, nullable=True)

    # Lease-based processing locks & heartbeats
    processing_started_at = Column(DateTime, nullable=True)
    heartbeat_at = Column(DateTime, nullable=True)
    lease_expires_at = Column(DateTime, nullable=True)
    worker_id = Column(String, nullable=True)
    attempt_count = Column(Integer, default=0, nullable=False)

    # Plaud AI Summary (fetched from Plaud cloud after workflow processing)
    plaud_ai_summary = Column(Text, nullable=True)

    # Plaud Cloud Workflow tracking
    plaud_workflow_id = Column(String, nullable=True)  # Active workflow ID
    plaud_workflow_status = Column(
        String, nullable=True
    )  # PENDING/PROCESSING/SUCCESS/FAILED
    plaud_workflow_submitted_at = Column(DateTime, nullable=True)
    plaud_workflow_completed_at = Column(DateTime, nullable=True)
    plaud_workflow_template_id = Column(String, nullable=True)
    plaud_workflow_error = Column(Text, nullable=True)
    plaud_extracted_data = Column(JSON, nullable=True)  # AI_ETL structured output
    time_is_estimated = Column(Boolean, nullable=True)
    time_estimate_reason = Column(Text, nullable=True)

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
        String, ForeignKey("chronos_recordings.recording_id"), nullable=False, index=True
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
    category_confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    # User-overridden category (takes precedence over Gemini-assigned)
    user_category_override = Column(String, nullable=True)

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


class ChronosExecutionRun(Base):
    """Top-level observable Chronos execution run.

    A run represents one user/system-triggered unit of work: a manual sync,
    scheduled pipeline pass, recording reprocess, search/ask request, etc. Child
    spans capture the exact provider/model/API work performed inside the run.
    """

    __tablename__ = "chronos_execution_runs"

    run_id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    trigger = Column(String, nullable=True)  # manual | scheduled | webhook | ios | dash | cli
    source = Column(String, nullable=True)  # sync | search | recording | system
    status = Column(String, default="running", nullable=False)
    title = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)
    host = Column(String, nullable=True)
    process_id = Column(Integer, nullable=True)
    entrypoint = Column(String, nullable=True)
    summary = Column(JSON, nullable=True)
    run_metadata = Column("metadata", JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    spans = relationship(
        "ChronosExecutionSpan",
        back_populates="run",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"ChronosExecutionRun(id={self.run_id}, status={self.status})"


class ChronosExecutionSpan(Base):
    """Granular observable operation inside a Chronos execution run."""

    __tablename__ = "chronos_execution_spans"

    span_id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:16])
    run_id = Column(
        String,
        ForeignKey("chronos_execution_runs.run_id"),
        nullable=True,
        index=True,
    )
    parent_span_id = Column(String, nullable=True, index=True)
    recording_id = Column(
        String,
        ForeignKey("chronos_recordings.recording_id"),
        nullable=True,
        index=True,
    )
    event_id = Column(
        String,
        ForeignKey("chronos_events.event_id"),
        nullable=True,
        index=True,
    )

    stage = Column(String, nullable=True)  # ingest | process | embed | index | graph | ask
    operation = Column(String, nullable=False)
    source = Column(String, nullable=True)  # xray source: gemini | openai | qdrant | local
    provider = Column(String, nullable=True)  # gemini | openai | local | qdrant | plaud
    model = Column(String, nullable=True)
    status = Column(String, default="running", nullable=False)
    level = Column(String, default="info", nullable=False)
    message = Column(Text, nullable=True)
    detail = Column(Text, nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)

    input_hash = Column(String, nullable=True)
    output_hash = Column(String, nullable=True)
    input_snippet = Column(Text, nullable=True)
    output_snippet = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    request_id = Column(String, nullable=True, index=True)
    retry_count = Column(Integer, default=0, nullable=False)
    span_metadata = Column("metadata", JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("ChronosExecutionRun", back_populates="spans")

    def __repr__(self) -> str:
        return f"ChronosExecutionSpan(id={self.span_id}, op={self.operation}, status={self.status})"


class ChronosWebhookEvent(Base):
    """Persisted incoming Plaud webhook events.

    Storing webhook payloads allows Chronos to audit incoming events and
    replay them into the pipeline (ingest/process) in a controlled way.
    """

    __tablename__ = "chronos_webhook_events"

    event_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    webhook_id = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    headers = Column(JSON, nullable=True)
    # Optional link to a recording if the event references one
    recording_id = Column(String, ForeignKey("chronos_recordings.recording_id"), nullable=True)

    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed = Column(String, default="new", nullable=False)  # new | processed | failed
    processed_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"ChronosWebhookEvent(id={self.event_id}, type={self.event_type}, received_at={self.received_at})"


class NotionMatchOverride(Base):
    """Persisted manual overrides mapping Notion page UUIDs to Chronos recording IDs."""

    __tablename__ = "notion_match_overrides"

    notion_page_id = Column(String, primary_key=True)
    chronos_recording_id = Column(String, nullable=False)


class ChronosRecordingArtifact(Base):
    """One artifact Plaud 4.0 serves for a recording, kept verbatim on disk.

    The sync stores a flattened transcript and the summary on the recording
    row. Everything else Plaud produces -- the outline, the polished
    transcript, highlight memos, the transcript's per-line timing and
    speaker JSON -- is fetched by scripts/plaud_v4_artifacts.py and kept
    here: the content on disk, the row saying what it is and where.
    """

    __tablename__ = "chronos_recording_artifacts"

    recording_id = Column(String, ForeignKey("chronos_recordings.recording_id"), primary_key=True)
    object_type = Column(String, primary_key=True)  # TRANSCRIPT, OUTLINE, POLISHED_TRANSCRIPT, MARK_MEMO, SUMMARY_BETA ...
    content_id = Column(String, nullable=True)
    mime_type = Column(String, nullable=True)
    path = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class JanitorTombstone(Base):
    """Recording ids the janitor deliberately deleted, so sync will not revive them.

    `upsert_chronos_recording` consults this table on *every* ingest, so a database
    that lacks it cannot ingest at all. The table was previously created by hand
    outside the repo, which left every fresh install and every rebuilt database
    raising `no such table: janitor_tombstones` on the first upsert. Declaring it
    here lets `init_db` create it like any other table; `create_all` skips it
    where it already exists, so deployments carrying the hand-made copy are
    untouched.

    Columns mirror that hand-made schema exactly. `recording_id` holds
    `notion:`-prefixed ids as well as Plaud ones, so it carries no foreign key,
    and `deleted_at` stays TEXT because existing rows hold ISO-8601 strings
    written by the out-of-band janitor.
    """

    __tablename__ = "janitor_tombstones"

    recording_id = Column(String, primary_key=True)
    deleted_at = Column(String, nullable=True)
    title = Column(String, nullable=True)
    reason = Column(String, nullable=True)

    def __repr__(self) -> str:
        return f"JanitorTombstone(recording_id={self.recording_id}, reason={self.reason})"
