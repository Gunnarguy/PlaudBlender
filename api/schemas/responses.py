"""
Pydantic response/request models for the REST API.

These mirror the dataclasses in data_service.py but as Pydantic models
for automatic FastAPI serialization and OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Generic Wrappers ────────────────────────────────────────


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""


class ErrorResponse(BaseModel):
    detail: str


# ── Events ──────────────────────────────────────────────────


class EventOut(BaseModel):
    id: str
    recording_id: str
    start_ts: str
    end_ts: str
    day_of_week: str
    hour_of_day: int
    clean_text: str
    category: str
    category_confidence: Optional[float] = None
    sentiment: Optional[float] = None
    keywords: List[str] = Field(default_factory=list)
    speaker: str = "self_talk"
    duration_seconds: float = 0.0

    model_config = {"from_attributes": True}


# ── Recordings ──────────────────────────────────────────────


class RecordingSummaryOut(BaseModel):
    recording_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: int = 0
    duration_formatted: Optional[str] = None
    top_category: str = "unknown"
    event_count: int = 0
    time_range_formatted: Optional[str] = None
    time_is_estimated: Optional[bool] = None
    time_estimate_reason: Optional[str] = None
    title: Optional[str] = None
    plaud_ai_summary: Optional[str] = None
    cloud_status: Optional[str] = None

    model_config = {"from_attributes": True}


class RecordingDetailOut(BaseModel):
    summary: RecordingSummaryOut
    events: List[EventOut] = Field(default_factory=list)
    category_percentages: Optional[Dict[str, float]] = None
    transcript: Optional[str] = None
    ai_summary: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    workflow_status: Optional[Dict[str, Any]] = None
    plaud_transcript: Optional[str] = None


# ── Days ────────────────────────────────────────────────────


class DaySummaryOut(BaseModel):
    date: str
    date_display: Optional[str] = None
    total_duration_seconds: float = 0
    recording_count: int = 0
    event_count: int = 0
    top_category: Optional[str] = None
    category_percentages: Optional[Dict[str, float]] = None
    top_keywords: Optional[List[str]] = None
    ai_summary: Optional[str] = None
    recordings: Optional[List[RecordingSummaryOut]] = None

    model_config = {"from_attributes": True}


class DaysResponse(BaseModel):
    days: List[DaySummaryOut]
    total: int = 0


# ── Search ──────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=50, ge=1, le=200)
    categories: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SearchResultOut(BaseModel):
    event: EventOut
    score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class AIAnswerOut(BaseModel):
    answer: str
    model: str = ""
    response_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[SearchResultOut]
    ai_answer: Optional[AIAnswerOut] = None
    total: int = 0


class AskRequest(BaseModel):
    question: str
    reasoning: Optional[str] = None  # none|low|medium|high|xhigh


# ── Topics ──────────────────────────────────────────────────


class TopicOut(BaseModel):
    name: str
    count: int


class TopicOccurrenceOut(BaseModel):
    event_id: str
    recording_id: str
    timestamp: str
    text_snippet: str
    category: str


class TopicTimelineOut(BaseModel):
    topic: str
    total_occurrences: int
    recording_count: int
    occurrences: List[TopicOccurrenceOut]


# ── Knowledge Graph ─────────────────────────────────────────


class GraphDataOut(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# ── Statistics ──────────────────────────────────────────────


class StatsOut(BaseModel):
    total_recordings: int = 0
    total_events: int = 0
    total_days: int = 0
    total_duration_hours: float = 0.0
    categories: Dict[str, int] = Field(default_factory=dict)
    sentiment_avg: Optional[float] = None
    top_keywords: Optional[List[Dict[str, Any]]] = (
        None  # [{"keyword": str, "count": int}]
    )
    categories_by_hour: Optional[Dict[str, Any]] = None
    sentiment_distribution: Optional[Dict[str, int]] = None
    recent_days: Optional[List[Dict[str, Any]]] = None


# ── Pipeline / Sync ─────────────────────────────────────────


class PipelineRunRequest(BaseModel):
    stage: str = "full"  # full|ingest|process|index|graph


class PipelineRunResponse(BaseModel):
    status: str
    message: str = ""


class WorkflowSubmitRequest(BaseModel):
    days_back: int = 7
    limit: int = 3
    template_id: Optional[str] = None
    model: str = "gemini"


class WorkflowRefreshRequest(BaseModel):
    days_back: int = 30
    limit: int = 10


class RecordingWorkflowRequest(BaseModel):
    template_id: Optional[str] = None
    model: str = "gemini"


class CategoryOverrideRequest(BaseModel):
    category: str


# ── Costs ───────────────────────────────────────────────────


class SessionCostOut(BaseModel):
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    by_model: Dict[str, Any] = Field(default_factory=dict)
    by_type: Dict[str, Any] = Field(default_factory=dict)
    session_minutes: float = 0.0


class CostHistoryOut(BaseModel):
    days: int
    total_cost_usd: float = 0.0
    total_calls: int = 0
    by_model: Dict[str, Any] = Field(default_factory=dict)
    by_day: Optional[List[Dict[str, Any]]] = None


# ── X-Ray ───────────────────────────────────────────────────


class XRayEventOut(BaseModel):
    seq: int
    ts: float
    source: str
    op: str
    message: str
    duration_ms: Optional[float] = None
    detail: Optional[str] = None
    level: str = "info"


class XRayEventsResponse(BaseModel):
    events: List[XRayEventOut]
    latest_seq: int = 0


# ── Auth ────────────────────────────────────────────────────


class AuthURLResponse(BaseModel):
    auth_url: str
    state: str


class TokenExchangeRequest(BaseModel):
    code: str
    state: Optional[str] = None


class TokenStatusOut(BaseModel):
    is_authenticated: bool = False
    has_access_token: bool = False
    expires_at: Optional[str] = None
    workspace_name: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# ── Notion ──────────────────────────────────────────────────


class NotionDatabaseSelectRequest(BaseModel):
    db_id: str


class NotionImportRequest(BaseModel):
    process: bool = True
    index: bool = True


class NotionRecordingOut(BaseModel):
    page_id: str
    title: str
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None
    url: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    source: str = "notion"
    matched_recording_id: Optional[str] = None


class NotionRecordingsResponse(BaseModel):
    recordings: List[NotionRecordingOut]
    total: int = 0
    has_more: bool = False


# ── Health ──────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class SystemStatusOut(BaseModel):
    database: Dict[str, Any] = Field(default_factory=dict)
    qdrant: Dict[str, Any] = Field(default_factory=dict)
    gemini: Dict[str, Any] = Field(default_factory=dict)
    openai: Dict[str, Any] = Field(default_factory=dict)
    plaud: Dict[str, Any] = Field(default_factory=dict)
    notion: Dict[str, Any] = Field(default_factory=dict)
