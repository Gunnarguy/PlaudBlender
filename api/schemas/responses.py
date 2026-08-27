"""
Pydantic response/request models for the REST API.

These mirror the dataclasses in data_service.py but as Pydantic models
for automatic FastAPI serialization and OpenAPI documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.chronos.pipeline_stages import (
    ACCEPTED_PIPELINE_STAGES,
    normalize_pipeline_stage,
)


# ── Generic Wrappers ────────────────────────────────────────


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""


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
    coverage_status: Optional[str] = None
    coverage_note: Optional[str] = None
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
    reasoning_summary: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[SearchResultOut]
    ai_answer: Optional[AIAnswerOut] = None
    total: int = 0


class AskRequest(BaseModel):
    question: str
    previous_response_id: Optional[str] = None
    model: Optional[str] = None
    reasoning: Optional[str] = None  # none|low|medium|high|xhigh
    reasoning_summary: Optional[str] = None  # off|auto
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(default=None, ge=16, le=65536)
    verbosity: Optional[str] = None  # low|medium|high
    service_tier: Optional[str] = None  # auto|default|flex|priority


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
    stage: str = Field(
        default="full",
        description=(
            "Pipeline stage to run. Aliases all_history and full_history normalize to backfill."
        ),
        json_schema_extra={"enum": list(ACCEPTED_PIPELINE_STAGES)},
    )
    days_back: int = Field(default=7, ge=1, le=365)

    @property
    def normalized_stage(self) -> str:
        return normalize_pipeline_stage(self.stage)


class PipelineRunResponse(BaseModel):
    status: str
    message: str = ""
    run_id: Optional[str] = None


class WorkflowSubmitRequest(BaseModel):
    days_back: int = 7
    limit: int = 3
    template_id: Optional[str] = None
    model: str = "openai"


class WorkflowRefreshRequest(BaseModel):
    days_back: int = 30
    limit: int = 10


class RecordingWorkflowRequest(BaseModel):
    template_id: Optional[str] = None
    model: str = "openai"


class UploadProcessRequest(BaseModel):
    file_paths: Optional[List[str]] = None
    template_id: Optional[str] = None
    model: str = "openai"


class UploadProcessItemOut(BaseModel):
    path: str
    file_id: Optional[str] = None
    workflow_id: Optional[str] = None
    error: Optional[str] = None


class UploadProcessResultOut(BaseModel):
    uploaded_count: int = 0
    error_count: int = 0
    uploaded: List[UploadProcessItemOut] = Field(default_factory=list)
    errors: List[UploadProcessItemOut] = Field(default_factory=list)


class SyncFailureItemOut(BaseModel):
    recording_id: Optional[str] = None
    source: Optional[str] = None
    title: Optional[str] = None
    error: str = ""
    reason: Optional[str] = None


class SyncFailureSummaryOut(BaseModel):
    actionable_count: int = 0
    archived_count: int = 0
    actionable: List[SyncFailureItemOut] = Field(default_factory=list)
    archived: List[SyncFailureItemOut] = Field(default_factory=list)


class StackControlResponse(BaseModel):
    action: str
    status: str
    message: str = ""
    output: str = ""
    public_url: Optional[str] = None


class BackupInfoOut(BaseModel):
    filename: str
    created_at: str
    size_bytes: int
    download_path: str
    message: str = ""


class CategoryOverrideRequest(BaseModel):
    category: str


# ── Server Settings ────────────────────────────────────────


class ServerSettingsFlagsOut(BaseModel):
    has_gemini_api_key: bool = False
    has_openai_api_key: bool = False
    has_qdrant_api_key: bool = False
    has_notion_token: bool = False
    has_notion_oauth: bool = False


class ServerSettingsOut(BaseModel):
    processing_provider: str = "openai"
    cleaning_model: str = ""
    analyst_model: str = ""
    embedding_model: str = ""
    openai_model: str = ""
    thinking_level: str = "high"
    openai_temperature: float = 0.7
    embedding_dim: int = 768
    plaud_language: str = "en"
    plaud_diarization: bool = True
    log_level: str = "INFO"
    custom_categories: str = ""
    notion_weekday_start: str = "07:30"
    notion_weekend_start: str = "12:00"
    qdrant_url: str = ""
    qdrant_collection_name: str = ""
    chronos_openai_enabled: bool = False
    chronos_local_llm_enabled: bool = False
    chronos_local_llm_provider: str = "ollama"
    chronos_local_llm_base_url: str = "http://127.0.0.1:11434"
    chronos_local_llm_model: str = "qwen2.5:0.5b"
    chronos_local_llm_max_context: int = 4096
    chronos_local_llm_allowed_tasks: str = "json_repair,entity_extract,classify,ask"
    chronos_poll_interval: int = 1800
    chronos_enable_notion_import: bool = True
    chronos_notion_import_batch_size: int = 25
    chronos_self_heal_limit: int = 10
    chronos_embed_batch_size: int = 20
    chronos_index_events_per_limit: int = 0
    chronos_autosync_process_limit: int = 10
    chronos_autosync_index_limit: int = 10
    chronos_autosync_index_timeout: int = 900
    chronos_autosync_graph_limit: int = 10
    chronos_autosync_max_load_avg: float = 3.5
    chronos_autosync_min_available_mb: int = 700
    chronos_autosync_max_swap_used_mb: int = 512
    chronos_autosync_defer_seconds: int = 90
    chronos_stats_enable_plaud_cloud: bool = False
    flags: ServerSettingsFlagsOut = Field(default_factory=ServerSettingsFlagsOut)


class ServerSettingsUpdateRequest(BaseModel):
    processing_provider: Optional[str] = None
    cleaning_model: Optional[str] = None
    analyst_model: Optional[str] = None
    embedding_model: Optional[str] = None
    openai_model: Optional[str] = None
    thinking_level: Optional[str] = None
    openai_temperature: Optional[float] = None
    embedding_dim: Optional[int] = None
    plaud_language: Optional[str] = None
    plaud_diarization: Optional[bool] = None
    log_level: Optional[str] = None
    custom_categories: Optional[str] = None
    notion_weekday_start: Optional[str] = None
    notion_weekend_start: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_collection_name: Optional[str] = None
    chronos_openai_enabled: Optional[bool] = None
    chronos_local_llm_enabled: Optional[bool] = None
    chronos_local_llm_provider: Optional[str] = None
    chronos_local_llm_base_url: Optional[str] = None
    chronos_local_llm_model: Optional[str] = None
    chronos_local_llm_max_context: Optional[int] = None
    chronos_local_llm_allowed_tasks: Optional[str] = None
    chronos_poll_interval: Optional[int] = None
    chronos_enable_notion_import: Optional[bool] = None
    chronos_notion_import_batch_size: Optional[int] = None
    chronos_self_heal_limit: Optional[int] = None
    chronos_embed_batch_size: Optional[int] = None
    chronos_index_events_per_limit: Optional[int] = None
    chronos_autosync_process_limit: Optional[int] = None
    chronos_autosync_index_limit: Optional[int] = None
    chronos_autosync_index_timeout: Optional[int] = None
    chronos_autosync_graph_limit: Optional[int] = None
    chronos_autosync_max_load_avg: Optional[float] = None
    chronos_autosync_min_available_mb: Optional[int] = None
    chronos_autosync_max_swap_used_mb: Optional[int] = None
    chronos_autosync_defer_seconds: Optional[int] = None
    chronos_stats_enable_plaud_cloud: Optional[bool] = None


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
    run_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    recording_id: Optional[str] = None
    event_id: Optional[str] = None
    stage: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class XRayEventsResponse(BaseModel):
    events: List[XRayEventOut]
    latest_seq: int = 0


class TraceRunOut(BaseModel):
    run_id: str
    trigger: Optional[str] = None
    source: Optional[str] = None
    status: str = "running"
    title: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: Optional[float] = None
    summary: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class TraceSpanOut(BaseModel):
    span_id: str
    run_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    recording_id: Optional[str] = None
    event_id: Optional[str] = None
    stage: Optional[str] = None
    operation: str
    source: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    status: str = "running"
    level: str = "info"
    message: Optional[str] = None
    detail: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: Optional[float] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    input_snippet: Optional[str] = None
    output_snippet: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    request_id: Optional[str] = None
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class TraceDagOut(BaseModel):
    run: Optional[TraceRunOut] = None
    spans: List[TraceSpanOut] = Field(default_factory=list)


class RecordingProcessingOut(BaseModel):
    recording_id: str
    status: str = "unknown"
    latest_error: Optional[str] = None
    runs: List[TraceRunOut] = Field(default_factory=list)
    spans: List[TraceSpanOut] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)


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
    batch_size: int = 0
    force: bool = False


class NotionMatchOverrideRequest(BaseModel):
    page_id: str
    recording_id: Optional[str] = None
    clear: bool = False


class NotionBulkMatchOverrideRequest(BaseModel):
    overrides: List[NotionMatchOverrideRequest]
    stop_on_error: bool = False


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
    local_llm: Dict[str, Any] = Field(default_factory=dict)
