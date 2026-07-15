"""Stable internal models for all public PLAUD transports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(kw_only=True)
class Provenance:
    source_transport: str
    source_operation: str
    source_version: str | None = None
    source_payload_hash: str | None = None
    retrieved_at: datetime = field(default_factory=utc_now)
    raw_payload_available: bool = False
    raw_payload: Any = field(default=None, repr=False)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_raw:
            value.pop("raw_payload", None)
        return _json_value(value)


@dataclass(kw_only=True)
class PlaudUser(Provenance):
    id: str | None = None
    email: str | None = None
    name: str | None = None


@dataclass(kw_only=True)
class PlaudFile(Provenance):
    id: str
    name: str | None = None
    created_at: str | None = None
    start_at: str | None = None
    duration_ms: float | None = None
    serial_number: str | None = None
    presigned_url: str | None = None


@dataclass
class PlaudFileListRequest:
    query: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    page: int = 1
    page_size: int = 20


@dataclass(kw_only=True)
class PlaudFilePage(Provenance):
    files: list[PlaudFile] = field(default_factory=list)
    page: int | None = None
    page_size: int | None = None
    total: int | None = None
    next_page: int | None = None


@dataclass(kw_only=True)
class PlaudNote(Provenance):
    file_id: str
    markdown: str | None = None
    action_items: list[Any] = field(default_factory=list)
    topics: list[Any] = field(default_factory=list)


@dataclass
class PlaudSpeaker:
    id: str | None = None
    label: str | None = None


@dataclass
class PlaudTranscriptSegment:
    start_seconds: float | None = None
    end_seconds: float | None = None
    text: str | None = None
    speaker: PlaudSpeaker | None = None
    language: str | None = None
    language_probability: float | None = None


@dataclass(kw_only=True)
class PlaudTranscript(Provenance):
    file_id: str
    text: str | None = None
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[PlaudTranscriptSegment] = field(default_factory=list)


@dataclass(kw_only=True)
class PlaudUploadSession(Provenance):
    file_id: str
    upload_id: str
    chunk_size: int
    parts: list[dict[str, Any]] = field(default_factory=list)
    download_url: str | None = None
    file_md5: str | None = None


@dataclass(kw_only=True)
class PlaudTranscriptionJob(Provenance):
    transcription_id: str
    status: str
    transcript: PlaudTranscript | None = None
    error: str | None = None


@dataclass
class PlaudIntegrationCapability:
    operation_id: str
    transport: str
    authentication_model: str
    safety: str
    implementation_status: str
    test_status: str
    source_file: str
    method: str | None = None
    path: str | None = None
    tool_name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    schema_hash: str | None = None
    discovered_at_runtime: bool = False
    last_successful_call_time: str | None = None
    last_failure: str | None = None
    last_latency_ms: int | None = None
    source_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class PlaudIntegrationStatus:
    account_rest: str
    official_mcp: str
    mcp_tool_count: int | None
    embedded_auth: str
    file_upload: str
    transcription: str
    region: str
    last_verified: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlaudCallEvent:
    timestamp: str
    correlation_id: str
    transport: str
    operation: str
    safety: str
    request_summary: str
    redacted_request: Any
    response_status: str | int | None
    redacted_response: Any
    duration_ms: int
    retry_count: int = 0
    schema_hash: str | None = None
    source_version: str | None = None
    error_classification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
