"""Compatibility wrapper around PlaudBlender's existing account REST client."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from .call_ledger import PlaudCallLedger, default_ledger
from .models import (
    PlaudCallEvent,
    PlaudFile,
    PlaudFileListRequest,
    PlaudFilePage,
    PlaudNote,
    PlaudSpeaker,
    PlaudTranscript,
    PlaudTranscriptSegment,
    PlaudUser,
    payload_hash,
    utc_now,
)
from .redaction import redact


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


class PlaudLegacyAccountAdapter:
    """Preserves the current third-party account REST behavior unchanged."""

    source_version = "compatibility-third-party-rest"

    def __init__(self, client=None, *, ledger: PlaudCallLedger = default_ledger):
        if client is None:
            from src.plaud_client import PlaudClient

            client = PlaudClient()
        self.client = client
        self.ledger = ledger

    def _call(self, operation: str, request: dict[str, Any], callback):
        correlation_id = str(uuid4())
        started = time.perf_counter()
        payload: Any = None
        status = "success"
        error_name: str | None = None
        try:
            payload = callback()
            return payload
        except Exception as exc:
            status = "error"
            error_name = type(exc).__name__
            raise
        finally:
            self.ledger.record(PlaudCallEvent(
                timestamp=utc_now().isoformat(), correlation_id=correlation_id,
                transport="plaud_account_rest", operation=operation, safety="read-only",
                request_summary=operation, redacted_request=redact(request),
                response_status=status, redacted_response=redact(payload),
                duration_ms=int((time.perf_counter() - started) * 1000),
                source_version=self.source_version, error_classification=error_name,
            ))

    def _provenance(self, operation: str, payload: Any) -> dict[str, Any]:
        return {
            "source_transport": "plaud_account_rest",
            "source_operation": operation,
            "source_version": self.source_version,
            "source_payload_hash": payload_hash(payload),
            "raw_payload_available": True,
            "raw_payload": payload,
        }

    def get_current_user(self) -> PlaudUser:
        payload = self._call("get_user", {}, self.client.get_user)
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        return PlaudUser(
            id=_first(data, "id", "user_id", "uuid"),
            email=_first(data, "email", "mail"),
            name=_first(data, "name", "nickname", "username"),
            **self._provenance("get_user", payload),
        )

    def list_files(self, request: PlaudFileListRequest) -> PlaudFilePage:
        payload = self._call(
            "list_recordings", {"page": request.page, "page_size": request.page_size},
            lambda: self.client.list_recordings(page=request.page, page_size=request.page_size),
        )
        if isinstance(payload, list):
            rows, meta = payload, {}
        else:
            data = payload.get("data", payload)
            rows = data if isinstance(data, list) else _first(data, "files", "recordings", "list", default=[])
            meta = data if isinstance(data, dict) else {}
        files = [self._normalize_file(row, "list_recordings") for row in rows if isinstance(row, dict)]
        return PlaudFilePage(
            files=files,
            page=_first(meta, "page", "current_page", default=request.page),
            page_size=_first(meta, "page_size", "limit", default=request.page_size),
            total=_first(meta, "total", "total_count"),
            next_page=_first(meta, "next_page"),
            **self._provenance("list_recordings", payload),
        )

    def _normalize_file(self, data: dict[str, Any], operation: str) -> PlaudFile:
        return PlaudFile(
            id=str(_first(data, "id", "file_id", "recording_id", default="")),
            name=_first(data, "name", "title", "filename"),
            created_at=_first(data, "created_at", "createdAt"),
            start_at=_first(data, "start_at", "startAt", "start_time"),
            duration_ms=_first(data, "duration", "duration_ms"),
            serial_number=_first(data, "serial_number", "serialNumber"),
            presigned_url=_first(data, "presigned_url", "download_url", "url"),
            **self._provenance(operation, data),
        )

    def get_file(self, file_id: str) -> PlaudFile:
        payload = self._call("get_recording", {"file_id": file_id}, lambda: self.client.get_recording(file_id))
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        return self._normalize_file(data, "get_recording")

    def get_note(self, file_id: str) -> PlaudNote:
        payload = self._call("get_summary", {"file_id": file_id}, lambda: self.client.get_summary(file_id))
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        return PlaudNote(
            file_id=file_id,
            markdown=_first(data, "markdown", "summary", "content", "text"),
            action_items=_first(data, "action_items", "actionItems", default=[]) or [],
            topics=_first(data, "topics", "key_topics", default=[]) or [],
            **self._provenance("get_summary", payload),
        )

    def get_transcript(self, file_id: str) -> PlaudTranscript:
        payload = self._call("get_transcript", {"file_id": file_id}, lambda: self.client.get_transcript(file_id))
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        rows = _first(data, "segments", "source_list", "results", default=[]) or []
        segments = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            speaker_id = _first(row, "speaker_id", "speaker", "speakerId")
            segments.append(
                PlaudTranscriptSegment(
                    start_seconds=_first(row, "start", "start_time"),
                    end_seconds=_first(row, "end", "end_time"),
                    text=_first(row, "text", "content"),
                    speaker=PlaudSpeaker(id=str(speaker_id), label=str(speaker_id)) if speaker_id else None,
                    language=_first(row, "language"),
                )
            )
        return PlaudTranscript(
            file_id=file_id,
            text=_first(data, "text", "transcript", "content"),
            language=_first(data, "language"),
            duration_seconds=_first(data, "duration"),
            segments=segments,
            **self._provenance("get_transcript", payload),
        )
