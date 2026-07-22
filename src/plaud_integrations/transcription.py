"""PLAUD transcription submit, polling, normalization, timeout, and failure handling."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import requests

from .call_ledger import PlaudCallLedger, default_ledger
from .embedded_auth import PlaudRegion
from .errors import PlaudConfigurationError, PlaudIntegrationError
from .models import (
    PlaudCallEvent, PlaudSpeaker, PlaudTranscript, PlaudTranscriptSegment,
    PlaudTranscriptionJob, payload_hash, utc_now,
)
from .redaction import redact

IN_PROGRESS = {"PENDING", "RECEIVED", "STARTED", "PROGRESS"}
FAILED = {"FAILURE", "REVOKED"}


class PlaudTranscriptionClient:
    PATH = "/open/partner/ai/transcriptions/"

    def __init__(
        self,
        client_id: str | None,
        api_key: str | None,
        *,
        region: str | PlaudRegion = PlaudRegion.US,
        session: requests.Session | None = None,
        timeout: float = 30,
        ledger: PlaudCallLedger = default_ledger,
    ):
        self.client_id = (client_id or "").strip()
        self.api_key = (api_key or "").strip()
        if not self.client_id or not self.api_key:
            raise PlaudConfigurationError("PLAUD_EMBEDDED_CLIENT_ID and PLAUD_EMBEDDED_API_KEY are required")
        self.region = PlaudRegion.parse(region)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.ledger = ledger

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Client-Id": self.client_id, "X-Client-Api-Key": self.api_key, "Content-Type": "application/json"}

    def _request(self, method: str, path: str, operation: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        correlation_id = str(uuid4())
        started = time.perf_counter()
        status: Any = None
        payload: Any = None
        error_name = None
        try:
            response = self.session.request(
                method, self.region.base_url + path, headers=self._headers, json=body, timeout=self.timeout
            )
            status = response.status_code
            payload = response.json() if response.content else {}
            response.raise_for_status()
            return payload
        except requests.RequestException as exc:
            error_name = type(exc).__name__
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f" body={exc.response.text[:300]}"
            raise PlaudIntegrationError(
                f"PLAUD transcription {operation} failed: {exc}{detail}", code="transcription_http_error",
                retryable=isinstance(exc, (requests.Timeout, requests.ConnectionError)),
            ) from exc
        finally:
            self.ledger.record(PlaudCallEvent(
                timestamp=utc_now().isoformat(), correlation_id=correlation_id,
                transport="plaud_embedded_rest", operation=operation,
                safety="mutating" if method == "POST" else "read-only",
                request_summary=f"{method} {path}", redacted_request=redact(body or {}),
                response_status=status, redacted_response=redact(payload),
                duration_ms=int((time.perf_counter() - started) * 1000), error_classification=error_name,
            ))

    @staticmethod
    def _job(payload: dict[str, Any]) -> PlaudTranscriptionJob:
        task_id = str(payload.get("transcription_id", ""))
        status = str(payload.get("status", "UNKNOWN")).upper()
        data = payload.get("data") or {}
        transcript = None
        if status == "SUCCESS":
            segments = []
            for item in data.get("results", []) or []:
                speaker_id = item.get("speaker_id")
                probability = item.get("language_probability", item.get("language_probabilitiy"))
                try:
                    probability = float(probability) if probability is not None else None
                except (TypeError, ValueError):
                    probability = None
                segments.append(PlaudTranscriptSegment(
                    start_seconds=item.get("start"), end_seconds=item.get("end"), text=item.get("text"),
                    speaker=PlaudSpeaker(id=speaker_id, label=speaker_id) if speaker_id else None,
                    language=item.get("language"), language_probability=probability,
                ))
            transcript = PlaudTranscript(
                file_id=task_id, text=data.get("text"), language=data.get("language"),
                duration_seconds=data.get("duration"), segments=segments,
                source_transport="plaud_embedded_rest", source_operation="getTranscription",
                source_version="openapi-0.0.1", source_payload_hash=payload_hash(payload),
                raw_payload_available=True, raw_payload=payload,
            )
        return PlaudTranscriptionJob(
            transcription_id=task_id, status=status, transcript=transcript,
            error=str(data.get("error")) if status in FAILED and data.get("error") else None,
            source_transport="plaud_embedded_rest", source_operation="getTranscription",
            source_version="openapi-0.0.1", source_payload_hash=payload_hash(payload),
            raw_payload_available=True, raw_payload=payload,
        )

    def submit(
        self,
        file_url: str,
        *,
        language: str = "auto",
        model: str = "plaud-fast-whisper",
        detection_level: str = "segment",
        decode_silence: bool = False,
        diarization: bool = False,
        return_embedding: bool = False,
    ) -> PlaudTranscriptionJob:
        body = {
            "file_url": file_url,
            "params": {
                "transcribe": {"language": language, "model": model, "detection_level": detection_level},
                "vad": {"decode_silence": decode_silence},
                "diarization": {"enabled": diarization, "return_embedding": return_embedding},
            },
        }
        return self._job(self._request("POST", self.PATH, "createTranscription", body))

    def get(self, transcription_id: str) -> PlaudTranscriptionJob:
        return self._job(self._request("GET", self.PATH + transcription_id, "getTranscription"))

    def poll(self, transcription_id: str, *, timeout_seconds: float = 600, interval_seconds: float = 2) -> PlaudTranscriptionJob:
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = self.get(transcription_id)
            if job.status == "SUCCESS":
                return job
            if job.status in FAILED:
                raise PlaudIntegrationError(
                    job.error or f"PLAUD transcription ended with {job.status}", code="transcription_failed"
                )
            if job.status not in IN_PROGRESS:
                raise PlaudIntegrationError(f"Unknown PLAUD transcription status: {job.status}", code="unknown_transcription_status")
            if time.monotonic() >= deadline:
                raise PlaudIntegrationError("PLAUD transcription polling timed out", code="transcription_timeout", status_code=504, retryable=True)
            time.sleep(min(interval_seconds, max(0, deadline - time.monotonic())))
