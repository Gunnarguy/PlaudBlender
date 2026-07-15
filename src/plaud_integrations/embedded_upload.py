"""PLAUD Embedded multipart upload orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

import requests

from .call_ledger import PlaudCallLedger, default_ledger
from .embedded_auth import PlaudRegion
from .errors import PlaudIntegrationError
from .models import PlaudCallEvent, PlaudUploadSession, payload_hash, utc_now
from .redaction import redact

ProgressCallback = Callable[[int, int], None]


class PlaudEmbeddedUploadClient:
    PRESIGN_PATH = "/open/partner/files/upload/generate-presigned-urls"
    COMPLETE_PATH = "/open/partner/files/upload/complete-upload"

    def __init__(
        self,
        user_access_token: str,
        *,
        region: str | PlaudRegion = PlaudRegion.US,
        session: requests.Session | None = None,
        timeout: float = 60,
        max_retries: int = 3,
        ledger: PlaudCallLedger = default_ledger,
    ):
        self.user_access_token = user_access_token
        self.region = PlaudRegion.parse(region)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.ledger = ledger

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.user_access_token}", "Content-Type": "application/json"}

    def _post(self, operation: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        correlation_id = str(uuid4())
        started = time.perf_counter()
        response_payload: Any = None
        status: Any = None
        retry_count = 0
        error_name = None
        try:
            for attempt in range(self.max_retries):
                retry_count = attempt
                try:
                    response = self.session.post(
                        self.region.base_url + path, headers=self._headers, json=body, timeout=self.timeout
                    )
                    status = response.status_code
                    response_payload = response.json() if response.content else {}
                    if response.status_code >= 500 and attempt + 1 < self.max_retries:
                        time.sleep(0.25 * (2**attempt))
                        continue
                    response.raise_for_status()
                    return response_payload
                except (requests.Timeout, requests.ConnectionError):
                    if attempt + 1 >= self.max_retries:
                        raise
                    time.sleep(0.25 * (2**attempt))
            raise AssertionError("unreachable")
        except requests.RequestException as exc:
            error_name = type(exc).__name__
            raise PlaudIntegrationError(
                f"PLAUD upload {operation} failed: {exc}", code="upload_http_error", retryable=True
            ) from exc
        finally:
            self.ledger.record(PlaudCallEvent(
                timestamp=utc_now().isoformat(), correlation_id=correlation_id,
                transport="plaud_embedded_rest", operation=operation, safety="mutating",
                request_summary=f"POST {path}", redacted_request=redact(body), response_status=status,
                redacted_response=redact(response_payload), duration_ms=int((time.perf_counter() - started) * 1000),
                retry_count=retry_count, error_classification=error_name,
            ))

    def generate_presigned_urls(self, filesize: int, filetype: str) -> PlaudUploadSession:
        if filesize <= 0 or filetype.lower().lstrip(".") not in {"mp3", "opus"}:
            raise PlaudIntegrationError("PLAUD upload requires a positive filesize and mp3 or opus filetype", code="invalid_upload", status_code=422)
        payload = self._post("generatePresignedUrls", self.PRESIGN_PATH, {"filesize": filesize, "filetype": filetype.lower().lstrip(".")})
        return PlaudUploadSession(
            file_id=payload["FileId"], upload_id=payload["UploadId"], chunk_size=int(payload["ChunkSize"]),
            parts=payload.get("Parts", []), source_transport="plaud_embedded_rest",
            source_operation="generatePresignedUrls", source_version="openapi-0.0.1",
            source_payload_hash=payload_hash(payload), raw_payload_available=True, raw_payload=payload,
        )

    def upload_part(self, presigned_url: str, chunk: bytes, *, part_number: int) -> dict[str, Any]:
        correlation_id = str(uuid4())
        started = time.perf_counter()
        last_error: Exception | None = None
        status: Any = None
        result: dict[str, Any] | None = None
        retry_count = 0
        error_name: str | None = None
        try:
            for attempt in range(self.max_retries):
                retry_count = attempt
                try:
                    response = self.session.put(presigned_url, data=chunk, timeout=self.timeout)
                    status = response.status_code
                    response.raise_for_status()
                    etag = response.headers.get("ETag")
                    if not etag:
                        raise PlaudIntegrationError("Presigned upload response did not include ETag", code="missing_etag")
                    result = {"PartNumber": part_number, "ETag": etag}
                    return result
                except (requests.RequestException, PlaudIntegrationError) as exc:
                    last_error = exc
                    if attempt + 1 < self.max_retries:
                        time.sleep(0.25 * (2**attempt))
            raise PlaudIntegrationError(
                f"Presigned part upload failed: {last_error}", code="part_upload_failed", retryable=True
            )
        except Exception as exc:
            error_name = type(exc).__name__
            raise
        finally:
            self.ledger.record(PlaudCallEvent(
                timestamp=utc_now().isoformat(), correlation_id=correlation_id,
                transport="plaud_upload", operation="uploadPresignedPart", safety="mutating",
                request_summary="PUT <presigned URL>",
                redacted_request={"part_number": part_number, "bytes": len(chunk)},
                response_status=status, redacted_response=redact(result),
                duration_ms=int((time.perf_counter() - started) * 1000), retry_count=retry_count,
                error_classification=error_name,
            ))

    def complete_upload(
        self, session: PlaudUploadSession, completed_parts: list[dict[str, Any]], filetype: str, file_md5: str | None = None
    ) -> PlaudUploadSession:
        body: dict[str, Any] = {
            "file_id": session.file_id, "upload_id": session.upload_id,
            "part_list": completed_parts, "filetype": filetype.lower().lstrip("."),
        }
        if file_md5:
            body["file_md5"] = file_md5
        payload = self._post("completeUpload", self.COMPLETE_PATH, body)
        session.download_url = payload.get("DownloadUrl")
        session.file_md5 = payload.get("FileMd5", file_md5)
        session.raw_payload = payload
        session.source_operation = "completeUpload"
        session.source_payload_hash = payload_hash(payload)
        return session

    def upload_file(self, path: str | Path, progress: ProgressCallback | None = None) -> PlaudUploadSession:
        path = Path(path)
        filetype = path.suffix.lower().lstrip(".")
        content = path.read_bytes()
        session = self.generate_presigned_urls(len(content), filetype)
        completed = []
        for part in session.parts:
            number = int(part["PartNumber"])
            start = (number - 1) * session.chunk_size
            chunk = content[start : start + session.chunk_size]
            completed.append(self.upload_part(part["PresignedUrl"], chunk, part_number=number))
            if progress:
                progress(min(start + len(chunk), len(content)), len(content))
        return self.complete_upload(session, completed, filetype, hashlib.md5(content).hexdigest())
