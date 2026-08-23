"""Authenticated API facade for separated public PLAUD integrations."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.auth.jwt import require_auth
from src.plaud_integrations.capability_manifest import load_manifest, write_manifest
from src.plaud_integrations.embedded_auth import PlaudEmbeddedAuthClient
from src.plaud_integrations.embedded_upload import PlaudEmbeddedUploadClient
from src.plaud_integrations.errors import PlaudIntegrationError
from src.plaud_integrations.mcp_account import EXPECTED_PUBLIC_TOOLS, PlaudMCPAccountAdapter, REVIEWED_TOOLS
from src.plaud_integrations.redaction import redact
from src.plaud_integrations.transcription import PlaudTranscriptionClient

# Shared adapter so the persistent MCP subprocess survives across
# requests — per-request construction re-spawned npx every call.
_MCP_ADAPTER = PlaudMCPAccountAdapter()

router = APIRouter(
    prefix="/api/v1/plaud/integrations",
    tags=["plaud-integrations"],
    dependencies=[Depends(require_auth)],
)


class MCPInvocationRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirm_mutating: bool = False


class UserTokenRequest(BaseModel):
    user_id: str = Field(min_length=6, max_length=120)
    expires_in: int = Field(default=86400, gt=0)


class PresignRequest(BaseModel):
    user_access_token: str
    filesize: int = Field(gt=0)
    filetype: str


class CompleteUploadRequest(BaseModel):
    user_access_token: str
    file_id: str
    upload_id: str
    part_list: list[dict[str, Any]]
    filetype: str
    file_md5: str | None = None


class TranscriptionRequest(BaseModel):
    file_url: str
    language: str = "auto"
    model: str = "plaud-fast-whisper"
    detection_level: str = "segment"
    decode_silence: bool = False
    diarization: bool = False
    return_embedding: bool = False


def _correlation_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid4())


def _settings() -> dict[str, str]:
    return {
        "client_id": os.getenv("PLAUD_EMBEDDED_CLIENT_ID", "").strip(),
        "secret_key": os.getenv("PLAUD_EMBEDDED_SECRET_KEY", "").strip(),
        "api_key": os.getenv("PLAUD_EMBEDDED_API_KEY", "").strip(),
        "region": os.getenv("PLAUD_EMBEDDED_REGION", "us").strip().lower(),
    }


def _auth_client() -> PlaudEmbeddedAuthClient:
    settings = _settings()
    return PlaudEmbeddedAuthClient(settings["client_id"], settings["secret_key"], region=settings["region"])


def _transcription_client() -> PlaudTranscriptionClient:
    settings = _settings()
    return PlaudTranscriptionClient(settings["client_id"], settings["api_key"], region=settings["region"])


def _raise(exc: Exception, correlation_id: str) -> None:
    if isinstance(exc, PlaudIntegrationError):
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict(correlation_id)) from exc
    raise HTTPException(
        status_code=502,
        detail={"error": {"code": "unexpected_integration_error", "message": str(exc), "retryable": False}, "correlation_id": correlation_id},
    ) from exc


@router.get("/status")
async def integration_status(request: Request):
    settings = _settings()
    legacy_configured = bool(os.getenv("PLAUD_CLIENT_ID") and os.getenv("PLAUD_CLIENT_SECRET"))
    mcp_auth = await asyncio.to_thread(_MCP_ADAPTER.authentication_status)
    manifest = load_manifest()
    discovered_mcp = [
        item for item in manifest.get("capabilities", [])
        if item.get("transport") == "plaud_mcp" and item.get("discovered_at_runtime")
    ]
    mcp_label = {
        "connected": "Connected",
        "reauthorization_required": "Reauthorization required",
    }.get(mcp_auth.state, "Unavailable")
    return {
        "account_rest": "Unverified" if legacy_configured else "Missing",
        "official_mcp": mcp_label,
        "mcp_auth": mcp_auth.to_dict(),
        "mcp_tool_count": len(discovered_mcp) if discovered_mcp else None,
        "embedded_auth": "Configured" if settings["client_id"] and settings["secret_key"] else "Missing",
        "file_upload": "Ready" if settings["client_id"] and settings["secret_key"] else "Missing prerequisites",
        "transcription": "Ready" if settings["client_id"] and settings["api_key"] else "Missing prerequisites",
        "region": settings["region"],
        "last_verified": manifest["generated_at"],
        "correlation_id": _correlation_id(request),
    }


@router.get("/capabilities")
async def capabilities(request: Request):
    return {**load_manifest(), "correlation_id": _correlation_id(request)}


@router.get("/files/{file_id}/audio")
async def stream_file_audio(file_id: str, request: Request):
    """Stream a recording's audio through the broker. Presigned URLs are
    redacted in every JSON surface (telemetry safety), so the broker
    dereferences the URL server-side and streams the bytes instead."""
    correlation_id = _correlation_id(request)
    try:
        result = await asyncio.to_thread(_MCP_ADAPTER.call_tool, "get_file", {"file_id": file_id})
        payload = result.structured_content or {}
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        if isinstance(data, list):
            data = data[0] if data else {}
        url = data.get("presigned_url") if isinstance(data, dict) else None
        if not url:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "no_audio", "message": "No presigned audio URL for this recording"},
                        "correlation_id": correlation_id},
            )
        import urllib.request as _urllib_request
        # AVPlayer will not scrub without byte-range support, so forward the
        # client's Range header to S3 and mirror the 206 back verbatim.
        upstream_request = _urllib_request.Request(url)
        range_header = request.headers.get("Range")
        if range_header:
            upstream_request.add_header("Range", range_header)
        upstream = await asyncio.to_thread(_urllib_request.urlopen, upstream_request, None, 120)
        # S3 serves these as binary/octet-stream, which leaves clients unable to
        # tell what the file is. Plaud stores audiofiles/{id}.mp3.
        content_type = upstream.headers.get("Content-Type", "") or ""
        if content_type in ("", "binary/octet-stream", "application/octet-stream"):
            content_type = "audio/mpeg"

        def _chunks():
            try:
                while True:
                    block = upstream.read(1 << 16)
                    if not block:
                        break
                    yield block
            finally:
                upstream.close()

        stream_headers = {
            "X-Correlation-Id": correlation_id,
            "Content-Disposition": f'inline; filename="{file_id}.mp3"',
            "Accept-Ranges": "bytes",
        }
        for passthrough in ("Content-Range", "Content-Length"):
            value = upstream.headers.get(passthrough)
            if value:
                stream_headers[passthrough] = value

        return StreamingResponse(
            _chunks(),
            status_code=206 if upstream.status == 206 else 200,
            media_type=content_type,
            headers=stream_headers,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise(exc, correlation_id)


@router.get("/mcp/tools")
async def mcp_tools(request: Request):
    correlation_id = _correlation_id(request)
    try:
        tools = await asyncio.to_thread(_MCP_ADAPTER.discover_tools)
        await asyncio.to_thread(lambda: write_manifest(discovered_mcp_tools=tools))
        return {"tools": [tool.to_dict() for tool in tools], "count": len(tools), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)


@router.get("/mcp/auth/status")
async def mcp_auth_status(request: Request):
    correlation_id = _correlation_id(request)
    try:
        status = await asyncio.to_thread(_MCP_ADAPTER.authentication_status, force_refresh=True)
        return {**status.to_dict(), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)


@router.post("/mcp/auth/reconnect")
async def reconnect_mcp_auth(request: Request):
    correlation_id = _correlation_id(request)
    try:
        status = await asyncio.to_thread(_MCP_ADAPTER.reconnect_from_account_oauth)
        return {**status.to_dict(), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)


@router.post("/mcp/tools/{tool_name}")
async def invoke_mcp_tool(tool_name: str, body: MCPInvocationRequest, request: Request):
    correlation_id = _correlation_id(request)
    if tool_name not in REVIEWED_TOOLS:
        raise HTTPException(status_code=400, detail={"error": {"code": "unknown_mcp_tool", "message": f"Tool is not reviewed: {tool_name}"}, "correlation_id": correlation_id})
    if EXPECTED_PUBLIC_TOOLS[tool_name] == "auth" and not body.confirm_mutating:
        raise HTTPException(status_code=409, detail={"error": {"code": "confirmation_required", "message": "Set confirm_mutating=true for authentication-changing MCP calls"}, "correlation_id": correlation_id})
    try:
        result = await asyncio.to_thread(_MCP_ADAPTER.call_tool, tool_name, body.arguments)
        return {
            "tool_name": result.tool_name,
            "structured_content": redact(result.structured_content),
            "text_content": redact(result.text_content),
            "duration_ms": result.duration_ms,
            "schema_hash": result.schema_hash,
            "correlation_id": correlation_id,
        }
    except Exception as exc:
        _raise(exc, correlation_id)


@router.post("/embedded/user-token")
async def issue_user_token(body: UserTokenRequest, request: Request):
    correlation_id = _correlation_id(request)
    try:
        client = _auth_client()
        partner = await asyncio.to_thread(client.acquire_partner_token)
        token = await asyncio.to_thread(client.issue_user_token, partner.access_token, body.user_id, body.expires_in)
        return {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "expires_in": token.expires_in,
            "expires_at": token.expires_at,
            "correlation_id": correlation_id,
        }
    except Exception as exc:
        _raise(exc, correlation_id)


@router.post("/embedded/uploads/presign")
async def presign_upload(body: PresignRequest, request: Request):
    correlation_id = _correlation_id(request)
    try:
        client = PlaudEmbeddedUploadClient(body.user_access_token, region=_settings()["region"])
        session = await asyncio.to_thread(client.generate_presigned_urls, body.filesize, body.filetype)
        return {**session.to_dict(include_raw=False), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)


@router.post("/embedded/uploads/complete")
async def complete_upload(body: CompleteUploadRequest, request: Request):
    correlation_id = _correlation_id(request)
    try:
        from src.plaud_integrations.models import PlaudUploadSession

        session = PlaudUploadSession(
            file_id=body.file_id, upload_id=body.upload_id, chunk_size=0,
            source_transport="plaud_embedded_rest", source_operation="completeUpload",
        )
        client = PlaudEmbeddedUploadClient(body.user_access_token, region=_settings()["region"])
        completed = await asyncio.to_thread(client.complete_upload, session, body.part_list, body.filetype, body.file_md5)
        return {**completed.to_dict(include_raw=False), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)


@router.post("/embedded/transcriptions")
async def submit_transcription(body: TranscriptionRequest, request: Request):
    correlation_id = _correlation_id(request)
    try:
        job = await asyncio.to_thread(
            _transcription_client().submit, body.file_url, language=body.language, model=body.model,
            detection_level=body.detection_level, decode_silence=body.decode_silence,
            diarization=body.diarization, return_embedding=body.return_embedding,
        )
        return {**job.to_dict(include_raw=False), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)


@router.get("/embedded/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: str, request: Request):
    correlation_id = _correlation_id(request)
    try:
        job = await asyncio.to_thread(_transcription_client().get, transcription_id)
        return {**job.to_dict(include_raw=False), "correlation_id": correlation_id}
    except Exception as exc:
        _raise(exc, correlation_id)
