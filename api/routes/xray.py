"""X-Ray Activity Monitor endpoints."""

from __future__ import annotations

import asyncio
import hmac
import os
import queue
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from api.schemas.responses import (
    TraceDagOut,
    TraceRunOut,
    TraceSpanOut,
    XRayEventsResponse,
    XRayEventOut,
)
from api.routes._ttl_cache import TTLCache

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/xray",
    tags=["xray"],
)

_cache = TTLCache()
_THROUGHPUT_TTL_SECONDS = 1.0


def _event_out(e: dict[str, Any]) -> XRayEventOut:
    """Convert an internal X-Ray event dict to the public response model."""
    return XRayEventOut(
        seq=e["seq"],
        ts=e["ts"],
        source=e["source"],
        op=e["op"],
        message=e["message"],
        duration_ms=e.get("duration_ms"),
        detail=e.get("detail"),
        level=e.get("level", "info"),
        run_id=e.get("run_id"),
        span_id=e.get("span_id"),
        parent_span_id=e.get("parent_span_id"),
        recording_id=e.get("recording_id"),
        event_id=e.get("event_id"),
        stage=e.get("stage"),
        provider=e.get("provider"),
        model=e.get("model"),
        status=e.get("status"),
        input_tokens=e.get("input_tokens"),
        output_tokens=e.get("output_tokens"),
        cost_usd=e.get("cost_usd"),
        request_id=e.get("request_id"),
        metadata=e.get("metadata"),
    )


def _run_out(run) -> TraceRunOut:
    return TraceRunOut(
        run_id=str(run.run_id),
        trigger=getattr(run, "trigger", None),
        source=getattr(run, "source", None),
        status=getattr(run, "status", "running"),
        title=getattr(run, "title", None),
        started_at=str(run.started_at) if getattr(run, "started_at", None) else None,
        ended_at=str(run.ended_at) if getattr(run, "ended_at", None) else None,
        duration_ms=getattr(run, "duration_ms", None),
        summary=getattr(run, "summary", None),
        metadata=getattr(run, "run_metadata", None),
        error_message=getattr(run, "error_message", None),
    )


def _span_out(span) -> TraceSpanOut:
    return TraceSpanOut(
        span_id=str(span.span_id),
        run_id=getattr(span, "run_id", None),
        parent_span_id=getattr(span, "parent_span_id", None),
        recording_id=getattr(span, "recording_id", None),
        event_id=getattr(span, "event_id", None),
        stage=getattr(span, "stage", None),
        operation=getattr(span, "operation", "unknown"),
        source=getattr(span, "source", None),
        provider=getattr(span, "provider", None),
        model=getattr(span, "model", None),
        status=getattr(span, "status", "running"),
        level=getattr(span, "level", "info"),
        message=getattr(span, "message", None),
        detail=getattr(span, "detail", None),
        started_at=str(span.started_at) if getattr(span, "started_at", None) else None,
        ended_at=str(span.ended_at) if getattr(span, "ended_at", None) else None,
        duration_ms=getattr(span, "duration_ms", None),
        input_hash=getattr(span, "input_hash", None),
        output_hash=getattr(span, "output_hash", None),
        input_snippet=getattr(span, "input_snippet", None),
        output_snippet=getattr(span, "output_snippet", None),
        input_tokens=getattr(span, "input_tokens", None),
        output_tokens=getattr(span, "output_tokens", None),
        cost_usd=getattr(span, "cost_usd", None),
        request_id=getattr(span, "request_id", None),
        retry_count=int(getattr(span, "retry_count", 0) or 0),
        metadata=getattr(span, "span_metadata", None),
        error_message=getattr(span, "error_message", None),
    )


@router.get("/events", response_model=XRayEventsResponse, dependencies=[Depends(require_auth)])
async def get_events(
    since_seq: int = 0,
    limit: int = 50,
    source: Optional[str] = None,
    level: Optional[str] = None,
    stage: Optional[str] = None,
    run_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
):
    """Poll X-Ray events (incremental via since_seq)."""
    from app_v2.services.xray import get_latest_seq, get_recent_events

    raw = get_recent_events(
        limit=limit,
        since_seq=since_seq,
        source=source,
        level=level,
        stage=stage,
        run_id=run_id,
        recording_id=recording_id,
        provider=provider,
        model=model,
        status=status,
    )
    events = [_event_out(e) for e in raw]
    latest = max([event.seq for event in events], default=get_latest_seq() if since_seq else 0)
    return XRayEventsResponse(events=events, latest_seq=latest)


@router.get("/throughput", dependencies=[Depends(require_auth)])
async def throughput(buckets: int = 30):
    """Per-second event throughput (last N seconds)."""
    from app_v2.services.xray import get_throughput

    bucket_count = max(1, min(int(buckets), 60))
    values = _cache.get_or_compute(
        ("throughput", bucket_count),
        _THROUGHPUT_TTL_SECONDS,
        lambda: get_throughput(buckets=bucket_count),
    )
    return {"buckets": values}


@router.post("/clear", dependencies=[Depends(require_auth)])
async def clear_events():
    """Clear all X-Ray events."""
    from app_v2.services.xray import clear_events as _clear

    _clear()
    return {"success": True}


@router.get("/runs", response_model=list[TraceRunOut], dependencies=[Depends(require_auth)])
async def runs(limit: int = 25):
    """Recent persisted execution runs."""
    from src.database.engine import SessionLocal
    from src.database.chronos_repository import list_execution_runs

    with SessionLocal() as session:
        return [_run_out(run) for run in list_execution_runs(session, limit=limit)]


@router.get("/spans", response_model=list[TraceSpanOut], dependencies=[Depends(require_auth)])
async def spans(
    run_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    stage: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100,
):
    """Recent persisted execution spans."""
    from src.database.engine import SessionLocal
    from src.database.chronos_repository import list_execution_spans

    with SessionLocal() as session:
        return [
            _span_out(span)
            for span in list_execution_spans(
                session,
                run_id=run_id,
                recording_id=recording_id,
                stage=stage,
                source=source,
                limit=limit,
            )
        ]


@router.get("/runs/{run_id}", response_model=TraceDagOut, dependencies=[Depends(require_auth)])
async def run_detail(run_id: str, limit: int = 500):
    """Execution run summary plus its span DAG."""
    from src.database.engine import SessionLocal
    from src.database.chronos_repository import get_execution_run, list_execution_spans

    with SessionLocal() as session:
        run = get_execution_run(session, run_id)
        spans = list_execution_spans(session, run_id=run_id, limit=limit)
        return TraceDagOut(
            run=_run_out(run) if run else None,
            spans=[_span_out(span) for span in reversed(spans)],
        )


@router.get(
    "/recordings/{recording_id}/lineage",
    response_model=TraceDagOut,
    dependencies=[Depends(require_auth)],
)
async def recording_lineage(recording_id: str, limit: int = 500):
    """All trace spans currently tied to a recording."""
    from src.database.engine import SessionLocal
    from src.database.chronos_repository import list_execution_spans

    with SessionLocal() as session:
        spans = list_execution_spans(session, recording_id=recording_id, limit=limit)
        return TraceDagOut(run=None, spans=[_span_out(span) for span in reversed(spans)])


def _filter_match(value: Optional[str], pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    allowed = {p.strip().lower() for p in pattern.split(",") if p.strip()}
    return str(value or "").lower() in allowed


def _ws_event_matches(
    event: dict[str, Any],
    *,
    source: Optional[str],
    level: Optional[str],
    stage: Optional[str],
    run_id: Optional[str],
    recording_id: Optional[str],
    provider: Optional[str],
    status: Optional[str],
) -> bool:
    return (
        _filter_match(event.get("source"), source)
        and _filter_match(event.get("level"), level)
        and _filter_match(event.get("stage"), stage)
        and _filter_match(event.get("provider"), provider)
        and _filter_match(event.get("status"), status)
        and (not run_id or event.get("run_id") == run_id)
        and (not recording_id or event.get("recording_id") == recording_id)
    )


async def _websocket_auth_ok(websocket: WebSocket) -> bool:
    api_key = os.getenv("CHRONOS_API_KEY", "")
    if not api_key:
        return True

    token = websocket.query_params.get("token") or ""
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
    if hmac.compare_digest(token, api_key):
        return True
    await websocket.close(code=1008)
    return False


@router.websocket("/ws")
async def xray_websocket(websocket: WebSocket):
    """Live X-Ray event stream for iOS/web clients.

    Polling remains the fallback; this endpoint simply fans out the same event
    payloads the polling endpoint returns.
    """
    if not await _websocket_auth_ok(websocket):
        return

    await websocket.accept()
    from app_v2.services.xray import subscribe_events, unsubscribe_events

    source = websocket.query_params.get("source")
    level = websocket.query_params.get("level")
    stage = websocket.query_params.get("stage")
    run_id = websocket.query_params.get("run_id")
    recording_id = websocket.query_params.get("recording_id")
    provider = websocket.query_params.get("provider")
    status = websocket.query_params.get("status")

    q = subscribe_events(maxsize=500)
    try:
        while True:
            try:
                event = await asyncio.to_thread(q.get, True, 15.0)
            except queue.Empty:
                await websocket.send_json({"type": "heartbeat", "ts": time.time()})
                continue

            if not _ws_event_matches(
                event,
                source=source,
                level=level,
                stage=stage,
                run_id=run_id,
                recording_id=recording_id,
                provider=provider,
                status=status,
            ):
                continue
            await websocket.send_json(_event_out(event).model_dump())
    except WebSocketDisconnect:
        return
    finally:
        unsubscribe_events(q)
