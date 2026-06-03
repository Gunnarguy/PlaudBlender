"""X-ray telemetry system — collects timing & diagnostic data from callbacks.

Usage in any callback:
    from app_v2.services.xray import xray_log, xray_timer

    # Simple log entry
    xray_log("search", "query", "Searching for 'meeting notes'")

    # Auto-timed block
    with xray_timer("search", "embed") as t:
        vector = embed(query)
    # t.ms available after the block
"""

import time
import threading
import queue
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class XRayEvent:
    """Single telemetry event."""

    seq: int  # monotonic sequence number (never resets while server is up)
    ts: float  # unix timestamp
    source: str  # e.g. "search", "nav", "sync", "graph"
    op: str  # e.g. "query", "embed", "fetch", "render"
    message: str  # human-readable description
    duration_ms: Optional[float] = None  # if timed
    detail: Optional[str] = None  # extra metadata (counts, scores, etc.)
    level: str = "info"  # info | warn | error | perf
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
    metadata: Optional[dict[str, Any]] = None


# Thread-safe ring buffer — 2000 events to match client-side capacity
_MAX_EVENTS = 2000
_events: deque = deque(maxlen=_MAX_EVENTS)
_lock = threading.Lock()
_seq_counter = 0  # monotonic sequence number
_subscribers: set[queue.Queue] = set()
_subscribers_lock = threading.Lock()

# ── Throughput tracking — rolling 60-second window of event counts ──
_THROUGHPUT_BUCKETS = 60  # 1-second buckets for 60 seconds
_throughput: deque = deque(maxlen=_THROUGHPUT_BUCKETS)
_throughput_lock = threading.Lock()
_last_bucket_ts: int = 0


def _record_throughput():
    """Record an event into the current 1-second throughput bucket."""
    global _last_bucket_ts
    now = int(time.time())
    with _throughput_lock:
        if now != _last_bucket_ts:
            # Fill gap buckets with 0
            gap = (
                min(now - _last_bucket_ts, _THROUGHPUT_BUCKETS)
                if _last_bucket_ts
                else 1
            )
            for _ in range(gap - 1):
                _throughput.append(0)
            _throughput.append(1)
            _last_bucket_ts = now
        else:
            if _throughput:
                _throughput[-1] += 1
            else:
                _throughput.append(1)
                _last_bucket_ts = now


def get_throughput(buckets: int = 30) -> list:
    """Return last N seconds of event counts for sparkline rendering."""
    now = int(time.time())
    with _throughput_lock:
        # Fill any gap between last recorded and now
        result = list(_throughput)
        if _last_bucket_ts and now > _last_bucket_ts:
            gap = min(now - _last_bucket_ts, _THROUGHPUT_BUCKETS)
            result.extend([0] * gap)
    # Return last N buckets
    return (
        result[-buckets:]
        if len(result) >= buckets
        else ([0] * (buckets - len(result))) + result
    )


def xray_log(
    source: str,
    op: str,
    message: str,
    duration_ms: Optional[float] = None,
    detail: Optional[str] = None,
    level: str = "info",
    run_id: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    event_id: Optional[str] = None,
    stage: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    request_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """Push a telemetry event into the ring buffer."""
    global _seq_counter
    if run_id is None or span_id is None:
        try:
            from src.chronos.trace_service import current_run_id, current_span_id

            if run_id is None:
                run_id = current_run_id()
            if span_id is None:
                span_id = current_span_id()
        except Exception:
            pass
    with _lock:
        _seq_counter += 1
        evt = XRayEvent(
            seq=_seq_counter,
            ts=time.time(),
            source=source,
            op=op,
            message=message,
            duration_ms=duration_ms,
            detail=detail,
            level=level,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            recording_id=recording_id,
            event_id=event_id,
            stage=stage,
            provider=provider,
            model=model,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
            metadata=metadata,
        )
        _events.append(evt)
        payload = asdict(evt)
    _record_throughput()
    _notify_subscribers(payload)


def _notify_subscribers(payload: dict[str, Any]) -> None:
    """Fan out an event to WebSocket subscribers without blocking producers."""
    with _subscribers_lock:
        subscribers = list(_subscribers)
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(payload)
        except queue.Full:
            try:
                subscriber.get_nowait()
                subscriber.put_nowait(payload)
            except Exception:
                pass


@contextmanager
def xray_timer(source: str, op: str, message: str = ""):
    """Context manager that auto-logs with duration."""

    class _Timer:
        ms: float = 0.0

    t = _Timer()
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t.ms = (time.perf_counter() - t0) * 1000
        xray_log(source, op, message or op, duration_ms=round(t.ms, 1))


def _as_filter_set(value) -> Optional[set[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",")]
    else:
        parts = [str(p).strip().lower() for p in value]
    return {p for p in parts if p}


def get_latest_seq() -> int:
    """Return the latest server-side event sequence."""
    with _lock:
        return int(_seq_counter)


def get_recent_events(
    limit: int = 50,
    since_seq: int = 0,
    *,
    source: Optional[str] = None,
    level: Optional[str] = None,
    stage: Optional[str] = None,
    run_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
) -> list:
    """Return recent events as dicts (newest first).

    If since_seq > 0, only return events with seq > since_seq (incremental).
    """
    limit = max(1, min(int(limit), _MAX_EVENTS))
    since_seq = max(int(since_seq), 0)
    sources = _as_filter_set(source)
    levels = _as_filter_set(level)
    stages = _as_filter_set(stage)
    providers = _as_filter_set(provider)
    models = _as_filter_set(model)
    statuses = _as_filter_set(status)

    def _matches(event: XRayEvent) -> bool:
        if sources and str(event.source).lower() not in sources:
            return False
        if levels and str(event.level).lower() not in levels:
            return False
        if stages and str(event.stage or "").lower() not in stages:
            return False
        if providers and str(event.provider or "").lower() not in providers:
            return False
        if models and str(event.model or "").lower() not in models:
            return False
        if statuses and str(event.status or "").lower() not in statuses:
            return False
        if run_id and event.run_id != run_id:
            return False
        if recording_id and event.recording_id != recording_id:
            return False
        return True

    with _lock:
        if since_seq >= _seq_counter:
            return []

        if since_seq > 0:
            items = []
            for event in reversed(_events):
                if event.seq <= since_seq:
                    break
                if _matches(event):
                    items.append(asdict(event))
                if len(items) >= limit:
                    break
            return items

        items = [event for event in reversed(_events) if _matches(event)]

    return [asdict(event) for event in items[:limit]]


def subscribe_events(maxsize: int = 500) -> queue.Queue:
    """Subscribe to live X-Ray events. Caller must later unsubscribe."""
    q: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
    with _subscribers_lock:
        _subscribers.add(q)
    return q


def unsubscribe_events(q: queue.Queue) -> None:
    """Remove a live X-Ray event subscriber."""
    with _subscribers_lock:
        _subscribers.discard(q)


def clear_events():
    """Clear the event buffer."""
    with _lock:
        _events.clear()
