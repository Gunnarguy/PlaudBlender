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
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional


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


# Thread-safe ring buffer — 2000 events to match client-side capacity
_MAX_EVENTS = 2000
_events: deque = deque(maxlen=_MAX_EVENTS)
_lock = threading.Lock()
_seq_counter = 0  # monotonic sequence number

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
):
    """Push a telemetry event into the ring buffer."""
    global _seq_counter
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
        )
        _events.append(evt)
    _record_throughput()


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


def get_recent_events(limit: int = 50, since_seq: int = 0) -> list:
    """Return recent events as dicts (newest first).

    If since_seq > 0, only return events with seq > since_seq (incremental).
    """
    with _lock:
        if since_seq > 0:
            items = [e for e in _events if e.seq > since_seq]
        else:
            items = list(_events)
    items.reverse()
    return [asdict(e) for e in items[:limit]]


def clear_events():
    """Clear the event buffer."""
    with _lock:
        _events.clear()
