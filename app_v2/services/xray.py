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
from dataclasses import dataclass, field, asdict
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


# Thread-safe ring buffer (last 200 events)
_MAX_EVENTS = 200
_events: deque = deque(maxlen=_MAX_EVENTS)
_lock = threading.Lock()
_seq_counter = 0  # monotonic sequence number


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
