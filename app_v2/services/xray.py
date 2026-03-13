"""X-ray telemetry system — collects timing & diagnostic data from callbacks.

Usage in any callback:
    from app_v2.services.xray import xray_log, xray_timer

    # Simple log entry
    xray_log("search", "query", f"Searching for '{query}'")

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


def xray_log(
    source: str,
    op: str,
    message: str,
    duration_ms: Optional[float] = None,
    detail: Optional[str] = None,
    level: str = "info",
):
    """Push a telemetry event into the ring buffer."""
    evt = XRayEvent(
        ts=time.time(),
        source=source,
        op=op,
        message=message,
        duration_ms=duration_ms,
        detail=detail,
        level=level,
    )
    with _lock:
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


def get_recent_events(limit: int = 50) -> list:
    """Return recent events as dicts (newest first)."""
    with _lock:
        items = list(_events)
    items.reverse()
    return [asdict(e) for e in items[:limit]]


def clear_events():
    """Clear the event buffer."""
    with _lock:
        _events.clear()
