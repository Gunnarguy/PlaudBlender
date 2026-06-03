"""Chronos execution tracing helpers.

This module is intentionally lightweight and dependency-minimal. It gives the
pipeline a single vocabulary for "what ran what" while also bridging every span
back into the existing X-Ray UI stream.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

_current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "chronos_run_id", default=None
)
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "chronos_span_id", default=None
)


@dataclass
class TraceSpanHandle:
    """Runtime handle returned from ``trace_span``."""

    span_id: str
    run_id: Optional[str]
    parent_span_id: Optional[str]
    started_at: float

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000


def current_run_id() -> Optional[str]:
    return _current_run_id.get()


def current_span_id() -> Optional[str]:
    return _current_span_id.get()


def safe_hash(value: Any, *, max_chars: int = 100_000) -> Optional[str]:
    """Return a SHA256 hash for safe provenance without storing full content."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if not text:
        return None
    return hashlib.sha256(text[:max_chars].encode("utf-8", errors="ignore")).hexdigest()


def safe_snippet(value: Any, *, max_chars: int = 500) -> Optional[str]:
    """Return a bounded, UI-safe snippet for trace debugging."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\x00", "").strip()
    if not text:
        return None
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def _with_session(func) -> None:
    """Run a best-effort DB operation without letting tracing break work."""
    try:
        from src.database.engine import SessionLocal

        with SessionLocal() as session:
            func(session)
    except Exception:
        # Trace persistence is diagnostic only; never break the user's pipeline.
        return


def start_trace_run(
    *,
    run_id: Optional[str] = None,
    trigger: Optional[str] = None,
    source: Optional[str] = None,
    title: Optional[str] = None,
    entrypoint: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    emit_xray: bool = True,
) -> str:
    """Start a new execution run and set it as the current context."""
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    _current_run_id.set(resolved_run_id)

    def _persist(session):
        from src.database.chronos_repository import create_execution_run

        create_execution_run(
            session,
            run_id=resolved_run_id,
            trigger=trigger,
            source=source,
            title=title,
            host=socket.gethostname(),
            process_id=os.getpid(),
            entrypoint=entrypoint,
            metadata=metadata,
        )

    _with_session(_persist)

    if emit_xray:
        try:
            from app_v2.services.xray import xray_log

            xray_log(
                source or "pipeline",
                "run-start",
                title or "Chronos run started",
                run_id=resolved_run_id,
                status="running",
                metadata=metadata,
            )
        except Exception:
            pass

    return resolved_run_id


def finish_trace_run(
    run_id: Optional[str] = None,
    *,
    status: str = "completed",
    summary: Optional[dict[str, Any]] = None,
    error_message: Optional[str] = None,
    emit_xray: bool = True,
) -> None:
    """Finish a run and clear the current context if it matches."""
    resolved_run_id = run_id or current_run_id()
    if not resolved_run_id:
        return

    def _persist(session):
        from src.database.chronos_repository import finish_execution_run

        finish_execution_run(
            session,
            resolved_run_id,
            status=status,
            summary=summary,
            error_message=error_message,
        )

    _with_session(_persist)

    if emit_xray:
        try:
            from app_v2.services.xray import xray_log

            xray_log(
                "pipeline",
                "run-done" if status == "completed" else "run-fail",
                "Chronos run finished" if status == "completed" else "Chronos run failed",
                run_id=resolved_run_id,
                status=status,
                level="error" if status == "failed" else "info",
                detail=error_message,
                metadata=summary,
            )
        except Exception:
            pass

    if current_run_id() == resolved_run_id:
        _current_run_id.set(None)


@contextmanager
def trace_span(
    *,
    operation: str,
    source: str = "pipeline",
    stage: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    message: Optional[str] = None,
    detail: Optional[str] = None,
    recording_id: Optional[str] = None,
    event_id: Optional[str] = None,
    run_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    input_value: Any = None,
    request_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    emit_start: bool = True,
) -> Iterator[TraceSpanHandle]:
    """Trace an operation as a persisted span and live X-Ray event pair."""
    resolved_run_id = run_id or current_run_id()
    resolved_parent = parent_span_id or current_span_id()
    span_id = uuid.uuid4().hex[:16]
    input_hash = safe_hash(input_value)
    input_snip = safe_snippet(input_value)
    started = time.perf_counter()

    def _persist_start(session):
        from src.database.chronos_repository import start_execution_span

        start_execution_span(
            session,
            span_id=span_id,
            run_id=resolved_run_id,
            parent_span_id=resolved_parent,
            recording_id=recording_id,
            event_id=event_id,
            stage=stage,
            operation=operation,
            source=source,
            provider=provider,
            model=model,
            message=message,
            detail=detail,
            input_hash=input_hash,
            input_snippet=input_snip,
            request_id=request_id,
            metadata=metadata,
        )

    _with_session(_persist_start)

    if emit_start:
        try:
            from app_v2.services.xray import xray_log

            xray_log(
                source,
                operation,
                message or operation,
                detail=detail,
                run_id=resolved_run_id,
                span_id=span_id,
                parent_span_id=resolved_parent,
                recording_id=recording_id,
                event_id=event_id,
                stage=stage,
                provider=provider,
                model=model,
                status="running",
                request_id=request_id,
                metadata=metadata,
            )
        except Exception:
            pass

    token = _current_span_id.set(span_id)
    handle = TraceSpanHandle(
        span_id=span_id,
        run_id=resolved_run_id,
        parent_span_id=resolved_parent,
        started_at=started,
    )

    try:
        yield handle
    except Exception as exc:
        elapsed = round(handle.elapsed_ms(), 1)
        err = str(exc)[:500]

        def _persist_error(session):
            from src.database.chronos_repository import finish_execution_span

            finish_execution_span(
                session,
                span_id,
                status="failed",
                level="error",
                error_message=err,
                metadata={"exception_type": exc.__class__.__name__},
            )

        _with_session(_persist_error)
        try:
            from app_v2.services.xray import xray_log

            xray_log(
                source,
                operation,
                f"{message or operation} failed: {err[:120]}",
                duration_ms=elapsed,
                detail=detail,
                level="error",
                run_id=resolved_run_id,
                span_id=span_id,
                parent_span_id=resolved_parent,
                recording_id=recording_id,
                event_id=event_id,
                stage=stage,
                provider=provider,
                model=model,
                status="failed",
                request_id=request_id,
            )
        except Exception:
            pass
        raise
    else:
        elapsed = round(handle.elapsed_ms(), 1)

        def _persist_ok(session):
            from src.database.chronos_repository import finish_execution_span

            finish_execution_span(session, span_id, status="completed", level="perf")

        _with_session(_persist_ok)
        try:
            from app_v2.services.xray import xray_log

            xray_log(
                source,
                operation,
                f"{message or operation} finished",
                duration_ms=elapsed,
                detail=detail,
                level="perf",
                run_id=resolved_run_id,
                span_id=span_id,
                parent_span_id=resolved_parent,
                recording_id=recording_id,
                event_id=event_id,
                stage=stage,
                provider=provider,
                model=model,
                status="completed",
                request_id=request_id,
                metadata=metadata,
            )
        except Exception:
            pass
    finally:
        _current_span_id.reset(token)
