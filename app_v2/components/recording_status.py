"""Shared recording system-status badges for cards and detail headers."""

from __future__ import annotations

from typing import Optional

from dash import html

from app_v2.services.data_service import RecordingSummary

_ACTIVE_WORKFLOW_STATUSES = {"PENDING", "PROCESSING", "RUNNING"}
_SUCCESS_WORKFLOW_STATUSES = {"SUCCESS", "COMPLETED"}
_FAILED_WORKFLOW_STATUSES = {"FAILED", "ERROR"}


def _badge(label: str, tone: str, title: Optional[str] = None) -> html.Span:
    props = {"className": f"recording-system-badge {tone}"}
    if title:
        props["title"] = title
    return html.Span(label, **props)


def _normalized_processing_status(summary: RecordingSummary) -> str:
    return str(getattr(summary, "processing_status", "completed") or "completed").strip().lower()


def _normalized_workflow_status(
    summary: RecordingSummary, workflow_status: Optional[dict] = None
) -> str:
    if isinstance(workflow_status, dict):
        candidate = str(workflow_status.get("status") or "").strip().upper()
        if candidate:
            return candidate
    return str(getattr(summary, "plaud_workflow_status", "") or "").strip().upper()


def _build_processing_badge(summary: RecordingSummary) -> html.Span:
    status = _normalized_processing_status(summary)
    mapping = {
        "completed": (
            "Sync ready",
            "success",
            "Gemini extraction finished and this recording is available in Chronos.",
        ),
        "processing": (
            "Sync processing",
            "info",
            "Chronos is currently extracting and indexing this recording.",
        ),
        "pending": (
            "Sync pending",
            "warn",
            "Chronos has the recording queued but has not extracted moments yet.",
        ),
        "failed": (
            "Sync failed",
            "danger",
            "Chronos hit an error while processing this recording.",
        ),
    }
    label, tone, title = mapping.get(
        status,
        (
            "Sync unknown",
            "muted",
            "Chronos could not determine the processing state for this recording.",
        ),
    )
    return _badge(label, tone, title)


def _build_plaud_badge(
    summary: RecordingSummary, workflow_status: Optional[dict] = None
) -> Optional[html.Span]:
    source = str(getattr(summary, "source", "") or "").strip().lower()
    workflow = _normalized_workflow_status(summary, workflow_status)
    has_plaud_ai = bool(
        getattr(summary, "has_plaud_ai", False) or getattr(summary, "plaud_ai_summary", None)
    )

    if workflow in _ACTIVE_WORKFLOW_STATUSES:
        template_id = ""
        if isinstance(workflow_status, dict):
            template_id = str(workflow_status.get("template_id") or "").strip()
        detail = "Plaud cloud workflow is still running for this recording."
        if template_id:
            detail = f"Plaud cloud workflow is still running ({template_id})."
        return _badge("Plaud AI running", "info", detail)

    if workflow in _FAILED_WORKFLOW_STATUSES:
        error_detail = ""
        if isinstance(workflow_status, dict):
            error_detail = str(workflow_status.get("error") or "").strip()
        title = error_detail or "Plaud workflow failed for this recording."
        return _badge("Plaud AI failed", "danger", title)

    if workflow in _SUCCESS_WORKFLOW_STATUSES:
        if has_plaud_ai:
            return _badge(
                "Plaud AI ready",
                "success",
                "Plaud cloud summary or transcript data is available for this recording.",
            )
        return _badge(
            "Plaud workflow done",
            "success",
            "Plaud cloud finished processing this recording.",
        )

    if has_plaud_ai:
        return _badge(
            "Plaud AI ready",
            "success",
            "Plaud cloud summary or transcript data is available for this recording.",
        )

    if source in {"plaud", "plaud_cloud"}:
        return _badge(
            "Plaud AI missing",
            "muted",
            "This Plaud cloud recording does not yet have Plaud AI output attached.",
        )

    return None


def _build_notion_badge(summary: RecordingSummary) -> Optional[html.Span]:
    state = str(getattr(summary, "notion_state", "") or "").strip().lower()
    if not state:
        return None

    page_title = str(getattr(summary, "notion_page_title", "") or "").strip()
    match_count = int(getattr(summary, "notion_match_count", 0) or 0)
    count_suffix = f" ×{match_count}" if match_count > 1 else ""

    if state == "imported":
        return _badge(
            "Notion imported",
            "notion",
            page_title or "This recording originated from a Notion page.",
        )

    if state == "imported-stale":
        return _badge(
            "Notion newer",
            "warn",
            page_title or "The source Notion page changed after this recording was imported.",
        )

    if state == "linked":
        return _badge(
            f"Notion linked{count_suffix}",
            "notion",
            page_title or "This recording is linked to a Notion page through the Chronos bridge.",
        )

    if state == "stale":
        return _badge(
            f"Notion newer{count_suffix}",
            "warn",
            page_title or "A linked Notion page has been edited since Chronos last synced it.",
        )

    if state == "chronos-only":
        return _badge(
            "Ready for Notion",
            "info",
            "This Chronos recording is not linked to Notion yet and can be pushed there.",
        )

    return None


def build_recording_system_strip(
    summary: RecordingSummary,
    workflow_status: Optional[dict] = None,
    *,
    detail: bool = False,
) -> Optional[html.Div]:
    """Render a shared strip of system badges for a recording."""

    items: list = [_build_processing_badge(summary)]

    plaud_badge = _build_plaud_badge(summary, workflow_status)
    if plaud_badge is not None:
        items.append(plaud_badge)

    notion_badge = _build_notion_badge(summary)
    if notion_badge is not None:
        items.append(notion_badge)

    notion_url = str(getattr(summary, "notion_page_url", "") or "").strip()
    if detail and notion_url:
        items.append(
            html.A(
                "Open Notion ↗",
                href=notion_url,
                target="_blank",
                rel="noreferrer noopener",
                className="detail-system-link",
            )
        )

    if not items:
        return None

    return html.Div(
        className="detail-system-strip" if detail else "recording-system-strip",
        children=items,
    )
