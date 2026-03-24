"""Notion integration endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.responses import (
    NotionDatabaseSelectRequest,
    NotionImportRequest,
    NotionRecordingOut,
    NotionRecordingsResponse,
    SuccessResponse,
)

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/notion",
    tags=["notion"],
    dependencies=[Depends(require_auth)],
)


def _get_notion_service():
    """Lazy-load NotionService singleton."""
    from src.notion_service import NotionService

    return NotionService()


def _get_notion_oauth():
    """Lazy-load NotionOAuthClient."""
    from src.notion_oauth import NotionOAuthClient

    return NotionOAuthClient()


@router.get("/status")
async def notion_status():
    """Notion connection status."""
    try:
        ns = _get_notion_service()
        status = ns.check_connection(quick=True)
        return {
            "is_connected": status.connected if hasattr(status, "connected") else False,
            "page_count": getattr(status, "total_pages", 0),
            "database_name": getattr(status, "database_title", None),
            "error": getattr(status, "error", None),
        }
    except Exception as e:
        return {"is_connected": False, "error": str(e)}


@router.get("/databases")
async def list_databases():
    """List Notion databases accessible to the integration."""
    ns = _get_notion_service()
    return ns.list_databases()


@router.post("/databases/select", response_model=SuccessResponse)
async def select_database(body: NotionDatabaseSelectRequest):
    """Set the active Notion database for sync."""
    ns = _get_notion_service()
    ns.set_database_id(body.db_id)
    return SuccessResponse(message=f"Database set to {body.db_id}")


@router.get("/recordings", response_model=NotionRecordingsResponse)
async def list_notion_recordings(
    limit: Optional[int] = Query(default=None, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
):
    """Fetch recordings from the Notion database.

    Supports optional pagination via limit/offset query params.
    If limit is omitted, all recordings are returned.
    """
    ns = _get_notion_service()
    # Always fetch all pages so we know the true total
    pages = ns.fetch_recordings(limit=2000)
    total = len(pages)
    # Apply offset/limit slicing
    if offset:
        pages = pages[offset:]
    if limit is not None:
        pages = pages[:limit]
    out = []
    for p in pages:
        out.append(
            NotionRecordingOut(
                page_id=p.page_id,
                title=p.title,
                created_time=getattr(p, "created_time", None),
                last_edited_time=getattr(p, "last_edited_time", None),
                url=getattr(p, "url", None),
                transcript=getattr(p, "transcript", None),
                summary=getattr(p, "summary", None),
                date=getattr(p, "date", None),
                duration=getattr(p, "duration", None),
                tags=getattr(p, "tags", None),
                category=getattr(p, "category", None),
                matched_recording_id=getattr(p, "matched_recording_id", None),
            )
        )
    return NotionRecordingsResponse(
        recordings=out,
        total=total,
        has_more=(offset + len(out)) < total,
    )


@router.post("/import", response_model=SuccessResponse)
async def import_from_notion(body: NotionImportRequest):
    """Import unmatched Notion recordings into Chronos."""
    from src.chronos.notion_bridge import import_all_unmatched
    from src.database import SessionLocal

    with SessionLocal() as session:
        imported, skipped, errors = import_all_unmatched(
            session, process=body.process, index=body.index
        )
    return SuccessResponse(message=f"Imported {imported} recordings from Notion")


@router.get("/import/progress")
async def notion_import_progress():
    """Get current Notion import progress."""
    from src.chronos.notion_bridge import get_import_progress

    progress = get_import_progress()
    return progress or {"status": "idle"}


@router.get("/coverage")
async def notion_coverage():
    """Calendar view of Notion vs Chronos coverage."""
    from src.chronos.notion_bridge import get_coverage_calendar
    from src.database import SessionLocal

    with SessionLocal() as session:
        return get_coverage_calendar(session)
