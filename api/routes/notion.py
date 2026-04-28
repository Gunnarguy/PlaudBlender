"""Notion integration endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas.responses import (
    NotionBulkMatchOverrideRequest,
    NotionDatabaseSelectRequest,
    NotionImportRequest,
    NotionMatchOverrideRequest,
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


BLIND_NOTION_IMPORT_LIMIT = 25


def _build_notion_import_preview(session, sample_size: int = 25) -> dict:
    from src.chronos.notion_bridge import (
        collapse_exact_notion_duplicates,
        match_notion_to_chronos,
    )
    from src.database.models import ChronosRecording

    ns = _get_notion_service()
    recordings = ns.fetch_recordings(limit=1000)

    completed_notion = set()
    failed_notion = set()
    for rec in (
        session.query(ChronosRecording)
        .filter(ChronosRecording.source == "notion")
        .all()
    ):
        if rec.recording_id.startswith("notion:"):
            page_id = rec.recording_id[7:]
            if rec.processing_status == "completed":
                completed_notion.add(page_id)
            elif rec.processing_status == "failed":
                failed_notion.add(page_id)

    matches = match_notion_to_chronos(recordings, session)

    pending = []
    matched_to_existing = 0
    for nrec in recordings:
        if nrec.page_id in completed_notion:
            continue
        if matches.get(nrec.page_id):
            matched_to_existing += 1
            continue
        pending.append(nrec)

    pending.sort(key=lambda n: n.date or n.created_time or "", reverse=True)
    effective_pending, duplicate_pages_collapsed = collapse_exact_notion_duplicates(pending)

    return {
        "total_pages": len(recordings),
        "completed_imports": len(completed_notion),
        "failed_imports": len(failed_notion),
        "matched_to_existing": matched_to_existing,
        "pending_import_raw": len(pending),
        "duplicate_pages_collapsed": duplicate_pages_collapsed,
        "pending_import": len(effective_pending),
        "blind_import_limit": BLIND_NOTION_IMPORT_LIMIT,
        "blocked_without_force": len(effective_pending) > BLIND_NOTION_IMPORT_LIMIT,
        "sample": [
            {
                "page_id": n.page_id,
                "title": n.title,
                "date": n.date or (n.created_time[:10] if n.created_time else None),
                "url": n.url,
            }
            for n in effective_pending[:sample_size]
        ],
    }


@router.get("/status")
async def notion_status():
    """Notion connection status."""
    try:
        ns = _get_notion_service()
        status = ns.check_connection(quick=False)
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
    from src.chronos.notion_bridge import match_notion_to_chronos
    from src.database import SessionLocal

    ns = _get_notion_service()
    # Always fetch all pages so we know the true total and can compute real match status.
    pages = ns.fetch_recordings(limit=2000)
    total = len(pages)

    with SessionLocal() as session:
        match_map = match_notion_to_chronos(pages, session)

    # Apply offset/limit slicing after computing the full match map.
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
                matched_recording_id=match_map.get(p.page_id),
            )
        )
    return NotionRecordingsResponse(
        recordings=out,
        total=total,
        has_more=(offset + len(out)) < total,
    )


@router.get("/match/review")
async def notion_match_review(limit: int = Query(default=25, ge=1, le=100)):
    """Review high-confidence Notion match candidates and duplicate groups."""
    from src.chronos.notion_bridge import build_notion_match_review
    from src.database import SessionLocal

    with SessionLocal() as session:
        return build_notion_match_review(session, limit=limit)


@router.post("/match/override", response_model=SuccessResponse)
async def notion_match_override(body: NotionMatchOverrideRequest):
    """Persist or clear a manual Notion → Chronos match override."""
    from src.chronos.notion_bridge import (
        clear_manual_notion_match_override,
        set_manual_notion_match_override,
    )
    from src.database import SessionLocal

    if body.clear or not body.recording_id:
        ok, message = clear_manual_notion_match_override(body.page_id)
        if not ok:
            raise HTTPException(status_code=400, detail=message)
        return SuccessResponse(message=message)

    with SessionLocal() as session:
        ok, message = set_manual_notion_match_override(
            session,
            page_id=body.page_id,
            recording_id=body.recording_id,
        )
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return SuccessResponse(message=message)


@router.post("/match/override/bulk")
async def notion_match_override_bulk(body: NotionBulkMatchOverrideRequest):
    """Apply multiple manual Notion → Chronos match overrides in one request."""
    from src.chronos.notion_bridge import (
        clear_manual_notion_match_override,
        set_manual_notion_match_override,
    )
    from src.database import SessionLocal

    results = []
    applied = 0
    cleared = 0
    failed = 0

    with SessionLocal() as session:
        for override in body.overrides:
            if override.clear or not override.recording_id:
                ok, message = clear_manual_notion_match_override(override.page_id)
                action = "clear"
            else:
                ok, message = set_manual_notion_match_override(
                    session,
                    page_id=override.page_id,
                    recording_id=override.recording_id,
                )
                action = "set"

            if ok:
                if action == "clear":
                    cleared += 1
                else:
                    applied += 1
            else:
                failed += 1

            results.append(
                {
                    "page_id": override.page_id,
                    "action": action,
                    "ok": ok,
                    "message": message,
                }
            )

            if body.stop_on_error and not ok:
                break

    return {
        "applied": applied,
        "cleared": cleared,
        "failed": failed,
        "results": results,
    }


@router.get("/import/preview")
async def notion_import_preview():
    """Preview what the backend would import from Notion right now."""
    from src.database import SessionLocal

    with SessionLocal() as session:
        return _build_notion_import_preview(session)


@router.post("/import/next-batch", response_model=SuccessResponse)
async def import_next_notion_batch(body: NotionImportRequest):
    """Import the next safe deduped Notion batch without requiring force."""
    from src.chronos.notion_bridge import import_all_unmatched
    from src.database import SessionLocal

    requested_batch_size = body.batch_size or BLIND_NOTION_IMPORT_LIMIT
    if requested_batch_size < 1:
        requested_batch_size = BLIND_NOTION_IMPORT_LIMIT
    if requested_batch_size > BLIND_NOTION_IMPORT_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Safe batch import only allows batch_size<={BLIND_NOTION_IMPORT_LIMIT}."
            ),
        )

    with SessionLocal() as session:
        preview = _build_notion_import_preview(session)
        pending_import = preview["pending_import"]
        if pending_import == 0:
            return SuccessResponse(message="No Notion pages are pending import")

        imported, failed, errors = import_all_unmatched(
            session,
            process=body.process,
            index=body.index,
            batch_size=requested_batch_size,
        )

    message = (
        f"Imported {imported} recordings from Notion using a safe batch of up to "
        f"{requested_batch_size}"
    )
    if failed:
        message += f" ({failed} failed)"
    return SuccessResponse(message=message)


@router.post("/import", response_model=SuccessResponse)
async def import_from_notion(body: NotionImportRequest):
    """Import unmatched Notion recordings into Chronos."""
    from src.chronos.notion_bridge import import_all_unmatched
    from src.database import SessionLocal

    with SessionLocal() as session:
        preview = _build_notion_import_preview(session)
        pending_import = preview["pending_import"]
        requested_batch_size = max(0, body.batch_size or 0)

        if pending_import > BLIND_NOTION_IMPORT_LIMIT and not body.force:
            if requested_batch_size == 0:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Bulk Notion import blocked: {pending_import} pages are pending import. "
                        f"Review /api/v1/notion/import/preview and retry with force=true or "
                        f"batch_size<={BLIND_NOTION_IMPORT_LIMIT}."
                    ),
                )
            if requested_batch_size > BLIND_NOTION_IMPORT_LIMIT:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Requested batch_size={requested_batch_size} exceeds the blind import limit of "
                        f"{BLIND_NOTION_IMPORT_LIMIT}. Use force=true if you intend a large batch."
                    ),
                )

        imported, failed, errors = import_all_unmatched(
            session,
            process=body.process,
            index=body.index,
            batch_size=requested_batch_size,
        )

    message = f"Imported {imported} recordings from Notion"
    if failed:
        message += f" ({failed} failed)"
    return SuccessResponse(message=message)


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
