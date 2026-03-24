"""Pipeline / sync endpoints."""

import subprocess
import sys

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_service
from api.schemas.responses import (
    PipelineRunRequest,
    PipelineRunResponse,
    RecordingWorkflowRequest,
    SuccessResponse,
    WorkflowRefreshRequest,
    WorkflowSubmitRequest,
)
from api.auth.jwt import require_auth
from app_v2.services.data_service import ChronosDataService

router = APIRouter(
    prefix="/api/v1/sync",
    tags=["sync"],
    dependencies=[Depends(require_auth)],
)


@router.get("/status")
async def pipeline_status():
    """Current pipeline run progress (if any)."""
    from src.chronos.pipeline_progress import read_progress

    progress = read_progress()
    if progress is None:
        return {"status": "idle"}
    return progress


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(body: PipelineRunRequest):
    """Trigger a pipeline run in a background subprocess.

    stage: full | ingest | process | index | graph
    """
    valid_stages = {"full", "ingest", "process", "index", "graph", "reindex"}
    if body.stage not in valid_stages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage: {body.stage}. Must be one of {valid_stages}",
        )

    cmd = [
        sys.executable,
        "scripts/chronos_pipeline.py",
        f"--{body.stage}",
    ]
    # Fire-and-forget subprocess
    subprocess.Popen(cmd, start_new_session=True)

    return PipelineRunResponse(
        status="started",
        message=f"Pipeline stage '{body.stage}' started",
    )


@router.post("/workflows/submit", response_model=SuccessResponse)
async def submit_workflows(
    body: WorkflowSubmitRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """Submit Plaud AI workflows for unprocessed recordings."""
    result = svc.submit_plaud_workflows(
        days_back=body.days_back,
        limit=body.limit,
        template_id=body.template_id,
        model=body.model,
    )
    return SuccessResponse(
        message=(
            f"Submitted {result.get('submitted', 0)} workflows"
            if isinstance(result, dict)
            else str(result)
        )
    )


@router.post("/workflows/refresh", response_model=SuccessResponse)
async def refresh_workflows(
    body: WorkflowRefreshRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """Refresh status of pending Plaud AI workflows."""
    result = svc.refresh_plaud_workflow_statuses(
        days_back=body.days_back,
        limit=body.limit,
    )
    return SuccessResponse(
        message=(
            f"Refreshed {result.get('refreshed', 0)} workflows"
            if isinstance(result, dict)
            else str(result)
        )
    )


@router.post(
    "/workflows/{recording_id}",
    response_model=SuccessResponse,
)
async def submit_single_workflow(
    recording_id: str,
    body: RecordingWorkflowRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """Submit a single recording for Plaud AI workflow."""
    result = svc.submit_single_recording_workflow(
        recording_id=recording_id,
        template_id=body.template_id,
        model=body.model,
    )
    return SuccessResponse(message=str(result) if result else "Workflow submitted")


@router.get("/workflows/{recording_id}/status")
async def workflow_status(
    recording_id: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Get workflow status for a specific recording."""
    status = svc.get_workflow_status_for_recording(recording_id)
    return {"recording_id": recording_id, "workflow_status": status}


@router.get("/db-stats")
async def sync_db_stats(svc: ChronosDataService = Depends(get_service)):
    """Recording status counts (pending, processed, failed, etc.)."""
    return svc.get_recording_db_stats()


@router.post("/reset-stuck", response_model=SuccessResponse)
async def reset_stuck(svc: ChronosDataService = Depends(get_service)):
    """Reset stuck processing recordings back to pending."""
    count = svc.reset_stuck_recordings()
    return SuccessResponse(message=f"Reset {count} stuck recordings")


@router.post("/refresh-cache", response_model=SuccessResponse)
async def refresh_cache(svc: ChronosDataService = Depends(get_service)):
    """Force-refresh the in-memory event cache."""
    svc.refresh_cache()
    return SuccessResponse(message="Cache refreshed")


@router.get("/upload-candidates")
async def upload_candidates(svc: ChronosDataService = Depends(get_service)):
    """Recordings eligible for Plaud cloud upload."""
    return svc.get_upload_candidates()
