"""Pipeline / sync endpoints."""

import subprocess
import sys
import time
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_service
from api.schemas.responses import (
    PipelineRunRequest,
    PipelineRunResponse,
    RecordingWorkflowRequest,
    SuccessResponse,
    SyncFailureSummaryOut,
    UploadProcessRequest,
    UploadProcessResultOut,
    WorkflowRefreshRequest,
    WorkflowSubmitRequest,
)
from api.auth.jwt import require_auth
from app_v2.services.data_service import ChronosDataService
from src.chronos.pipeline_progress import progress, read_progress
from src.chronos.pipeline_stages import invalid_pipeline_stage_message

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

    stage: full | backfill | ingest | process | index | graph | reindex
    aliases: all_history | full_history -> backfill
    """
    try:
        stage = body.normalized_stage
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=invalid_pipeline_stage_message(body.stage),
        )

    current = read_progress()
    if current and current.get("status") == "running":
        pipeline_active = _pipeline_process_active()
        started_at = float(current.get("started_at") or 0)
        age_minutes = ((time.time() - started_at) / 60) if started_at else 0
        if pipeline_active and age_minutes < 30:
            return PipelineRunResponse(
                status="already_running",
                message="Pipeline already running — watch the progress panel instead of starting another run.",
            )

        progress.finish_run(error=f"Stale run auto-cleared after {age_minutes:.0f} min")

    cmd = [
        sys.executable,
        "scripts/chronos_pipeline.py",
        f"--{stage}",
    ]
    if stage in {"full", "ingest", "backfill"}:
        cmd.extend(["--days-back", str(body.days_back)])

    trace_run_id = uuid.uuid4().hex[:12]
    env = os.environ.copy()
    env["CHRONOS_TRACE_RUN_ID"] = trace_run_id
    env["CHRONOS_TRACE_TRIGGER"] = "api"

    # Fire-and-forget subprocess
    subprocess.Popen(cmd, start_new_session=True, env=env)

    return PipelineRunResponse(
        status="started",
        run_id=trace_run_id,
        message=(
            f"Smart sync started for the last {body.days_back} days"
            if stage == "full"
            else "Full-history Plaud backfill started"
            if stage == "backfill"
            else (
                f"Pipeline stage '{stage}' started"
                + (
                    f" for the last {body.days_back} days"
                    if stage == "ingest"
                    else ""
                )
            )
        ),
    )


def _pipeline_process_active() -> bool:
    """True when another chronos pipeline process is currently alive."""
    current_pid = os.getpid()
    parent_pid = os.getppid()
    proc_root = Path("/proc")

    try:
        proc_dirs = list(proc_root.iterdir())
    except Exception:
        # If procfs is unavailable, assume active so we avoid accidental overlap.
        return True

    for proc_dir in proc_dirs:
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == current_pid or pid == parent_pid:
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        except Exception:
            continue
        if b"scripts/chronos_pipeline.py" in cmdline:
            return True
    return False


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


@router.get("/failures", response_model=SyncFailureSummaryOut)
async def sync_failures(
    include_archived: bool = False,
    svc: ChronosDataService = Depends(get_service),
):
    """Summarize retryable sync failures.

    Archived dead-end failures are hidden by default and only included when
    `include_archived=true` is passed explicitly.
    """
    return svc.get_sync_failure_summary(include_archived=include_archived)


@router.post("/reset-stuck", response_model=SuccessResponse)
async def reset_stuck(svc: ChronosDataService = Depends(get_service)):
    """Reset stuck processing recordings back to pending."""
    count = svc.reset_stuck_recordings()
    try:
        from src.chronos.pipeline_progress import progress
        progress.finish_run(error="Manually reset stuck pipeline")
    except Exception as e:
        pass
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


@router.post("/upload/process", response_model=UploadProcessResultOut)
async def upload_and_process_candidates(
    body: UploadProcessRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """Upload local audio files to Plaud cloud and submit workflows."""
    file_paths = body.file_paths or [
        item["path"] for item in svc.get_upload_candidates()
    ]
    if not file_paths:
        return UploadProcessResultOut(uploaded_count=0, error_count=0)

    result = svc.upload_and_process_files(
        file_paths=file_paths,
        template_id=body.template_id,
        model=body.model,
    )
    uploaded = result.get("uploaded", [])
    errors = result.get("errors", [])
    return UploadProcessResultOut(
        uploaded_count=len(uploaded),
        error_count=len(errors),
        uploaded=uploaded,
        errors=errors,
    )
