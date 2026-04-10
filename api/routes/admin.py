"""Administrative endpoints for stack control and backups."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.auth.jwt import require_auth
from api.schemas.responses import BackupInfoOut, StackControlResponse

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_auth)],
)

ROOT = Path(__file__).resolve().parents[2]
CHRONOS_SCRIPT = ROOT / "chronos"
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "exports"
LOG_DIR = ROOT / ".logs"


def _run_chronos(args: list[str], timeout: int = 90) -> StackControlResponse:
    cmd = ["bash", str(CHRONOS_SCRIPT), *args]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()
    status = "ok" if result.returncode == 0 else "failed"
    return StackControlResponse(
        action=" ".join(args),
        status=status,
        message=f"Command exited with code {result.returncode}",
        output=output or "No output",
    )


def _get_public_url() -> str | None:
    response = _run_chronos(["url"], timeout=10)
    if response.status != "ok":
        return None
    output = (response.output or "").strip()
    if output.startswith("http"):
        return output.rsplit("/api/v1/auth/notion/callback", 1)[0]
    return None


def _backup_info(path: Path, message: str = "") -> BackupInfoOut:
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return BackupInfoOut(
        filename=path.name,
        created_at=created_at,
        size_bytes=stat.st_size,
        download_path=f"/api/v1/admin/backups/{path.name}",
        message=message,
    )


@router.post("/stack/status", response_model=StackControlResponse)
async def stack_status():
    response = _run_chronos(["status"], timeout=15)
    response.public_url = _get_public_url()
    return response


@router.post("/stack/ensure-public", response_model=StackControlResponse)
async def ensure_public_stack():
    response = _run_chronos(["start"], timeout=45)
    response.action = "ensure-public"
    response.public_url = _get_public_url()
    return response


@router.post("/stack/restart-public", response_model=StackControlResponse)
async def restart_public_stack():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    command = (
        f"cd {ROOT} && "
        "(sleep 1; bash ./chronos stop; bash ./chronos start) "
        f">> {LOG_DIR / 'admin-restart.log'} 2>&1"
    )
    subprocess.Popen(
        ["/bin/sh", "-c", command],
        cwd=str(ROOT),
        start_new_session=True,
    )
    return StackControlResponse(
        action="restart-public",
        status="scheduled",
        message="Restart scheduled. The API may be briefly unavailable for a few seconds.",
        output="Queued: chronos stop && chronos start",
    )


@router.get("/backups", response_model=list[BackupInfoOut])
async def list_backups():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backups = sorted(BACKUP_DIR.glob("chronos_backup_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_backup_info(path) for path in backups]


@router.post("/backups", response_model=BackupInfoOut)
async def create_backup():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / "brain.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="brain.db not found")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_path = BACKUP_DIR / f"chronos_backup_{timestamp}.zip"
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size,
        "api_log_exists": (ROOT / ".logs" / "api.log").exists(),
    }

    temp_db_copy = BACKUP_DIR / f"brain_{timestamp}.db"
    shutil.copy2(db_path, temp_db_copy)
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(temp_db_copy, arcname="brain.db")
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
    finally:
        temp_db_copy.unlink(missing_ok=True)

    return _backup_info(archive_path, message="Backup created")


@router.get("/backups/{filename}")
async def download_backup(filename: str):
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = BACKUP_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(path, filename=filename, media_type="application/zip")