"""Administrative endpoints for stack control and backups."""

from __future__ import annotations

import json
import socket
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


def _run_status_command(
    args: list[str], timeout: int = 3
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


async def _run_chronos_async(
    args: list[str], timeout: int = 90
) -> StackControlResponse:
    import asyncio

    cmd = ["bash", str(CHRONOS_SCRIPT), *args]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
    except asyncio.TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")

    output = ((stdout or "") + ("\n" + stderr if stderr else "")).strip()
    status = "ok" if process.returncode == 0 else "failed"
    return StackControlResponse(
        action=" ".join(args),
        status=status,
        message=f"Command exited with code {process.returncode}",
        output=output,
    )


async def _get_public_url_async() -> str | None:
    response = await _run_chronos_async(["url"], timeout=10)
    if response.status != "ok":
        return None
    output = (response.output or "").strip()
    if output.startswith("http"):
        return output.rsplit("/api/v1/auth/notion/callback", 1)[0]
    return None


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
    output = (
        (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    ).strip()
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


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _systemd_state(unit_name: str) -> tuple[str, bool | None, bool | None, str]:
    try:
        active = _run_status_command(["systemctl", "is-active", unit_name])
        enabled = _run_status_command(["systemctl", "is-enabled", unit_name])
    except FileNotFoundError:
        return "unknown", None, None, "systemctl unavailable"
    except Exception as exc:
        return "unknown", None, None, str(exc)

    state = (active.stdout or active.stderr or "unknown").strip()
    enabled_state = (enabled.stdout or enabled.stderr or "unknown").strip()
    healthy = active.returncode == 0 and state == "active"
    is_enabled = enabled.returncode == 0 and enabled_state in {"enabled", "static"}
    detail = f"systemd active={state}; enabled={enabled_state}"
    return state, healthy, is_enabled, detail


def _service_status(
    *,
    name: str,
    display_name: str,
    unit_name: str,
    category: str,
    port: int | None = None,
    url: str | None = None,
) -> dict:
    state, healthy, enabled, detail = _systemd_state(unit_name)
    port_reachable = _is_port_open(port) if port else None
    if port is not None and port_reachable is not None:
        healthy = bool(healthy and port_reachable)
        detail = f"{detail}; port {port} {'reachable' if port_reachable else 'closed'}"

    return {
        "name": name,
        "display_name": display_name,
        "category": category,
        "state": state,
        "healthy": healthy,
        "enabled": enabled,
        "detail": detail,
        "unit_name": unit_name,
        "url": url,
        "port": port,
        "last_transition_at": None,
    }


def _port_status(name: str, port: int, url: str) -> dict:
    reachable = _is_port_open(port)
    return {
        "name": name,
        "port": port,
        "protocol": "tcp",
        "reachable": reachable,
        "url": url,
        "detail": "reachable" if reachable else "closed",
    }


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
    response = await _run_chronos_async(["status"], timeout=15)
    response.public_url = await _get_public_url_async()
    return response


@router.post("/stack/ensure-public", response_model=StackControlResponse)
async def ensure_public_stack():
    response = await _run_chronos_async(["start"], timeout=45)
    response.action = "ensure-public"
    response.public_url = await _get_public_url_async()
    return response


@router.post("/stack/restart-public", response_model=StackControlResponse)
async def restart_public_stack():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / "admin-restart.log"
    log_file = open(log_file_path, "a")
    subprocess.Popen(
        ["bash", "-c", "sleep 1; bash ./chronos stop; bash ./chronos start"],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    return StackControlResponse(
        action="restart-public",
        status="scheduled",
        message="Restart scheduled. The API may be briefly unavailable for a few seconds.",
        output="Queued: chronos stop && chronos start",
    )


@router.get("/runtime")
async def runtime_snapshot():
    """Detailed runtime diagnostics for native clients."""
    captured_at = datetime.now(timezone.utc).isoformat()
    public_url = _get_public_url()

    services = [
        _service_status(
            name="qdrant",
            display_name="Qdrant",
            unit_name="chronos-qdrant.service",
            category="storage",
            port=6333,
            url="http://127.0.0.1:6333",
        ),
        _service_status(
            name="api",
            display_name="Chronos API",
            unit_name="chronos-api.service",
            category="api",
            port=8000,
            url="http://127.0.0.1:8000",
        ),
        _service_status(
            name="ui",
            display_name="Chronos UI",
            unit_name="chronos-ui.service",
            category="ui",
            port=8050,
            url="http://127.0.0.1:8050",
        ),
        _service_status(
            name="auto_sync",
            display_name="Auto Sync",
            unit_name="chronos-auto-sync.service",
            category="sync",
            port=8090,
            url="http://127.0.0.1:8090",
        ),
        _service_status(
            name="ngrok",
            display_name="ngrok",
            unit_name="chronos-ngrok.service",
            category="access",
            url=public_url,
        ),
    ]

    ports = [
        _port_status("qdrant", 6333, "http://127.0.0.1:6333"),
        _port_status("api", 8000, "http://127.0.0.1:8000"),
        _port_status("ui", 8050, "http://127.0.0.1:8050"),
        _port_status("webhook", 8090, "http://127.0.0.1:8090"),
    ]

    passed_checks = sum(1 for service in services if service.get("healthy") is True)
    warning_count = sum(1 for service in services if service.get("healthy") is None)
    failure_count = sum(1 for service in services if service.get("healthy") is False)
    ok = failure_count == 0

    watchdog_state, watchdog_healthy, watchdog_enabled, watchdog_detail = (
        _systemd_state("chronos-watchdog.timer")
    )

    plaud_auth = {
        "state": "unknown",
        "detail": "Plaud auth status unavailable",
        "is_authenticated": False,
        "workspace_name": None,
    }
    try:
        from src.plaud_oauth import PlaudOAuthClient

        token_status = PlaudOAuthClient().token_status_with_recovery(
            attempt_recovery=False
        )
        is_authenticated = bool(token_status.get("is_authenticated"))
        plaud_auth = {
            "state": "connected" if is_authenticated else "disconnected",
            "detail": (
                "Plaud token is authenticated"
                if is_authenticated
                else "Plaud token is not authenticated"
            ),
            "is_authenticated": is_authenticated,
            "workspace_name": token_status.get("workspace_name"),
        }
    except Exception as exc:
        plaud_auth["detail"] = str(exc)

    notes = []
    try:
        from src.config import get_settings
        from src.notion_oauth import NotionOAuthClient

        settings = get_settings()
        notion_status = NotionOAuthClient().token_status
        if notion_status.get("is_authenticated") and settings.notion_database_id:
            notes.append("Notion OAuth and data source are configured")
    except Exception:
        pass

    signals = [
        {
            "source": "runtime",
            "level": "info" if ok else "warning",
            "title": "Runtime Snapshot",
            "message": f"{passed_checks} services healthy, {failure_count} failing",
            "service": None,
            "timestamp": captured_at,
        }
    ]

    return {
        "captured_at": captured_at,
        "runtime_health": {
            "ok": ok,
            "summary": "Runtime healthy" if ok else "Runtime needs attention",
            "detail": f"{passed_checks} passed, {warning_count} unknown, {failure_count} failed",
            "passed_checks": passed_checks,
            "warning_count": warning_count,
            "failure_count": failure_count,
        },
        "runtime_manager": {
            "name": "systemd",
            "mode": "systemd",
            "healthy": watchdog_healthy,
            "detail": watchdog_detail,
            "version": None,
            "watchdog_enabled": watchdog_enabled,
            "watchdog_status": watchdog_state,
            "last_verified_at": captured_at,
        },
        "access": {
            "preferred_kind": "public" if public_url else "local",
            "preferred_label": "Public tunnel" if public_url else "Local API",
            "preferred_ui_url": public_url or "http://127.0.0.1:8050",
            "preferred_api_url": public_url or "http://127.0.0.1:8000",
            "entries": [
                {"label": "Local API", "url": "http://127.0.0.1:8000", "kind": "local"},
                {"label": "Local UI", "url": "http://127.0.0.1:8050", "kind": "local"},
                *(
                    [{"label": "Public tunnel", "url": public_url, "kind": "public"}]
                    if public_url
                    else []
                ),
            ],
        },
        "services": services,
        "ports": ports,
        "signals": signals,
        "plaud_auth": plaud_auth,
        "notes": notes,
    }


@router.get("/backups", response_model=list[BackupInfoOut])
async def list_backups():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backups = sorted(
        BACKUP_DIR.glob("chronos_backup_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
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
