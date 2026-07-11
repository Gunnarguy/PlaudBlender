"""Navigation callbacks - view switching and main content updates."""

import ipaddress
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from dash import Input, Output, State, callback, ctx, html, no_update, ALL, dcc
from dash.exceptions import PreventUpdate
import logging

from app_v2.services import get_data_service
from app_v2.components import (
    create_day_view,
    create_topics_grid,
    create_stats_view,
    create_graph_view,
    create_topic_timeline_view,
    create_search_results,
)

logger = logging.getLogger(__name__)

DEFAULT_PREFERENCES = {
    "auto_refresh_enabled": False,
    "auto_refresh_seconds": 60,
    "default_view": "timeline",
}
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_NOTION_RUNTIME_STATUS_TTL = 30.0
_notion_runtime_status_cache: dict | None = None
_notion_runtime_status_ts = 0.0


def merge_preferences(preferences):
    """Merge user preferences with defaults and coerce safe values."""
    merged = dict(DEFAULT_PREFERENCES)
    if isinstance(preferences, dict):
        merged.update(preferences)

    seconds = merged.get("auto_refresh_seconds", 60)
    try:
        if isinstance(seconds, float):
            import math
            if math.isnan(seconds):
                seconds = 60
            else:
                seconds = int(seconds)
        else:
            seconds = int(seconds)
    except (TypeError, ValueError):
        seconds = 60
    merged["auto_refresh_seconds"] = max(15, min(300, seconds))

    enabled = merged.get("auto_refresh_enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("true", "1", "yes", "on")
    else:
        enabled = bool(enabled)
    merged["auto_refresh_enabled"] = enabled

    default_view = str(merged.get("default_view", "timeline"))
    allowed_views = {
        "timeline",
        "days",
        "topics",
        "graph",
        "stats",
        "system",
        "sync",
        "settings",
    }
    merged["default_view"] = (
        default_view if default_view in allowed_views else "timeline"
    )
    return merged


def _is_tailscale_ipv4(candidate: str) -> bool:
    """Return True when an IPv4 address belongs to Tailscale's CGNAT range."""
    try:
        parsed = ipaddress.ip_address(str(candidate).strip())
    except ValueError:
        return False
    return parsed.version == 4 and parsed in _TAILSCALE_CGNAT


def _is_lan_ipv4(candidate: str) -> bool:
    """Return True for a non-loopback private IPv4 address."""
    try:
        parsed = ipaddress.ip_address(str(candidate).strip())
    except ValueError:
        return False
    return parsed.version == 4 and parsed.is_private and not parsed.is_loopback


def _list_host_ipv4_addresses() -> list[str]:
    """Best-effort inventory of host IPv4 addresses for access links."""
    addresses: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        try:
            parsed = ipaddress.ip_address(str(candidate).strip())
        except ValueError:
            return
        if parsed.version != 4 or parsed.is_loopback:
            return
        value = str(parsed)
        if value not in seen:
            seen.add(value)
            addresses.append(value)

    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            for part in (result.stdout or "").split():
                _add(part)
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(
            socket.gethostname(), None, family=socket.AF_INET, type=socket.SOCK_STREAM
        ):
            _add(info[4][0])
    except Exception:
        pass

    return addresses


def _pick_preferred_lan_ipv4(ip_values: list[str]) -> str | None:
    """Choose the least-weird LAN IPv4 from the host inventory."""
    candidates = [
        ip for ip in ip_values if _is_lan_ipv4(ip) and not _is_tailscale_ipv4(ip)
    ]
    if not candidates:
        return None

    def _sort_key(value: str) -> tuple[int, int, str]:
        prefix_rank = 2
        if value.startswith("10."):
            prefix_rank = 0
        elif value.startswith("192.168."):
            prefix_rank = 1
        try:
            last_octet = int(value.rsplit(".", 1)[-1])
        except ValueError:
            last_octet = 999
        bridge_penalty = 1 if last_octet == 1 else 0
        return (prefix_rank, bridge_penalty, value)

    return sorted(candidates, key=_sort_key)[0]


def _build_access_targets(ip_values: list[str]) -> dict:
    """Summarize local, LAN, and Tailscale endpoints for the current host."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in ip_values:
        try:
            parsed = ipaddress.ip_address(str(raw).strip())
        except ValueError:
            continue
        if parsed.version != 4 or parsed.is_loopback:
            continue
        value = str(parsed)
        if value not in seen:
            seen.add(value)
            normalized.append(value)

    tailscale_ip = next((ip for ip in normalized if _is_tailscale_ipv4(ip)), None)
    lan_ip = _pick_preferred_lan_ipv4(normalized)

    entries = []

    def _append_pair(label: str, host: str, kind: str) -> None:
        entries.append(
            {
                "label": f"{label} UI",
                "url": f"http://{host}:8050",
                "kind": kind,
            }
        )
        entries.append(
            {
                "label": f"{label} API",
                "url": f"http://{host}:8000/api/v1/health",
                "kind": kind,
            }
        )

    if tailscale_ip:
        _append_pair("Tailscale", tailscale_ip, "tailscale")
    if lan_ip and lan_ip != tailscale_ip:
        _append_pair("LAN", lan_ip, "lan")
    _append_pair("Localhost", "127.0.0.1", "local")

    if tailscale_ip:
        preferred_kind = "tailscale"
        preferred_label = "Tailscale"
        preferred_host = tailscale_ip
    elif lan_ip:
        preferred_kind = "lan"
        preferred_label = "LAN"
        preferred_host = lan_ip
    else:
        preferred_kind = "local"
        preferred_label = "Localhost"
        preferred_host = "127.0.0.1"

    return {
        "preferred_kind": preferred_kind,
        "preferred_label": preferred_label,
        "preferred_ui_url": f"http://{preferred_host}:8050",
        "preferred_api_url": f"http://{preferred_host}:8000/api/v1/health",
        "tailscale_ip": tailscale_ip,
        "lan_ip": lan_ip,
        "entries": entries,
    }


def _get_notion_runtime_status() -> dict:
    """Return a cached, shell-friendly summary of the Notion bridge state."""
    global _notion_runtime_status_cache, _notion_runtime_status_ts

    now = time.monotonic()
    if (
        _notion_runtime_status_cache is not None
        and (now - _notion_runtime_status_ts) < _NOTION_RUNTIME_STATUS_TTL
    ):
        return dict(_notion_runtime_status_cache)

    status = {
        "connected": False,
        "database_found": False,
        "has_credentials": False,
        "has_oauth": False,
        "workspace_name": "",
        "database_title": "",
        "label": "Setup",
        "detail": "Set up Notion OAuth or a token to enable the knowledge bridge",
    }

    try:
        from src.notion_oauth import NotionOAuthClient

        oauth = NotionOAuthClient()
        token_status = oauth.token_status
        status["has_oauth"] = bool(token_status.get("is_authenticated"))
        status["has_credentials"] = bool(token_status.get("has_credentials"))
        status["workspace_name"] = token_status.get("workspace_name") or ""

        if status["has_credentials"] or status["has_oauth"]:
            try:
                from src.notion_service import get_notion_service

                svc = get_notion_service()
                connection = svc.check_connection(quick=True)
                status["connected"] = bool(connection.connected)
                status["database_found"] = bool(connection.database_found)
                status["database_title"] = connection.database_title or ""

                if connection.connected and connection.database_found:
                    status["label"] = "Connected"
                    if status["workspace_name"]:
                        status["detail"] = (
                            f"{connection.database_title or 'Data source'} in {status['workspace_name']}"
                        )
                    else:
                        status["detail"] = (
                            connection.database_title or "Notion data source live"
                        )
                elif status["has_oauth"]:
                    status["label"] = "Select DB"
                    status["detail"] = (
                        connection.error
                        or "OAuth is ready, but Chronos still needs a Notion data source selected"
                    )
                else:
                    status["label"] = "Connect"
                    status["detail"] = (
                        connection.error
                        or "Credentials are present, but the Notion bridge is not live yet"
                    )
            except Exception as exc:
                status["label"] = "Check"
                status["detail"] = str(exc)[:120]
    except Exception as exc:
        status["label"] = "Error"
        status["detail"] = str(exc)[:120]

    _notion_runtime_status_cache = dict(status)
    _notion_runtime_status_ts = now
    return status


def _get_cached_notion_bridge_snapshot() -> dict | None:
    """Read the cached Notion bridge payload without forcing a fetch."""
    try:
        from app_v2.callbacks.notion import _get_cached_notion_data

        cached = _get_cached_notion_data()
        return cached if isinstance(cached, dict) else None
    except Exception as exc:
        logger.debug("Could not read cached Notion bridge payload: %s", exc)
        return None


_notion_index_cache: dict[str, dict] | None = None
_notion_index_cache_ts: float = 0.0


def _build_notion_recording_index(snapshot: dict | None) -> dict[str, dict]:
    """Invert cached Notion data into recording-centric page metadata."""
    global _notion_index_cache, _notion_index_cache_ts
    if not snapshot:
        return {}

    now = time.time()
    if _notion_index_cache is not None and (now - _notion_index_cache_ts) < 300.0:
        return _notion_index_cache

    recordings = snapshot.get("recordings") or []
    match_map = snapshot.get("match_map") or {}
    stale_map = snapshot.get("stale_map") or {}
    index: dict[str, dict] = {}

    def _register(recording_id: str, page: dict) -> None:
        if not recording_id:
            return
        rid = str(recording_id).strip()
        page_id = str(page.get("page_id") or "").strip()
        if not rid or not page_id:
            return

        bucket = index.setdefault(rid, {"pages": [], "has_stale": False})
        if any(existing.get("page_id") == page_id for existing in bucket["pages"]):
            return

        page_info = {
            "page_id": page_id,
            "title": str(page.get("title") or "").strip(),
            "url": str(page.get("url") or "").strip(),
            "stale": bool(stale_map.get(page_id)),
        }
        bucket["pages"].append(page_info)
        bucket["has_stale"] = bool(bucket["has_stale"] or page_info["stale"])

    for page in recordings:
        page_id = str(page.get("page_id") or "").strip()
        if not page_id:
            continue

        _register(f"notion:{page_id}", page)

        matched_recording_id = match_map.get(page_id)
        if matched_recording_id:
            _register(str(matched_recording_id), page)

    _notion_index_cache = index
    _notion_index_cache_ts = now
    return index


def _annotate_recording_summary_with_bridge_state(
    summary,
    notion_snapshot: dict | None,
    notion_index: dict[str, dict] | None = None,
):
    """Decorate a recording summary with Notion linkage metadata."""
    if summary is None:
        return None

    notion_index = notion_index or {}
    summary.notion_state = None
    summary.notion_page_id = None
    summary.notion_page_url = None
    summary.notion_page_title = None
    summary.notion_match_count = 0

    recording_id = str(getattr(summary, "recording_id", "") or "").strip()
    source = str(getattr(summary, "source", "") or "").strip().lower()
    index_entry = notion_index.get(recording_id)

    if index_entry:
        pages = index_entry.get("pages") or []
        if pages:
            first_page = pages[0]
            summary.notion_page_id = first_page.get("page_id")
            summary.notion_page_url = first_page.get("url")
            summary.notion_page_title = first_page.get("title")
            summary.notion_match_count = len(pages)

        has_stale = bool(index_entry.get("has_stale"))
        if source == "notion" or recording_id.startswith("notion:"):
            summary.notion_state = "imported-stale" if has_stale else "imported"
        else:
            summary.notion_state = "stale" if has_stale else "linked"
        return summary

    if source == "notion" or recording_id.startswith("notion:"):
        page_id = recording_id.split("notion:", 1)[-1] if recording_id.startswith("notion:") else ""
        summary.notion_page_id = page_id or None
        summary.notion_state = "imported"
        return summary

    status = (notion_snapshot or {}).get("status") or {}
    if (
        status.get("connected")
        and status.get("database_found")
        and str(getattr(summary, "processing_status", "") or "").strip().lower()
        == "completed"
    ):
        summary.notion_state = "chronos-only"

    return summary


def _annotate_days_with_bridge_state(
    days,
    notion_snapshot: dict | None,
    notion_index: dict[str, dict] | None = None,
):
    """Apply cached Notion linkage metadata to all timeline recordings."""
    notion_index = notion_index or {}
    for day in days or []:
        for summary in getattr(day, "recordings", []) or []:
            _annotate_recording_summary_with_bridge_state(
                summary, notion_snapshot, notion_index
            )
    return days


def _build_workspace_pulse(
    runtime_status: dict, current_view: str | None, active_workflows_count: int
) -> html.Div:
    """Build a shell-level summary that ties Timeline, Sync, System, and Notion together."""
    activity = runtime_status.get("activity", {})
    notion = runtime_status.get("notion", {})
    access = runtime_status.get("access", {})
    view_label = str(current_view or "timeline").replace("-", " ").title()
    access_label = access.get("preferred_label", "Localhost")

    if not runtime_status.get("plaud_ok"):
        tone = "warn"
        title = "Plaud intake needs attention"
        body = (
            "Chronos is up, but the Plaud side is not fully linked. Open Sync to reconnect the ingest side and resume the pipeline."
        )
    elif notion.get("has_credentials") and not notion.get("connected"):
        tone = "info"
        title = "Notion bridge needs one more step"
        body = (
            "The Notion integration is partly configured, but the live bridge is not complete yet. "
            "Open Notion to finish OAuth or choose the correct data source so both systems stay in step."
        )
    elif activity.get("active"):
        tone = "info"
        title = "Background sync is active"
        body = activity.get("detail") or activity.get("summary") or (
            "Chronos is working through background sync and import tasks."
        )
    elif active_workflows_count > 0:
        tone = "info"
        title = f"{active_workflows_count} workflow{'s' if active_workflows_count != 1 else ''} still running"
        body = (
            "Background processing is active. Keep browsing anywhere in Chronos, or open Sync if you want the detailed progress view."
        )
    elif notion.get("connected"):
        tone = "ok"
        title = "Chronos is connected end-to-end"
        body = (
            f"Plaud intake, local search, and the Notion bridge are all live. You are in {view_label}, and private access prefers {access_label}."
        )
    else:
        tone = "subtle"
        title = "Chronos shell is live"
        body = (
            f"Use the shared shell to move between Timeline, Sync, System, and Notion without losing operational context. Private access prefers {access_label}."
        )

    meta_items = [
        html.Span(f"View: {view_label}", className="app-pulse-meta-item"),
        html.Span(f"Access: {access_label}", className="app-pulse-meta-item"),
        html.Span(
            f"Notion: {notion.get('label', 'Setup')}",
            className="app-pulse-meta-item",
        ),
    ]
    activity_summary = str(activity.get("summary") or "").strip()
    if activity_summary and activity_summary != "Idle":
        meta_items.append(
            html.Span(
                f"Activity: {activity_summary}",
                className="app-pulse-meta-item",
            )
        )
    database_title = notion.get("database_title")
    if database_title:
        meta_items.append(
            html.Span(
                f"Data source: {database_title}",
                className="app-pulse-meta-item",
            )
        )

    return html.Div(
        className=f"app-pulse-card tone-{tone}",
        children=[
            html.Div(
                className="app-pulse-copy",
                children=[
                    html.Span("Workspace Pulse", className="app-pulse-kicker"),
                    html.H3(title, className="app-pulse-title"),
                    html.P(body, className="app-pulse-body"),
                    html.Div(className="app-pulse-meta", children=meta_items),
                ],
            )
        ],
    )


def _local_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.35) -> bool:
    """Return True when a local TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _systemd_unit_state(unit_name: str) -> tuple[str, str]:
    """Return (active_state, enabled_state) for a systemd unit when available."""
    if platform.system() != "Linux" or not shutil.which("systemctl"):
        return ("unavailable", "unavailable")

    try:
        active = subprocess.run(
            ["systemctl", "is-active", unit_name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        enabled = subprocess.run(
            ["systemctl", "is-enabled", unit_name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        active_state = (active.stdout or active.stderr or "").strip() or "unknown"
        enabled_state = (enabled.stdout or enabled.stderr or "").strip() or "unknown"
        return (active_state, enabled_state)
    except Exception as exc:
        return ("error", str(exc)[:80])


def _get_local_runtime_status() -> dict:
    """Return lightweight runtime status for the host running the Dash UI."""
    systemd_available = platform.system() == "Linux" and bool(shutil.which("systemctl"))
    auto_sync_active, auto_sync_enabled = _systemd_unit_state("chronos-auto-sync.service")
    watchdog_active, watchdog_enabled = _systemd_unit_state("chronos-watchdog.timer")

    managed_states = {"enabled", "enabled-runtime", "linked", "alias", "static", "indirect"}
    active_states = {"active", "activating", "reloading"}
    systemd_managed_auto_sync = systemd_available and (
        auto_sync_enabled in managed_states or auto_sync_active in active_states
    )

    plaud_ok = False
    plaud_label = "Missing"
    plaud_detail = "No local Plaud token"
    try:
        from src.plaud_oauth import PlaudOAuthClient

        token_status = PlaudOAuthClient().token_status
        plaud_ok = bool(token_status.get("is_authenticated"))
        mins = token_status.get("expires_in_minutes")
        if plaud_ok:
            plaud_label = "Linked"
            if mins is not None:
                plaud_detail = f"Token valid for ~{int(mins)} min"
            else:
                plaud_detail = "Token active"
        elif token_status.get("has_refresh_token"):
            plaud_label = "Refreshable"
            plaud_detail = "Access token expired; refresh token present"
    except Exception as exc:
        plaud_label = "Error"
        plaud_detail = str(exc)[:80]

    if systemd_managed_auto_sync:
        manager_label = "systemd"
        auto_sync_label = auto_sync_active.title()
        auto_sync_ok = auto_sync_active in active_states
        auto_sync_detail = (
            f"chronos-auto-sync.service: {auto_sync_active} ({auto_sync_enabled})"
        )
    else:
        manager_label = "embedded"
        auto_sync_label = "In-App"
        auto_sync_ok = _local_port_open(8090)
        auto_sync_detail = (
            "Managed inside the Dash app on this host"
            if not systemd_available
            else "No active systemd auto-sync unit detected"
        )

    watchdog_ok = watchdog_active in active_states
    if systemd_available and watchdog_enabled != "unavailable":
        watchdog_label = watchdog_active.title()
        watchdog_detail = f"chronos-watchdog.timer: {watchdog_active} ({watchdog_enabled})"
    else:
        watchdog_label = "N/A"
        watchdog_detail = "systemd timer not available on this host"

    activity = _summarize_background_activity()

    return {
        "manager_label": manager_label,
        "manager_detail": (
            "Dedicated systemd services own the pipeline"
            if systemd_managed_auto_sync
            else "Dash app owns the local background sync worker"
        ),
        "systemd_managed_auto_sync": systemd_managed_auto_sync,
        "auto_sync_ok": auto_sync_ok,
        "auto_sync_label": auto_sync_label,
        "auto_sync_detail": auto_sync_detail,
        "watchdog_ok": watchdog_ok,
        "watchdog_label": watchdog_label,
        "watchdog_detail": watchdog_detail,
        "activity": activity,
        "plaud_ok": plaud_ok,
        "plaud_label": plaud_label,
        "plaud_detail": plaud_detail,
        "ports": [
            {"label": "UI", "port": 8050, "ok": _local_port_open(8050)},
            {"label": "API", "port": 8000, "ok": _local_port_open(8000)},
            {"label": "Qdrant", "port": 6333, "ok": _local_port_open(6333)},
            {"label": "Webhook", "port": 8090, "ok": _local_port_open(8090)},
        ],
        "notion": _get_notion_runtime_status(),
        "access": _build_access_targets(_list_host_ipv4_addresses()),
    }


def _summarize_background_activity() -> dict:
    """Summarize active pipeline and Notion import work for the shell UI."""
    pipeline_data = None
    notion_data = None

    try:
        from src.chronos.pipeline_progress import read_progress

        pipeline_data = read_progress() or None
    except Exception:
        pipeline_data = None

    try:
        from src.chronos.notion_bridge import get_import_progress

        notion_data = get_import_progress() or None
    except Exception:
        notion_data = None

    summary_parts: list[str] = []
    detail_parts: list[str] = []

    if pipeline_data and str(pipeline_data.get("status", "")).lower() == "running":
        current_phase = str(pipeline_data.get("current_phase") or "pipeline")
        current_phase = current_phase.replace("-", " ").title()
        phase_info = next(
            (
                phase
                for phase in pipeline_data.get("phases", [])
                if str(phase.get("name") or "")
                == str(pipeline_data.get("current_phase") or "")
            ),
            {},
        )
        completed = int(phase_info.get("completed_items") or 0)
        total = int(phase_info.get("total_items") or 0)
        pipeline_summary = (
            f"Pipeline {current_phase} {completed}/{total}"
            if total > 0
            else f"Pipeline {current_phase}"
        )
        summary_parts.append(pipeline_summary)
        pipeline_step = str(phase_info.get("current_step") or "").strip()
        pipeline_item = str(phase_info.get("current_item") or "").strip()
        pipeline_detail = pipeline_step or pipeline_summary
        if pipeline_item:
            pipeline_detail = f"{pipeline_detail} · {pipeline_item}"
        detail_parts.append(pipeline_detail)

    if notion_data and str(notion_data.get("status", "")).lower() == "running":
        completed = int(notion_data.get("completed") or 0)
        total = int(notion_data.get("total") or 0)
        notion_summary = (
            f"Notion {completed}/{total}" if total > 0 else "Notion import"
        )
        summary_parts.append(notion_summary)
        current_title = str(notion_data.get("current_title") or "").strip()
        if len(current_title) > 60:
            current_title = f"{current_title[:57]}..."
        detail_parts.append(
            f"Notion import is processing {current_title}"
            if current_title
            else notion_summary
        )

    active = bool(summary_parts)
    return {
        "active": active,
        "summary": " · ".join(summary_parts) if active else "Idle",
        "detail": " | ".join(detail_parts)
        if detail_parts
        else "No background work is active right now.",
    }


def _read_log_tail(log_name: str, max_lines: int = 12) -> list[str]:
    """Read the last few lines from a local log file if it exists."""
    log_path = Path(__file__).resolve().parents[2] / "logs" / log_name
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception as exc:
        return [f"Could not read {log_name}: {str(exc)[:80]}"]

    trimmed = [line.strip() for line in lines if line.strip()]
    return trimmed[-max_lines:]


def create_system_view(service) -> html.Div:
    """Create a dedicated host/runtime diagnostics page for Pi and desktop installs."""
    from src.config import get_settings

    runtime_status = _get_local_runtime_status()
    access_summary = runtime_status.get("access", {})
    checks = _check_services(get_settings())
    db_stats = service.get_recording_db_stats()
    stats = service.get_stats()

    service_units = [
        ("UI", "chronos-ui.service"),
        ("Auto-Sync", "chronos-auto-sync.service"),
        ("API", "chronos-api.service"),
        ("MCP", "chronos-mcp.service"),
        ("Qdrant", "chronos-qdrant.service"),
        ("Watchdog", "chronos-watchdog.timer"),
    ]

    unit_rows = []
    for label, unit_name in service_units:
        active_state, enabled_state = _systemd_unit_state(unit_name)
        ok = active_state in {"active", "activating", "reloading"}
        unit_rows.append(
            html.Div(
                className="setting-row",
                children=[
                    html.Label(f"{label}:"),
                    html.Span(
                        active_state.title(),
                        className=f"status-badge {'connected' if ok else 'disconnected'}",
                    ),
                    html.Span(
                        f"{unit_name} ({enabled_state})",
                        className="status-detail",
                    ),
                ],
            )
        )

    connectivity_rows = []
    for label, key in [
        ("Plaud", "plaud"),
        ("Gemini", "gemini"),
        ("OpenAI", "openai"),
        ("SQLite", "sqlite"),
        ("Qdrant", "qdrant"),
        ("Webhook Listener", "webhook_listener"),
        ("Webhook Config", "webhook_config"),
    ]:
        ok, detail = checks.get(key, (False, "Unavailable"))
        connectivity_rows.append(
            html.Div(
                className="setting-row",
                children=[
                    html.Label(f"{label}:"),
                    html.Span(
                        "OK" if ok else "Check",
                        className=f"status-badge {'connected' if ok else 'disconnected'}",
                    ),
                    html.Span(detail, className="status-detail"),
                ],
            )
        )

    port_badges = []
    for port in runtime_status.get("ports", []):
        port_badges.append(
            html.Div(
                className="status-stat",
                children=[
                    html.Span(
                        "UP" if port["ok"] else "DOWN",
                        className="big-number",
                        style={
                            "fontSize": "0.95rem",
                            "color": "#10b981" if port["ok"] else "#ef4444",
                        },
                    ),
                    html.Span(
                        f"{port['label']}:{port['port']}",
                        className="stat-label",
                    ),
                ],
            )
        )

    watchdog_lines = _read_log_tail("watchdog.log")
    auto_sync_lines = _read_log_tail("auto_sync.log")

    access_rows = []
    for entry in access_summary.get("entries", []):
        access_rows.append(
            html.Div(
                className="setting-row",
                children=[
                    html.Label(f"{entry['label']}:"),
                    html.A(
                        entry["url"],
                        href=entry["url"],
                        target="_blank",
                        rel="noreferrer",
                        className="runtime-access-link",
                    ),
                ],
            )
        )

    if access_summary.get("preferred_kind") == "tailscale":
        access_note = (
            "Private mesh access through Tailscale is the recommended way to reach this Chronos host remotely."
        )
    elif access_summary.get("preferred_kind") == "lan":
        access_note = "Chronos is reachable on the local network, but private mesh access is preferred when available."
    else:
        access_note = "Only localhost access is available from this host right now."

    quick_commands = [
        "~/PlaudBlender/deploy/verify-pi.sh",
        "~/PlaudBlender/deploy/update-pi.sh",
        "systemctl status chronos-ui chronos-auto-sync chronos-api chronos-mcp chronos-qdrant",
        "journalctl -u chronos-auto-sync -f",
        "journalctl -u chronos-watchdog.service -n 50 --no-pager",
    ]

    pending_count = int(db_stats.get("pending", 0) or 0)
    processing_count = int(db_stats.get("processing", 0) or 0)
    completed_count = int(db_stats.get("completed", 0) or 0)
    failed_count = int(db_stats.get("failed", 0) or 0)

    return html.Div(
        className="sync-view system-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("🖥 System", className="view-title"),
                    html.P(
                        "Live host diagnostics for the machine serving Chronos. On the Pi, this is the actual systemd-managed runtime.",
                        className="view-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="sync-dashboard-grid",
                children=[
                    html.Div(
                        className="sync-main-column",
                        children=[
                            html.Div(
                                className="sync-status-card runtime-status-card",
                                children=[
                                    html.H4("Runtime Overview"),
                                    html.Div(
                                        className="status-stats",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Span(
                                                        runtime_status["manager_label"],
                                                        className="big-number",
                                                        style={
                                                            "fontSize": "0.95rem",
                                                            "color": "#60a5fa",
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Manager",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        runtime_status[
                                                            "auto_sync_label"
                                                        ],
                                                        className="big-number",
                                                        style={
                                                            "fontSize": "0.95rem",
                                                            "color": (
                                                                "#10b981"
                                                                if runtime_status[
                                                                    "auto_sync_ok"
                                                                ]
                                                                else "#ef4444"
                                                            ),
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Auto-Sync",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        runtime_status[
                                                            "watchdog_label"
                                                        ],
                                                        className="big-number",
                                                        style={
                                                            "fontSize": "0.95rem",
                                                            "color": (
                                                                "#10b981"
                                                                if runtime_status[
                                                                    "watchdog_ok"
                                                                ]
                                                                else "#94a3b8"
                                                            ),
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Watchdog",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        runtime_status["plaud_label"],
                                                        className="big-number",
                                                        style={
                                                            "fontSize": "0.95rem",
                                                            "color": (
                                                                "#10b981"
                                                                if runtime_status[
                                                                    "plaud_ok"
                                                                ]
                                                                else "#ef4444"
                                                            ),
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Plaud Auth",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="auto-sync-details",
                                        children=[
                                            html.Span(
                                                runtime_status["manager_detail"],
                                                className="sync-detail-text",
                                            ),
                                            html.Span(
                                                " · ", className="sync-detail-sep"
                                            ),
                                            html.Span(
                                                runtime_status["auto_sync_detail"],
                                                className="sync-detail-text",
                                            ),
                                            html.Span(
                                                " · ", className="sync-detail-sep"
                                            ),
                                            html.Span(
                                                runtime_status["watchdog_detail"],
                                                className="sync-detail-text",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="status-stats",
                                        style={"marginTop": "12px"},
                                        children=port_badges,
                                    ),
                                    html.Div(
                                        className="auto-sync-details",
                                        children=[
                                            html.Span(
                                                runtime_status["plaud_detail"],
                                                className="sync-detail-text",
                                            )
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sync-status-card",
                                children=[
                                    html.H4("Service Diagnostics"),
                                    *unit_rows,
                                    html.H4(
                                        "Connectivity Checks",
                                        style={"marginTop": "14px"},
                                    ),
                                        "OpenAI below powers Ask Chronos and the default transcript extraction path. Embeddings below are OpenAI embeddings.",
                                ],
                            ),
                            html.Div(
                                className="sync-status-card",
                                children=[
                                    html.H4("Access Paths"),
                                    html.P(access_note, className="sync-note"),
                                    html.Div(
                                        className="setting-row",
                                        children=[
                                            html.Label("Preferred:"),
                                            html.Span(
                                                access_summary.get(
                                                    "preferred_label", "Localhost"
                                                ),
                                                className="status-detail",
                                            ),
                                        ],
                                    ),
                                    *access_rows,
                                ],
                            ),
                            html.Div(
                                className="sync-status-card",
                                children=[
                                    html.H4("Quick Commands"),
                                    html.P(
                                        "These run on the Pi itself. The verify script is the single-command red/green report.",
                                        className="sync-note",
                                    ),
                                    html.Pre(
                                        "\n".join(quick_commands),
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "wordBreak": "break-word",
                                            "fontSize": "0.85rem",
                                            "background": "rgba(9, 105, 218, 0.04)",
                                            "border": "1px solid rgba(9, 105, 218, 0.1)",
                                            "borderRadius": "12px",
                                            "padding": "12px",
                                            "margin": "10px 0 0 0",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="sync-side-column",
                        children=[
                            html.Div(
                                className="sync-status-card",
                                children=[
                                    html.H4("Pipeline Snapshot"),
                                    html.Div(
                                        className="status-stats",
                                        children=[
                                            html.Div(
                                                [
                                                    html.Span(
                                                        str(pending_count),
                                                        className="big-number",
                                                        style={"color": "#f59e0b"},
                                                    ),
                                                    html.Span(
                                                        "Pending",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        str(processing_count),
                                                        className="big-number",
                                                        style={"color": "#3b82f6"},
                                                    ),
                                                    html.Span(
                                                        "Processing",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        str(completed_count),
                                                        className="big-number",
                                                        style={"color": "#10b981"},
                                                    ),
                                                    html.Span(
                                                        "Completed",
                                                        className="stat-label",
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                            html.Div(
                                                [
                                                    html.Span(
                                                        str(failed_count),
                                                        className="big-number",
                                                        style={
                                                            "color": (
                                                                "#ef4444"
                                                                if failed_count
                                                                else "#94a3b8"
                                                            )
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Failed", className="stat-label"
                                                    ),
                                                ],
                                                className="status-stat",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="auto-sync-details",
                                        children=[
                                            html.Span(
                                                f"Events indexed: {stats.total_events}",
                                                className="sync-detail-text",
                                            ),
                                            html.Span(
                                                " · ", className="sync-detail-sep"
                                            ),
                                            html.Span(
                                                f"Topics: {stats.total_topics}",
                                                className="sync-detail-text",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sync-status-card",
                                children=[
                                    html.H4("Recent Watchdog Activity"),
                                    html.P(
                                        "Latest lines from logs/watchdog.log on this host.",
                                        className="sync-note",
                                    ),
                                    html.Pre(
                                        (
                                            "\n".join(watchdog_lines)
                                            if watchdog_lines
                                            else "No watchdog log entries yet."
                                        ),
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "wordBreak": "break-word",
                                            "fontSize": "0.82rem",
                                            "background": "rgba(15, 23, 42, 0.03)",
                                            "border": "1px solid rgba(148, 163, 184, 0.18)",
                                            "borderRadius": "12px",
                                            "padding": "12px",
                                            "margin": "10px 0 0 0",
                                            "maxHeight": "280px",
                                            "overflowY": "auto",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                className="sync-status-card",
                                children=[
                                    html.H4("Recent Auto-Sync Activity"),
                                    html.P(
                                        "Latest lines from logs/auto_sync.log on this host.",
                                        className="sync-note",
                                    ),
                                    html.Pre(
                                        (
                                            "\n".join(auto_sync_lines)
                                            if auto_sync_lines
                                            else "No auto-sync log entries yet."
                                        ),
                                        style={
                                            "whiteSpace": "pre-wrap",
                                            "wordBreak": "break-word",
                                            "fontSize": "0.82rem",
                                            "background": "rgba(15, 23, 42, 0.03)",
                                            "border": "1px solid rgba(148, 163, 184, 0.18)",
                                            "borderRadius": "12px",
                                            "padding": "12px",
                                            "margin": "10px 0 0 0",
                                            "maxHeight": "280px",
                                            "overflowY": "auto",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_sync_view(service) -> html.Div:
    """Create the sync view with full pipeline controls and auto-sync status."""
    stats = service.get_stats()
    db_stats = service.get_recording_db_stats()
    workflow_stats = service.get_plaud_workflow_stats(days_back=30)
    failure_summary = service.get_sync_failure_summary(limit=5)
    runtime_status = _get_local_runtime_status()

    pending = db_stats.get("pending", 0)
    processing = db_stats.get("processing", 0)
    failed = db_stats.get("failed", 0)
    completed = db_stats.get("completed", 0)
    total = int(db_stats.get("total", pending + processing + failed + completed) or 0)
    actionable_failed = int(failure_summary.get("actionable_count", 0) or 0)

    # Auto-sync status
    auto_sync_children = []
    if not runtime_status.get("systemd_managed_auto_sync"):
        try:
            from src.plaud_auto_sync import get_auto_sync

            sync_svc = get_auto_sync()
            status = sync_svc.get_status()
            is_running = status.get("running", False)
            devices = status.get("connected_devices", 0)
            pending_jobs = status.get("pending_jobs", 0)
            total_syncs = status.get("total_syncs", 0)
            last_sync = status.get("last_sync")
            config = status.get("config", {})

            auto_sync_children = [
                html.Div(
                    className="sync-status-card auto-sync-card",
                    children=[
                        html.H4("⚡ Auto-Sync"),
                        html.Div(
                            className="status-stats",
                            children=[
                                html.Div(
                                    [
                                        html.Span(
                                            "●",
                                            className="big-number",
                                            style={
                                                "color": (
                                                    "#10b981" if is_running else "#ef4444"
                                                )
                                            },
                                        ),
                                        html.Span(
                                            "Running" if is_running else "Stopped",
                                            className="stat-label",
                                        ),
                                    ],
                                    className="status-stat",
                                ),
                                html.Div(
                                    [
                                        html.Span(str(devices), className="big-number"),
                                        html.Span("USB Devices", className="stat-label"),
                                    ],
                                    className="status-stat",
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            str(pending_jobs), className="big-number"
                                        ),
                                        html.Span("Pending Jobs", className="stat-label"),
                                    ],
                                    className="status-stat",
                                ),
                                html.Div(
                                    [
                                        html.Span(str(total_syncs), className="big-number"),
                                        html.Span("Total Syncs", className="stat-label"),
                                    ],
                                    className="status-stat",
                                ),
                            ],
                        ),
                        html.Div(
                            className="auto-sync-details",
                            children=[
                                html.Span(
                                    f"Last sync: {last_sync or 'Never'}",
                                    className="sync-detail-text",
                                ),
                                html.Span(" · ", className="sync-detail-sep"),
                                html.Span(
                                    f"USB: {'on' if config.get('sync_on_usb') else 'off'}",
                                    className="sync-detail-text",
                                ),
                                html.Span(" · ", className="sync-detail-sep"),
                                html.Span(
                                    f"Webhook: {'on' if config.get('sync_on_webhook') else 'off'}",
                                    className="sync-detail-text",
                                ),
                                html.Span(" · ", className="sync-detail-sep"),
                                html.Span(
                                    f"Webhook server: {'port ' + str(status.get('webhook_port', 8090)) if status.get('webhook_server_running') else 'off'}",
                                    className="sync-detail-text",
                                ),
                                html.Span(" · ", className="sync-detail-sep"),
                                html.Span(
                                    f"Cloud poll: {'every ' + str(config.get('poll_interval_minutes', 15)) + 'm' if config.get('enable_scheduled_poll') else 'off'}",
                                    className="sync-detail-text",
                                ),
                                *(
                                    [
                                        html.Span(" · ", className="sync-detail-sep"),
                                        html.Span(
                                            f"Last poll: {status.get('last_poll', 'Never')}",
                                            className="sync-detail-text",
                                        ),
                                    ]
                                    if status.get("last_poll")
                                    else []
                                ),
                            ],
                        ),
                        *(
                            [
                                html.H5(
                                    "Recent Activity",
                                    style={"marginTop": "12px", "marginBottom": "6px"},
                                ),
                                html.Div(
                                    className="sync-history",
                                    children=[
                                        html.Div(
                                            className=f"sync-history-item {job.status}",
                                            children=[
                                                html.Span(
                                                    {
                                                        "completed": "✅",
                                                        "failed": "❌",
                                                        "running": "🔄",
                                                        "timeout": "⏰",
                                                        "error": "⚠️",
                                                        "pending": "⏳",
                                                    }.get(job.status, "•"),
                                                    className="history-icon",
                                                ),
                                                html.Span(
                                                    job.trigger.value.replace(
                                                        "_", " "
                                                    ).title(),
                                                    className="history-trigger",
                                                ),
                                                html.Span(
                                                    job.timestamp.strftime("%H:%M:%S"),
                                                    className="history-time",
                                                ),
                                                html.Span(
                                                    (job.result or "")[:60],
                                                    className="history-result",
                                                ),
                                            ],
                                        )
                                        for job in reversed(sync_svc.sync_history[-10:])
                                    ],
                                ),
                            ]
                            if sync_svc.sync_history
                            else []
                        ),
                    ],
                ),
            ]
        except Exception as e:
            logger.debug(f"Auto-sync status unavailable: {e}")

    # Plaud cloud stats row
    plaud_cloud_children = []

    # Failed recording details (actionable vs archived)
    failed_details_children = []
    actionable_rows = failure_summary.get("actionable", [])
    if actionable_rows:
        failed_details_children.append(
            html.Div(
                className="failed-recordings-detail",
                style={
                    "marginTop": "10px",
                    "borderTop": "1px solid var(--border-color, #e2e8f0)",
                    "paddingTop": "10px",
                },
                children=[
                    html.Span(
                        "🔍 Actionable Failures:",
                        style={
                            "fontWeight": "600",
                            "fontSize": "0.85rem",
                            "color": "#ef4444",
                        },
                    ),
                    html.Ul(
                        style={
                            "margin": "4px 0 0 0",
                            "paddingLeft": "18px",
                            "fontSize": "0.8rem",
                            "color": "#94a3b8",
                        },
                        children=[
                            html.Li(
                                f"{row['recording_id'][:16]}… — {row['error'][:80]}"
                            )
                            for row in actionable_rows
                        ],
                    ),
                ],
            )
        )
    if stats.plaud_cloud_stats:
        cs = stats.plaud_cloud_stats
        cloud_total = cs.get("total_count", 0)
        cloud_hours = cs.get("total_duration_hours", 0)
        local_plaud_completed = 0
        try:
            from src.database.engine import SessionLocal as _SL
            import sqlalchemy as sa

            _db = _SL()
            try:
                row = _db.execute(
                    sa.text(
                        "SELECT COUNT(*) FROM chronos_recordings "
                        "WHERE source = 'plaud' AND processing_status = 'completed'"
                    )
                ).fetchone()
                local_plaud_completed = int(row[0] or 0) if row else 0
            finally:
                _db.close()
        except Exception:
            local_plaud_completed = completed

        synced_pct = (
            min(100.0, (local_plaud_completed / cloud_total * 100))
            if cloud_total
            else 0
        )

        plaud_cloud_children = [
            html.Div(
                className="status-stats",
                style={
                    "marginTop": "10px",
                    "borderTop": "1px solid var(--border-color, #e2e8f0)",
                    "paddingTop": "10px",
                },
                children=[
                    html.Div(
                        [
                            html.Span(str(cloud_total), className="big-number"),
                            html.Span("Plaud Cloud", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(f"{cloud_hours:.1f}", className="big-number"),
                            html.Span("Cloud Hours", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{synced_pct:.0f}%",
                                className="big-number",
                                style={
                                    "color": "#10b981" if synced_pct > 80 else "#f59e0b"
                                },
                            ),
                            html.Span("Synced", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                ],
            ),
        ]

    workflow_last_run = workflow_stats.get("last_submitted_at") or "Never"

    # --- Plaud auth status ---
    auth_connected = False
    auth_detail = "Not configured"
    auth_expiry_text = ""
    try:
        from src.plaud_oauth import PlaudOAuthClient

        _oauth = PlaudOAuthClient()
        _ts = _oauth.token_status
        auth_connected = _ts.get("is_authenticated", False)
        if auth_connected:
            mins = _ts.get("expires_in_minutes")
            if mins is not None:
                auth_expiry_text = f"Token expires in {int(mins)} min"
            else:
                auth_expiry_text = "Token active"
            auth_detail = auth_expiry_text
        else:
            auth_detail = "Not connected — click Connect to authenticate"
    except Exception as _ae:
        auth_detail = f"Config error: {_ae}"

    auth_section = html.Div(
        className="sync-status-card plaud-auth-card",
        children=[
            html.H4("🔐 Plaud Account"),
            html.Div(
                className="plaud-auth-row",
                children=[
                    html.Div(
                        className="plaud-auth-status",
                        children=[
                            html.Span(
                                "●",
                                className="auth-dot",
                                style={
                                    "color": "#10b981" if auth_connected else "#ef4444",
                                    "fontSize": "1.2rem",
                                    "marginRight": "8px",
                                },
                            ),
                            html.Span(
                                "Connected" if auth_connected else "Disconnected",
                                className="auth-status-text",
                                style={
                                    "fontWeight": "600",
                                    "color": "#10b981" if auth_connected else "#ef4444",
                                },
                            ),
                        ],
                    ),
                    html.Span(
                        auth_detail,
                        className="auth-detail-text",
                        style={
                            "color": "#94a3b8",
                            "fontSize": "0.85rem",
                        },
                    ),
                ],
            ),
            html.Div(
                className="plaud-auth-actions",
                children=[
                    html.A(
                        className="sync-action-btn plaud-connect-btn"
                        + (" connected" if auth_connected else ""),
                        href="/auth/plaud",
                        children=[
                            html.Span(
                                "🔗" if not auth_connected else "🔄",
                                className="btn-icon",
                            ),
                            html.Span(
                                (
                                    "Connect Plaud Account"
                                    if not auth_connected
                                    else "Reconnect"
                                ),
                                className="btn-text",
                            ),
                        ],
                    ),
                ],
            ),
            *(
                [
                    html.P(
                        [
                            "First time? Register ",
                            html.Code("https://localhost:8050/auth/plaud/callback"),
                            " as a redirect URI in your ",
                            html.A(
                                "Plaud Developer Portal",
                                href="https://platform.plaud.ai/developer/portal",
                                target="_blank",
                                style={"color": "#60a5fa"},
                            ),
                            ".",
                        ],
                        className="sync-note",
                        style={"marginTop": "10px"},
                    ),
                ]
                if not auth_connected
                else []
            ),
        ],
    )

    runtime_port_children = []
    for idx, port_status in enumerate(runtime_status.get("ports", [])):
        if idx:
            runtime_port_children.append(
                html.Span(" · ", className="sync-detail-sep")
            )
        runtime_port_children.append(
            html.Span(
                f"{port_status['label']}:{port_status['port']} {'up' if port_status['ok'] else 'down'}",
                className="sync-detail-text",
                style={"color": "#10b981" if port_status["ok"] else "#ef4444"},
            )
        )

    runtime_card = html.Div(
        className="sync-status-card runtime-status-card",
        children=[
            html.H4("🖥 Local Runtime"),
            html.Div(
                className="status-stats",
                children=[
                    html.Div(
                        [
                            html.Span(
                                runtime_status["manager_label"],
                                className="big-number",
                                style={"fontSize": "0.95rem", "color": "#60a5fa"},
                            ),
                            html.Span("Manager", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                runtime_status["auto_sync_label"],
                                className="big-number",
                                style={
                                    "fontSize": "0.95rem",
                                    "color": "#10b981"
                                    if runtime_status["auto_sync_ok"]
                                    else "#ef4444",
                                },
                            ),
                            html.Span("Auto-Sync", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                runtime_status["watchdog_label"],
                                className="big-number",
                                style={
                                    "fontSize": "0.95rem",
                                    "color": "#10b981"
                                    if runtime_status["watchdog_ok"]
                                    else "#94a3b8",
                                },
                            ),
                            html.Span("Watchdog", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                runtime_status["plaud_label"],
                                className="big-number",
                                style={
                                    "fontSize": "0.95rem",
                                    "color": "#10b981"
                                    if runtime_status["plaud_ok"]
                                    else "#ef4444",
                                },
                            ),
                            html.Span("Plaud Auth", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                ],
            ),
            html.Div(
                className="auto-sync-details",
                children=[
                    html.Span(
                        runtime_status["manager_detail"],
                        className="sync-detail-text",
                    ),
                    html.Span(" · ", className="sync-detail-sep"),
                    html.Span(
                        runtime_status["auto_sync_detail"],
                        className="sync-detail-text",
                    ),
                    html.Span(" · ", className="sync-detail-sep"),
                    html.Span(
                        runtime_status["watchdog_detail"],
                        className="sync-detail-text",
                    ),
                ],
            ),
            html.Div(className="auto-sync-details", children=runtime_port_children),
            html.Div(
                className="auto-sync-details",
                children=[
                    html.Span(
                        runtime_status["plaud_detail"],
                        className="sync-detail-text",
                    )
                ],
            ),
        ],
    )

    # Build active workflows list for monitoring dashboard
    active_workflows = workflow_stats.get("active_workflows", [])
    active_workflow_cards = []
    for wf in active_workflows[:10]:
        wf_status = str(wf.get("status", "PENDING")).upper()
        completed_tasks = wf.get("completed_tasks", 0)
        total_tasks = wf.get("total_tasks", 0)
        pct = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        status_color = {
            "SUCCESS": "#10b981",
            "COMPLETED": "#10b981",
            "FAILED": "#ef4444",
            "ERROR": "#ef4444",
            "PROCESSING": "#3b82f6",
            "RUNNING": "#3b82f6",
        }.get(wf_status, "#f59e0b")

        active_workflow_cards.append(
            html.Div(
                className="workflow-monitor-card",
                children=[
                    html.Div(
                        className="wf-card-header",
                        children=[
                            html.Span(
                                wf.get("title", wf.get("recording_id", "")[:16]),
                                className="wf-card-title",
                            ),
                            html.Span(
                                wf_status,
                                className="wf-status-badge",
                                style={"backgroundColor": status_color},
                            ),
                        ],
                    ),
                    html.Div(
                        className="wf-card-progress",
                        children=[
                            html.Div(
                                className="wf-progress-bar",
                                children=[
                                    html.Div(
                                        className="wf-progress-fill",
                                        style={"width": f"{pct:.0f}%"},
                                    ),
                                ],
                            ),
                            html.Span(
                                f"{completed_tasks}/{total_tasks} tasks",
                                className="wf-progress-text",
                            ),
                        ],
                    ),
                    *(
                        [
                            html.Span(
                                f"Template: {wf.get('template_id', 'summary-only')}",
                                className="wf-card-detail",
                            ),
                        ]
                        if wf.get("template_id")
                        else []
                    ),
                ],
            )
        )

    workflow_monitor_section = []
    if active_workflow_cards:
        workflow_monitor_section = [
            html.Div(
                className="sync-status-card workflow-monitor-card-container",
                children=[
                    html.H4("🔄 Active Workflows"),
                    html.Div(
                        className="workflow-monitor-grid",
                        children=active_workflow_cards,
                    ),
                ],
            ),
        ]

    # Upload candidates section
    upload_section = []
    try:
        upload_candidates = service.get_upload_candidates()
        if upload_candidates:
            upload_section = [
                html.Div(
                    className="sync-status-card upload-section",
                    children=[
                        html.H4("📤 Upload Local Recordings"),
                        html.P(
                            f"Found {len(upload_candidates)} local audio file(s) in data/raw/usb_import/ not yet in Plaud cloud.",
                            className="sync-note",
                        ),
                        html.Div(
                            className="upload-candidates-list",
                            children=[
                                html.Div(
                                    className="upload-candidate-row",
                                    children=[
                                        html.Span(
                                            f"📎 {f.get('name', 'Unknown')} ({f.get('size_mb', 0):.1f} MB, {f.get('format', '?')})",
                                            className="upload-candidate-name",
                                        ),
                                    ],
                                )
                                for f in upload_candidates[:20]
                            ],
                        ),
                        html.Button(
                            id="upload-files-btn",
                            className="sync-action-btn",
                            children=[
                                html.Span("📤", className="btn-icon"),
                                html.Span(
                                    f"Upload & Process ({len(upload_candidates)} files)",
                                    className="btn-text",
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        else:
            # Hidden button so Dash callbacks don't fail
            upload_section = [
                html.Button(
                    id="upload-files-btn",
                    style={"display": "none"},
                ),
            ]
    except Exception:
        upload_section = [
            html.Button(
                id="upload-files-btn",
                style={"display": "none"},
            ),
        ]

    # Build pending recordings detail section for Sync view
    pipeline_pending_details_children = []
    if pending + processing > 0:
        try:
            pending_recs = service.get_pending_recording_details()
            if pending_recs:
                pipeline_pending_details_children.append(
                    html.Div(
                        className="failed-recordings-detail",
                        style={
                            "marginTop": "10px",
                            "borderTop": "1px solid var(--border-color, #e2e8f0)",
                            "paddingTop": "10px",
                        },
                        children=[
                            html.Span(
                                "\u23f3 Waiting to Process:",
                                style={
                                    "fontWeight": "600",
                                    "fontSize": "0.85rem",
                                    "color": "#f59e0b",
                                },
                            ),
                            html.P(
                                "These recordings have been fetched from Plaud but not yet processed by the local OpenAI pipeline. "
                                "Run Full Sync to process them.",
                                style={
                                    "margin": "4px 0 4px 0",
                                    "fontSize": "0.78rem",
                                    "color": "#94a3b8",
                                },
                            ),
                            html.Ul(
                                style={
                                    "margin": "4px 0 0 0",
                                    "paddingLeft": "18px",
                                    "fontSize": "0.8rem",
                                    "color": "#94a3b8",
                                },
                                children=[
                                    html.Li(
                                        [
                                            html.Span(
                                                (rec["title"][:40] + "\u2026" if len(rec["title"]) > 40 else rec["title"])
                                                if rec["title"] != "Untitled"
                                                else rec["recording_id"][:16] + "\u2026",
                                                style={"color": "#cbd5e1"},
                                            ),
                                            html.Span(
                                                f" \u2014 {rec['date']}" if rec["date"] else "",
                                            ),
                                            html.Span(
                                                f" ({int(rec['duration_seconds']) // 60}m {int(rec['duration_seconds']) % 60}s)",
                                                style={"color": "#64748b"},
                                            ),
                                        ]
                                    )
                                    for rec in pending_recs[:8]
                                ],
                            ),
                        ],
                    )
                )
        except Exception:
            pass

    pipeline_status_card = html.Div(
        className="sync-status-card sync-pipeline-card",
        children=[
            html.H4("Pipeline Status"),
            html.Div(
                className="status-stats",
                children=[
                    html.Div(
                        [
                            html.Span(str(total), className="big-number"),
                            html.Span("Total", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(completed),
                                className="big-number",
                                style={"color": "#10b981"},
                            ),
                            html.Span("Completed", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(pending),
                                className="big-number",
                                style={"color": "#f59e0b"},
                            ),
                            html.Span("Pending", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(processing),
                                className="big-number",
                                style={"color": "#3b82f6"},
                            ),
                            html.Span("Processing", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(actionable_failed),
                                className="big-number",
                                style={"color": "#ef4444" if actionable_failed else "#94a3b8"},
                            ),
                            html.Span("Failed", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                ],
            ),
            html.Div(
                className="status-stats sync-status-subgrid",
                children=[
                    html.Div(
                        [
                            html.Span(str(stats.total_events), className="big-number"),
                            html.Span("Events in Qdrant", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(str(stats.total_days), className="big-number"),
                            html.Span("Days", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(f"{stats.total_duration_hours:.1f}", className="big-number"),
                            html.Span("Hours Recorded", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                ],
            ),
            *plaud_cloud_children,
            *failed_details_children,
            *pipeline_pending_details_children,
        ],
    )

    plaud_enrichment_card = html.Div(
        className="sync-status-card sync-enrichment-card",
        children=[
            html.H4("☁️ Plaud Cloud Enrichment"),
            html.Div(
                className="status-stats",
                children=[
                    html.Div(
                        [
                            html.Span(
                                str(workflow_stats.get("with_ai_summary", 0)),
                                className="big-number",
                                style={"color": "#10b981"},
                            ),
                            html.Span("AI Summaries", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(workflow_stats.get("ready_for_enrichment", 0)),
                                className="big-number",
                                style={"color": "#f59e0b"},
                            ),
                            html.Span("Cloud Ready", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(workflow_stats.get("workflow_pending", 0)),
                                className="big-number",
                                style={"color": "#3b82f6"},
                            ),
                            html.Span("In Flight", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                    html.Div(
                        [
                            html.Span(
                                str(workflow_stats.get("workflow_failed", 0)),
                                className="big-number",
                                style={"color": "#ef4444"},
                            ),
                            html.Span("Failed", className="stat-label"),
                        ],
                        className="status-stat",
                    ),
                ],
            ),
            html.Div(
                className="auto-sync-details",
                children=[
                    html.Span(
                        f"Recent window: {workflow_stats.get('recent_recordings', 0)} recordings",
                        className="sync-detail-text",
                    ),
                    html.Span(" · ", className="sync-detail-sep"),
                    html.Span(
                        f"Last submit: {workflow_last_run}",
                        className="sync-detail-text",
                    ),
                ],
            ),
        ],
    )

    actions_card = html.Div(
        className="sync-options sync-operations-card",
        children=[
            html.H4("Actions"),
            html.Div(
                className="sync-action-group",
                children=[
                    html.Label("Days to fetch back:"),
                    dcc.Slider(
                        id="sync-days-slider",
                        min=1,
                        max=30,
                        step=1,
                        value=7,
                        marks={1: "1", 7: "7", 14: "14", 30: "30"},
                        className="sync-slider",
                    ),
                    html.Button(
                        id="do-sync-btn",
                        className="sync-action-btn",
                        children=[
                            html.Span("🚀", className="btn-icon"),
                            html.Span(
                                "Smart Sync (Recent Fetch → Process → Index)",
                                className="btn-text",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="sync-action-group",
                style={"marginTop": "15px"},
                children=[
                    html.Label("Enrichment batch size:"),
                    dcc.Slider(
                        id="plaud-workflow-limit",
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={1: "1", 3: "3", 5: "5", 10: "10"},
                        className="sync-slider",
                    ),
                    html.Label("Summary Template:"),
                    dcc.Dropdown(
                        id="plaud-template-select",
                        options=[
                            {"label": "Summary Only (no ETL)", "value": ""},
                            {
                                "label": "📋 General Summary — Key points and action items",
                                "value": "general",
                            },
                            {
                                "label": "📝 Meeting Notes — Attendees, decisions, action items",
                                "value": "meeting",
                            },
                            {
                                "label": "💡 Brainstorm — Ideas grouped by theme",
                                "value": "brainstorm",
                            },
                            {
                                "label": "📅 Daily Log — Timeline of activities",
                                "value": "daily_log",
                            },
                            {
                                "label": "🎤 Interview — Q&A format with key quotes",
                                "value": "interview",
                            },
                        ],
                        value="",
                        clearable=False,
                        className="sync-dropdown",
                        placeholder="Select a template…",
                    ),
                    html.Label("AI Model:"),
                    dcc.Dropdown(
                        id="plaud-model-select",
                        options=[
                            {"label": "OpenAI (GPT)", "value": "openai"},
                            {"label": "Gemini (Google)", "value": "gemini"},
                            {"label": "Claude (Anthropic)", "value": "claude"},
                        ],
                        value="openai",
                        clearable=False,
                        className="sync-dropdown",
                    ),
                    html.Label("Custom ETL template ID (optional):"),
                    dcc.Input(
                        id="plaud-template-id",
                        type="text",
                        placeholder="tpl_your_template_id (overrides dropdown)",
                        className="sync-text-input",
                    ),
                    html.Div(
                        className="sync-button-row",
                        children=[
                            html.Button(
                                id="run-plaud-workflows-btn",
                                className="sync-action-btn",
                                children=[
                                    html.Span("☁️", className="btn-icon"),
                                    html.Span(
                                        "Submit Plaud AI Workflows",
                                        className="btn-text",
                                    ),
                                ],
                            ),
                            html.Button(
                                id="refresh-plaud-workflows-btn",
                                className="sync-action-btn secondary",
                                children=[
                                    html.Span("🔄", className="btn-icon"),
                                    html.Span(
                                        "Refresh Plaud Workflow Status",
                                        className="btn-text",
                                    ),
                                ],
                                disabled=(
                                    workflow_stats.get("workflow_pending", 0) == 0
                                ),
                            ),
                        ],
                    ),
                    html.P(
                        "Targets recent recordings that already finished local processing but do not have an optional Plaud Cloud AI summary yet. Select a template for AI_ETL structured extraction, or leave on 'Summary Only' for just the AI summary.",
                        className="sync-note",
                    ),
                ],
            ),
            html.Div(
                className="sync-action-group",
                style={"marginTop": "15px"},
                children=[
                    html.Button(
                        id="reset-stuck-btn",
                        className="sync-action-btn secondary",
                        children=[
                            html.Span("🔧", className="btn-icon"),
                            html.Span(
                                f"Reset Stuck / Retry ({processing + actionable_failed} recordings)",
                                className="btn-text",
                            ),
                        ],
                        disabled=(processing + actionable_failed == 0),
                    ),
                ],
            ),
            html.Div(id="sync-result", className="sync-result"),
            html.Div(
                id="pipeline-progress-panel",
                className="pipeline-progress-panel",
            ),
        ],
    )

    return html.Div(
        className="sync-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("🔄 Sync & Process", className="view-title"),
                    html.P(
                        "Fetch, process, enrich, and index your Plaud recordings from one operations dashboard.",
                        className="view-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="sync-dashboard-grid",
                children=[
                    html.Div(
                        className="sync-main-column",
                        children=[
                            auth_section,
                            actions_card,
                            *upload_section,
                        ],
                    ),
                    html.Div(
                        className="sync-side-column",
                        children=[
                            runtime_card,
                            pipeline_status_card,
                            plaud_enrichment_card,
                            *auto_sync_children,
                            *workflow_monitor_section,
                        ],
                    ),
                ],
            ),
        ],
    )


def _check_services(settings):
    """Run all connectivity checks, return dict of (ok, detail) tuples."""
    import subprocess
    from datetime import datetime
    from urllib.parse import urlparse

    checks = {}

    # Docker daemon
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            checks["docker"] = (True, f"Docker {result.stdout.strip()}")
        else:
            checks["docker"] = (
                False,
                (result.stderr or result.stdout or "docker info failed").strip()[:80],
            )
    except Exception as e:
        checks["docker"] = (False, str(e)[:80])

    # Plaud auth — attempt refresh if expired so status is accurate
    try:
        from src.plaud_oauth import PlaudOAuthClient

        oauth = PlaudOAuthClient()
        has_token = bool(getattr(oauth, "_access_token", None))
        has_refresh = bool(getattr(oauth, "_refresh_token", None))
        expiry = getattr(oauth, "_token_expiry", None)

        if has_token and expiry:
            minutes = int((expiry - datetime.now()).total_seconds() / 60)
            if minutes > 0:
                checks["plaud"] = (True, f"Token valid for ~{minutes} min")
            elif has_refresh:
                # Token expired but we can refresh — try it now
                try:
                    oauth.refresh_access_token()
                    new_expiry = getattr(oauth, "_token_expiry", None)
                    if new_expiry:
                        mins = int((new_expiry - datetime.now()).total_seconds() / 60)
                        checks["plaud"] = (
                            True,
                            f"Token refreshed — valid for ~{mins} min",
                        )
                    else:
                        checks["plaud"] = (True, "Token refreshed")
                except Exception:
                    checks["plaud"] = (
                        False,
                        "Token expired; refresh failed — use Connect in Sync view",
                    )
            else:
                checks["plaud"] = (
                    False,
                    "Token expired; no refresh token — use Connect in Sync view",
                )
        elif has_token:
            checks["plaud"] = (True, "Token present")
        else:
            checks["plaud"] = (False, "Not authenticated — use Connect in Sync view")
    except Exception as e:
        checks["plaud"] = (False, str(e)[:80])

    # Gemini
    try:
        from src.chronos.genai_helpers import get_genai_client

        if not settings.gemini_api_key:
            checks["gemini"] = (False, "CHRONOS_GEMINI_API_KEY not set")
        else:
            client = get_genai_client()
            available_models = set()
            for model in client.models.list():
                name = getattr(model, "name", None)
                if not name:
                    continue
                if name.startswith("models/"):
                    available_models.add(name.split("/", 1)[1])
                else:
                    available_models.add(name)

            required_models = [
                settings.chronos_cleaning_model,
                settings.chronos_analyst_model,
                settings.chronos_embedding_model,
            ]
            missing_models = [
                model_name
                for model_name in required_models
                if model_name and model_name not in available_models
            ]

            if missing_models:
                checks["gemini"] = (
                    False,
                    "API key valid, but configured models are unavailable: "
                    + ", ".join(missing_models),
                )
            else:
                checks["gemini"] = (
                    True,
                    f"API key valid — {len(available_models)} models visible; configured models available",
                )
    except Exception as e:
        checks["gemini"] = (False, str(e)[:80])

    # OpenAI. Skip active probes unless current routing can use OpenAI; settings
    # polling should not make Responses API calls just to draw a status dot.
    try:
        provider = (settings.chronos_processing_provider or "").strip().lower()
        routed_models = [
            settings.chronos_cleaning_model,
            settings.chronos_analyst_model,
            settings.chronos_embedding_model,
        ]
        openai_routed = provider in {"openai", "auto"} or any(
            str(model or "").strip().lower().startswith(("gpt-", "o", "text-embedding"))
            for model in routed_models
        )
        if not getattr(settings, "chronos_openai_enabled", False):
            checks["openai"] = (True, "Disabled by CHRONOS_OPENAI_ENABLED=0; no quota will be used")
        elif not settings.openai_api_key:
            checks["openai"] = (False, "OPENAI_API_KEY not set")
        elif not openai_routed:
            checks["openai"] = (True, "Configured but not used by current Chronos routing")
        else:
            from src.chronos.openai_service import OpenAIResponseService

            openai_ok, openai_detail = OpenAIResponseService().check_connection(quick=True)
            checks["openai"] = (openai_ok, openai_detail)
    except Exception as e:
        checks["openai"] = (False, str(e)[:80])

    # SQLite
    try:
        from src.database.engine import SessionLocal
        from src.database.models import ChronosRecording, ChronosEvent

        db = SessionLocal()
        try:
            rec_count = db.query(ChronosRecording).count()
            event_count = db.query(ChronosEvent).count()
            checks["sqlite"] = (True, f"{rec_count} recordings, {event_count} events")
        finally:
            db.close()
    except Exception as e:
        checks["sqlite"] = (False, str(e)[:80])

    # Qdrant
    try:
        from qdrant_client import QdrantClient as QC

        qc = QC(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=3)
        info = qc.get_collection(settings.qdrant_collection_name)
        points = getattr(info, "points_count", 0)
        dim = getattr(getattr(info, "config", None), "params", None)
        dim_str = ""
        if dim:
            vec_cfg = getattr(dim, "vectors", None)
            if vec_cfg is not None and hasattr(vec_cfg, "size"):
                dim_str = f" (dim={vec_cfg.size})"
        checks["qdrant"] = (
            True,
            f"{points} points in {settings.qdrant_collection_name}{dim_str}",
        )
    except Exception as e:
        checks["qdrant"] = (False, str(e)[:80])

    # Webhook listener
    try:
        import requests as req

        resp = req.get("http://127.0.0.1:8090/health", timeout=2)
        if resp.ok:
            payload = resp.json()
            checks["webhook_listener"] = (
                True,
                f"Live on :8090 (events: {payload.get('events_received', 0)})",
            )
        else:
            checks["webhook_listener"] = (False, f"Health returned {resp.status_code}")
    except Exception:
        checks["webhook_listener"] = (False, "Not running — start via Auto-Sync task")

    # Notion auth
    try:
        from src.notion_oauth import NotionOAuthClient

        nclient = NotionOAuthClient()
        if nclient.is_authenticated:
            wname = nclient.token_status.get("workspace_name", "")
            checks["notion"] = (True, f"OAuth: {wname}" if wname else "OAuth connected")
        elif nclient.has_credentials:
            checks["notion"] = (False, "Not connected — use Connect below")
        else:
            token_ok = bool(settings.notion_token)
            if token_ok:
                checks["notion"] = (True, "Static token set")
            else:
                checks["notion"] = (False, "No token — set NOTION_CLIENT_ID or NOTION_TOKEN")
    except Exception as e:
        checks["notion"] = (False, str(e)[:80])

    # Webhook config
    wh_ok = bool(settings.plaud_webhook_secret and settings.plaud_webhook_url)
    if wh_ok:
        webhook_url = settings.plaud_webhook_url or ""
        parsed = urlparse(webhook_url)
        host = (parsed.hostname or "").lower()
        is_local = host in {"localhost", "127.0.0.1", "0.0.0.0"}
        if parsed.scheme == "https" and not is_local:
            checks["webhook_config"] = (True, f"Public HTTPS: {webhook_url}")
        elif is_local:
            checks["webhook_config"] = (
                True,
                f"Local ({webhook_url}) — use ngrok for Plaud delivery",
            )
        else:
            checks["webhook_config"] = (True, f"{webhook_url} — needs public HTTPS")
    else:
        checks["webhook_config"] = (
            False,
            "Set PLAUD_WEBHOOK_SECRET + PLAUD_WEBHOOK_URL in .env",
        )

    return checks


def _notion_auth_row(checks) -> html.Div:
    """Build the Notion Auth status row for Settings, with a Connect link when needed."""
    notion_ok, notion_detail = checks.get("notion", (False, "Check unavailable"))
    children = [
        html.Label("Notion Auth:"),
        html.Span(
            "✅ Connected" if notion_ok else "❌ Not Connected",
            className=f"status-badge {'connected' if notion_ok else 'disconnected'}",
        ),
        html.Span(notion_detail, className="status-detail"),
    ]
    if not notion_ok:
        # Only show Connect button if OAuth credentials exist
        try:
            from src.notion_oauth import NotionOAuthClient
            if NotionOAuthClient().has_credentials:
                children.append(
                    html.A(
                        "Connect →",
                        href="/auth/notion",
                        className="plaud-connect-btn",
                        style={
                            "marginLeft": "8px",
                            "fontSize": "0.8rem",
                            "padding": "2px 10px",
                        },
                    )
                )
        except Exception:
            pass
    return html.Div(className="setting-row", children=children)


def create_settings_view(preferences=None) -> html.Div:
    """Create the full settings view: connections, models, parameters, controls."""
    from src.config import get_settings

    settings = get_settings()
    prefs = merge_preferences(preferences)
    checks = _check_services(settings)

    def status_row(label, ok, detail):
        return html.Div(
            className="setting-row",
            children=[
                html.Label(label),
                html.Span(
                    "✅ Connected" if ok else "❌ Not Connected",
                    className=f"status-badge {'connected' if ok else 'disconnected'}",
                ),
                html.Span(detail, className="status-detail"),
            ],
        )

    def param_row(label, value, note=None):
        """Read-only parameter display row."""
        children = [
            html.Label(label, className="param-label"),
            html.Span(str(value), className="param-value"),
        ]
        if note:
            children.append(html.Span(note, className="param-note"))
        return html.Div(className="setting-param-row", children=children)

    # ── Section: Service Connections ──────────────────────────────────────
    # Build Plaud Auth row with inline Connect link when not authenticated
    plaud_ok, plaud_detail = checks["plaud"]
    plaud_row_children = [
        html.Label("Plaud Auth:"),
        html.Span(
            "✅ Connected" if plaud_ok else "❌ Not Connected",
            className=f"status-badge {'connected' if plaud_ok else 'disconnected'}",
        ),
        html.Span(plaud_detail, className="status-detail"),
    ]
    if not plaud_ok:
        plaud_row_children.append(
            html.A(
                "Connect →",
                href="/auth/plaud",
                className="plaud-connect-btn",
                style={
                    "marginLeft": "8px",
                    "fontSize": "0.8rem",
                    "padding": "2px 10px",
                },
            )
        )
    plaud_auth_row = html.Div(className="setting-row", children=plaud_row_children)

    connection_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🔗 Service Connections"),
            status_row("Docker:", *checks["docker"]),
            plaud_auth_row,
            _notion_auth_row(checks),
            status_row("Gemini AI:", *checks["gemini"]),
            status_row("OpenAI:", *checks["openai"]),
            status_row("SQLite:", *checks["sqlite"]),
            status_row("Qdrant:", *checks["qdrant"]),
            status_row("Webhook Listener:", *checks["webhook_listener"]),
            status_row("Webhook Config:", *checks["webhook_config"]),
        ],
    )

    # ── Section: AI Models ───────────────────────────────────────────────
    from src.chronos.cost_tracker import get_pricing

    def _model_label(name: str, note: str = "") -> str:
        """Build a dropdown label with pricing info."""
        p = get_pricing(name)
        if p["tier"] == "free":
            price = "FREE"
        else:
            price = f"${p['input_per_mtok']:.2f}/${p['output_per_mtok']:.2f} per MTok"
        suffix = f" — {note}" if note else ""
        return f"{name} ({price}{suffix})"

    models_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🧠 AI Models"),
            html.P(
                "Model selection for each processing stage. "
                "Change via .env or the controls below. "
                "Prices: input/output per 1M tokens. "
                "Processing Provider controls transcript cleanup/event extraction. "
                "Use local mode to avoid cloud AI calls; OpenAI is inert unless CHRONOS_OPENAI_ENABLED=1.",
                className="setting-note",
            ),
            html.Div(
                className="settings-grid",
                children=[
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Processing Provider"),
                            dcc.Dropdown(
                                id="setting-processing-provider",
                                options=[
                                    {
                                        "label": "gemini (recommended low-cost path)",
                                        "value": "gemini",
                                    },
                                    {
                                        "label": "local (Ollama, no cloud quota)",
                                        "value": "local",
                                    },
                                    {
                                        "label": "openai (paid, requires CHRONOS_OPENAI_ENABLED=1)",
                                        "value": "openai",
                                    },
                                    {
                                        "label": "auto (Gemini → local → explicit OpenAI)",
                                        "value": "auto",
                                    },
                                ],
                                value=settings.chronos_processing_provider,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Controls transcript cleanup and event extraction. Local mode builds searchable transcript chunks without Gemini or OpenAI.",
                                className="param-note",
                            ),
                        ],
                    ),
                    # Cleaning model
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Cleaning Model"),
                            dcc.Dropdown(
                                id="setting-cleaning-model",
                                options=[
                                    {
                                        "label": _model_label("gemini-2.5-flash", "default extraction"),
                                        "value": "gemini-2.5-flash",
                                    },
                                    {
                                        "label": _model_label("gemini-3-flash-preview", "higher ceiling, preview"),
                                        "value": "gemini-3-flash-preview",
                                    },
                                    {
                                        "label": _model_label("gemini-3.1-pro-preview", "deep paid fallback"),
                                        "value": "gemini-3.1-pro-preview",
                                    },
                                    {
                                        "label": _model_label("gpt-5.5", "default extraction"),
                                        "value": "gpt-5.5",
                                    },
                                    {
                                        "label": _model_label("gpt-5.5-pro", "maximum quality"),
                                        "value": "gpt-5.5-pro",
                                    },
                                    {
                                        "label": _model_label("gpt-5.4", "previous flagship"),
                                        "value": "gpt-5.4",
                                    },
                                    {
                                        "label": _model_label("gpt-5.4-mini", "lower cost"),
                                        "value": "gpt-5.4-mini",
                                    },
                                ],
                                value=settings.chronos_cleaning_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Default model for transcript cleanup and event extraction",
                                className="param-note",
                            ),
                        ],
                    ),
                    # Analyst model
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Analyst Model"),
                            dcc.Dropdown(
                                id="setting-analyst-model",
                                options=[
                                    {
                                        "label": _model_label("gemini-2.5-flash", "default analysis"),
                                        "value": "gemini-2.5-flash",
                                    },
                                    {
                                        "label": _model_label("gemini-3.1-pro-preview", "deeper Gemini analysis"),
                                        "value": "gemini-3.1-pro-preview",
                                    },
                                    {
                                        "label": _model_label("gpt-5.5", "default analysis"),
                                        "value": "gpt-5.5",
                                    },
                                    {
                                        "label": _model_label("gpt-5.5-pro", "maximum quality"),
                                        "value": "gpt-5.5-pro",
                                    },
                                    {
                                        "label": _model_label("gpt-5.4", "previous flagship"),
                                        "value": "gpt-5.4",
                                    },
                                    {
                                        "label": _model_label("gpt-5.4-pro", "previous max quality"),
                                        "value": "gpt-5.4-pro",
                                    },
                                ],
                                value=settings.chronos_analyst_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Default model for Ask Chronos and deeper analysis",
                                className="param-note",
                            ),
                        ],
                    ),
                    # Embedding model
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Embedding Model"),
                            dcc.Dropdown(
                                id="setting-embedding-model",
                                options=[
                                    {
                                        "label": "gemini-embedding-2 (default)",
                                        "value": "gemini-embedding-2",
                                    },
                                    {
                                        "label": "gemini-embedding-001 (text-only fallback)",
                                        "value": "gemini-embedding-001",
                                    },
                                    {
                                        "label": "nomic-embed-text (local Ollama, no cloud quota)",
                                        "value": "nomic-embed-text",
                                    },
                                    {
                                        "label": "text-embedding-3-large (paid, requires CHRONOS_OPENAI_ENABLED=1)",
                                        "value": "text-embedding-3-large",
                                    },
                                    {
                                        "label": "text-embedding-3-small (paid, requires CHRONOS_OPENAI_ENABLED=1)",
                                        "value": "text-embedding-3-small",
                                    },
                                ],
                                value=settings.chronos_embedding_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "⚠️ Changing requires full re-index (incompatible spaces)",
                                className="param-note warning",
                            ),
                        ],
                    ),
                    # OpenAI model
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("OpenAI Model"),
                            dcc.Dropdown(
                                id="setting-openai-model",
                                options=[
                                    {
                                        "label": _model_label(
                                            "gpt-5.5", "flagship reasoning"
                                        ),
                                        "value": "gpt-5.5",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-5.5-pro", "maximum quality"
                                        ),
                                        "value": "gpt-5.5-pro",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-5.4", "previous flagship"
                                        ),
                                        "value": "gpt-5.4",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-5.4-pro", "smartest"
                                        ),
                                        "value": "gpt-5.4-pro",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-5.4-mini", "strong mini, 400K ctx"
                                        ),
                                        "value": "gpt-5.4-mini",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-5.4-nano", "fastest/cheapest"
                                        ),
                                        "value": "gpt-5.4-nano",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-5", "previous reasoning"
                                        ),
                                        "value": "gpt-5",
                                    },
                                    {
                                        "label": _model_label(
                                            "gpt-4.1", "non-reasoning legacy"
                                        ),
                                        "value": "gpt-4.1",
                                    },
                                ],
                                value=settings.openai_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Global OpenAI fallback model. Cleaning and Analyst above can pin extraction and Ask Chronos separately.",
                                className="param-note",
                            ),
                        ],
                    ),
                    # Thinking level
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Thinking Level"),
                            dcc.Dropdown(
                                id="setting-thinking-level",
                                options=[
                                    {"label": "Minimal", "value": "minimal"},
                                    {"label": "Low", "value": "low"},
                                    {"label": "Medium", "value": "medium"},
                                    {"label": "High", "value": "high"},
                                ],
                                value=settings.chronos_thinking_level,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Flash: minimal–high | Pro: low/high",
                                className="param-note",
                            ),
                        ],
                    ),
                    # OpenAI temperature
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("OpenAI Temperature"),
                            html.Div(
                                className="setting-control",
                                children=[
                                    dcc.Slider(
                                        id="setting-openai-temp",
                                        min=0,
                                        max=2,
                                        step=0.1,
                                        value=settings.openai_temperature,
                                        marks={
                                            0: "0",
                                            0.5: "0.5",
                                            1: "1.0",
                                            1.5: "1.5",
                                            2: "2.0",
                                        },
                                    ),
                                    html.Span(
                                        f"{settings.openai_temperature}",
                                        id="setting-openai-temp-label",
                                        className="param-note",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ── Section: Embedding Config ────────────────────────────────────────
    embedding_section = html.Div(
        className="settings-section",
        children=[
            html.H4("📐 Embedding Configuration"),
            html.P(
                "Matryoshka Representation Learning (MRL) allows dimensionality "
                "reduction. Lower dims = faster search, slightly less accuracy.",
                className="setting-note",
            ),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Embedding Dimensions"),
                    dcc.Dropdown(
                        id="setting-embedding-dim",
                        options=[
                            {
                                "label": "128  — Fastest, lowest accuracy",
                                "value": "128",
                            },
                            {"label": "256  — Fast", "value": "256"},
                            {"label": "512  — Balanced", "value": "512"},
                            {"label": "768  — Default, good balance ✓", "value": "768"},
                            {"label": "1024 — Higher accuracy", "value": "1024"},
                            {"label": "1536 — OpenAI-compatible", "value": "1536"},
                            {"label": "2048 — High accuracy", "value": "2048"},
                            {
                                "label": "3072 — Maximum (native, no L2 norm)",
                                "value": "3072",
                            },
                        ],
                        value=str(settings.chronos_embedding_dim),
                        clearable=False,
                        className="settings-dropdown",
                    ),
                    html.Span(
                        "⚠️ Changing requires full re-index",
                        className="param-note warning",
                    ),
                ],
            ),
            param_row(
                "Multimodal Support",
                "Text + Audio (WAV/MP3 ≤80s)",
                "gemini-embedding-2",
            ),
            param_row(
                "Task Types",
                "RETRIEVAL_DOCUMENT (index) | RETRIEVAL_QUERY (search) | QUESTION_ANSWERING (RAG)",
            ),
            param_row(
                "L2 Normalization", "Auto-applied when dim < 3072 (MRL requirement)"
            ),
        ],
    )

    # ── Section: Qdrant Vector Store ─────────────────────────────────────
    qdrant_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🔮 Qdrant Vector Store"),
            param_row("URL", settings.qdrant_url),
            param_row("Collection", settings.qdrant_collection_name),
            param_row("Distance Metric", "COSINE"),
            param_row("Timeout", f"{settings.qdrant_timeout_seconds}s"),
            param_row(
                "API Key", "••••" if settings.qdrant_api_key else "Not set (local mode)"
            ),
            html.P(
                "Payload indexes: day_of_week, hour_of_day, timestamp, category, "
                "start_ts_unix, recording_id",
                className="param-note",
                style={"marginTop": "8px"},
            ),
        ],
    )

    # ── Section: Plaud Device & API ──────────────────────────────────────
    # Fetch Plaud user info and webhooks (non-blocking)
    plaud_user_info = None
    plaud_webhooks = []
    try:
        from src.plaud_client import PlaudClient
        from src.plaud_admin import PlaudAdminClient

        plaud = PlaudClient()
        if plaud.oauth.is_authenticated:
            try:
                plaud_user_info = plaud.get_user()
            except Exception:
                pass
            try:
                admin = PlaudAdminClient(plaud)
                plaud_webhooks = admin.list_webhooks()
            except Exception:
                pass
    except Exception:
        pass

    # Build user info display
    plaud_user_children = []
    if plaud_user_info:
        name = plaud_user_info.get("name") or plaud_user_info.get("username") or "—"
        email = plaud_user_info.get("email", "—")
        plaud_user_children = [
            param_row("Plaud User", name),
            param_row("Email", email),
        ]

    # Build webhook list
    webhook_children = []
    if plaud_webhooks:
        webhook_children = [
            html.H5("Registered Webhooks", style={"marginTop": "12px"}),
            html.Div(
                className="webhook-list",
                children=[
                    html.Div(
                        className="webhook-item",
                        children=[
                            html.Span(
                                wh.get("url", "unknown"),
                                className="webhook-url",
                            ),
                            html.Span(
                                (
                                    f" ({', '.join(wh.get('events', []))})"
                                    if wh.get("events")
                                    else ""
                                ),
                                className="webhook-events",
                            ),
                        ],
                    )
                    for wh in plaud_webhooks
                ],
            ),
        ]
    else:
        webhook_children = [
            html.P(
                "No webhooks registered. Configure PLAUD_WEBHOOK_URL in .env to receive real-time sync events.",
                className="param-note",
                style={"marginTop": "8px"},
            ),
        ]

    plaud_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🎙️ Plaud Device & API"),
            *plaud_user_children,
            param_row("API Base URL", settings.plaud_api_base_url),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Default Language"),
                    dcc.Dropdown(
                        id="setting-plaud-language",
                        options=[
                            {"label": "English", "value": "en"},
                            {"label": "Spanish", "value": "es"},
                            {"label": "French", "value": "fr"},
                            {"label": "German", "value": "de"},
                            {"label": "Chinese", "value": "zh"},
                            {"label": "Japanese", "value": "ja"},
                            {"label": "Korean", "value": "ko"},
                        ],
                        value=settings.plaud_default_language,
                        clearable=False,
                        className="settings-dropdown",
                    ),
                ],
            ),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Speaker Diarization"),
                    dcc.Checklist(
                        id="setting-plaud-diarization",
                        options=[
                            {
                                "label": "Enable speaker identification",
                                "value": "enabled",
                            }
                        ],
                        value=["enabled"] if settings.plaud_enable_diarization else [],
                        className="pref-checklist",
                    ),
                ],
            ),
            param_row("Workflow Timeout", f"{settings.plaud_workflow_timeout}s"),
            param_row("Client ID", "••••" if settings.plaud_client_id else "Not set"),
            param_row("Webhook URL", settings.plaud_webhook_url or "Not configured"),
            *webhook_children,
        ],
    )

    notion_import_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🗓️ Notion Import Defaults"),
            html.P(
                "Fallback start times used when a Notion page only provides a recording date.",
                className="setting-note",
            ),
            html.Div(
                className="settings-grid",
                children=[
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Weekday Fallback Start"),
                            dcc.Input(
                                id="setting-notion-weekday-start",
                                type="text",
                                value=settings.notion_weekday_start_time,
                                placeholder="07:30",
                                className="settings-input",
                                debounce=True,
                            ),
                            html.Span(
                                "24-hour HH:MM, used Monday-Friday",
                                className="param-note",
                            ),
                        ],
                    ),
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Weekend Fallback Start"),
                            dcc.Input(
                                id="setting-notion-weekend-start",
                                type="text",
                                value=settings.notion_weekend_start_time,
                                placeholder="12:00",
                                className="settings-input",
                                debounce=True,
                            ),
                            html.Span(
                                "24-hour HH:MM, used Saturday-Sunday",
                                className="param-note",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # ── Section: Data & Storage ──────────────────────────────────────────
    storage_section = html.Div(
        className="settings-section collapsible",
        children=[
            html.H4("📁 Data & Storage"),
            param_row("Database", settings.database_url.replace("sqlite:///", "")),
            param_row("Raw Audio Dir", settings.chronos_raw_audio_dir),
            param_row("Processed Dir", settings.chronos_processed_dir),
            param_row("Graph Cache", settings.chronos_graph_cache_dir),
        ],
    )

    # ── Section: Logging ─────────────────────────────────────────────────
    logging_section = html.Div(
        className="settings-section",
        children=[
            html.H4("📝 Logging"),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Log Level"),
                    dcc.Dropdown(
                        id="setting-log-level",
                        options=[
                            {"label": "DEBUG — Everything", "value": "DEBUG"},
                            {"label": "INFO — Normal", "value": "INFO"},
                            {"label": "WARNING — Issues only", "value": "WARNING"},
                            {"label": "ERROR — Errors only", "value": "ERROR"},
                        ],
                        value=settings.log_level,
                        clearable=False,
                        className="settings-dropdown",
                    ),
                ],
            ),
            param_row(
                "Verbose Mode",
                "On" if settings.verbose else "Off",
                "Set PB_VERBOSE=1 to enable",
            ),
            param_row("Gemini API Version", settings.gemini_api_version),
        ],
    )

    # ── Section: Categories ─────────────────────────────────────────────
    from app_v2.components import CATEGORIES, CATEGORY_COLORS, CATEGORY_LABELS

    builtin_cat_rows = []
    for cat in CATEGORIES:
        color = CATEGORY_COLORS.get(cat, "#374151")
        label = CATEGORY_LABELS.get(cat, cat)
        builtin_cat_rows.append(
            html.Div(
                className="category-def-row",
                children=[
                    html.Span(
                        className="category-color-swatch",
                        style={"backgroundColor": color},
                    ),
                    html.Span(label, className="category-def-label"),
                    html.Span(cat, className="category-def-key"),
                ],
            )
        )

    # Load custom categories from env
    custom_cats_raw = os.environ.get("CHRONOS_CUSTOM_CATEGORIES", "")

    categories_section = html.Div(
        className="settings-section",
        children=[
            html.H4("📂 Categories"),
            html.P(
                "Built-in event categories used by the Gemini processor.",
                className="setting-note",
            ),
            html.Div(
                className="category-def-grid",
                children=builtin_cat_rows,
            ),
            html.Hr(className="settings-divider"),
            html.H5("Custom Categories", className="settings-subsection-title"),
            html.P(
                "Define custom categories as comma-separated values (e.g. exercise, commute, learning). "
                "These will be available in the category dropdown. Save to .env to persist.",
                className="setting-note",
            ),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Custom Categories"),
                    dcc.Input(
                        id="custom-categories-input",
                        type="text",
                        value=custom_cats_raw,
                        placeholder="exercise, commute, learning, social",
                        className="settings-input wide",
                        debounce=True,
                    ),
                ],
            ),
        ],
    )

    # ── Section: UI Preferences ──────────────────────────────────────────
    ui_prefs_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🎛️ UI Preferences"),
            html.P(
                "Saved in your browser (localStorage).",
                className="setting-note",
            ),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Auto-refresh"),
                    dcc.Checklist(
                        id="pref-auto-refresh-enabled",
                        options=[
                            {"label": "Enable background refresh", "value": "enabled"}
                        ],
                        value=["enabled"] if prefs["auto_refresh_enabled"] else [],
                        className="pref-checklist",
                    ),
                ],
            ),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Refresh interval"),
                    html.Div(
                        className="setting-control",
                        children=[
                            dcc.Slider(
                                id="pref-auto-refresh-seconds",
                                min=15,
                                max=300,
                                step=15,
                                value=prefs["auto_refresh_seconds"],
                                marks={15: "15s", 60: "60s", 120: "2m", 300: "5m"},
                            ),
                            html.Span(
                                f"{prefs['auto_refresh_seconds']} seconds",
                                id="pref-refresh-seconds-label",
                                className="setting-note",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="setting-control-row",
                children=[
                    html.Label("Default landing view"),
                    dcc.Dropdown(
                        id="pref-default-view",
                        options=[
                            {"label": "Timeline", "value": "timeline"},
                            {"label": "Topics", "value": "topics"},
                            {"label": "Graph", "value": "graph"},
                            {"label": "Stats", "value": "stats"},
                            {"label": "Sync", "value": "sync"},
                            {"label": "Settings", "value": "settings"},
                        ],
                        value=prefs["default_view"],
                        clearable=False,
                        searchable=False,
                        className="pref-dropdown",
                    ),
                ],
            ),
            html.Div(
                className="settings-action-buttons",
                children=[
                    html.Button(
                        "Save Preferences",
                        id="save-preferences-btn",
                        className="settings-action-btn",
                    ),
                    html.Button(
                        "Reset to Defaults",
                        id="reset-preferences-btn",
                        className="settings-action-btn secondary",
                    ),
                ],
            ),
            html.Div(
                id="preferences-save-status",
                className="setting-note",
                children="Adjust and save to apply settings.",
            ),
        ],
    )

    # ── Section: Save Settings to .env ───────────────────────────────────
    save_section = html.Div(
        className="settings-section settings-save-section",
        children=[
            html.H4("💾 Apply Configuration"),
            html.P(
                "Write model/parameter changes back to .env. Takes effect on next app restart.",
                className="setting-note",
            ),
            html.Div(
                className="settings-action-buttons",
                children=[
                    html.Button(
                        "Save to .env",
                        id="save-env-btn",
                        className="settings-action-btn primary-action",
                    ),
                ],
            ),
            html.Div(id="env-save-status", className="setting-note", children=""),
        ],
    )

    # ── Section: Stack Control ───────────────────────────────────────────
    stack_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🛠️ Stack Control"),
            html.P(
                "Start/stop/analyze services from the dashboard.",
                className="setting-note",
            ),
            html.Div(
                className="settings-action-buttons",
                children=[
                    html.Button(
                        "Status", id="ctl-status-btn", className="settings-action-btn"
                    ),
                    html.Button(
                        "Analyze", id="ctl-analyze-btn", className="settings-action-btn"
                    ),
                    html.Button(
                        "Start", id="ctl-start-btn", className="settings-action-btn"
                    ),
                    html.Button(
                        "Start Public",
                        id="ctl-start-public-btn",
                        className="settings-action-btn",
                    ),
                    html.Button(
                        "Stop",
                        id="ctl-stop-btn",
                        className="settings-action-btn secondary",
                    ),
                    html.Button(
                        "Restart Public",
                        id="ctl-restart-btn",
                        className="settings-action-btn secondary",
                    ),
                ],
            ),
            html.Pre(
                "Click Status or Analyze to inspect the running stack.",
                id="ctl-output",
                className="ctl-output",
            ),
        ],
    )

    # ── Section: About ───────────────────────────────────────────────────
    about_section = html.Div(
        className="settings-section",
        children=[
            html.H4("ℹ️ About"),
            html.P("Chronos v2.0 — Recording Lifecycle Intelligence"),
            html.P(
                "Transform your Plaud voice recordings into searchable knowledge.",
                className="about-desc",
            ),
            html.Div(
                className="about-stack",
                children=[
                    html.Span("Gemini Embedding 2", className="tech-badge"),
                    html.Span("Qdrant", className="tech-badge"),
                    html.Span("OpenAI Responses", className="tech-badge"),
                    html.Span("Dash", className="tech-badge"),
                    html.Span("Plaud API", className="tech-badge"),
                    html.Span("FastMCP", className="tech-badge"),
                ],
            ),
        ],
    )

    return html.Div(
        className="settings-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("⚙️ Settings", className="view-title"),
                    html.P(
                        "System configuration, models, and service status",
                        className="view-subtitle",
                    ),
                ],
            ),
            connection_section,
            models_section,
            embedding_section,
            qdrant_section,
            plaud_section,
            notion_import_section,
            storage_section,
            logging_section,
            categories_section,
            ui_prefs_section,
            save_section,
            stack_section,
            about_section,
        ],
    )


def register_navigation_callbacks(app):
    """Register navigation-related callbacks."""

    @app.callback(
        Output("pref-refresh-seconds-label", "children"),
        Input("pref-auto-refresh-seconds", "value"),
        prevent_initial_call=True,
    )
    def update_refresh_seconds_label(seconds):
        if seconds is None:
            raise PreventUpdate
        return f"{int(seconds)} seconds"

    @app.callback(
        Output("app-preferences", "data"),
        Output("preferences-save-status", "children"),
        Input("save-preferences-btn", "n_clicks"),
        Input("reset-preferences-btn", "n_clicks"),
        State("pref-auto-refresh-enabled", "value"),
        State("pref-auto-refresh-seconds", "value"),
        State("pref-default-view", "value"),
        prevent_initial_call=True,
    )
    def persist_preferences(
        save_clicks,
        reset_clicks,
        auto_refresh_value,
        auto_refresh_seconds,
        default_view,
    ):
        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        if triggered == "reset-preferences-btn":
            return dict(DEFAULT_PREFERENCES), "Preferences reset to defaults."

        if triggered == "save-preferences-btn":
            updated = merge_preferences(
                {
                    "auto_refresh_enabled": bool(auto_refresh_value),
                    "auto_refresh_seconds": auto_refresh_seconds,
                    "default_view": default_view,
                }
            )
            return updated, "Preferences saved."

        raise PreventUpdate

    @app.callback(
        Output("auto-refresh", "interval"),
        Output("auto-refresh", "disabled"),
        Input("app-preferences", "data"),
        prevent_initial_call=False,
    )
    def apply_refresh_preferences(preferences):
        prefs = merge_preferences(preferences)
        return prefs["auto_refresh_seconds"] * 1000, (not prefs["auto_refresh_enabled"])

    @app.callback(
        Output("ctl-output", "children"),
        Input("ctl-status-btn", "n_clicks"),
        Input("ctl-analyze-btn", "n_clicks"),
        Input("ctl-start-btn", "n_clicks"),
        Input("ctl-start-public-btn", "n_clicks"),
        Input("ctl-stop-btn", "n_clicks"),
        Input("ctl-restart-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_stack_control(
        status_clicks,
        analyze_clicks,
        start_clicks,
        start_public_clicks,
        stop_clicks,
        restart_clicks,
    ):
        """Run stack control commands from settings and show command output."""
        import subprocess
        import sys
        from datetime import datetime
        from pathlib import Path

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        action_map = {
            "ctl-status-btn": ["status"],
            "ctl-analyze-btn": ["analyze"],
            "ctl-start-btn": ["start"],
            "ctl-start-public-btn": ["start", "--public"],
            "ctl-stop-btn": ["stop"],
            "ctl-restart-btn": ["restart", "--public"],
        }

        args = action_map.get(triggered)
        if not args:
            raise PreventUpdate

        root = Path(__file__).resolve().parents[2]
        cmd = [sys.executable, "scripts/chronos_ctl.py", *args]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            output = (result.stdout or "") + (
                "\n" + result.stderr if result.stderr else ""
            )
            output = output.strip() or "No output"
            header = (
                f"$ {' '.join(cmd)}\n"
                f"[{datetime.now().strftime('%H:%M:%S')}] exit={result.returncode}"
            )
            return f"{header}\n\n{output}"
        except Exception as e:
            return f"Stack control failed: {e}"

    @app.callback(
        Output("setting-openai-temp-label", "children"),
        Input("setting-openai-temp", "value"),
        prevent_initial_call=True,
    )
    def update_openai_temp_label(value):
        if value is None:
            raise PreventUpdate
        return f"{value:.1f}"

    @app.callback(
        Output("env-save-status", "children"),
        Input("save-env-btn", "n_clicks"),
        State("setting-processing-provider", "value"),
        State("setting-cleaning-model", "value"),
        State("setting-analyst-model", "value"),
        State("setting-embedding-model", "value"),
        State("setting-openai-model", "value"),
        State("setting-thinking-level", "value"),
        State("setting-openai-temp", "value"),
        State("setting-embedding-dim", "value"),
        State("setting-plaud-language", "value"),
        State("setting-plaud-diarization", "value"),
        State("setting-log-level", "value"),
        State("custom-categories-input", "value"),
        State("setting-notion-weekday-start", "value"),
        State("setting-notion-weekend-start", "value"),
        prevent_initial_call=True,
    )
    def save_env_settings(
        n_clicks,
        processing_provider,
        cleaning_model,
        analyst_model,
        embedding_model,
        openai_model,
        thinking_level,
        openai_temp,
        embedding_dim,
        plaud_language,
        plaud_diarization,
        log_level,
        custom_categories,
        notion_weekday_start,
        notion_weekend_start,
    ):
        """Write changed settings back to .env file."""
        if not n_clicks:
            raise PreventUpdate

        def _valid_hhmm(value: str) -> bool:
            try:
                hour_text, minute_text = str(value).strip().split(":", 1)
                hour = int(hour_text)
                minute = int(minute_text)
                return 0 <= hour <= 23 and 0 <= minute <= 59
            except (AttributeError, TypeError, ValueError):
                return False

        if not _valid_hhmm(notion_weekday_start or ""):
            return "❌ Weekday fallback start must be HH:MM in 24-hour time"
        weekday_hour, weekday_minute = [
            int(part) for part in str(notion_weekday_start).strip().split(":", 1)
        ]
        if (weekday_hour, weekday_minute) > (8, 0):
            return "❌ Weekday fallback start must be 08:00 or earlier"
        if not _valid_hhmm(notion_weekend_start or ""):
            return "❌ Weekend fallback start must be HH:MM in 24-hour time"

        from pathlib import Path

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            return "❌ .env file not found"

        # Map of env var name → new value
        updates = {
            "CHRONOS_PROCESSING_PROVIDER": processing_provider,
            "CHRONOS_CLEANING_MODEL": cleaning_model,
            "CHRONOS_ANALYST_MODEL": analyst_model,
            "CHRONOS_EMBEDDING_MODEL": embedding_model,
            "OPENAI_MODEL": openai_model,
            "CHRONOS_THINKING_LEVEL": thinking_level,
            "OPENAI_TEMPERATURE": str(openai_temp) if openai_temp is not None else None,
            "CHRONOS_EMBEDDING_DIM": embedding_dim,
            "PLAUD_DEFAULT_LANGUAGE": plaud_language,
            "PLAUD_ENABLE_DIARIZATION": (
                "1" if plaud_diarization and "enabled" in plaud_diarization else "0"
            ),
            "PB_LOG_LEVEL": log_level,
            "CHRONOS_CUSTOM_CATEGORIES": custom_categories or "",
            "NOTION_WEEKDAY_START_TIME": notion_weekday_start or "07:30",
            "NOTION_WEEKEND_START_TIME": notion_weekend_start or "12:00",
        }

        try:
            lines = env_path.read_text().splitlines()
            existing_keys = {}
            new_lines = []
            changed = 0

            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, _, old_val = stripped.partition("=")
                    key = key.strip()
                    old_val = old_val.strip()
                    if key in updates and updates[key] is not None:
                        new_val = str(updates[key])
                        new_lines.append(f"{key}={new_val}")
                        existing_keys[key] = True
                        if old_val != new_val:
                            changed += 1
                        continue
                new_lines.append(line)

            # Append any new keys not already in .env
            for key, value in updates.items():
                if key not in existing_keys and value is not None:
                    new_lines.append(f"{key}={value}")
                    changed += 1

            env_path.write_text("\n".join(new_lines) + "\n")

            if changed == 0:
                return "✅ Settings unchanged — nothing to save"
            return f"✅ Saved {changed} changed setting{'s' if changed != 1 else ''} to .env — restart app to apply"

        except Exception as e:
            return f"❌ Failed to save: {e}"

    @app.callback(
        Output("content-container", "children"),
        Output("current-view", "data"),
        Output("detail-panel", "children"),
        Output("detail-panel", "className"),
        Input({"type": "nav-item", "view": ALL}, "n_clicks"),
        Input("shell-open-notion", "n_clicks"),
        Input("shell-open-sync", "n_clicks"),
        Input("shell-open-system", "n_clicks"),
        Input("selected-recording", "data"),
        Input("selected-topic", "data"),
        Input("search-query", "data"),
        Input("auto-refresh", "n_intervals"),
        State("current-view", "data"),
        State("app-preferences", "data"),
        prevent_initial_call=False,
    )
    def update_main_content(
        nav_clicks,
        shell_notion_clicks,
        shell_sync_clicks,
        shell_system_clicks,
        selected_recording,
        selected_topic,
        search_query,
        n_intervals,
        current_view,
        preferences,
    ):
        """Update main content based on navigation and state."""
        import time as _time
        from app_v2.services.xray import xray_log, xray_timer

        _t0 = _time.perf_counter()
        triggered = ctx.triggered_id
        if triggered == "auto-refresh":
            return no_update, no_update, no_update, no_update
        service = None

        def get_service():
            nonlocal service
            if service is None:
                service = get_data_service()
            return service

        logger.info(f"Navigation callback triggered by: {triggered}")
        logger.info(f"selected_recording: {selected_recording}")

        # Determine what triggered the callback
        prefs = merge_preferences(preferences)
        view = current_view or prefs["default_view"]
        if (
            triggered is None
            and current_view == "timeline"
            and prefs["default_view"] != "timeline"
        ):
            view = prefs["default_view"]

        if triggered == "shell-open-notion":
            view = "notion"
            xray_log("nav", "switch", "Opening Notion from the workspace pulse")
        elif triggered == "shell-open-sync":
            view = "sync"
            xray_log("nav", "switch", "Opening Sync from the runtime strip")
        elif triggered == "shell-open-system":
            view = "system"
            xray_log("nav", "switch", "Opening System from the runtime strip")
        elif isinstance(triggered, dict) and triggered.get("type") == "nav-item":
            view = triggered.get("view", "timeline")
            xray_log("nav", "switch", f"You tapped {view}")

        notion_snapshot = _get_cached_notion_bridge_snapshot()
        notion_index = _build_notion_recording_index(notion_snapshot)

        def load_timeline_days():
            days = get_service().get_days()
            return _annotate_days_with_bridge_state(days, notion_snapshot, notion_index)

        # Handle search query
        if search_query and triggered == "search-query":
            results = get_service().search(search_query)
            return (
                create_search_results(results, search_query),
                "search",
                [],
                "detail-panel",
            )

        # Handle topic selection
        if selected_topic and triggered == "selected-topic":
            timeline = get_service().get_topic_timeline(selected_topic)
            xray_log("nav", "topic", f"Showing everything about '{selected_topic}'")
            return (
                create_topic_timeline_view(timeline),
                "topic-detail",
                [],
                "detail-panel",
            )

        # Handle recording selection
        detail_content = []
        detail_class = "detail-panel"

        if selected_recording:
            from app_v2.components import create_recording_detail

            rec_id = selected_recording.get("id")
            xray_log("nav", "detail", f"Opening a recording")
            with xray_timer("nav", "fetch", "Grabbing that recording"):
                detail = get_service().get_recording_detail(rec_id)
            if detail:
                logger.info(f"Got detail with {len(detail.events)} events")
                xray_log("nav", "detail", f"This recording has {len(detail.events)} moments in it")
                _annotate_recording_summary_with_bridge_state(
                    detail.summary, notion_snapshot, notion_index
                )
                transcript = get_service().get_transcript(rec_id)
                ai_summary = get_service().get_ai_summary(rec_id)
                extracted_data = get_service().get_extracted_data(rec_id)
                workflow_status = get_service().get_workflow_status_for_recording(
                    rec_id
                )
                plaud_transcript = get_service().get_plaud_workflow_transcript(rec_id)
                detail_content = create_recording_detail(
                    detail,
                    selected_recording.get("date", ""),
                    transcript=transcript,
                    highlight_event_id=selected_recording.get("scroll_to_event"),
                    ai_summary=ai_summary,
                    extracted_data=extracted_data,
                    workflow_status=workflow_status,
                    plaud_transcript=plaud_transcript,
                )
                detail_class = "detail-panel open"
            else:
                # No Qdrant events found — check if this is a pending recording
                try:
                    from src.database.engine import SessionLocal as _SL2
                    from src.database.models import ChronosRecording as _CR2

                    _db2 = _SL2()
                    try:
                        _pending_rec = _db2.query(_CR2).filter_by(recording_id=rec_id).first()
                    finally:
                        _db2.close()
                    if _pending_rec and str(_pending_rec.processing_status) in ("pending", "processing"):
                        _status = str(_pending_rec.processing_status)
                        _dur_s = int(_pending_rec.duration_seconds or 0)
                        _dur_text = f"{_dur_s // 60}:{_dur_s % 60:02d}"
                        _status_label = "\u23f3 Waiting to process" if _status == "pending" else "\ud83d\udd04 Processing\u2026"
                        detail_content = html.Div(
                            className="recording-detail pending-detail",
                            children=[
                                html.Div(
                                    className="detail-header",
                                    children=[
                                        html.H3(
                                            str(_pending_rec.title or "Untitled Recording"),
                                            className="detail-title",
                                        ),
                                        html.Span(
                                            _status_label,
                                            className="detail-status-badge",
                                            style={"color": "#f59e0b", "fontWeight": "600", "fontSize": "0.9rem"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="detail-meta",
                                    style={"padding": "16px", "color": "#94a3b8"},
                                    children=[
                                        html.P(f"Duration: {_dur_text}"),
                                        html.P(
                                            "This recording hasn't been analyzed yet. "
                                            "Run Full Sync (Sync view) to extract events and make it searchable.",
                                        ),
                                        html.P(
                                            "If you've already run Full Sync and this recording is still pending, "
                                            "Plaud may not have a transcript available for it yet.",
                                            style={"fontSize": "0.85rem", "marginTop": "8px"},
                                        ),
                                    ],
                                ),
                            ],
                        )
                        detail_class = "detail-panel open"
                    else:
                        logger.warning("No detail returned!")
                        xray_log("nav", "detail", "Hmm, couldn't find that recording", level="warn")
                except Exception as _de:
                    logger.warning(f"Could not look up pending recording: {_de}")
                    xray_log("nav", "detail", "Hmm, couldn't find that recording", level="warn")

        # Render main content based on view
        with xray_timer("nav", "render", f"Drawing the {view} screen"):
            if view == "timeline":
                days = load_timeline_days()
                content = create_day_view(days)
                xray_log("nav", "data", f"Your timeline covers {len(days)} days of recordings")
            elif view == "days":
                days = load_timeline_days()
                content = create_day_view(days)
                view = "timeline"
            elif view == "search":
                if search_query:
                    results = get_service().search(search_query)
                    content = create_search_results(results, search_query)
                else:
                    days = load_timeline_days()
                    content = create_day_view(days)
                    view = "timeline"
            elif view == "topic-detail":
                if selected_topic:
                    timeline = get_service().get_topic_timeline(selected_topic)
                    content = create_topic_timeline_view(timeline)
                else:
                    topics = get_service().get_all_topics()
                    content = create_topics_grid(topics)
                    view = "topics"
            elif view == "topics":
                topics = get_service().get_all_topics()
                content = create_topics_grid(topics)
            elif view == "graph":
                graph_data = get_service().get_graph_data()
                content = create_graph_view(graph_data)
                xray_log(
                    "nav",
                    "data",
                    f"Your knowledge map has {len(graph_data.nodes)} ideas connected by {len(graph_data.edges)} links",
                )
            elif view == "stats":
                stats = get_service().get_stats()
                content = create_stats_view(stats)
            elif view == "system":
                content = create_system_view(get_service())
            elif view == "sync":
                content = create_sync_view(get_service())
            elif view == "notion":
                from app_v2.callbacks.notion import (
                    _do_full_fetch_data,
                    _get_cached_notion_data,
                )
                from app_v2.components.notion import create_notion_view
                from src.config import get_settings

                databases = []
                settings = get_settings()
                has_token = bool(settings.notion_token)
                has_db = bool(settings.notion_database_id)

                if has_token and not has_db:
                    try:
                        from src.notion_service import get_notion_service
                        svc = get_notion_service()
                        databases = svc.list_databases()
                        xray_log("nav", "data", f"Notion: found {len(databases)} data sources")

                        # Auto-select the one with the most properties
                        if databases:
                            best = max(databases, key=lambda d: d.get("property_count", 0))
                            svc.set_database_id(best["id"])
                            settings.notion_database_id = best["id"]
                            has_db = True
                            xray_log("nav", "data",
                                     f"Auto-selected '{best['title']}' ({best['property_count']} properties)")
                    except Exception as e:
                        xray_log("nav", "data", f"Notion: could not list data sources: {e}")

                if has_db:
                    # Serve from cache if available (instant tab switch)
                    cached = _get_cached_notion_data()
                    if cached is not None:
                        xray_log("nav", "data", "Notion tab loaded from cache")
                        content = create_notion_view(**cached)
                    else:
                        # Cold load — show skeleton, auto-fetch populates data
                        xray_log("nav", "data", "Loading Notion data for dashboard")
                        try:
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=1
                            ) as pool:
                                future = pool.submit(_do_full_fetch_data)
                                notion_data = future.result(timeout=45)
                            content = create_notion_view(**notion_data)
                        except concurrent.futures.TimeoutError:
                            logger.warning("Notion pre-fetch timed out after 45s")
                            xray_log(
                                "nav",
                                "data",
                                "Notion API is slow — showing empty view, hit Refresh to retry",
                                level="warn",
                            )
                            content = create_notion_view(databases=databases)
                        except Exception as e:
                            logger.warning(f"Notion pre-fetch failed: {e}")
                            xray_log(
                                "nav", "data", f"Notion fetch error: {e}", level="warn"
                            )
                            content = create_notion_view(databases=databases)
                else:
                    content = create_notion_view(databases=databases)
            elif view == "settings":
                content = create_settings_view(preferences=prefs)
            else:
                days = load_timeline_days()
                content = create_day_view(days)

        _total_ms = (_time.perf_counter() - _t0) * 1000
        xray_log(
            "nav",
            "total",
            f"Done! Showing {view}",
            duration_ms=round(_total_ms, 1),
        )

        return content, view, detail_content, detail_class

    @app.callback(
        Output({"type": "nav-item", "view": ALL}, "className"),
        Input("current-view", "data"),
        State({"type": "nav-item", "view": ALL}, "id"),
        prevent_initial_call=True)
    def update_nav_active(current_view, nav_ids):
        """Update active state of navigation items."""
        if not nav_ids:
            raise PreventUpdate

        classes = []
        for nav_id in nav_ids:
            view = nav_id.get("view")
            base_class = "nav-item"
            # Add sync-btn class for sync button
            if view == "sync":
                base_class = "nav-item sync-btn"

            if view == current_view:
                classes.append(f"{base_class} active")
            else:
                classes.append(base_class)

        return classes

    @app.callback(
        Output("app-runtime-status", "children"),
        Output("app-runtime-access-link", "children"),
        Output("app-runtime-access-link", "href"),
        Output("app-runtime-access-link", "style"),
        Output("app-workspace-pulse", "children"),
        Input("current-view", "data"),
        Input("auto-refresh", "n_intervals"),
        Input("active-workflows-count", "data"),
        prevent_initial_call=False,
    )
    def update_runtime_strip(current_view, _refresh_count, active_workflows_count):
        """Keep the app shell aware of runtime health and preferred access."""
        runtime_status = _get_local_runtime_status()
        access_summary = runtime_status.get("access", {})
        notion_status = runtime_status.get("notion", {})
        active_count = max(int(active_workflows_count or 0), 0)

        ui_ok = any(
            port.get("label") == "UI" and port.get("ok")
            for port in runtime_status.get("ports", [])
        )
        api_ok = any(
            port.get("label") == "API" and port.get("ok")
            for port in runtime_status.get("ports", [])
        )

        pretty_view = str(current_view or "timeline").replace("-", " ").title()
        preferred_kind = access_summary.get("preferred_kind", "local")
        preferred_label = access_summary.get("preferred_label", "Localhost")

        pills = [
            html.Span(f"View: {pretty_view}", className="app-runtime-pill subtle"),
            html.Span(
                f"Access: {preferred_label}",
                className=(
                    "app-runtime-pill info"
                    if preferred_kind == "tailscale"
                    else "app-runtime-pill subtle"
                ),
            ),
            html.Span(
                f"Plaud: {runtime_status['plaud_label']}",
                className=(
                    "app-runtime-pill ok"
                    if runtime_status.get("plaud_ok")
                    else "app-runtime-pill warn"
                ),
            ),
            html.Span(
                f"Sync: {runtime_status['auto_sync_label']}",
                className=(
                    "app-runtime-pill ok"
                    if runtime_status.get("auto_sync_ok")
                    else "app-runtime-pill warn"
                ),
            ),
            html.Span(
                f"Activity: {runtime_status.get('activity', {}).get('summary', 'Idle')}",
                className=(
                    "app-runtime-pill info"
                    if runtime_status.get("activity", {}).get("active")
                    else "app-runtime-pill subtle"
                ),
            ),
            html.Span(
                f"Notion: {notion_status.get('label', 'Setup')}",
                className=(
                    "app-runtime-pill ok"
                    if notion_status.get("connected")
                    else (
                        "app-runtime-pill warn"
                        if notion_status.get("has_credentials")
                        else "app-runtime-pill subtle"
                    )
                ),
            ),
            html.Span(
                f"API: {'Live' if api_ok else 'Offline'}",
                className=("app-runtime-pill ok" if api_ok else "app-runtime-pill warn"),
            ),
            html.Span(
                (
                    f"Workflows: {active_count} active"
                    if active_count
                    else "Workflows: idle"
                ),
                className=(
                    "app-runtime-pill info" if active_count else "app-runtime-pill subtle"
                ),
            ),
        ]

        preferred_ui_url = access_summary.get("preferred_ui_url", "#")
        if preferred_kind == "tailscale":
            link_label = "Open via Tailscale"
        elif preferred_kind == "lan":
            link_label = "Open on LAN"
        else:
            link_label = "Open locally"

        link_style = {"display": "inline-flex"} if ui_ok else {"display": "none"}
        pulse = _build_workspace_pulse(runtime_status, current_view, active_count)

        return pills, link_label, preferred_ui_url, link_style, pulse

    @app.callback(
        Output("pipeline-progress-poll", "disabled"),
        Input("current-view", "data"),
        prevent_initial_call=False,
    )
    def toggle_pipeline_progress_poll(current_view):
        """Only poll pipeline progress while the Sync view is active."""
        return current_view != "sync"

    @app.callback(
        Output("sync-result", "children"),
        Input("do-sync-btn", "n_clicks"),
        Input("reset-stuck-btn", "n_clicks"),
        Input("run-plaud-workflows-btn", "n_clicks"),
        Input("refresh-plaud-workflows-btn", "n_clicks"),
        Input("upload-files-btn", "n_clicks"),
        State("sync-days-slider", "value"),
        State("plaud-workflow-limit", "value"),
        State("plaud-template-id", "value"),
        State("plaud-template-select", "value"),
        State("plaud-model-select", "value"),
        prevent_initial_call=True,
    )
    def perform_sync(
        sync_clicks,
        reset_clicks,
        run_plaud_clicks,
        refresh_plaud_clicks,
        upload_clicks,
        days_back,
        workflow_limit,
        template_id,
        template_select,
        model_select,
    ):
        """Perform full pipeline sync or reset stuck recordings."""
        triggered = ctx.triggered_id

        def render_detail_list(items, formatter):
            if not items:
                return []
            return [html.Ul([html.Li(formatter(item)) for item in items[:5]])]

        if triggered == "reset-stuck-btn" and reset_clicks:
            try:
                service = get_data_service()
                count = service.reset_stuck_recordings()
                service.refresh_cache()
                return html.Div(
                    className="sync-success",
                    children=[
                        html.Span("🔧 Reset Complete!", className="success-icon"),
                        html.P(
                            f"Reset {count} stuck/actionable recordings to pending."
                        ),
                        html.P("Run Full Sync to process them.", className="sync-note"),
                    ],
                )
            except Exception as e:
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Reset Failed", className="error-icon"),
                        html.P(str(e)),
                    ],
                )

        if triggered == "run-plaud-workflows-btn" and run_plaud_clicks:
            service = get_data_service()
            # Custom template ID overrides dropdown selection
            effective_template = template_id or template_select or None
            effective_model = model_select or "gemini"
            result = service.submit_plaud_workflows(
                days_back=days_back or 7,
                limit=workflow_limit or 3,
                template_id=effective_template,
                model=effective_model,
            )

            submitted = result.get("submitted", [])
            errors = result.get("errors", [])
            skipped = result.get("skipped", [])
            template_text = result.get("template_id") or "summary-only"
            model_text = effective_model

            if errors and not submitted:
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Plaud Submission Failed", className="error-icon"),
                        html.P(errors[0].get("error", "Unknown error")),
                    ],
                )

            return html.Div(
                className="sync-success",
                children=[
                    html.Span("☁️ Plaud Workflows Submitted", className="success-icon"),
                    html.P(
                        f"Submitted {len(submitted)} workflow(s) — template: {template_text}, model: {model_text}."
                    ),
                    html.P(
                        f"Skipped {len(skipped)} recording(s) already summarized or in flight.",
                        className="sync-note",
                    ),
                ]
                + render_detail_list(
                    submitted,
                    lambda item: f"{item.get('title', item.get('recording_id', '')[:16])} → {item.get('workflow_id')}",
                )
                + render_detail_list(
                    errors,
                    lambda item: f"{item.get('recording_id') or 'global'}: {item.get('error')}",
                ),
            )

        if triggered == "refresh-plaud-workflows-btn" and refresh_plaud_clicks:
            service = get_data_service()
            result = service.refresh_plaud_workflow_statuses(
                days_back=max(days_back or 7, 1),
                limit=workflow_limit or 3,
            )
            completed = result.get("completed", [])
            pending = result.get("pending", [])
            failed = result.get("failed", [])

            if failed and not completed and not pending:
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Plaud Refresh Failed", className="error-icon"),
                        html.P(failed[0].get("error", "Unknown error")),
                    ],
                )

            return html.Div(
                className="sync-success",
                children=[
                    html.Span("🔄 Plaud Status Refreshed", className="success-icon"),
                    html.P(
                        f"Completed {len(completed)}, still running {len(pending)}, failed {len(failed)}."
                    ),
                ]
                + render_detail_list(
                    completed,
                    lambda item: f"{item.get('recording_id')} completed",
                )
                + render_detail_list(
                    pending,
                    lambda item: f"{item.get('recording_id')} → {item.get('current_task') or 'processing'}",
                )
                + render_detail_list(
                    failed,
                    lambda item: f"{item.get('recording_id') or 'global'}: {item.get('error')}",
                ),
            )

        if triggered == "upload-files-btn" and upload_clicks:
            service = get_data_service()
            candidates = service.get_upload_candidates()
            if not candidates:
                return html.Div(
                    className="sync-info",
                    children=[
                        html.Span("📤 No Files to Upload", className="success-icon"),
                        html.P("No local audio files found in data/raw/usb_import/."),
                    ],
                )

            file_paths = [c["path"] for c in candidates]
            effective_template = template_id or template_select or None
            effective_model = model_select or "gemini"

            result = service.upload_and_process_files(
                file_paths=file_paths,
                template_id=effective_template,
                model=effective_model,
            )

            uploaded = result.get("uploaded", [])
            errors = result.get("errors", [])

            if errors and not uploaded:
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Upload Failed", className="error-icon"),
                        html.P(errors[0].get("error", "Unknown error")),
                    ],
                )

            return html.Div(
                className="sync-success",
                children=[
                    html.Span("📤 Upload Complete", className="success-icon"),
                    html.P(
                        f"Uploaded {len(uploaded)} file(s), {len(errors)} error(s)."
                    ),
                ]
                + render_detail_list(
                    uploaded,
                    lambda item: f"{item.get('path', '').rsplit('/', 1)[-1]} → {item.get('workflow_id', 'submitted')}",
                )
                + render_detail_list(
                    errors,
                    lambda item: f"{item.get('path', '').rsplit('/', 1)[-1]}: {item.get('error')}",
                ),
            )

        if triggered == "do-sync-btn" and sync_clicks:
            # Launch pipeline in a background thread so the UI stays
            # responsive and the 2-second progress poller can update live.
            import threading

            from src.chronos.pipeline_progress import progress, read_progress
            from app_v2.services.xray import xray_log

            xray_log(
                "sync",
                "start",
                f"Kicking off a smart sync — checking the last {days_back or 7} days",
            )

            # Don't start a second run if one is already going
            # (but auto-clear stale runs older than 30 minutes)
            cur = read_progress()
            if cur and cur.get("status") == "running":
                import time as _t

                started = cur.get("started_at", 0)
                age_minutes = (_t.time() - started) / 60 if started else 999
                if age_minutes < 30:
                    return html.Div(
                        className="sync-info",
                        children=[
                            html.Span(
                                "⏳ Pipeline Already Running", className="success-icon"
                            ),
                            html.P("Watch the progress panel below."),
                        ],
                    )
                else:
                    # Stale run — auto-clear it
                    progress.finish_run(
                        error=f"Stale run auto-cleared after {age_minutes:.0f} min"
                    )
                    xray_log(
                        "sync",
                        "reset",
                        f"Previous sync was stuck for {age_minutes:.0f} min — clearing it",
                        level="warn",
                    )

            _days = days_back or 7

            def _run_pipeline():
                from src.chronos.ingest_service import ChronosIngestService
                from src.chronos.transcript_processor import TranscriptProcessor
                from src.chronos.embedding_service import ChronosEmbeddingService
                from src.chronos.qdrant_client import ChronosQdrantClient
                from src.database.engine import SessionLocal
                from src.database.chronos_repository import get_pending_chronos_recordings
                from src.database.models import ChronosEvent as ChronosEventModel
                from app_v2.services.xray import xray_log
                from src.chronos.trace_service import finish_trace_run, start_trace_run
                import time as _time

                db = SessionLocal()
                trace_run_id = None
                try:
                    active_phases = ["ingest", "process", "index"]
                    trace_run_id = start_trace_run(
                        trigger="manual",
                        source="sync",
                        title="Dash smart sync",
                        entrypoint="app_v2.callbacks.navigation",
                        metadata={"phases": active_phases, "days_back": _days},
                    )
                    progress.start_run(phases=active_phases, trigger="manual")

                    # Phase 1: Ingest
                    progress.start_phase("ingest")
                    progress.update(step="Fetching recording list from Plaud…")
                    xray_log(
                        "pipeline",
                        "ingest",
                        f"Step 1 — Checking your Plaud device for new recordings (last {_days} days)",
                    )
                    _p1 = _time.perf_counter()
                    ingest_svc = ChronosIngestService(db_session=db)
                    try:
                        success, failed = ingest_svc.ingest_recent_recordings(
                            days_back=_days, fetch_all_pages=True
                        )
                        progress.finish_phase(summary=f"{success} ingested, {failed} failed")
                        xray_log(
                            "pipeline",
                            "ingest",
                            f"Got {success} new recordings from Plaud" + (f" ({failed} had problems)" if failed else ""),
                            duration_ms=(_time.perf_counter() - _p1) * 1000,
                        )
                    except Exception as auth_err:
                        progress.finish_phase(summary=f"⚠️ {auth_err}")
                        xray_log(
                            "pipeline",
                            "ingest",
                            f"Plaud import failed: {auth_err}",
                            level="error",
                            duration_ms=(_time.perf_counter() - _p1) * 1000,
                        )

                    # Phase 2: Process
                    pending = get_pending_chronos_recordings(db)
                    progress.start_phase("process", total_items=len(pending))
                    xray_log(
                        "pipeline",
                        "process",
                        f"Step 2 — Sending {len(pending)} recordings to Gemini AI for analysis",
                    )
                    _p2 = _time.perf_counter()
                    if pending:
                        processor = TranscriptProcessor(db_session=db)
                        processed = proc_failed = 0
                        for i, rec in enumerate(pending):
                            rec_id = str(rec.recording_id)
                            rec_label = rec_id[:20]

                            def _recording_progress(step: str, detail: str = ""):
                                progress.update(
                                    step=f"Recording {i+1}/{len(pending)}: {step}",
                                    item=(detail or rec_label)[:120],
                                )

                            progress.update(
                                step=f"Recording {i+1}/{len(pending)}: Gemini AI…",
                                item=rec_label,
                            )
                            try:
                                if processor.process_recording_id(
                                    rec_id,
                                    progress_callback=_recording_progress,
                                ):
                                    processed += 1
                                else:
                                    proc_failed += 1
                            except Exception as e:
                                proc_failed += 1
                                progress.update(
                                    step=f"Recording {i+1}/{len(pending)}: failed",
                                    item=str(e)[:120],
                                )
                            progress.advance(item=rec_label)
                        progress.finish_phase(summary=f"{processed} processed, {proc_failed} failed")
                        xray_log(
                            "pipeline",
                            "process",
                            f"Gemini finished — understood {processed} recordings" + (f" ({proc_failed} confused it)" if proc_failed else ""),
                            duration_ms=(_time.perf_counter() - _p2) * 1000,
                        )
                    else:
                        progress.finish_phase(summary="No pending recordings")
                        xray_log(
                            "pipeline",
                            "process",
                            "Every recording is already analyzed — nothing new",
                            duration_ms=(_time.perf_counter() - _p2) * 1000,
                        )

                    # Phase 3: Index
                    progress.start_phase("index")
                    xray_log(
                        "pipeline", "index", "Step 3 — Making everything searchable"
                    )
                    _p3 = _time.perf_counter()
                    try:
                        embedder = ChronosEmbeddingService()
                        qdrant = ChronosQdrantClient()
                        unindexed = (
                            db.query(ChronosEventModel)
                            .filter(ChronosEventModel.qdrant_point_id.is_(None))
                            .all()
                        )
                        progress.update(total=len(unindexed), step="Generating embeddings…")
                        if unindexed:
                            texts = [str(e.clean_text) for e in unindexed]
                            vectors = embedder.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
                            from src.models.chronos_schemas import ChronosEvent as CE
                            indexed = 0
                            for event, vector in zip(unindexed, vectors):
                                try:
                                    schema_event = CE(
                                        event_id=str(event.event_id),
                                        recording_id=str(event.recording_id),
                                        start_ts=event.start_ts,
                                        end_ts=event.end_ts,
                                        day_of_week=str(event.day_of_week).capitalize(),
                                        hour_of_day=int(event.hour_of_day),
                                        clean_text=str(event.clean_text),
                                        category=str(event.category),
                                        category_confidence=(
                                            float(event.category_confidence)
                                            if getattr(
                                                event, "category_confidence", None
                                            )
                                            else None
                                        ),
                                        sentiment=float(event.sentiment or 0.0),
                                        keywords=list(event.keywords or []),
                                        speaker=str(event.speaker or "unknown"),
                                        raw_transcript_snippet=(
                                            str(event.raw_transcript_snippet)
                                            if event.raw_transcript_snippet
                                            else None
                                        ),
                                        gemini_reasoning=(
                                            str(event.gemini_reasoning)
                                            if event.gemini_reasoning
                                            else None
                                        ),
                                    )
                                    point_id = qdrant.upsert_event(schema_event, vector)
                                    event.qdrant_point_id = point_id
                                    db.commit()
                                    indexed += 1
                                    progress.advance(item=f"{indexed} indexed")
                                except Exception as e:
                                    logger.error(f"Index error: {e}")
                                    progress.advance(item=f"error: {e}")
                            progress.finish_phase(summary=f"{indexed} events indexed")
                            xray_log(
                                "pipeline",
                                "index",
                                f"Made {indexed} new moments searchable",
                                duration_ms=(_time.perf_counter() - _p3) * 1000,
                            )
                        else:
                            progress.finish_phase(summary="All events already indexed")
                            xray_log(
                                "pipeline",
                                "index",
                                "Everything is already searchable — nothing to add",
                                duration_ms=(_time.perf_counter() - _p3) * 1000,
                            )
                    except Exception as e:
                        progress.finish_phase(error=str(e)[:100])
                        xray_log(
                            "pipeline",
                            "index",
                            f"Couldn't make things searchable: {str(e)[:80]}",
                            level="error",
                            duration_ms=(_time.perf_counter() - _p3) * 1000,
                        )

                    # Refresh cache
                    try:
                        service = get_data_service()
                        service.refresh_cache()
                    except Exception:
                        pass
                    progress.finish_run()
                    finish_trace_run(
                        trace_run_id,
                        status="completed",
                        summary={"phases": active_phases},
                    )
                    xray_log("pipeline", "done", "All done! Everything is synced and up to date ✔")
                except Exception as e:
                    logger.error(f"Pipeline thread error: {e}")
                    progress.finish_run(error=str(e))
                    finish_trace_run(trace_run_id, status="failed", error_message=str(e))
                    xray_log("pipeline", "done", f"Something went wrong during sync: {e}", level="error")
                finally:
                    db.close()

            t = threading.Thread(target=_run_pipeline, daemon=True, name="pipeline-sync")
            t.start()

            return html.Div(
                className="sync-info",
                children=[
                    html.Span("🚀 Pipeline Started!", className="success-icon"),
                    html.P("Watch the live progress panel below."),
                ],
            )

        raise PreventUpdate

    # ------------------------------------------------------------------
    # Live pipeline-progress polling
    # ------------------------------------------------------------------
    @app.callback(
        Output("pipeline-progress-panel", "children"),
        Input("pipeline-progress-poll", "n_intervals"),
        prevent_initial_call=True)
    def poll_pipeline_progress(n):
        """Read pipeline_progress.json and render a live progress panel."""
        from src.chronos.pipeline_progress import read_progress

        data = read_progress()

        # Nothing to show
        if data is None:
            return []

        status = data.get("status", "idle")
        age = data.get("age_seconds", 9999)

        # Hide stale completed/failed runs (>5 min)
        if status in ("completed", "failed", "idle") and age > 300:
            return []

        phases = data.get("phases", [])
        trigger = data.get("trigger", "")
        elapsed = data.get("elapsed_seconds", 0)
        if status == "running" and data.get("started_at"):
            elapsed = time.time() - data.get("started_at")


        # Phase icons
        phase_icons = {
            "ingest": "\U0001f4e5",
            "process": "\U0001f9e0",
            "index": "\U0001f4ca",
            "graph": "\U0001f578\ufe0f",
            "refresh-workflows": "\U0001f504",
        }

        # Build per-phase cards
        phase_cards = []
        for ph in phases:
            ph_name = ph.get("name", "")
            ph_status = ph.get("status", "pending")
            total = ph.get("total_items", 0)
            completed = ph.get("completed_items", 0)
            step = ph.get("current_step", "")
            item = ph.get("current_item", "")
            ph_elapsed = ph.get("elapsed_seconds", 0)
            if ph_status == "running" and ph.get("started_at"):
                ph_elapsed = time.time() - ph.get("started_at")
            summary = ph.get("summary", "")
            error = ph.get("error", "")

            icon = phase_icons.get(ph_name, "\u2699\ufe0f")

            # Status indicator
            if ph_status == "running":
                status_badge = html.Span(
                    "\u25cf RUNNING", className="pp-badge pp-running"
                )
            elif ph_status == "completed":
                status_badge = html.Span(
                    "\u2713 DONE", className="pp-badge pp-completed"
                )
            elif ph_status == "failed":
                status_badge = html.Span(
                    "\u2717 FAILED", className="pp-badge pp-failed"
                )
            else:
                status_badge = html.Span(
                    "\u25cb PENDING", className="pp-badge pp-pending"
                )

            # Progress bar
            pct = (completed / total * 100) if total > 0 else 0
            progress_bar = (
                html.Div(
                    className=f"pp-bar-track {'pp-bar-active' if ph_status == 'running' else ''}",
                    children=html.Div(
                        className="pp-bar-fill",
                        style={"width": f"{pct:.0f}%"},
                    ),
                )
                if total > 0
                else None
            )

            # Detail line
            detail_parts = []
            if total > 0:
                detail_parts.append(f"{completed}/{total} items")
            if ph_elapsed > 0:
                detail_parts.append(f"{ph_elapsed:.1f}s")
            if step:
                detail_parts.append(step)
            detail_line = " \u00b7 ".join(detail_parts) if detail_parts else None

            # Item line (currently processing)
            item_line = None
            if item and ph_status == "running":
                display_item = item if len(item) <= 60 else item[:57] + "\u2026"
                item_line = html.Div(display_item, className="pp-item")

            # Summary (for completed phases)
            summary_line = None
            if summary and ph_status in ("completed", "failed"):
                summary_line = html.Div(summary, className="pp-summary")

            # Error line
            error_line = None
            if error:
                error_line = html.Div(f"Error: {error}", className="pp-error")

            card_children = [
                html.Div(
                    className="pp-card-header",
                    children=[
                        html.Span(
                            f"{icon} {ph_name.replace('-', ' ').title()}",
                            className="pp-phase-name",
                        ),
                        status_badge,
                    ],
                ),
            ]
            if progress_bar:
                card_children.append(progress_bar)
            if detail_line:
                card_children.append(html.Div(detail_line, className="pp-detail"))
            if item_line:
                card_children.append(item_line)
            if summary_line:
                card_children.append(summary_line)
            if error_line:
                card_children.append(error_line)

            phase_cards.append(
                html.Div(className=f"pp-card pp-{ph_status}", children=card_children)
            )

        # Overall header
        if status == "running":
            header_text = f"Pipeline Running \u2014 {elapsed:.0f}s"
            header_class = "pp-header pp-header-running"
        elif status == "completed":
            header_text = f"Pipeline Complete \u2014 {elapsed:.1f}s total"
            header_class = "pp-header pp-header-completed"
        elif status == "failed":
            header_text = f"Pipeline Failed \u2014 {elapsed:.1f}s"
            header_class = "pp-header pp-header-failed"
        else:
            header_text = "Pipeline Idle"
            header_class = "pp-header"

        trigger_label = f" ({trigger})" if trigger else ""

        return html.Div(
            className="pp-container",
            children=[
                html.Div(
                    className=header_class,
                    children=[
                        html.Span(header_text, className="pp-header-text"),
                        html.Span(trigger_label, className="pp-trigger"),
                    ],
                ),
                html.Div(className="pp-phases", children=phase_cards),
            ],
        )

    # ------------------------------------------------------------------
    # Workflow auto-poll: refresh active workflows every 10s
    # ------------------------------------------------------------------
    @app.callback(
        Output("active-workflows-count", "data"),
        Output("workflow-poll", "disabled"),
        Input("workflow-poll", "n_intervals"),
        Input("run-plaud-workflows-btn", "n_clicks"),
        Input("upload-files-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def poll_workflow_status(n_intervals, run_clicks, upload_clicks):
        """Auto-refresh active Plaud workflows and enable/disable polling."""
        try:
            service = get_data_service()
            stats = service.get_plaud_workflow_stats()
            active = stats.get("active_workflows", [])
            active_count = len(active)

            if active_count > 0:
                # There are active workflows — keep polling, refresh their statuses
                try:
                    service.refresh_plaud_workflow_statuses(days_back=7, limit=20)
                except Exception:
                    pass
                return active_count, False  # keep polling
            else:
                # No active workflows — disable polling until next submission
                return 0, True
        except Exception:
            return 0, True

    # ── Cost tracker live-refresh ──────────────────────────────

    @app.callback(
        Output("cost-live-container", "children"),
        Input("cost-refresh-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def refresh_cost_display(_n):
        """Refresh cost cards every 15 seconds while Stats view is open."""
        from app_v2.components.stats import create_cost_section

        section = create_cost_section()
        # Return only the live container children (skip the outer wrapper)
        container = None
        for child in section.children or []:
            if getattr(child, "id", None) == "cost-live-container":
                container = child
                break
        if container:
            return container.children
        raise PreventUpdate
