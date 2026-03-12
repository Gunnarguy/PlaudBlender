"""Navigation callbacks - view switching and main content updates."""

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
    "auto_refresh_enabled": True,
    "auto_refresh_seconds": 60,
    "default_view": "timeline",
}


def merge_preferences(preferences):
    """Merge user preferences with defaults and coerce safe values."""
    merged = dict(DEFAULT_PREFERENCES)
    if isinstance(preferences, dict):
        merged.update(preferences)

    seconds = merged.get("auto_refresh_seconds", 60)
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        seconds = 60
    merged["auto_refresh_seconds"] = max(15, min(300, seconds))
    merged["auto_refresh_enabled"] = bool(merged.get("auto_refresh_enabled", True))

    default_view = str(merged.get("default_view", "timeline"))
    allowed_views = {"timeline", "days", "topics", "graph", "stats", "sync", "settings"}
    merged["default_view"] = (
        default_view if default_view in allowed_views else "timeline"
    )
    return merged


def create_sync_view(service) -> html.Div:
    """Create the sync view with full pipeline controls and auto-sync status."""
    stats = service.get_stats()
    db_stats = service.get_recording_db_stats()
    workflow_stats = service.get_plaud_workflow_stats(days_back=30)

    pending = db_stats.get("pending", 0)
    processing = db_stats.get("processing", 0)
    failed = db_stats.get("failed", 0)
    completed = db_stats.get("completed", 0)
    total = pending + processing + failed + completed

    # Auto-sync status
    auto_sync_children = []
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
                    # Sync history
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

    # Failed recording details
    failed_details_children = []
    if failed > 0:
        try:
            from src.database.engine import SessionLocal as _SL
            import sqlalchemy as sa

            _db = _SL()
            try:
                failed_rows = _db.execute(
                    sa.text(
                        "SELECT recording_id, error_message FROM chronos_recordings "
                        "WHERE processing_status = 'failed' LIMIT 5"
                    )
                ).fetchall()
                if failed_rows:
                    failed_details_children = [
                        html.Div(
                            className="failed-recordings-detail",
                            style={
                                "marginTop": "10px",
                                "borderTop": "1px solid var(--border-color, #e2e8f0)",
                                "paddingTop": "10px",
                            },
                            children=[
                                html.Span(
                                    "🔍 Failed Recordings:",
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
                                            f"{row[0][:16]}… — {(row[1] or 'Unknown error')[:80]}"
                                        )
                                        for row in failed_rows
                                    ],
                                ),
                            ],
                        ),
                    ]
            finally:
                _db.close()
        except Exception:
            pass
    if stats.plaud_cloud_stats:
        cs = stats.plaud_cloud_stats
        cloud_total = cs.get("total_count", 0)
        cloud_hours = cs.get("total_duration_hours", 0)
        synced_pct = (completed / cloud_total * 100) if cloud_total else 0

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

    return html.Div(
        className="sync-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("🔄 Sync & Process", className="view-title"),
                    html.P(
                        "Fetch, process, and index your Plaud recordings",
                        className="view-subtitle",
                    ),
                ],
            ),
            # Pipeline status dashboard
            html.Div(
                className="sync-status-card",
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
                                        str(failed),
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
                        className="status-stats",
                        style={
                            "marginTop": "10px",
                            "borderTop": "1px solid var(--border-color, #e2e8f0)",
                            "paddingTop": "10px",
                        },
                        children=[
                            html.Div(
                                [
                                    html.Span(
                                        str(stats.total_events), className="big-number"
                                    ),
                                    html.Span(
                                        "Events in Qdrant", className="stat-label"
                                    ),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        str(stats.total_days), className="big-number"
                                    ),
                                    html.Span("Days", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        f"{stats.total_duration_hours:.1f}",
                                        className="big-number",
                                    ),
                                    html.Span("Hours Recorded", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                        ],
                    ),
                    # Plaud cloud stats (if available)
                    *plaud_cloud_children,
                    # Failed recording error details (if any)
                    *failed_details_children,
                ],
            ),
            html.Div(
                className="sync-status-card",
                children=[
                    html.H4("Plaud Cloud Enrichment"),
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
                                        str(
                                            workflow_stats.get(
                                                "ready_for_enrichment", 0
                                            )
                                        ),
                                        className="big-number",
                                        style={"color": "#f59e0b"},
                                    ),
                                    html.Span("Ready", className="stat-label"),
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
            ),
            # Auto-sync status (if available)
            *auto_sync_children,
            # Action buttons
            html.Div(
                className="sync-options",
                children=[
                    html.H4("Actions"),
                    # Full pipeline sync
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
                                        "Full Sync (Fetch → Process → Index)",
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
                            html.Label("Plaud cloud enrichment batch size:"),
                            dcc.Slider(
                                id="plaud-workflow-limit",
                                min=1,
                                max=10,
                                step=1,
                                value=3,
                                marks={1: "1", 3: "3", 5: "5", 10: "10"},
                                className="sync-slider",
                            ),
                            html.Label("Custom ETL template ID (optional):"),
                            dcc.Input(
                                id="plaud-template-id",
                                type="text",
                                placeholder="tpl_your_template_id",
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
                                            workflow_stats.get("workflow_pending", 0)
                                            == 0
                                        ),
                                    ),
                                ],
                            ),
                            html.P(
                                "This targets recent recordings missing a Plaud AI summary. Add a template only when you want AI_ETL structured output too.",
                                className="sync-note",
                            ),
                        ],
                    ),
                    # Reset stuck recordings
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
                                        f"Reset Stuck Recordings ({processing} stuck)",
                                        className="btn-text",
                                    ),
                                ],
                                disabled=(processing == 0),
                            ),
                        ],
                    ),
                    html.Div(id="sync-result", className="sync-result"),
                    # Live pipeline progress panel
                    html.Div(
                        id="pipeline-progress-panel",
                        className="pipeline-progress-panel",
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
                        "Token expired; refresh failed — run plaud_setup.py",
                    )
            else:
                checks["plaud"] = (
                    False,
                    "Token expired; no refresh token — run plaud_setup.py",
                )
        elif has_token:
            checks["plaud"] = (True, "Token present")
        else:
            checks["plaud"] = (False, "Run plaud_setup.py to authenticate")
    except Exception as e:
        checks["plaud"] = (False, str(e)[:80])

    # Gemini
    gemini_ok = bool(settings.gemini_api_key)
    checks["gemini"] = (
        gemini_ok,
        (
            f"API key set — {settings.chronos_cleaning_model}"
            if gemini_ok
            else "GEMINI_API_KEY not set"
        ),
    )

    # OpenAI
    openai_ok = bool(settings.openai_api_key)
    checks["openai"] = (
        openai_ok,
        (
            f"API key set — {settings.openai_model}"
            if openai_ok
            else "OPENAI_API_KEY not set"
        ),
    )

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
    connection_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🔗 Service Connections"),
            status_row("Docker:", *checks["docker"]),
            status_row("Plaud Auth:", *checks["plaud"]),
            status_row("Gemini AI:", *checks["gemini"]),
            status_row("OpenAI:", *checks["openai"]),
            status_row("SQLite:", *checks["sqlite"]),
            status_row("Qdrant:", *checks["qdrant"]),
            status_row("Webhook Listener:", *checks["webhook_listener"]),
            status_row("Webhook Config:", *checks["webhook_config"]),
        ],
    )

    # ── Section: AI Models ───────────────────────────────────────────────
    models_section = html.Div(
        className="settings-section",
        children=[
            html.H4("🧠 AI Models"),
            html.P(
                "Model selection for each processing stage. "
                "Change via .env or the controls below.",
                className="setting-note",
            ),
            html.Div(
                className="settings-grid",
                children=[
                    # Cleaning model
                    html.Div(
                        className="setting-control-row",
                        children=[
                            html.Label("Cleaning Model"),
                            dcc.Dropdown(
                                id="setting-cleaning-model",
                                options=[
                                    {
                                        "label": "gemini-3-flash-preview (free)",
                                        "value": "gemini-3-flash-preview",
                                    },
                                    {
                                        "label": "gemini-2.5-flash (free, stable)",
                                        "value": "gemini-2.5-flash",
                                    },
                                    {
                                        "label": "gemini-3.1-pro-preview (best)",
                                        "value": "gemini-3.1-pro-preview",
                                    },
                                    {
                                        "label": "gemini-2.5-pro (stable thinking)",
                                        "value": "gemini-2.5-pro",
                                    },
                                ],
                                value=settings.chronos_cleaning_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Processes raw transcripts → clean events",
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
                                        "label": "gemini-3.1-pro-preview (best)",
                                        "value": "gemini-3.1-pro-preview",
                                    },
                                    {
                                        "label": "gemini-2.5-pro (stable thinking)",
                                        "value": "gemini-2.5-pro",
                                    },
                                    {
                                        "label": "gemini-3-flash-preview (free)",
                                        "value": "gemini-3-flash-preview",
                                    },
                                    {
                                        "label": "gemini-2.5-flash (stable fast)",
                                        "value": "gemini-2.5-flash",
                                    },
                                ],
                                value=settings.chronos_analyst_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Deep analysis, MCP ask_chronos, graph extraction",
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
                                        "label": "gemini-embedding-2-preview (multimodal)",
                                        "value": "gemini-embedding-2-preview",
                                    },
                                    {
                                        "label": "gemini-embedding-001 (text-only legacy)",
                                        "value": "gemini-embedding-001",
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
                                        "label": "gpt-5.4 (flagship — $2.50/$15 MTok, 1.05M ctx)",
                                        "value": "gpt-5.4",
                                    },
                                    {
                                        "label": "gpt-5.4-pro (smartest — precise/detailed)",
                                        "value": "gpt-5.4-pro",
                                    },
                                    {
                                        "label": "gpt-5-mini (cost-effective — $0.25/$2 MTok)",
                                        "value": "gpt-5-mini",
                                    },
                                    {
                                        "label": "gpt-5-nano (fastest/cheapest)",
                                        "value": "gpt-5-nano",
                                    },
                                    {
                                        "label": "gpt-5 (previous reasoning)",
                                        "value": "gpt-5",
                                    },
                                    {
                                        "label": "gpt-4.1 (non-reasoning legacy)",
                                        "value": "gpt-4.1",
                                    },
                                ],
                                value=settings.openai_model,
                                clearable=False,
                                className="settings-dropdown",
                            ),
                            html.Span(
                                "Used for RAG responses (Responses API)",
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
                "gemini-embedding-2-preview",
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
            storage_section,
            logging_section,
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
        prevent_initial_call=True,
    )
    def save_env_settings(
        n_clicks,
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
    ):
        """Write changed settings back to .env file."""
        if not n_clicks:
            raise PreventUpdate

        from pathlib import Path

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            return "❌ .env file not found"

        # Map of env var name → new value
        updates = {
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
        }

        try:
            lines = env_path.read_text().splitlines()
            existing_keys = set()
            new_lines = []

            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in updates and updates[key] is not None:
                        new_lines.append(f"{key}={updates[key]}")
                        existing_keys.add(key)
                        continue
                new_lines.append(line)

            # Append any new keys not already in .env
            for key, value in updates.items():
                if key not in existing_keys and value is not None:
                    new_lines.append(f"{key}={value}")

            env_path.write_text("\n".join(new_lines) + "\n")

            changed = len(updates)
            return f"✅ Saved {changed} settings to .env — restart app to apply"

        except Exception as e:
            return f"❌ Failed to save: {e}"

    @app.callback(
        Output("content-container", "children"),
        Output("current-view", "data"),
        Output("detail-panel", "children"),
        Output("detail-panel", "className"),
        Input({"type": "nav-item", "view": ALL}, "n_clicks"),
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
        selected_recording,
        selected_topic,
        search_query,
        n_intervals,
        current_view,
        preferences,
    ):
        """Update main content based on navigation and state."""
        triggered = ctx.triggered_id
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

        if isinstance(triggered, dict) and triggered.get("type") == "nav-item":
            view = triggered.get("view", "timeline")

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

            logger.info(
                f"Fetching recording detail for: {selected_recording.get('id')}"
            )
            detail = get_service().get_recording_detail(selected_recording.get("id"))
            if detail:
                logger.info(f"Got detail with {len(detail.events)} events")
                transcript = get_service().get_transcript(selected_recording.get("id"))
                ai_summary = get_service().get_ai_summary(selected_recording.get("id"))
                detail_content = create_recording_detail(
                    detail,
                    selected_recording.get("date", ""),
                    transcript=transcript,
                    highlight_event_id=selected_recording.get("scroll_to_event"),
                    ai_summary=ai_summary,
                )
                detail_class = "detail-panel open"
            else:
                logger.warning("No detail returned!")

        # Render main content based on view
        if view == "timeline":
            days = get_service().get_days()
            content = create_day_view(days)
        elif view == "days":
            # Legacy compat — treat 'days' as 'timeline'
            days = get_service().get_days()
            content = create_day_view(days)
            view = "timeline"
        elif view == "search":
            # Preserve search results when opening a detail panel
            if search_query:
                results = get_service().search(search_query)
                content = create_search_results(results, search_query)
            else:
                days = get_service().get_days()
                content = create_day_view(days)
                view = "timeline"
        elif view == "topic-detail":
            # Preserve topic timeline when opening a detail panel
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
        elif view == "stats":
            stats = get_service().get_stats()
            content = create_stats_view(stats)
        elif view == "sync":
            content = create_sync_view(get_service())
        elif view == "settings":
            content = create_settings_view(preferences=prefs)
        else:
            days = get_service().get_days()
            content = create_day_view(days)

        return content, view, detail_content, detail_class

    @app.callback(
        Output({"type": "nav-item", "view": ALL}, "className"),
        Input("current-view", "data"),
        State({"type": "nav-item", "view": ALL}, "id"),
    )
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
        Output("sync-result", "children"),
        Input("do-sync-btn", "n_clicks"),
        Input("reset-stuck-btn", "n_clicks"),
        Input("run-plaud-workflows-btn", "n_clicks"),
        Input("refresh-plaud-workflows-btn", "n_clicks"),
        State("sync-days-slider", "value"),
        State("plaud-workflow-limit", "value"),
        State("plaud-template-id", "value"),
        prevent_initial_call=True,
    )
    def perform_sync(
        sync_clicks,
        reset_clicks,
        run_plaud_clicks,
        refresh_plaud_clicks,
        days_back,
        workflow_limit,
        template_id,
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
                return html.Div(
                    className="sync-success",
                    children=[
                        html.Span("🔧 Reset Complete!", className="success-icon"),
                        html.P(f"Reset {count} stuck recordings to pending."),
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
            result = service.submit_plaud_workflows(
                days_back=days_back or 7,
                limit=workflow_limit or 3,
                template_id=template_id,
            )

            submitted = result.get("submitted", [])
            errors = result.get("errors", [])
            skipped = result.get("skipped", [])
            template_text = result.get("template_id") or "summary-only"

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
                        f"Submitted {len(submitted)} workflow(s) using {template_text}."
                    ),
                    html.P(
                        f"Skipped {len(skipped)} recording(s) already summarized or already in flight.",
                        className="sync-note",
                    ),
                ]
                + render_detail_list(
                    submitted,
                    lambda item: f"{item.get('recording_id')} → {item.get('workflow_id')}",
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

        if triggered == "do-sync-btn" and sync_clicks:
            try:
                from src.chronos.ingest_service import ChronosIngestService
                from src.chronos.transcript_processor import TranscriptProcessor
                from src.chronos.embedding_service import ChronosEmbeddingService
                from src.chronos.qdrant_client import ChronosQdrantClient
                from src.chronos.pipeline_progress import progress
                from src.database.engine import SessionLocal
                from src.database.chronos_repository import (
                    get_pending_chronos_recordings,
                    get_chronos_events_by_recording,
                )
                from src.database.models import ChronosEvent as ChronosEventModel
                import time as _time

                db = SessionLocal()
                steps = []
                timings = []
                try:
                    active_phases = ["ingest", "process", "index"]
                    progress.start_run(phases=active_phases, trigger="manual")

                    # Phase 1: Ingest from Plaud
                    t0 = _time.monotonic()
                    progress.start_phase("ingest")
                    progress.update(step="Fetching recording list from Plaud…")
                    ingest_svc = ChronosIngestService(db_session=db)
                    auth_warning = None
                    try:
                        success, failed = ingest_svc.ingest_recent_recordings(
                            days_back=days_back or 7, fetch_all_pages=True
                        )
                    except Exception as auth_err:
                        # Catch ALL ingest errors (auth, network, API) —
                        # don't let ingest failure kill process/index phases.
                        auth_warning = str(auth_err)
                        success, failed = 0, 0
                    t_ingest = _time.monotonic() - t0
                    timings.append(("Ingest", t_ingest))
                    if auth_warning:
                        steps.append(f"⚠️ Plaud Auth Failed: {auth_warning}")
                    else:
                        steps.append(f"📥 Ingested: {success} new, {failed} failed")
                    progress.finish_phase(
                        summary=f"{success} ingested, {failed} failed"
                    )

                    # Phase 2: Process pending through Gemini
                    t0 = _time.monotonic()
                    pending = get_pending_chronos_recordings(db)
                    progress.start_phase("process", total_items=len(pending))
                    if pending:
                        processor = TranscriptProcessor(db_session=db)
                        processed = 0
                        proc_failed = 0
                        proc_errors = []
                        for i, rec in enumerate(pending):
                            rec_id = str(rec.recording_id)
                            progress.update(
                                step=f"Recording {i+1}/{len(pending)}: Gemini AI…",
                                item=rec_id[:20],
                            )
                            try:
                                ok = processor.process_recording_id(rec_id)
                                if ok:
                                    processed += 1
                                    progress.advance(
                                        item=rec_id[:20], step=f"✅ {processed} done"
                                    )
                                else:
                                    proc_failed += 1
                                    proc_errors.append(rec_id[:16])
                                    progress.advance(
                                        item=rec_id[:20], step=f"❌ failed"
                                    )
                            except Exception as e:
                                logger.error(f"Process error: {e}")
                                proc_failed += 1
                                proc_errors.append(f"{rec_id[:16]}: {str(e)[:40]}")
                                progress.advance(item=rec_id[:20])
                        t_process = _time.monotonic() - t0
                        timings.append(("Process", t_process))
                        step_msg = f"🧠 Processed: {processed} recordings"
                        if proc_failed:
                            step_msg += f" ({proc_failed} failed)"
                        steps.append(step_msg)
                        if proc_errors:
                            steps.append(f"   └ Errors: {'; '.join(proc_errors[:3])}")
                        progress.finish_phase(
                            summary=f"{processed} processed, {proc_failed} failed"
                        )
                    else:
                        t_process = _time.monotonic() - t0
                        timings.append(("Process", t_process))
                        steps.append("🧠 No pending recordings to process")
                        progress.finish_phase(summary="No pending recordings")

                    # Phase 3: Index to Qdrant
                    t0 = _time.monotonic()
                    progress.start_phase("index")
                    try:
                        embedder = ChronosEmbeddingService()
                        qdrant = ChronosQdrantClient()

                        # Find events not yet in Qdrant
                        all_events = (
                            db.query(ChronosEventModel)
                            .filter(ChronosEventModel.qdrant_point_id.is_(None))
                            .all()
                        )
                        unindexed = all_events
                        progress.update(
                            total=len(unindexed), step="Generating embeddings…"
                        )

                        if unindexed:
                            texts = [str(e.clean_text) for e in unindexed]
                            vectors = embedder.embed_batch(
                                texts, task_type="RETRIEVAL_DOCUMENT"
                            )

                            from src.models.chronos_schemas import ChronosEvent as CE

                            indexed = 0
                            for event, vector in zip(unindexed, vectors):
                                try:
                                    schema_event = CE(
                                        event_id=str(event.event_id),
                                        recording_id=str(event.recording_id),
                                        start_ts=event.start_ts,  # type: ignore[arg-type]
                                        end_ts=event.end_ts,  # type: ignore[arg-type]
                                        day_of_week=str(event.day_of_week),  # type: ignore[arg-type]
                                        hour_of_day=int(event.hour_of_day),  # type: ignore[arg-type]
                                        clean_text=str(event.clean_text),
                                        category=str(event.category),  # type: ignore[arg-type]
                                        sentiment=float(event.sentiment or 0.0),  # type: ignore[arg-type]
                                        keywords=list(event.keywords or []),  # type: ignore[arg-type]
                                        speaker=str(event.speaker or "unknown"),  # type: ignore[arg-type]
                                        raw_transcript_snippet=str(event.raw_transcript_snippet) if event.raw_transcript_snippet else None,  # type: ignore[truthy-bool]
                                        gemini_reasoning=str(event.gemini_reasoning) if event.gemini_reasoning else None,  # type: ignore[truthy-bool]
                                    )
                                    point_id = qdrant.upsert_event(schema_event, vector)
                                    event.qdrant_point_id = point_id  # type: ignore[assignment]
                                    db.commit()
                                    indexed += 1
                                except Exception as e:
                                    logger.error(f"Index error: {e}")
                            steps.append(f"📊 Indexed: {indexed} events to Qdrant")
                            progress.finish_phase(summary=f"{indexed} events indexed")
                        else:
                            steps.append("📊 All events already indexed")
                            progress.finish_phase(summary="All events already indexed")
                        t_index = _time.monotonic() - t0
                        timings.append(("Index", t_index))
                    except Exception as e:
                        t_index = _time.monotonic() - t0
                        timings.append(("Index", t_index))
                        steps.append(f"📊 Indexing error: {str(e)[:60]}")
                        progress.finish_phase(error=str(e)[:100])

                    # Refresh the data service cache
                    service = get_data_service()
                    service.refresh_cache()
                    progress.finish_run()

                    # Build telemetry section
                    total_time = sum(t for _, t in timings)
                    telemetry_children = [
                        html.Div(
                            className="sync-telemetry",
                            children=[
                                html.Span("⏱ ", className="telemetry-icon"),
                                html.Span(
                                    " · ".join(
                                        f"{name}: {t:.1f}s" for name, t in timings
                                    )
                                    + f" · Total: {total_time:.1f}s",
                                    className="telemetry-text",
                                ),
                            ],
                        )
                    ]

                    # Build failed recording details (if any)
                    failed_details = []
                    try:
                        import sqlalchemy as sa

                        failed_rows = db.execute(
                            sa.text(
                                "SELECT recording_id, error_message FROM chronos_recordings "
                                "WHERE processing_status = 'failed' LIMIT 5"
                            )
                        ).fetchall()
                        if failed_rows:
                            failed_details = [
                                html.Div(
                                    className="sync-failed-details",
                                    children=[
                                        html.Span(
                                            "🔍 Failed recordings:",
                                            style={"fontWeight": "600"},
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    f"{row[0][:16]}… — {(row[1] or 'Unknown error')[:80]}",
                                                    className="failed-detail-item",
                                                )
                                                for row in failed_rows
                                            ]
                                        ),
                                    ],
                                )
                            ]
                    except Exception:
                        pass

                    return html.Div(
                        className="sync-success",
                        children=[
                            html.Span(
                                "✅ Full Pipeline Complete!", className="success-icon"
                            ),
                        ]
                        + [html.P(step) for step in steps]
                        + telemetry_children
                        + failed_details
                        + [
                            html.P(
                                "Refresh the page to see updated data.",
                                className="sync-note",
                            ),
                        ],
                    )
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Sync error: {e}")
                try:
                    from src.chronos.pipeline_progress import progress as _p

                    _p.finish_run(error=str(e))
                except Exception:
                    pass
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Pipeline Failed", className="error-icon"),
                        html.P(str(e)),
                    ],
                )

        raise PreventUpdate

    # ------------------------------------------------------------------
    # Live pipeline-progress polling
    # ------------------------------------------------------------------
    @app.callback(
        Output("pipeline-progress-panel", "children"),
        Input("pipeline-progress-poll", "n_intervals"),
    )
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
