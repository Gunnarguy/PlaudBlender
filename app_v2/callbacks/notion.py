"""Notion integration callbacks.

Handles:
- Database discovery and selection
- Auto-fetch on page load
- Fetching recordings from Notion
- Smart matching against Chronos recordings
- Coverage calendar generation
- Import single / import all to Chronos pipeline
- Write-back Chronos enrichments to Notion
- Page detail display
- Client-side search / filter / sort
"""

import logging
import time
import threading
from dash import Input, Output, State, callback_context as ctx, html, dcc, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)

# ── Notion data cache (avoids re-fetching on every tab switch) ────
_notion_cache_lock = threading.Lock()
_notion_cache: dict | None = None
_notion_cache_ts: float = 0.0
_NOTION_CACHE_TTL = 60  # seconds


def _get_cached_notion_data() -> dict | None:
    """Return cached Notion data if still fresh, else None."""
    with _notion_cache_lock:
        if _notion_cache and (time.monotonic() - _notion_cache_ts) < _NOTION_CACHE_TTL:
            return _notion_cache
    return None


def _set_notion_cache(data: dict):
    """Store Notion data in cache."""
    global _notion_cache, _notion_cache_ts
    with _notion_cache_lock:
        _notion_cache = data
        _notion_cache_ts = time.monotonic()


def invalidate_notion_cache():
    """Clear the Notion data cache (called on Refresh, import, etc.)."""
    global _notion_cache, _notion_cache_ts
    with _notion_cache_lock:
        _notion_cache = None
        _notion_cache_ts = 0.0


def _build_progress_display(progress: dict):
    """Build a live progress bar + label from import progress data."""
    total = progress.get("total", 1)
    completed = progress.get("completed", 0)
    failed = progress.get("failed", 0)
    done_count = completed + failed
    pct = int((done_count / max(total, 1)) * 100)
    current = progress.get("current_title", "")
    idx = progress.get("current_index", 0)

    return html.Div(
        className="notion-import-result notion-import-running",
        children=[
            html.Span(
                f"🔄 Importing recordings through Gemini... ({completed} done, {failed} failed)"
            ),
            html.Div(
                className="notion-progress-bar-container",
                children=[
                    html.Div(
                        className="notion-progress-bar", style={"width": f"{pct}%"}
                    ),
                ],
            ),
            html.Span(f"[{idx}/{total}] {current}", className="notion-progress-label"),
        ],
    )


def register_notion_callbacks(app):
    """Register all Notion-related callbacks."""

    # ── Auto-fetch on page load ───────────────────────────────────
    @app.callback(
        Output("notion-fetch-btn", "n_clicks", allow_duplicate=True),
        Input("notion-auto-fetch-trigger", "n_intervals"),
        State("notion-recordings-store", "data"),
        prevent_initial_call=True,
    )
    def auto_fetch_on_load(n_intervals, existing_data):
        """Auto-trigger fetch when Notion page first loads (if DB configured)."""
        if existing_data:
            raise PreventUpdate  # Already have data, don't re-fetch
        return 1  # Trigger the fetch button

    # ── Discover databases ────────────────────────────────────────
    @app.callback(
        Output("content-container", "children", allow_duplicate=True),
        Input("notion-discover-dbs-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def discover_databases(n_clicks):
        """List all accessible Notion databases for selection."""
        if not n_clicks:
            raise PreventUpdate

        from app_v2.services.xray import xray_log

        xray_log("data", "notion", "Discovering Notion data sources...")

        try:
            from src.notion_service import get_notion_service
            from app_v2.components.notion import create_notion_view

            svc = get_notion_service()
            databases = svc.list_databases()
            xray_log("data", "notion", f"Found {len(databases)} accessible data sources")

            return create_notion_view(databases=databases)

        except Exception as e:
            logger.error(f"Error discovering databases: {e}")
            xray_log("data", "notion", f"Database discovery failed: {e}", level="error")
            from app_v2.components.notion import create_notion_view
            return create_notion_view()

    # ── Select database ───────────────────────────────────────────
    @app.callback(
        Output("content-container", "children", allow_duplicate=True),
        Input({"type": "notion-select-db", "db_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select_database(n_clicks_list):
        """Select a database and save to .env."""
        if not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            raise PreventUpdate

        db_id = triggered.get("db_id")
        if not db_id:
            raise PreventUpdate

        from app_v2.services.xray import xray_log

        try:
            from src.notion_service import get_notion_service
            svc = get_notion_service()
            svc.set_database_id(db_id)
            xray_log("data", "notion", f"Selected database {db_id[:8]}... — saved to .env")

            # Now fetch data with the new database
            return _do_full_fetch()

        except Exception as e:
            logger.error(f"Error selecting database: {e}")
            xray_log("data", "notion", f"Error selecting database: {e}", level="error")
            raise PreventUpdate

    # ── Fetch recordings (main data loader) ───────────────────────
    @app.callback(
        Output("notion-recordings-store", "data"),
        Output("notion-status-store", "data"),
        Output("notion-match-map-store", "data"),
        Output("notion-coverage-store", "data"),
        Output("notion-page-detail", "children"),
        Output("notion-databases-store", "data"),
        Input("notion-fetch-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def fetch_notion_recordings(n_clicks):
        """Fetch recordings from Notion, compute matches and coverage."""
        if not n_clicks:
            raise PreventUpdate

        import concurrent.futures
        from app_v2.services.xray import xray_log

        invalidate_notion_cache()
        xray_log("data", "notion", "Refreshing Notion data...")

        def _do_fetch():
            from app_v2.services.xray import xray_timer
            from src.notion_service import get_notion_service

            svc = get_notion_service()

            # Always discover available databases
            databases_data = []
            try:
                databases = svc.list_databases()
                databases_data = databases
                xray_log("data", "notion", f"Found {len(databases)} accessible databases")
            except Exception as e:
                logger.warning(f"Database discovery failed (non-fatal): {e}")

            # Check connection first
            with xray_timer("data", "notion", "Checking Notion connection"):
                status = svc.check_connection()

            status_dict = {
                "connected": status.connected,
                "database_found": status.database_found,
                "database_title": status.database_title,
                "total_pages": status.total_pages,
                "error": status.error,
                "schema": status.schema,
            }

            if not status.connected:
                xray_log("data", "notion", f"Notion connection failed: {status.error}", level="warn")
                return [], status_dict, {}, [], _build_error_message(status.error), databases_data

            # Fetch recordings
            with xray_timer("data", "notion", f"Pulling recordings from '{status.database_title}'"):
                recordings = svc.fetch_recordings(limit=500)

            xray_log("data", "notion", f"Found {len(recordings)} recordings in Notion")

            # Serialize recordings for the store
            recordings_data = []
            for rec in recordings:
                recordings_data.append({
                    "page_id": rec.page_id,
                    "title": rec.title,
                    "created_time": rec.created_time,
                    "last_edited_time": rec.last_edited_time,
                    "url": rec.url,
                    "transcript": rec.transcript,
                    "summary": rec.summary,
                    "date": rec.date,
                    "duration": rec.duration,
                    "tags": rec.tags,
                    "category": rec.category,
                    "source": rec.source,
                    "properties": rec.properties,
                })

            # Smart matching
            match_map = {}
            coverage = []
            try:
                from src.chronos.notion_bridge import match_notion_to_chronos, get_coverage_calendar
                from src.database.engine import SessionLocal

                with xray_timer("data", "notion-match", "Matching Notion pages to Chronos recordings"):
                    db = SessionLocal()
                    try:
                        match_map = match_notion_to_chronos(recordings, db)
                        matched = sum(1 for v in match_map.values() if v)
                        xray_log("data", "notion-match",
                                 f"Smart match: {matched} linked, {len(match_map) - matched} unique to Notion")

                        coverage = get_coverage_calendar(db, days=90)
                    finally:
                        db.close()
            except Exception as e:
                logger.warning(f"Smart matching failed (non-fatal): {e}")

            return recordings_data, status_dict, match_map, coverage, [], databases_data

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_fetch)
                return future.result(timeout=45)
        except concurrent.futures.TimeoutError:
            logger.warning("Notion fetch timed out after 45s")
            xray_log(
                "data",
                "notion",
                "Notion API timed out — try again in a moment",
                level="warn",
            )
            return (
                [],
                {"connected": False, "error": "Request timed out"},
                {},
                [],
                _build_error_message(
                    "Notion API timed out after 45 seconds. Try again."
                ),
                [],
            )
        except Exception as e:
            logger.error(f"Error fetching Notion recordings: {e}")
            xray_log("data", "notion", f"Error connecting to Notion: {e}", level="error")
            return [], {"connected": False, "error": str(e)}, {}, [], _build_error_message(str(e)), []

    # ── Re-render view when data changes ──────────────────────────
    @app.callback(
        Output("content-container", "children", allow_duplicate=True),
        Input("notion-recordings-store", "data"),
        Input("notion-status-store", "data"),
        Input("notion-match-map-store", "data"),
        Input("notion-coverage-store", "data"),
        Input("notion-databases-store", "data"),
        State("current-view", "data"),
        prevent_initial_call=True,
    )
    def refresh_notion_view(recordings_data, status_data, match_map, coverage, databases_data, current_view):
        """Re-render the Notion view when data is fetched."""
        if current_view != "notion":
            raise PreventUpdate

        from app_v2.components.notion import create_notion_view
        from app_v2.services.data_service import get_data_service

        # Get Chronos recording IDs for interplay comparison
        chronos_ids = set()
        try:
            service = get_data_service()
            days = service.get_days()
            for day in days:
                for rec in day.recordings:
                    if rec.recording_id:
                        chronos_ids.add(rec.recording_id)
        except Exception as e:
            logger.warning(f"Could not fetch Chronos IDs: {e}")

        return create_notion_view(
            status=status_data,
            recordings=recordings_data,
            chronos_recording_ids=chronos_ids,
            match_map=match_map or {},
            coverage_calendar=coverage or [],
            databases=databases_data or [],
        )

    # ── Show page detail on click ─────────────────────────────────
    @app.callback(
        Output("notion-page-detail", "children", allow_duplicate=True),
        Input({"type": "notion-rec-click", "page_id": __import__("dash").ALL}, "n_clicks"),
        State("notion-recordings-store", "data"),
        State("notion-match-map-store", "data"),
        prevent_initial_call=True,
    )
    def show_notion_page_detail(n_clicks_list, recordings_data, match_map):
        """Show detail panel when a Notion recording is clicked."""
        if not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            raise PreventUpdate

        page_id = triggered.get("page_id")
        if not page_id:
            raise PreventUpdate

        from app_v2.services.xray import xray_log, xray_timer
        from app_v2.components.notion import create_notion_page_detail

        # Find the recording in our stored data
        rec = None
        for r in (recordings_data or []):
            if r.get("page_id") == page_id:
                rec = r
                break

        if not rec:
            raise PreventUpdate

        # Fetch page body content
        body_text = ""
        try:
            from src.notion_service import get_notion_service
            svc = get_notion_service()
            with xray_timer("data", "notion", f"Reading page content"):
                body_text = svc.fetch_page_content(page_id)
            xray_log("data", "notion", f"Loaded content for '{rec.get('title', 'Untitled')}'")
        except Exception as e:
            logger.warning(f"Could not fetch page content: {e}")

        in_chronos = bool((match_map or {}).get(page_id))
        matched_recording_id = (match_map or {}).get(page_id, "") or ""
        return create_notion_page_detail(
            rec,
            body_text=body_text,
            in_chronos=in_chronos,
            matched_recording_id=matched_recording_id,
        )

    # ── Import single recording to Chronos ────────────────────────
    @app.callback(
        Output("notion-import-progress", "children", allow_duplicate=True),
        Input({"type": "notion-import-one", "page_id": __import__("dash").ALL}, "n_clicks"),
        State("notion-recordings-store", "data"),
        prevent_initial_call=True,
    )
    def import_one_to_chronos(n_clicks_list, recordings_data):
        """Import a single Notion recording into Chronos pipeline."""
        if not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            raise PreventUpdate

        page_id = triggered.get("page_id")
        if not page_id:
            raise PreventUpdate

        from app_v2.services.xray import xray_log
        from src.database.engine import SessionLocal

        # Find title for display
        title = "Unknown"
        for r in (recordings_data or []):
            if r.get("page_id") == page_id:
                title = r.get("title", "Unknown")
                break

        xray_log("pipeline", "notion-import", f"Importing '{title}' to Chronos...")

        try:
            from src.chronos.notion_bridge import import_notion_recording

            db = SessionLocal()
            try:
                ok, msg = import_notion_recording(page_id, db, process=True, index=True)
            finally:
                db.close()

            invalidate_notion_cache()
            if ok:
                return html.Div(
                    className="notion-import-result notion-import-success",
                    children=[
                        html.Span("✅ "),
                        html.Span(msg),
                        html.Button(
                            "View in Timeline →",
                            id="notion-goto-timeline",
                            className="notion-goto-timeline-btn",
                            n_clicks=0,
                        ),
                    ],
                )
            else:
                return html.Div(
                    className="notion-import-result notion-import-error",
                    children=[
                        html.Span("❌ "),
                        html.Span(msg),
                    ],
                )
        except Exception as e:
            logger.error(f"Import failed: {e}", exc_info=True)
            return html.Div(
                className="notion-import-result notion-import-error",
                children=[html.Span(f"❌ Import error: {str(e)}")],
            )

    # ── Import ALL unmatched recordings (background thread) ─────
    @app.callback(
        [
            Output("notion-import-progress", "children", allow_duplicate=True),
            Output("notion-import-poll", "disabled", allow_duplicate=True),
        ],
        Input("notion-import-all-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def import_all_to_chronos(n_clicks):
        """Launch batch import of all unmatched Notion recordings in a background thread."""
        if not n_clicks:
            raise PreventUpdate

        import threading
        from app_v2.services.xray import xray_log
        from src.chronos.notion_bridge import get_import_progress

        # Check if already running
        progress = get_import_progress()
        if progress and progress.get("status") == "running":
            return _build_progress_display(progress), False

        xray_log("pipeline", "notion-import", "Starting batch import of all Notion-only recordings...")

        def _run_import():
            from src.database.engine import SessionLocal
            from src.chronos.notion_bridge import import_all_unmatched

            db = SessionLocal()
            try:
                import_all_unmatched(db, process=True, index=True)
            except Exception as e:
                logger.error(f"Background import failed: {e}", exc_info=True)
            finally:
                db.close()

        t = threading.Thread(target=_run_import, daemon=True, name="notion-import")
        t.start()

        # Return initial progress display + enable polling
        return (
            html.Div(
                className="notion-import-result notion-import-running",
                children=[
                    html.Span(
                        "🔄 Import started — processing recordings through Gemini..."
                    ),
                    html.Div(
                        className="notion-progress-bar-container",
                        children=[
                            html.Div(
                                className="notion-progress-bar", style={"width": "0%"}
                            ),
                        ],
                    ),
                    html.Span("Preparing...", className="notion-progress-label"),
                ],
            ),
            False,
        )  # False = enable the poll interval

    # ── Resume import (retry failed + continue remaining) ────────
    @app.callback(
        [
            Output("notion-import-progress", "children", allow_duplicate=True),
            Output("notion-import-poll", "disabled", allow_duplicate=True),
        ],
        Input("notion-import-resume-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def resume_import(n_clicks):
        """Resume a previously interrupted or partially failed import."""
        if not n_clicks:
            raise PreventUpdate

        import threading
        from app_v2.services.xray import xray_log
        from src.chronos.notion_bridge import get_import_progress, _clear_progress

        # Clear old progress to allow a fresh run
        _clear_progress()
        xray_log(
            "pipeline",
            "notion-import",
            "Resuming import — retrying failed + remaining recordings...",
        )

        def _run_import():
            from src.database.engine import SessionLocal
            from src.chronos.notion_bridge import import_all_unmatched

            db = SessionLocal()
            try:
                import_all_unmatched(db, process=True, index=True)
            except Exception as e:
                logger.error(f"Background resume import failed: {e}", exc_info=True)
            finally:
                db.close()

        t = threading.Thread(
            target=_run_import, daemon=True, name="notion-import-resume"
        )
        t.start()

        return (
            html.Div(
                className="notion-import-result notion-import-running",
                children=[
                    html.Span("🔄 Resuming import — retrying failed and remaining..."),
                    html.Div(
                        className="notion-progress-bar-container",
                        children=[
                            html.Div(
                                className="notion-progress-bar", style={"width": "0%"}
                            ),
                        ],
                    ),
                    html.Span("Preparing...", className="notion-progress-label"),
                ],
            ),
            False,
        )

    # ── Poll import progress ──────────────────────────────────────
    @app.callback(
        [
            Output("notion-import-progress", "children", allow_duplicate=True),
            Output("notion-import-poll", "disabled", allow_duplicate=True),
        ],
        Input("notion-import-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_import_progress(n_intervals):
        """Poll the batch import progress file and update the UI."""
        from src.chronos.notion_bridge import get_import_progress

        progress = get_import_progress()
        if not progress:
            raise PreventUpdate

        status = progress.get("status", "unknown")

        if status == "running":
            return _build_progress_display(progress), False  # keep polling

        elif status == "done":
            invalidate_notion_cache()
            completed = progress.get("completed", 0)
            failed = progress.get("failed", 0)
            total = progress.get("total", 0)
            errors = progress.get("errors", [])

            children = [
                html.Div(
                    className=(
                        "notion-import-result notion-import-success"
                        if completed
                        else "notion-import-result notion-import-error"
                    ),
                    children=[
                        html.Span(f"Batch import complete: "),
                        html.Strong(f"{completed} succeeded"),
                        html.Span(f", {failed} failed") if failed else None,
                        html.Span(f" out of {total}"),
                    ],
                ),
            ]
            if failed > 0:
                children.append(
                    html.Button(
                        f"🔄 Resume — Retry {failed} Failed",
                        id="notion-import-resume-btn",
                        className="sync-action-btn notion-resume-btn",
                        n_clicks=0,
                    )
                )
            if errors:
                children.append(
                    html.Div(
                        className="notion-import-errors",
                        children=[html.P(e, className="notion-muted") for e in errors[:5]],
                    )
                )
            if completed:
                children.append(
                    html.Button(
                        "View in Timeline →",
                        id="notion-goto-timeline",
                        className="notion-goto-timeline-btn",
                        n_clicks=0,
                    )
                )

            return html.Div(children=children), True  # stop polling

        raise PreventUpdate

    # ── Write-back enrichments to Notion ──────────────────────────
    @app.callback(
        Output("notion-import-progress", "children", allow_duplicate=True),
        Input({"type": "notion-writeback", "page_id": __import__("dash").ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def write_back_to_notion(n_clicks_list):
        """Push Chronos AI enrichments back to a Notion page."""
        if not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            raise PreventUpdate

        page_id = triggered.get("page_id")
        if not page_id:
            raise PreventUpdate

        from app_v2.services.xray import xray_log
        from src.database.engine import SessionLocal

        xray_log("data", "notion-writeback", "Pushing Chronos insights back to Notion...")

        try:
            from src.chronos.notion_bridge import write_back_to_notion as _writeback

            db = SessionLocal()
            try:
                ok, msg = _writeback(page_id, db)
            finally:
                db.close()

            if ok:
                return html.Div(
                    className="notion-import-result notion-import-success",
                    children=[html.Span(f"📤 {msg}")],
                )
            else:
                return html.Div(
                    className="notion-import-result notion-import-error",
                    children=[html.Span(f"⚠️ {msg}")],
                )
        except Exception as e:
            logger.error(f"Write-back failed: {e}", exc_info=True)
            return html.Div(
                className="notion-import-result notion-import-error",
                children=[html.Span(f"❌ Write-back error: {str(e)}")],
            )

    # ── Batch write-back all matched ──────────────────────────────
    @app.callback(
        Output("notion-import-progress", "children", allow_duplicate=True),
        Input("notion-writeback-all-btn", "n_clicks"),
        State("notion-match-map-store", "data"),
        prevent_initial_call=True,
    )
    def write_back_all_to_notion(n_clicks, match_map):
        """Push Chronos AI enrichments back to ALL matched Notion pages."""
        if not n_clicks or not match_map:
            raise PreventUpdate

        from app_v2.services.xray import xray_log
        from src.database.engine import SessionLocal

        matched_count = sum(1 for v in match_map.values() if v)
        xray_log(
            "data",
            "notion-writeback",
            f"Starting batch write-back for {matched_count} matched recordings",
        )

        try:
            from src.chronos.notion_bridge import write_back_all_matched

            db = SessionLocal()
            try:
                success, failed, errors = write_back_all_matched(match_map, db)
            finally:
                db.close()

            children = [
                html.Div(
                    className=(
                        "notion-import-result notion-import-success"
                        if success
                        else "notion-import-result notion-import-error"
                    ),
                    children=[
                        html.Span(f"📤 Batch write-back: "),
                        html.Strong(f"{success} succeeded"),
                        html.Span(f", {failed} failed") if failed else None,
                    ],
                ),
            ]
            if errors:
                children.append(
                    html.Div(
                        className="notion-import-errors",
                        children=[
                            html.P(e, className="notion-muted") for e in errors[:5]
                        ],
                    )
                )
            return html.Div(children=children)
        except Exception as e:
            logger.error(f"Batch write-back failed: {e}", exc_info=True)
            return html.Div(
                className="notion-import-result notion-import-error",
                children=[html.Span(f"❌ Batch write-back error: {str(e)}")],
            )

    # ── Navigate to Timeline after import ────────────────────────
    @app.callback(
        Output("current-view", "data", allow_duplicate=True),
        Output("selected-recording", "data", allow_duplicate=True),
        Input("notion-goto-timeline", "n_clicks"),
        prevent_initial_call=True,
    )
    def goto_timeline_after_import(n_clicks):
        """Navigate to Timeline view after a successful Notion import."""
        if not n_clicks:
            raise PreventUpdate
        from app_v2.services.xray import xray_log
        xray_log("nav", "switch", "Switching to Timeline after Notion import")
        return "timeline", None

    # ── Deep-link: detail panel → Timeline ────────────────────────
    @app.callback(
        Output("current-view", "data", allow_duplicate=True),
        Output("selected-recording", "data", allow_duplicate=True),
        Input("notion-detail-goto-timeline", "n_clicks"),
        State("notion-detail-matched-rec-id", "data"),
        prevent_initial_call=True,
    )
    def goto_timeline_from_detail(n_clicks, recording_id):
        """Navigate to a specific recording in Timeline view."""
        if not n_clicks or not recording_id:
            raise PreventUpdate
        from app_v2.services.xray import xray_log

        xray_log("nav", "switch", f"Deep-linking to matched recording in Timeline")
        return "timeline", recording_id

    # ── Client-side search/filter/sort ───────────────────────────
    @app.callback(
        Output("content-container", "children", allow_duplicate=True),
        Input("notion-search-input", "value"),
        Input("notion-sort-dropdown", "value"),
        Input({"type": "notion-filter-cat", "category": ALL}, "n_clicks"),
        Input("notion-filter-all", "n_clicks"),
        State("notion-recordings-store", "data"),
        State("notion-status-store", "data"),
        State("notion-match-map-store", "data"),
        State("notion-coverage-store", "data"),
        State("notion-databases-store", "data"),
        State("current-view", "data"),
        prevent_initial_call=True,
    )
    def filter_recordings(
        search_text,
        sort_value,
        cat_clicks,
        all_clicks,
        recordings_data,
        status_data,
        match_map,
        coverage,
        databases_data,
        current_view,
    ):
        """Filter and sort recordings based on search, category, and sort."""
        if current_view != "notion" or not recordings_data:
            raise PreventUpdate

        from app_v2.components.notion import create_notion_view
        from app_v2.services.data_service import get_data_service

        filtered = list(recordings_data)

        # Determine active category filter from trigger
        active_category = None
        triggered = ctx.triggered_id
        if isinstance(triggered, dict) and triggered.get("type") == "notion-filter-cat":
            active_category = triggered.get("category")

        # Apply category filter
        if active_category:
            filtered = [
                r
                for r in filtered
                if (r.get("category", "") or "uncategorized") == active_category
            ]

        # Apply search filter
        if search_text:
            q = search_text.lower()
            filtered = [
                r for r in filtered
                if q in r.get("title", "").lower()
                or q in r.get("transcript", "").lower()
                or q in r.get("summary", "").lower()
                or q in r.get("category", "").lower()
                or any(q in t.lower() for t in r.get("tags", []))
            ]

        # Apply sort
        if sort_value == "date-desc":
            filtered.sort(key=lambda r: r.get("date", "") or r.get("created_time", ""), reverse=True)
        elif sort_value == "date-asc":
            filtered.sort(key=lambda r: r.get("date", "") or r.get("created_time", ""))
        elif sort_value == "title-asc":
            filtered.sort(key=lambda r: r.get("title", "").lower())
        elif sort_value == "title-desc":
            filtered.sort(key=lambda r: r.get("title", "").lower(), reverse=True)

        # Get Chronos IDs
        chronos_ids = set()
        try:
            service = get_data_service()
            days = service.get_days()
            for day in days:
                for rec in day.recordings:
                    if rec.recording_id:
                        chronos_ids.add(rec.recording_id)
        except Exception:
            pass

        return create_notion_view(
            status=status_data,
            recordings=filtered,
            chronos_recording_ids=chronos_ids,
            match_map=match_map or {},
            coverage_calendar=coverage or [],
            databases=databases_data or [],
            active_category=active_category,
        )

    # ── Sync Engine: Preview Reformat ────────────────────────────
    @app.callback(
        Output("notion-sync-result", "children", allow_duplicate=True),
        Output("notion-sync-confirm-area", "style", allow_duplicate=True),
        Input("notion-reformat-preview-btn", "n_clicks"),
        State("notion-match-map-store", "data"),
        prevent_initial_call=True,
    )
    def preview_reformat(n_clicks, match_map):
        if not n_clicks:
            raise PreventUpdate

        from src.database.engine import SessionLocal
        from src.chronos.notion_bridge import reformat_all_notion_pages
        from app_v2.services.xray import xray_log

        xray_log("data", "notion-reformat", "Running standardize preview...")

        db = SessionLocal()
        try:
            result = reformat_all_notion_pages(db, match_map or {}, dry_run=True)
        finally:
            db.close()

        total = result.get("total", 0)
        reformatted = result.get("reformatted", 0)
        skipped = result.get("skipped", 0)
        errors = result.get("errors", [])
        changes_by_type = result.get("changes_by_type", {})
        diffs = result.get("diffs", [])

        # Build preview summary
        children = [
            html.Div(
                className="notion-import-result",
                children=[
                    html.H4(
                        f"Standardize Preview — {reformatted} of {total} pages need changes"
                    ),
                    html.P(f"{skipped} pages already normalized, {len(errors)} errors"),
                ],
            ),
        ]

        if changes_by_type:
            children.append(
                html.Div(
                    className="notion-sync-changes-summary",
                    children=[
                        html.Span(f"{k}: {v}", className="notion-sync-change-badge")
                        for k, v in changes_by_type.items()
                    ],
                )
            )

        # Show sample diffs (first 10)
        if diffs:
            diff_items = []
            for d in diffs[:10]:
                items = []
                for prop, val in d.get("changes", {}).items():
                    old = val.get("old") or "—"
                    new = val.get("new") or "—"
                    items.append(html.Li(f"{prop}: {old} → {new}"))
                diff_items.append(
                    html.Div(
                        className="notion-sync-diff-item",
                        children=[
                            html.Span(
                                d.get("page_id", "")[:8] + "…", className="notion-muted"
                            ),
                            html.Ul(items),
                        ],
                    )
                )
            if len(diffs) > 10:
                diff_items.append(
                    html.P(f"… and {len(diffs) - 10} more", className="notion-muted")
                )
            children.append(
                html.Div(className="notion-sync-diff-list", children=diff_items)
            )

        if errors:
            children.append(
                html.Div(
                    className="notion-import-result notion-import-errors",
                    children=[html.P(e) for e in errors[:5]],
                )
            )

        show_confirm = {"display": "flex"} if reformatted > 0 else {"display": "none"}
        return html.Div(children=children), show_confirm

    # ── Sync Engine: Execute Reformat ────────────────────────────
    @app.callback(
        Output("notion-sync-result", "children", allow_duplicate=True),
        Output("notion-sync-confirm-area", "style", allow_duplicate=True),
        Output("notion-sync-poll", "disabled", allow_duplicate=True),
        Input("notion-reformat-execute-btn", "n_clicks"),
        State("notion-match-map-store", "data"),
        prevent_initial_call=True,
    )
    def execute_reformat(n_clicks, match_map):
        if not n_clicks:
            raise PreventUpdate

        import threading
        from src.database.engine import SessionLocal
        from src.chronos.notion_bridge import reformat_all_notion_pages
        from app_v2.services.xray import xray_log

        xray_log(
            "data", "notion-reformat", "Executing standardize on all Notion pages..."
        )

        def _run():
            db = SessionLocal()
            try:
                reformat_all_notion_pages(db, match_map or {}, dry_run=False)
            finally:
                db.close()

        threading.Thread(target=_run, daemon=True).start()

        return (
            html.Div(
                className="notion-import-result notion-import-running",
                children=[
                    html.Span("Standardizing Notion pages… check progress below.")
                ],
            ),
            {"display": "none"},
            False,  # Enable poll
        )

    # ── Sync Engine: Preview Push ────────────────────────────────
    @app.callback(
        Output("notion-sync-result", "children", allow_duplicate=True),
        Output("notion-sync-confirm-area", "style", allow_duplicate=True),
        Input("notion-push-preview-btn", "n_clicks"),
        State("notion-match-map-store", "data"),
        prevent_initial_call=True,
    )
    def preview_push(n_clicks, match_map):
        if not n_clicks:
            raise PreventUpdate

        from src.database.engine import SessionLocal
        from src.chronos.notion_bridge import push_all_chronos_only
        from app_v2.services.xray import xray_log

        xray_log("data", "notion-push", "Running push preview...")

        db = SessionLocal()
        try:
            result = push_all_chronos_only(db, match_map or {}, dry_run=True)
        finally:
            db.close()

        total = result.get("total", 0)
        created = result.get("created", 0)
        errors = result.get("errors", [])
        pages = result.get("pages", [])

        children = [
            html.Div(
                className="notion-import-result",
                children=[
                    html.H4(
                        f"Push Preview — {created} Chronos recordings to create in Notion"
                    ),
                    html.P(f"Out of {total} unmatched Plaud recordings"),
                ],
            ),
        ]

        if pages:
            page_items = []
            for p in pages[:15]:
                props_str = ", ".join(p.get("properties", {}).keys())
                page_items.append(
                    html.Div(
                        className="notion-sync-diff-item",
                        children=[
                            html.Strong(p.get("title", "Untitled")),
                            html.Span(f" — {props_str}", className="notion-muted"),
                        ],
                    )
                )
            if len(pages) > 15:
                page_items.append(
                    html.P(f"… and {len(pages) - 15} more", className="notion-muted")
                )
            children.append(
                html.Div(className="notion-sync-diff-list", children=page_items)
            )

        if errors:
            children.append(
                html.Div(
                    className="notion-import-result notion-import-errors",
                    children=[html.P(e) for e in errors[:5]],
                )
            )

        show_confirm = {"display": "flex"} if created > 0 else {"display": "none"}
        return html.Div(children=children), show_confirm

    # ── Sync Engine: Execute Push ────────────────────────────────
    @app.callback(
        Output("notion-sync-result", "children", allow_duplicate=True),
        Output("notion-sync-confirm-area", "style", allow_duplicate=True),
        Output("notion-sync-poll", "disabled", allow_duplicate=True),
        Input("notion-push-execute-btn", "n_clicks"),
        State("notion-match-map-store", "data"),
        prevent_initial_call=True,
    )
    def execute_push(n_clicks, match_map):
        if not n_clicks:
            raise PreventUpdate

        import threading
        from src.database.engine import SessionLocal
        from src.chronos.notion_bridge import push_all_chronos_only
        from app_v2.services.xray import xray_log

        xray_log("data", "notion-push", "Pushing Chronos recordings to Notion...")

        def _run():
            db = SessionLocal()
            try:
                push_all_chronos_only(db, match_map or {}, dry_run=False)
            finally:
                db.close()

        threading.Thread(target=_run, daemon=True).start()

        return (
            html.Div(
                className="notion-import-result notion-import-running",
                children=[
                    html.Span(
                        "Pushing Chronos recordings to Notion… check progress below."
                    )
                ],
            ),
            {"display": "none"},
            False,  # Enable poll
        )

    # ── Sync Engine: Poll Progress ───────────────────────────────
    @app.callback(
        Output("notion-sync-result", "children", allow_duplicate=True),
        Output("notion-sync-poll", "disabled", allow_duplicate=True),
        Input("notion-sync-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_sync_progress(n_intervals):
        from src.chronos.notion_bridge import get_reformat_progress

        progress = get_reformat_progress()
        if not progress:
            raise PreventUpdate

        status = progress.get("status", "")
        total = progress.get("total", 1)
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        mode = progress.get("mode", "")
        current = progress.get("current_title", "")
        idx = progress.get("current_index", 0)
        done_count = completed + failed
        pct = int((done_count / max(total, 1)) * 100)

        if status == "done":
            errs = progress.get("errors", [])
            result_div = html.Div(
                className="notion-import-result",
                children=[
                    html.H4(
                        f"{'Standardize' if 'run' in mode or 'execute' in mode else 'Push'} Complete"
                    ),
                    html.P(f"{completed} succeeded, {failed} failed out of {total}"),
                ]
                + ([html.P(e) for e in errs[:5]] if errs else []),
            )
            return result_div, True  # Disable poll

        return (
            html.Div(
                className="notion-import-result notion-import-running",
                children=[
                    html.Span(
                        f"{'Standardizing' if 'execute' in mode else 'Pushing'} — {completed} done, {failed} failed"
                    ),
                    html.Div(
                        className="notion-progress-bar-container",
                        children=[
                            html.Div(
                                className="notion-progress-bar",
                                style={"width": f"{pct}%"},
                            )
                        ],
                    ),
                    html.Span(
                        f"[{idx}/{total}] {current}", className="notion-progress-label"
                    ),
                ],
            ),
            False,  # Keep polling
        )


def _do_full_fetch_data(force=False):
    """Fetch all Notion data and return kwargs dict for create_notion_view.

    Uses a 60s TTL cache so tab switching is instant.
    Pass force=True to bypass cache (Refresh button).
    """
    if not force:
        cached = _get_cached_notion_data()
        if cached is not None:
            logger.debug("Notion tab: serving from cache")
            return cached

    import concurrent.futures
    from app_v2.services.xray import xray_log, xray_timer
    from app_v2.services.data_service import get_data_service
    from src.notion_service import get_notion_service

    svc = get_notion_service()

    databases_data = []
    # Only list databases when no DB is configured (avoids slow/hanging API call)
    from src.config import get_settings as _gs

    if not _gs().notion_database_id:
        try:
            databases_data = svc.list_databases()
            xray_log(
                "data", "notion", f"Found {len(databases_data)} accessible databases"
            )
        except Exception:
            pass

    with xray_timer("data", "notion", "Checking Notion connection"):
        status = svc.check_connection(quick=True)

    status_dict = {
        "connected": status.connected,
        "database_found": status.database_found,
        "database_title": status.database_title,
        "total_pages": status.total_pages,  # 0 initially — set from fetch below
        "error": status.error,
        "schema": status.schema,
    }

    recordings_data = []
    match_map = {}
    coverage = []
    stale_map = {}
    chronos_ids = set()

    if status.connected:
        with xray_timer(
            "data", "notion", f"Pulling recordings from '{status.database_title}'"
        ):
            recordings = svc.fetch_recordings(limit=500)

        xray_log("data", "notion", f"Found {len(recordings)} recordings in Notion")

        # Set total_pages from actual fetch (skipped expensive count in quick mode)
        status_dict["total_pages"] = len(recordings)

        for rec in recordings:
            recordings_data.append(
                {
                    "page_id": rec.page_id,
                    "title": rec.title,
                    "created_time": rec.created_time,
                    "last_edited_time": rec.last_edited_time,
                    "url": rec.url,
                    "transcript": rec.transcript,
                    "summary": rec.summary,
                    "date": rec.date,
                    "duration": rec.duration,
                    "tags": rec.tags,
                    "category": rec.category,
                    "source": rec.source,
                    "properties": rec.properties,
                }
            )

        # Run matching + coverage + chronos IDs in parallel
        def _do_matching():
            _match_map, _coverage, _stale_map = {}, [], {}
            try:
                from src.chronos.notion_bridge import (
                    match_notion_to_chronos,
                    get_coverage_calendar,
                    detect_stale_imports,
                )
                from src.database.engine import SessionLocal

                db = SessionLocal()
                try:
                    with xray_timer(
                        "data",
                        "notion-match",
                        "Matching Notion pages to Chronos recordings",
                    ):
                        _match_map = match_notion_to_chronos(recordings, db)
                        matched = sum(1 for v in _match_map.values() if v)
                        xray_log(
                            "data",
                            "notion-match",
                            f"Smart match: {matched} linked, {len(_match_map) - matched} unique to Notion",
                        )
                    _coverage = get_coverage_calendar(
                        db, days=90, notion_recordings=recordings
                    )
                    _stale_map = detect_stale_imports(recordings, _match_map, db)
                    if _stale_map:
                        xray_log(
                            "data",
                            "notion-match",
                            f"{len(_stale_map)} recordings edited in Notion since import",
                        )
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Smart matching failed (non-fatal): {e}")
            return _match_map, _coverage, _stale_map

        def _do_chronos_ids():
            _ids = set()
            try:
                service = get_data_service()
                days = service.get_days()
                for day in days:
                    for rec in day.recordings:
                        if rec.recording_id:
                            _ids.add(rec.recording_id)
            except Exception:
                pass
            return _ids

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            match_future = pool.submit(_do_matching)
            ids_future = pool.submit(_do_chronos_ids)
            match_map, coverage, stale_map = match_future.result(timeout=30)
            chronos_ids = ids_future.result(timeout=10)

        xray_log(
            "data",
            "notion-match",
            f"Matched {sum(1 for v in match_map.values() if v)} of {len(match_map)} Notion pages to Chronos recordings",
        )
    else:
        # Not connected — still get chronos IDs for the view
        try:
            service = get_data_service()
            days = service.get_days()
            for day in days:
                for rec in day.recordings:
                    if rec.recording_id:
                        chronos_ids.add(rec.recording_id)
        except Exception:
            pass

    result = {
        "status": status_dict,
        "recordings": recordings_data,
        "chronos_recording_ids": chronos_ids,
        "match_map": match_map,
        "coverage_calendar": coverage,
        "databases": databases_data,
        "stale_map": stale_map,
    }
    _set_notion_cache(result)
    return result


def _do_full_fetch():
    """Perform a full Notion fetch and return a rendered view."""
    from app_v2.services.xray import xray_log, xray_timer
    from app_v2.components.notion import create_notion_view
    from app_v2.services.data_service import get_data_service
    from src.notion_service import get_notion_service

    svc = get_notion_service()

    # Discover databases
    databases_data = []
    try:
        databases_data = svc.list_databases()
    except Exception:
        pass

    # Check connection
    status = svc.check_connection()
    status_dict = {
        "connected": status.connected,
        "database_found": status.database_found,
        "database_title": status.database_title,
        "total_pages": status.total_pages,
        "error": status.error,
        "schema": status.schema,
    }

    recordings_data = []
    match_map = {}
    coverage = []
    stale_map = {}
    chronos_ids = set()

    if status.connected:
        # Fetch recordings
        recordings = svc.fetch_recordings(limit=500)
        for rec in recordings:
            recordings_data.append({
                "page_id": rec.page_id,
                "title": rec.title,
                "created_time": rec.created_time,
                "last_edited_time": rec.last_edited_time,
                "url": rec.url,
                "transcript": rec.transcript,
                "summary": rec.summary,
                "date": rec.date,
                "duration": rec.duration,
                "tags": rec.tags,
                "category": rec.category,
                "source": rec.source,
                "properties": rec.properties,
            })

        try:
            from src.chronos.notion_bridge import (
                match_notion_to_chronos,
                get_coverage_calendar,
                detect_stale_imports,
            )
            from src.database.engine import SessionLocal

            db = SessionLocal()
            try:
                match_map = match_notion_to_chronos(recordings, db)
                coverage = get_coverage_calendar(db, days=90)
                stale_map = detect_stale_imports(recordings, match_map, db)
            finally:
                db.close()
        except Exception:
            pass

        xray_log("data", "notion", f"Loaded {len(recordings_data)} recordings from '{status.database_title}'")

    # Get Chronos IDs
    try:
        service = get_data_service()
        days = service.get_days()
        for day in days:
            for rec in day.recordings:
                if rec.recording_id:
                    chronos_ids.add(rec.recording_id)
    except Exception:
        pass

    return create_notion_view(
        status=status_dict,
        recordings=recordings_data,
        chronos_recording_ids=chronos_ids,
        match_map=match_map,
        coverage_calendar=coverage,
        databases=databases_data,
        stale_map=stale_map,
    )


def _build_error_message(error: str) -> html.Div:
    """Build an error message component."""
    return html.Div(
        className="notion-error-panel",
        children=[
            html.Span("⚠️ ", className="error-icon"),
            html.Span(f"Error: {error}"),
            html.Div(
                className="notion-error-help",
                children=[
                    html.P("Make sure you have:"),
                    html.Ol([
                        html.Li("Created a Notion integration at notion.so/profile/integrations"),
                        html.Li("Added NOTION_TOKEN=ntn_xxx to your .env file"),
                        html.Li("Added NOTION_DATABASE_ID=your-database-id to your .env file"),
                        html.Li("Shared your database with the integration"),
                    ]),
                ],
            ),
        ],
    )
