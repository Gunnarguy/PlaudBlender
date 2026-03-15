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
from dash import Input, Output, State, callback_context as ctx, html, dcc, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)


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

        from app_v2.services.xray import xray_log, xray_timer

        xray_log("data", "notion", "Connecting to Notion...")

        try:
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
        return create_notion_page_detail(rec, body_text=body_text, in_chronos=in_chronos)

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

    # ── Import ALL unmatched recordings ───────────────────────────
    @app.callback(
        Output("notion-import-progress", "children", allow_duplicate=True),
        Input("notion-import-all-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def import_all_to_chronos(n_clicks):
        """Import all unmatched Notion recordings to Chronos."""
        if not n_clicks:
            raise PreventUpdate

        from app_v2.services.xray import xray_log
        from src.database.engine import SessionLocal

        xray_log("pipeline", "notion-import", "Starting batch import of all Notion-only recordings...")

        try:
            from src.chronos.notion_bridge import import_all_unmatched

            db = SessionLocal()
            try:
                successes, failures, errors = import_all_unmatched(
                    db, process=True, index=True,
                )
            finally:
                db.close()

            children = [
                html.Div(
                    className="notion-import-result notion-import-success" if successes else "notion-import-result notion-import-error",
                    children=[
                        html.Span(f"Batch import complete: "),
                        html.Strong(f"{successes} succeeded"),
                        html.Span(f", {failures} failed") if failures else None,
                        html.Button(
                            "View in Timeline →",
                            id="notion-goto-timeline",
                            className="notion-goto-timeline-btn",
                            n_clicks=0,
                        ) if successes else None,
                    ],
                ),
            ]
            if errors:
                children.append(
                    html.Div(
                        className="notion-import-errors",
                        children=[html.P(e, className="notion-muted") for e in errors[:5]],
                    )
                )

            return html.Div(children=children)

        except Exception as e:
            logger.error(f"Batch import failed: {e}", exc_info=True)
            return html.Div(
                className="notion-import-result notion-import-error",
                children=[html.Span(f"❌ Batch import error: {str(e)}")],
            )

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

    # ── Client-side search/filter/sort ───────────────────────────
    @app.callback(
        Output("content-container", "children", allow_duplicate=True),
        Input("notion-search-input", "value"),
        Input("notion-sort-dropdown", "value"),
        State("notion-recordings-store", "data"),
        State("notion-status-store", "data"),
        State("notion-match-map-store", "data"),
        State("notion-coverage-store", "data"),
        State("notion-databases-store", "data"),
        State("current-view", "data"),
        prevent_initial_call=True,
    )
    def filter_recordings(search_text, sort_value, recordings_data, status_data,
                          match_map, coverage, databases_data, current_view):
        """Filter and sort recordings based on search input and sort selection."""
        if current_view != "notion" or not recordings_data:
            raise PreventUpdate

        from app_v2.components.notion import create_notion_view
        from app_v2.services.data_service import get_data_service

        filtered = list(recordings_data)

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
        )


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
            from src.chronos.notion_bridge import match_notion_to_chronos, get_coverage_calendar
            from src.database.engine import SessionLocal

            db = SessionLocal()
            try:
                match_map = match_notion_to_chronos(recordings, db)
                coverage = get_coverage_calendar(db, days=90)
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
