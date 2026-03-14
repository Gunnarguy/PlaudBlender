"""Notion integration callbacks.

Handles fetching recordings from Notion, displaying page details,
and coordinating with the Chronos data service.
"""

import logging
from dash import Input, Output, State, callback_context as ctx, html, no_update
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)


def register_notion_callbacks(app):
    """Register all Notion-related callbacks."""

    @app.callback(
        Output("notion-recordings-store", "data"),
        Output("notion-status-store", "data"),
        Output("notion-page-detail", "children"),
        Input("notion-fetch-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def fetch_notion_recordings(n_clicks):
        """Fetch recordings from Notion when button is clicked."""
        if not n_clicks:
            raise PreventUpdate

        from app_v2.services.xray import xray_log, xray_timer

        xray_log("data", "notion", "Connecting to Notion...")

        try:
            from src.notion_service import get_notion_service

            svc = get_notion_service()

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
                return [], status_dict, _build_error_message(status.error)

            # Fetch recordings
            with xray_timer("data", "notion", f"Pulling recordings from '{status.database_title}'"):
                recordings = svc.fetch_recordings(limit=500)

            xray_log(
                "data", "notion",
                f"Found {len(recordings)} recordings in Notion"
            )

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

            return recordings_data, status_dict, []

        except Exception as e:
            logger.error(f"Error fetching Notion recordings: {e}")
            xray_log("data", "notion", f"Error connecting to Notion: {e}", level="error")
            return [], {"connected": False, "error": str(e)}, _build_error_message(str(e))

    @app.callback(
        Output("content-container", "children", allow_duplicate=True),
        Input("notion-recordings-store", "data"),
        Input("notion-status-store", "data"),
        State("current-view", "data"),
        prevent_initial_call=True,
    )
    def refresh_notion_view(recordings_data, status_data, current_view):
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
        )

    @app.callback(
        Output("notion-page-detail", "children", allow_duplicate=True),
        Input({"type": "notion-rec-click", "page_id": __import__("dash").ALL}, "n_clicks"),
        State("notion-recordings-store", "data"),
        prevent_initial_call=True,
    )
    def show_notion_page_detail(n_clicks_list, recordings_data):
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

        return create_notion_page_detail(rec, body_text=body_text)


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
