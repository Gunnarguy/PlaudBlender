"""Day view callbacks - day expansion and recording selection."""

from dash import Input, Output, State, ctx, html, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import logging

from app_v2.services import get_data_service
from app_v2.components import create_recording_card

logger = logging.getLogger(__name__)


def register_day_view_callbacks(app):
    """Register day view callbacks."""

    @app.callback(
        Output({"type": "day-recordings", "date": MATCH}, "style"),
        Output({"type": "day-header", "date": MATCH}, "children"),
        Input({"type": "day-header", "date": MATCH}, "n_clicks"),
        State({"type": "day-recordings", "date": MATCH}, "style"),
        State({"type": "day-header", "date": MATCH}, "children"),
        prevent_initial_call=True,
    )
    def toggle_day_expansion(n_clicks, current_style, current_children):
        """Toggle day card expansion."""
        if not n_clicks:
            raise PreventUpdate

        # Toggle visibility
        is_visible = current_style.get("display") == "block"
        new_style = {"display": "none" if is_visible else "block"}

        # Update expand icon in header
        # Find and update the expand icon
        new_children = []
        for child in current_children:
            if hasattr(child, "className") and child.className == "expand-icon":
                new_children.append(
                    html.Span(
                        "▼" if not is_visible else "▶",
                        className="expand-icon",
                    )
                )
            else:
                new_children.append(child)

        return new_style, new_children

    @app.callback(
        Output("selected-recording", "data"),
        Input({"type": "recording-card", "id": ALL, "date": ALL}, "n_clicks"),
        State({"type": "recording-card", "id": ALL, "date": ALL}, "id"),
        prevent_initial_call=True,
    )
    def select_recording(n_clicks, card_ids):
        """Handle recording card click."""
        if not any(n_clicks):
            raise PreventUpdate

        # Find which card was clicked
        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        recording_id = triggered.get("id")
        date = triggered.get("date")

        logger.info(f"Recording selected: {recording_id} from {date}")

        return {"id": recording_id, "date": date}

    @app.callback(
        Output("selected-recording", "data", allow_duplicate=True),
        Input({"type": "back-btn", "date": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def close_recording_detail(n_clicks):
        """Handle back button click to close recording detail."""
        if not any(n_clicks):
            raise PreventUpdate

        return None

    @app.callback(
        Output("selected-topic", "data"),
        Input({"type": "topic-card", "topic": ALL}, "n_clicks"),
        State({"type": "topic-card", "topic": ALL}, "id"),
        prevent_initial_call=True,
    )
    def select_topic(n_clicks, card_ids):
        """Handle topic card click."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        topic = triggered.get("topic")
        logger.info(f"Topic selected: {topic}")

        return topic

    @app.callback(
        Output("selected-recording", "data", allow_duplicate=True),
        Input({"type": "occurrence-card", "id": ALL, "recording_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select_occurrence(n_clicks):
        """Handle occurrence card click in topic timeline."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        recording_id = triggered.get("recording_id")
        event_id = triggered.get("id")
        logger.info(f"Occurrence selected: event={event_id} recording={recording_id}")

        return {"id": recording_id, "scroll_to_event": event_id}

    @app.callback(
        Output("selected-topic", "data", allow_duplicate=True),
        Input("back-to-topics-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_topic_detail(n_clicks):
        """Handle back button to return to topics grid."""
        if not n_clicks:
            raise PreventUpdate

        return None
