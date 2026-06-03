"""Day view callbacks - day expansion and recording selection."""

from dash import Input, Output, State, ctx, html, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import logging

from app_v2.services import get_data_service
from app_v2.components import create_recording_card
from app_v2.components.topics import create_topic_card

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
        from app_v2.services.xray import xray_log

        xray_log(
            "day", "select", f"You picked a recording from {date}"
        )

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

    @app.callback(
        Output("topics-grid-container", "children"),
        Input("topic-sort-select", "value"),
        prevent_initial_call=True,
    )
    def sort_topics(sort_value):
        """Re-sort topics grid based on dropdown selection."""
        if not sort_value:
            raise PreventUpdate

        service = get_data_service()
        topics = service.get_all_topics()

        if sort_value == "freq-desc":
            topics.sort(key=lambda x: -x[1])
        elif sort_value == "freq-asc":
            topics.sort(key=lambda x: x[1])
        elif sort_value == "alpha-asc":
            topics.sort(key=lambda x: x[0].lower())
        elif sort_value == "alpha-desc":
            topics.sort(key=lambda x: x[0].lower(), reverse=True)

        return [create_topic_card(t, c) for t, c in topics[:100]]

    # ── Timeline range filter → re-render day cards ──────────────
    @app.callback(
        Output("days-list", "children"),
        Input("timeline-range-select", "value"),
        prevent_initial_call=True,
    )
    def filter_timeline_range(range_days):
        """Filter day cards by selected time range."""
        from datetime import datetime, timedelta
        from app_v2.components.day_view import create_day_card
        from app_v2.services.xray import xray_log

        service = get_data_service()
        days = service.get_days()

        if range_days and range_days > 0:
            cutoff = (datetime.now() - timedelta(days=range_days)).strftime("%Y-%m-%d")
            days = [d for d in days if d.date >= cutoff]
            xray_log(
                "nav",
                "filter",
                f"Showing last {range_days} days ({len(days)} with recordings)",
            )
        else:
            xray_log("nav", "filter", f"Showing all time ({len(days)} days)")

        days = [
            day
            for day in days
            if day.recording_count > 0 or day.event_count > 0 or bool(day.recordings)
        ]

        # Enforce chronological ordering (newest first)
        days.sort(key=lambda d: d.date, reverse=True)

        if not days:
            return [
                html.Div("No recordings in this range", className="empty-state-text")
            ]

        return [create_day_card(day, expanded=(i == 0)) for i, day in enumerate(days)]

    # ── Heatmap cell click → scroll to matching day card ─────────
    app.clientside_callback(
        """
        function() {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered || !ctx.triggered.length)
                return window.dash_clientside.no_update;

            var t = ctx.triggered[0];
            if (!t.value) return window.dash_clientside.no_update;

            try {
                var propId = JSON.parse(t.prop_id.split('.')[0]);
                var targetDate = propId.date;
            } catch(e) {
                return window.dash_clientside.no_update;
            }

            // Find the day-header with matching date
            var headers = document.querySelectorAll('.day-header');
            for (var i = 0; i < headers.length; i++) {
                try {
                    var hid = JSON.parse(headers[i].id);
                    if (hid.type === 'day-header' && hid.date === targetDate) {
                        var card = headers[i].closest('.day-card');
                        if (card) {
                            card.scrollIntoView({behavior: 'smooth', block: 'start'});
                            // Expand if collapsed
                            var recs = card.querySelector('.day-recordings');
                            if (recs && recs.style.display === 'none') {
                                headers[i].click();
                            }
                            // Brief highlight pulse
                            card.classList.add('heatmap-highlight');
                            setTimeout(function() {
                                card.classList.remove('heatmap-highlight');
                            }, 2000);
                        }
                        return targetDate;
                    }
                } catch(e) {}
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("heatmap-scroll-target", "data"),
        Input({"type": "heatmap-cell", "date": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
