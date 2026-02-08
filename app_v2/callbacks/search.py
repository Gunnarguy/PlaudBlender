"""Search callbacks."""

from dash import Input, Output, State, html, no_update, ALL, ctx
from dash.exceptions import PreventUpdate
import logging

from app_v2.services import get_data_service
from app_v2.components import create_search_results

logger = logging.getLogger(__name__)


def register_search_callbacks(app):
    """Register search-related callbacks."""

    @app.callback(
        Output("search-filters", "style"),
        Input("toggle-filters-btn", "n_clicks"),
        State("search-filters", "style"),
        prevent_initial_call=True,
    )
    def toggle_search_filters(n_clicks, current_style):
        """Toggle search filter visibility."""
        if not n_clicks:
            raise PreventUpdate
        current = (current_style or {}).get("display", "none")
        new_display = "none" if current == "flex" else "flex"
        return {"display": new_display}

    @app.callback(
        Output("search-query", "data"),
        Output("search-results-container", "children"),
        Output("search-results-container", "style"),
        Input("search-input", "n_submit"),
        Input("search-btn", "n_clicks"),
        State("search-input", "value"),
        State("filter-category", "value"),
        State("filter-date-range", "start_date"),
        State("filter-date-range", "end_date"),
        prevent_initial_call=True,
    )
    def perform_search(n_submit, n_clicks, query, categories, start_date, end_date):
        """Handle search submission with filters."""
        if not query or (not n_submit and not n_clicks):
            raise PreventUpdate

        query = query.strip()
        if not query:
            return None, [], {"display": "none"}

        logger.info(
            f"Search query: {query}, categories={categories}, dates={start_date}-{end_date}"
        )

        # Perform filtered search
        service = get_data_service()
        results = service.search(
            query,
            limit=20,
            categories=categories or None,
            start_date=start_date[:10] if start_date else None,
            end_date=end_date[:10] if end_date else None,
        )

        if not results:
            return (
                query,
                [
                    html.Div(
                        [
                            html.P(
                                f"No results found for '{query}'",
                                className="no-results",
                            ),
                            html.P(
                                "Try different keywords or check spelling.",
                                className="no-results-hint",
                            ),
                        ],
                        className="no-results-container",
                    )
                ],
                {"display": "block"},
            )

        # Create results UI
        results_ui = [create_search_results(results, query)]

        return query, results_ui, {"display": "block"}

    @app.callback(
        Output("search-results-container", "style", allow_duplicate=True),
        Output("search-input", "value"),
        Output("clear-search-btn", "style"),
        Input("clear-search-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_search(n_clicks):
        """Clear search results."""
        if not n_clicks:
            raise PreventUpdate

        return {"display": "none"}, "", {"display": "none"}

    @app.callback(
        Output("selected-recording", "data", allow_duplicate=True),
        Input(
            {"type": "search-result", "recording_id": ALL, "event_id": ALL}, "n_clicks"
        ),
        State({"type": "search-result", "recording_id": ALL, "event_id": ALL}, "id"),
        prevent_initial_call=True,
    )
    def select_search_result(n_clicks, result_ids):
        """Handle search result click to open recording."""
        if not any(n_clicks):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered:
            raise PreventUpdate

        recording_id = triggered.get("recording_id")
        event_id = triggered.get("event_id")

        logger.info(
            f"Search result selected: recording={recording_id}, event={event_id}"
        )

        # Return recording info with optional scroll-to-event hint
        return {"id": recording_id, "scroll_to_event": event_id}
