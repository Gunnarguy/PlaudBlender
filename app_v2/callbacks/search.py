"""Search callbacks."""

from dash import Input, Output, State, html, dcc, no_update, ALL, ctx
from dash.exceptions import PreventUpdate
import logging
import time

from app_v2.services import get_data_service
from app_v2.components import create_search_results

logger = logging.getLogger(__name__)


def _build_ai_answer_section(question: str, results) -> html.Div:
    """Try OpenAI Responses API for a conversational answer. Returns Div or None."""
    from src.config import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        return None

    try:
        from src.chronos.openai_service import OpenAIResponseService

        svc = OpenAIResponseService()
        if not svc.available:
            return None

        context_events = []
        for r in results:
            e = r.event
            context_events.append(
                {
                    "date": e.start_ts.strftime("%Y-%m-%d"),
                    "time": f"{e.start_ts.strftime('%H:%M')}-{e.end_ts.strftime('%H:%M')}",
                    "category": e.category,
                    "text": e.clean_text,
                }
            )

        t0 = time.time()
        result = svc.ask(question, context_events)
        latency_ms = int((time.time() - t0) * 1000)

        if "error" in result:
            logger.warning(f"OpenAI ask failed: {result['error']}")
            return None

        usage = result.get("usage", {})
        model = result.get("model", settings.openai_model)

        return html.Div(
            className="ai-answer-section",
            children=[
                html.Div(
                    className="ai-answer-header",
                    children=[
                        html.Span("🧠", className="ai-icon"),
                        html.Span("Chronos AI", className="ai-label"),
                        html.Span(
                            f"{model} · {latency_ms}ms · {usage.get('total_tokens', '?')} tokens",
                            className="ai-meta",
                        ),
                    ],
                ),
                dcc.Markdown(
                    result["answer"],
                    className="ai-answer-body",
                ),
            ],
        )
    except Exception as exc:
        logger.warning(f"AI answer generation failed: {exc}")
        return None


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

        from app_v2.services.xray import xray_log, xray_timer

        logger.info(
            f"Search query: {query}, categories={categories}, dates={start_date}-{end_date}"
        )
        xray_log(
            "search", "query", f"Query: '{query}'", detail=f"cats={categories or 'all'}"
        )

        # Perform filtered search
        service = get_data_service()
        with xray_timer("search", "vector", "Qdrant semantic search") as t_search:
            results = service.search(
                query,
                limit=20,
                categories=categories or None,
                start_date=start_date[:10] if start_date else None,
                end_date=end_date[:10] if end_date else None,
            )

        xray_log(
            "search",
            "results",
            f"{len(results)} matches",
            duration_ms=round(t_search.ms, 1),
            detail=f"top={results[0].score:.2f}" if results else None,
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
        results_ui = []

        # Add AI conversational answer (if OpenAI configured)
        xray_log("search", "ai", "Generating AI answer...")
        with xray_timer("search", "openai", "OpenAI Responses API") as t_ai:
            ai_section = _build_ai_answer_section(query, results)
        if ai_section:
            results_ui.append(ai_section)
            xray_log(
                "search",
                "ai",
                "AI answer ready",
                duration_ms=round(t_ai.ms, 1),
                level="perf",
            )
        else:
            xray_log(
                "search", "ai", "AI answer skipped (no key or failed)", level="warn"
            )

        results_ui.append(create_search_results(results, query))

        return query, results_ui, {"display": "block"}

    @app.callback(
        Output("search-results-container", "style", allow_duplicate=True),
        Output("search-input", "value"),
        Output("clear-search-btn", "style"),
        Output("filter-category", "value"),
        Output("filter-date-range", "start_date"),
        Output("filter-date-range", "end_date"),
        Output("search-filters", "style", allow_duplicate=True),
        Input("clear-search-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_search(n_clicks):
        """Clear search results and reset all filters."""
        if not n_clicks:
            raise PreventUpdate

        return (
            {"display": "none"},  # hide results
            "",  # clear input
            {"display": "none"},  # hide clear button
            [],  # reset category filter
            None,  # reset start date
            None,  # reset end date
            {"display": "none"},  # hide filters panel
        )

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
