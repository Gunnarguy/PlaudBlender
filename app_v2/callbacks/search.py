"""Search callbacks."""

from dash import Input, Output, State, html, dcc, no_update, ALL, ctx
from dash.exceptions import PreventUpdate
import logging
import time

from app_v2.services import get_data_service
from app_v2.components import create_search_results
from src.chronos.ask_context import build_context_from_results

logger = logging.getLogger(__name__)


def _ai_config_chip(label: str, value) -> html.Span:
    return html.Span(f"{label}: {value}", className="ai-config-chip")


def _build_ai_answer_section(
    question: str,
    results,
    model_choice: str | None = None,
    reasoning_choice: str | None = None,
    verbosity_choice: str | None = None,
    reasoning_summary_choice: str | None = None,
    temperature=None,
    top_p=None,
    max_output_tokens=None,
    service_tier: str | None = None,
) -> html.Div | None:
    """Try OpenAI Responses API for a conversational answer. Returns Div or None."""
    try:
        from src.config import get_settings
        from src.chronos.ask_service import ChronosAskService

        settings = get_settings()
        svc = ChronosAskService()
        if not svc.available:
            return None

        context_events = build_context_from_results(get_data_service(), results)

        t0 = time.time()
        result = svc.ask(
            question,
            context_events,
            model=model_choice,
            reasoning=None
            if reasoning_choice in {None, "", "default"}
            else reasoning_choice,
            reasoning_summary=None
            if reasoning_summary_choice in {None, "", "off"}
            else reasoning_summary_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            verbosity=None
            if verbosity_choice in {None, "", "default"}
            else verbosity_choice,
            service_tier=service_tier,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if "error" in result:
            logger.warning(f"OpenAI ask failed: {result['error']}")
            return None

        usage = result.get("usage", {})
        model = result.get("model", settings.openai_model)
        config = result.get("config", {})
        provider = config.get("provider", "openai")
        config_row = [
            _ai_config_chip("provider", provider),
            _ai_config_chip("model", model),
        ]

        if config.get("reasoning"):
            config_row.append(_ai_config_chip("reasoning", config["reasoning"]))
        if config.get("verbosity"):
            config_row.append(_ai_config_chip("verbosity", config["verbosity"]))
        if config.get("reasoning_summary"):
            config_row.append(
                _ai_config_chip("summary", config["reasoning_summary"])
            )
        if config.get("temperature") is not None:
            config_row.append(_ai_config_chip("temp", config["temperature"]))
        if config.get("top_p") is not None:
            config_row.append(_ai_config_chip("top_p", config["top_p"]))
        if config.get("max_output_tokens") is not None:
            config_row.append(
                _ai_config_chip("max_output", config["max_output_tokens"])
            )
        if config.get("service_tier"):
            config_row.append(_ai_config_chip("tier", config["service_tier"]))
        if config.get("fallback_from"):
            config_row.append(_ai_config_chip("fallback", config["fallback_from"]))
        if config.get("incomplete_reason"):
            config_row.append(_ai_config_chip("status", config["incomplete_reason"]))
        if usage.get("reasoning_tokens"):
            config_row.append(
                _ai_config_chip("reasoning_tokens", usage["reasoning_tokens"])
            )

        reasoning_summary = result.get("reasoning_summary")

        return html.Div(
            className="ai-answer-section",
            children=[
                html.Div(
                    className="ai-answer-header",
                    children=[
                        html.Span("🧠", className="ai-icon"),
                        html.Span(
                            f"Chronos AI · {provider.title()}",
                            className="ai-label",
                        ),
                        html.Span(
                            f"{model} · {latency_ms}ms · {usage.get('total_tokens', '?')} tokens",
                            className="ai-meta",
                        ),
                    ],
                ),
                html.Div(className="ai-config-row", children=config_row),
                dcc.Markdown(
                    result["answer"],
                    className="ai-answer-body",
                ),
                html.Details(
                    className="ai-reasoning-summary",
                    children=[
                        html.Summary("Reasoning summary"),
                        dcc.Markdown(
                            reasoning_summary,
                            className="ai-answer-body ai-reasoning-summary-body",
                        ),
                    ],
                )
                if reasoning_summary
                else None,
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
        State("filter-model", "value"),
        State("filter-reasoning", "value"),
        State("filter-verbosity", "value"),
        State("filter-reasoning-summary", "value"),
        State("filter-temperature", "value"),
        State("filter-top-p", "value"),
        State("filter-max-output-tokens", "value"),
        State("filter-service-tier", "value"),
        prevent_initial_call=True,
    )
    def perform_search(
        n_submit,
        n_clicks,
        query,
        categories,
        start_date,
        end_date,
        filter_model,
        filter_reasoning,
        filter_verbosity,
        filter_reasoning_summary,
        filter_temperature,
        filter_top_p,
        filter_max_output_tokens,
        filter_service_tier,
    ):
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
            "search", "query", f"Looking for anything about '{query}'"
        )

        # Perform filtered search
        service = get_data_service()
        with xray_timer("search", "vector", "Scanning your recordings for matches") as t_search:
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
            f"Found {len(results)} matching moments" + (f" — top match is {results[0].score:.0%} relevant" if results else ""),
            duration_ms=round(t_search.ms, 1),
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
        xray_log("search", "ai", "Asking GPT to explain what it all means…")
        with xray_timer("search", "openai", "GPT is reading your results and writing a summary") as t_ai:
            ai_section = _build_ai_answer_section(
                query,
                results,
                model_choice=filter_model,
                reasoning_choice=filter_reasoning,
                verbosity_choice=filter_verbosity,
                reasoning_summary_choice=filter_reasoning_summary,
                temperature=filter_temperature,
                top_p=filter_top_p,
                max_output_tokens=filter_max_output_tokens,
                service_tier=filter_service_tier,
            )
        if ai_section:
            results_ui.append(ai_section)
            xray_log(
                "search",
                "ai",
                "GPT wrote you an answer",
                duration_ms=round(t_ai.ms, 1),
            )
        else:
            xray_log(
                "search", "ai", "GPT couldn't answer (no key or something broke)", level="warn"
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
