"""Search component and results."""

from dash import html, dcc
from typing import List

from app_v2.services.data_service import SearchResult


_AI_MODEL_OPTIONS = [
    {"label": "GPT-5.5", "value": "gpt-5.5"},
    {"label": "GPT-5.5 Pro", "value": "gpt-5.5-pro"},
    {"label": "GPT-5.4", "value": "gpt-5.4"},
    {"label": "GPT-5.4 Mini", "value": "gpt-5.4-mini"},
    {"label": "GPT-5.4 Nano", "value": "gpt-5.4-nano"},
]

_REASONING_OPTIONS = [
    {"label": "Model default", "value": "default"},
    {"label": "None", "value": "none"},
    {"label": "Low", "value": "low"},
    {"label": "Medium", "value": "medium"},
    {"label": "High", "value": "high"},
    {"label": "xHigh", "value": "xhigh"},
]

_VERBOSITY_OPTIONS = [
    {"label": "Model default", "value": "default"},
    {"label": "Low", "value": "low"},
    {"label": "Medium", "value": "medium"},
    {"label": "High", "value": "high"},
]

_REASONING_SUMMARY_OPTIONS = [
    {"label": "Off", "value": "off"},
    {"label": "Auto", "value": "auto"},
]

_SERVICE_TIER_OPTIONS = [
    {"label": "Auto", "value": "auto"},
    {"label": "Default", "value": "default"},
    {"label": "Flex", "value": "flex"},
    {"label": "Priority", "value": "priority"},
]


def create_search_bar() -> html.Div:
    """Create the search bar component with filter controls."""
    return html.Div(
        className="search-bar",
        children=[
            html.Div(
                className="search-row",
                children=[
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Search your recordings...",
                        className="search-input",
                        debounce=True,
                    ),
                    html.Button(
                        id="search-btn",
                        className="search-btn",
                        children=[html.Span("🔍", className="search-icon")],
                    ),
                    html.Button(
                        id="toggle-filters-btn",
                        className="search-btn filter-toggle",
                        children=[html.Span("⚙️", className="search-icon")],
                        title="Toggle search filters",
                    ),
                ],
            ),
            html.Div(
                id="search-filters",
                className="search-filters",
                style={"display": "none"},
                children=[
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Ask model:", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-model",
                                options=_AI_MODEL_OPTIONS,
                                value="gpt-5.5",
                                clearable=False,
                                className="filter-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Reasoning:", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-reasoning",
                                options=_REASONING_OPTIONS,
                                value="default",
                                clearable=False,
                                className="filter-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Verbosity:", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-verbosity",
                                options=_VERBOSITY_OPTIONS,
                                value="default",
                                clearable=False,
                                className="filter-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Reasoning summary:", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-reasoning-summary",
                                options=_REASONING_SUMMARY_OPTIONS,
                                value="off",
                                clearable=False,
                                className="filter-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Temperature:", className="filter-label"),
                            dcc.Input(
                                id="filter-temperature",
                                type="number",
                                placeholder="API default",
                                min=0.0,
                                max=2.0,
                                step=0.1,
                                className="filter-number-input",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Top-p:", className="filter-label"),
                            dcc.Input(
                                id="filter-top-p",
                                type="number",
                                placeholder="API default",
                                min=0.0,
                                max=1.0,
                                step=0.05,
                                className="filter-number-input",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Max output tokens:", className="filter-label"),
                            dcc.Input(
                                id="filter-max-output-tokens",
                                type="number",
                                placeholder="API default",
                                min=16,
                                step=256,
                                className="filter-number-input",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Service tier:", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-service-tier",
                                options=_SERVICE_TIER_OPTIONS,
                                value="auto",
                                clearable=False,
                                className="filter-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Category:", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-category",
                                options=[
                                    {"label": "Work", "value": "work"},
                                    {"label": "Personal", "value": "personal"},
                                    {"label": "Meeting", "value": "meeting"},
                                    {"label": "Deep Work", "value": "deep_work"},
                                    {"label": "Reflection", "value": "reflection"},
                                    {"label": "Idea", "value": "idea"},
                                    {"label": "Break", "value": "break"},
                                ],
                                multi=True,
                                placeholder="All categories",
                                className="filter-dropdown",
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-group",
                        children=[
                            html.Label("Date range:", className="filter-label"),
                            dcc.DatePickerRange(
                                id="filter-date-range",
                                className="filter-date-picker",
                                display_format="MMM D",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_search_result_card(result: SearchResult) -> html.Div:
    """Create a search result card."""
    event = result.event

    # Format date and time
    date_str = event.start_ts.strftime("%b %d")
    time_str = event.start_ts.strftime("%I:%M %p")

    # Score as percentage
    score_pct = result.score * 100

    return html.Div(
        id={
            "type": "search-result",
            "event_id": event.id,
            "recording_id": event.recording_id,
        },
        className="search-result-card",
        children=[
            # Header
            html.Div(
                className="result-header",
                children=[
                    html.Span(f"📅 {date_str}", className="result-recording"),
                    html.Span(f"🕐 {time_str}", className="result-meta"),
                    html.Span(
                        event.category,
                        className=f"category-pill cat-{event.category}",
                    ),
                    html.Span(f"{score_pct:.0f}%", className="result-score"),
                ],
            ),
            # Text
            html.P(
                (
                    event.clean_text[:200] + "..."
                    if len(event.clean_text) > 200
                    else event.clean_text
                ),
                className="result-text",
            ),
        ],
    )


def create_search_results(results: List[SearchResult], query: str) -> html.Div:
    """Create the search results view."""
    if not results:
        return html.Div(
            className="search-results-container empty-state",
            children=[
                html.Span("🔍", className="empty-icon"),
                html.P(f'No results found for "{query}"'),
            ],
        )

    return html.Div(
        className="search-results-container",
        children=[
            html.Div(
                className="search-results-header",
                children=[
                    html.H3(f'Results for "{query}"', className="search-results-title"),
                    html.Span(f"{len(results)} matches", className="result-score"),
                ],
            ),
            html.Div(
                className="search-results-list",
                children=[create_search_result_card(result) for result in results],
            ),
        ],
    )
