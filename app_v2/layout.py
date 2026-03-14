"""Main layout for Chronos UI."""

from dash import html, dcc

from app_v2.components.sidebar import create_sidebar
from app_v2.components.search import create_search_bar


def create_layout() -> html.Div:
    """Create the main app layout."""
    return html.Div(
        className="chronos-app",
        children=[
            # Stores for state management
            dcc.Store(id="current-view", data="timeline"),
            dcc.Store(id="selected-recording", data=None),
            dcc.Store(id="selected-topic", data=None),
            dcc.Store(id="search-query", data=None),
            dcc.Store(id="days-data", data=None),
            dcc.Store(id="app-preferences", storage_type="local", data=None),
            # X-ray telemetry store (event bus)
            dcc.Store(id="xray-events", data=[]),
            # X-ray refresh ticker (polls server-side buffer every 1s when open)
            dcc.Interval(id="xray-poll", interval=1000, n_intervals=0, disabled=True),
            # Auto-refresh interval (every 60 seconds)
            dcc.Interval(id="auto-refresh", interval=60000, n_intervals=0),
            # Pipeline progress polling (2s while running, self-disables when idle)
            dcc.Interval(
                id="pipeline-progress-poll",
                interval=2000,
                n_intervals=0,
                disabled=False,
            ),
            # Workflow status polling (10s, active when workflows are in flight)
            dcc.Interval(
                id="workflow-poll",
                interval=10000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Store(id="active-workflows-count", data=0),
            # Sidebar navigation
            create_sidebar(),
            # Main content area
            html.Main(
                className="main-content",
                children=[
                    # Header with search
                    html.Header(
                        className="main-header",
                        children=[
                            create_search_bar(),
                            html.Button(
                                "✕",
                                id="clear-search-btn",
                                className="clear-search-btn",
                                style={"display": "none"},
                            ),
                        ],
                    ),
                    # Search results overlay
                    html.Div(
                        id="search-results-container",
                        className="search-results-container",
                        style={"display": "none"},
                    ),
                    # Content container (switches based on view)
                    html.Div(
                        id="content-container",
                        className="content-container",
                    ),
                ],
            ),
            # Detail panel (slides in when recording selected)
            html.Aside(
                id="detail-panel",
                className="detail-panel",
            ),
            # Category save status toast
            html.Div(id="category-save-status", className="save-status-toast"),
            # X-ray drawer — collapsible activity monitor
            html.Div(
                className="xray-drawer collapsed",
                id="xray-drawer",
                children=[
                    html.Div(
                        className="xray-handle",
                        id="xray-toggle",
                        children=[
                            html.Span("⚡", className="xray-icon"),
                            html.Span("ACTIVITY", className="xray-title"),
                            html.Span("", id="xray-badge", className="xray-badge"),
                            # Live stats strip
                            html.Div(
                                className="xray-live-stats",
                                id="xray-live-stats",
                                children=[],
                            ),
                            html.Span("▲", id="xray-chevron", className="xray-chevron"),
                        ],
                    ),
                    html.Div(
                        className="xray-body",
                        id="xray-body",
                        children=[
                            html.Div(
                                className="xray-toolbar",
                                children=[
                                    # Filter tabs
                                    html.Div(
                                        className="xray-filters",
                                        children=[
                                            html.Button("All", id="xray-filter-all", className="xray-filter-btn active", **{"data-filter": "all"}),
                                            html.Button("Search", id="xray-filter-search", className="xray-filter-btn", **{"data-filter": "search"}),
                                            html.Button("Nav", id="xray-filter-nav", className="xray-filter-btn", **{"data-filter": "nav"}),
                                            html.Button("Graph", id="xray-filter-graph", className="xray-filter-btn", **{"data-filter": "graph"}),
                                            html.Button("Errors", id="xray-filter-errors", className="xray-filter-btn", **{"data-filter": "error"}),
                                        ],
                                    ),
                                    html.Div(
                                        className="xray-toolbar-right",
                                        children=[
                                            html.Span(
                                                "", id="xray-count", className="xray-count"
                                            ),
                                            html.Button(
                                                "Clear",
                                                id="xray-clear-btn",
                                                className="xray-btn",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id="xray-log", className="xray-log"),
                        ],
                    ),
                ],
            ),
            # Loading overlay
            dcc.Loading(
                id="loading-overlay",
                type="circle",
                fullscreen=True,
                color="#6366f1",
                children=[html.Div(id="loading-target")],
            ),
        ],
    )
