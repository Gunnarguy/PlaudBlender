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
            # X-ray Activity Monitor — floating PiP panel
            html.Div(id="xray-pip", className="xray-pip"),
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
