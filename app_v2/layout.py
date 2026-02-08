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
            dcc.Store(id="current-view", data="days"),
            dcc.Store(id="selected-recording", data=None),
            dcc.Store(id="selected-topic", data=None),
            dcc.Store(id="search-query", data=None),
            dcc.Store(id="days-data", data=None),
            # Auto-refresh interval (every 60 seconds)
            dcc.Interval(id="auto-refresh", interval=60000, n_intervals=0),
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
