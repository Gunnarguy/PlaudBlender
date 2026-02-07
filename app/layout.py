"""Main Dash Layout - The Glorious Chronos UI."""

from dash import html, dcc

from app.components.graph import create_graph_component
from app.components.timeline import create_timeline_component
from app.components.search import create_search_component
from app.components.details import create_details_component
from app.components.upload import create_upload_modal


def create_layout() -> html.Div:
    """Create the main application layout.

    Returns:
        The root Dash layout component
    """
    return html.Div(
        className="chronos-app",
        children=[
            # Hidden stores for state management
            dcc.Store(id="selected-event-store", data=None),
            dcc.Store(id="selected-node-store", data=None),
            dcc.Store(id="search-results-store", data=[]),
            dcc.Store(id="graph-data-store", data={"nodes": [], "edges": []}),
            dcc.Store(id="timeline-data-store", data=[]),
            dcc.Interval(
                id="auto-refresh", interval=60000, n_intervals=0
            ),  # 1 min refresh
            # Header
            html.Header(
                className="app-header",
                children=[
                    html.Div(
                        className="header-left",
                        children=[
                            html.H1("⏳ CHRONOS", className="app-title"),
                            html.Span(
                                "Your Knowledge Timeline", className="app-subtitle"
                            ),
                        ],
                    ),
                    # Search bar (center)
                    create_search_component(),
                    # Header actions (right)
                    html.Div(
                        className="header-actions",
                        children=[
                            html.Button(
                                "⬆️ Upload",
                                id="upload-btn",
                                className="btn btn-secondary",
                                title="Upload audio recording",
                            ),
                            html.Button(
                                "⚙️",
                                id="settings-btn",
                                className="btn btn-icon",
                                title="Settings",
                            ),
                        ],
                    ),
                ],
            ),
            # Main content area
            html.Main(
                className="app-main",
                children=[
                    # Left sidebar - Timeline
                    html.Aside(
                        className="sidebar-left",
                        children=[
                            create_timeline_component(),
                        ],
                    ),
                    # Center - Knowledge Graph
                    html.Section(
                        className="main-content",
                        children=[
                            create_graph_component(),
                        ],
                    ),
                    # Right panel - Details
                    html.Aside(
                        className="sidebar-right",
                        children=[
                            create_details_component(),
                        ],
                    ),
                ],
            ),
            # Footer / Status bar
            html.Footer(
                className="app-footer",
                children=[
                    html.Div(
                        id="status-bar",
                        className="status-bar",
                        children=[
                            html.Span(id="status-events", children="Events: —"),
                            html.Span(id="status-nodes", children="Nodes: —"),
                            html.Span(id="status-edges", children="Edges: —"),
                            html.Span(
                                id="status-connection",
                                children="● Connected",
                                className="status-ok",
                            ),
                        ],
                    ),
                ],
            ),
            # Modals
            create_upload_modal(),
            # Settings modal
            html.Div(
                id="settings-modal",
                className="modal",
                style={"display": "none"},
                children=[
                    html.Div(className="modal-overlay", id="settings-overlay"),
                    html.Div(
                        className="modal-content",
                        children=[
                            html.Div(
                                className="modal-header",
                                children=[
                                    html.H3("⚙️ Settings"),
                                    html.Button(
                                        "×",
                                        id="settings-close-btn",
                                        className="modal-close",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="modal-body",
                                children=[
                                    html.Div(
                                        className="setting-group",
                                        children=[
                                            html.H4("Plaud Sync"),
                                            html.Label("Days to sync:"),
                                            dcc.Slider(
                                                id="sync-days-slider",
                                                min=1,
                                                max=30,
                                                value=7,
                                                marks={
                                                    1: "1",
                                                    7: "7",
                                                    14: "14",
                                                    30: "30",
                                                },
                                            ),
                                            html.Button(
                                                "🔐 Re-authenticate with Plaud",
                                                id="reauth-btn",
                                                className="btn btn-secondary",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="setting-group",
                                        children=[
                                            html.H4("Graph Display"),
                                            html.Label("Max nodes to display:"),
                                            dcc.Slider(
                                                id="max-nodes-slider",
                                                min=50,
                                                max=500,
                                                value=200,
                                                marks={
                                                    50: "50",
                                                    100: "100",
                                                    200: "200",
                                                    500: "500",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="setting-group",
                                        children=[
                                            html.H4("System Info"),
                                            html.Div(id="system-info"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # Notifications container
            html.Div(id="notifications", className="notifications-container"),
        ],
    )
