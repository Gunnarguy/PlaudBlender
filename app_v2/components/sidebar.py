"""Sidebar navigation component."""

from dash import html, dcc


def create_sidebar() -> html.Div:
    """Create the sidebar navigation component."""
    return html.Div(
        className="sidebar",
        children=[
            # Logo/Brand
            html.Div(
                className="sidebar-header",
                children=[
                    html.Div(
                        className="logo",
                        children=[
                            html.Span("⏳", className="logo-icon"),
                            html.Span("Chronos", className="logo-text"),
                        ],
                    ),
                ],
            ),
            # Navigation items
            html.Nav(
                className="nav-menu",
                children=[
                    html.Button(
                        id={"type": "nav-item", "view": "timeline"},
                        className="nav-item active",
                        children=[
                            html.Span("⏱️", className="nav-icon"),
                            html.Span("Timeline", className="nav-label"),
                        ],
                    ),
                    html.Button(
                        id={"type": "nav-item", "view": "topics"},
                        className="nav-item",
                        children=[
                            html.Span("💡", className="nav-icon"),
                            html.Span("Topics", className="nav-label"),
                        ],
                    ),
                    html.Button(
                        id={"type": "nav-item", "view": "graph"},
                        className="nav-item",
                        children=[
                            html.Span("🕸️", className="nav-icon"),
                            html.Span("Graph", className="nav-label"),
                        ],
                    ),
                    html.Button(
                        id={"type": "nav-item", "view": "stats"},
                        className="nav-item",
                        children=[
                            html.Span("📊", className="nav-icon"),
                            html.Span("Stats", className="nav-label"),
                        ],
                    ),
                ],
            ),
            # Bottom actions
            html.Div(
                className="sidebar-footer",
                children=[
                    html.Button(
                        id={"type": "nav-item", "view": "sync"},
                        className="nav-item sync-btn",
                        children=[
                            html.Span("🔄", className="nav-icon"),
                            html.Span("Sync", className="nav-label"),
                        ],
                    ),
                    html.Button(
                        id={"type": "nav-item", "view": "settings"},
                        className="nav-item",
                        children=[
                            html.Span("⚙️", className="nav-icon"),
                            html.Span("Settings", className="nav-label"),
                        ],
                    ),
                ],
            ),
        ],
    )
