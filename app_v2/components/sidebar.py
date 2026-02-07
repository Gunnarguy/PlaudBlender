"""Sidebar navigation component."""

from dash import html, dcc


def create_sidebar() -> html.Div:
    """Create the sidebar navigation component."""
    return html.Div(
        className="sidebar",
        children=[
            # Logo/Brand
            html.Div(
                className="sidebar-brand",
                children=[
                    html.Span("⏳", className="brand-icon"),
                    html.Span("Chronos", className="brand-text"),
                ],
            ),
            # Navigation items
            html.Nav(
                className="sidebar-nav",
                children=[
                    html.Button(
                        id={"type": "nav-item", "view": "days"},
                        className="nav-item active",
                        children=[
                            html.Span("📅", className="nav-icon"),
                            html.Span("Days", className="nav-label"),
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
            # Spacer
            html.Div(className="sidebar-spacer"),
            # Bottom actions
            html.Div(
                className="sidebar-actions",
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
