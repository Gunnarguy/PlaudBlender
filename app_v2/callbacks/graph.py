"""Graph callbacks — node interaction, layout switching."""

from dash import Input, Output, State, callback, ctx, html, no_update, ALL
from dash.exceptions import PreventUpdate
import logging

logger = logging.getLogger(__name__)


def register_graph_callbacks(app):
    """Register knowledge graph interaction callbacks."""

    @app.callback(
        Output("knowledge-graph", "layout"),
        Input("graph-layout-select", "value"),
        prevent_initial_call=True,
    )
    def change_graph_layout(layout_name):
        """Change the graph layout algorithm."""
        if not layout_name:
            raise PreventUpdate

        # Tuned for a hub-spoke category graph
        layout_configs = {
            "cose-bilkent": {
                "name": "cose-bilkent",
                "animate": True,
                "animationDuration": 500,
                "nodeDimensionsIncludeLabels": True,
                "nodeRepulsion": 8000,
                "idealEdgeLength": 80,
                "edgeElasticity": 0.45,
                "nestingFactor": 0.1,
                "gravity": 0.15,
                "gravityRange": 3.8,
                "numIter": 3000,
                "tile": True,
                "fit": True,
                "padding": 50,
            },
            "dagre": {
                "name": "dagre",
                "animate": True,
                "animationDuration": 500,
                "rankDir": "TB",
                "rankerFunction": "tight-tree",
                "nodeSep": 40,
                "rankSep": 80,
                "fit": True,
                "padding": 40,
            },
            "circle": {
                "name": "circle",
                "animate": True,
                "animationDuration": 500,
                "fit": True,
                "padding": 40,
            },
            "concentric": {
                "name": "concentric",
                "animate": True,
                "animationDuration": 500,
                "fit": True,
                "padding": 40,
                "minNodeSpacing": 60,
            },
        }

        from app_v2.services.xray import xray_log

        xray_log("graph", "layout", f"Rearranged the knowledge map to '{layout_name}' style")
        return layout_configs.get(layout_name, {"name": layout_name, "fit": True})

    @app.callback(
        Output("graph-node-detail", "children"),
        Output("graph-clicked-keyword", "data"),
        Input("knowledge-graph", "tapNodeData"),
        prevent_initial_call=True,
    )
    def show_node_detail(node_data):
        """Show detailed info when a graph node is clicked."""
        if not node_data:
            raise PreventUpdate

        from app_v2.services.xray import xray_log

        node_type = node_data.get("type", "unknown")
        label = node_data.get("full_label") or node_data.get("label", "Unknown")
        # Store keyword for the Find Recordings button
        clicked_keyword = label.split(" (")[0] if " (" in label else label
        count = node_data.get("count", node_data.get("mention_count", 0))
        xray_log("graph", "node-tap", f"You clicked on '{label}' ({node_type})", detail=f"shows up {count} times")
        categories = node_data.get("categories", "")
        sentiment = node_data.get("sentiment")
        related = node_data.get("related_keywords", "")
        recordings = node_data.get("recording_count", 0)

        type_icons = {
            "topic": "💬",
            "category": "📁",
            "date": "📅",
            "person": "👤",
            "project": "📌",
            "organization": "🏢",
            "location": "📍",
        }
        type_colors = {
            "topic": "#475569",
            "category": "#8b5cf6",
            "date": "#475569",
            "person": "#3b82f6",
            "project": "#f59e0b",
            "organization": "#10b981",
            "location": "#f97316",
        }

        icon = type_icons.get(node_type.lower(), "●")
        color = type_colors.get(node_type.lower(), "#6366f1")

        children: list = [
            html.Div(
                className="node-detail-header",
                children=[
                    html.Span(
                        icon,
                        className="node-detail-icon",
                        style={"fontSize": "1.5rem"},
                    ),
                    html.Div(
                        children=[
                            html.H4(label, className="node-detail-name"),
                            html.Span(
                                node_type.capitalize(),
                                className="node-detail-type",
                                style={"color": color},
                            ),
                        ],
                    ),
                ],
            ),
        ]

        # Stats row
        stats_items = []
        if count:
            unit = "event" if node_type == "category" else "mention"
            stats_items.append(
                html.Span(
                    f"{count} {unit}{'s' if count != 1 else ''}",
                    className="node-stat",
                )
            )
        if recordings:
            stats_items.append(
                html.Span(
                    f"{recordings} recording{'s' if recordings != 1 else ''}",
                    className="node-stat",
                )
            )
        if stats_items:
            children.append(html.Div(className="node-stats-row", children=stats_items))

        # Sentiment
        if sentiment is not None and sentiment != 0:
            sent_val = float(sentiment)
            if sent_val > 0.1:
                sent_emoji, sent_label = "😊", "positive"
            elif sent_val < -0.1:
                sent_emoji, sent_label = "😔", "negative"
            else:
                sent_emoji, sent_label = "😐", "neutral"
            children.append(
                html.Div(
                    className="node-sentiment",
                    children=[
                        html.Span(sent_emoji, className="node-sentiment-icon"),
                        html.Span(
                            f"Avg: {sent_val:+.2f} ({sent_label})",
                            className="node-sentiment-text",
                        ),
                    ],
                )
            )

        # Categories
        if categories:
            cat_list = [c.strip() for c in str(categories).split(",") if c.strip()]
            if cat_list:
                from app_v2.components import CATEGORY_COLORS as CAT_COLORS

                children.append(
                    html.Div(
                        className="node-categories",
                        children=[
                            html.Span("Appears in:", className="node-section-label"),
                            html.Div(
                                className="node-category-pills",
                                children=[
                                    html.Span(
                                        c.replace("_", " ").title(),
                                        className="category-pill",
                                        style={
                                            "backgroundColor": CAT_COLORS.get(
                                                c, "#374151"
                                            )
                                        },
                                    )
                                    for c in cat_list
                                ],
                            ),
                        ],
                    )
                )

        # Related keywords
        if related:
            kw_list = [k.strip() for k in str(related).split(",") if k.strip()]
            if kw_list:
                children.append(
                    html.Div(
                        className="node-related",
                        children=[
                            html.Span(
                                "Related topics:", className="node-section-label"
                            ),
                            html.Div(
                                className="node-related-tags",
                                children=[
                                    html.Span(kw, className="keyword-tag small")
                                    for kw in kw_list[:8]
                                ],
                            ),
                        ],
                    )
                )

        # Add "Find Recordings" button
        children.append(
            html.Button(
                "Find Recordings",
                id="graph-find-recordings-btn",
                className="graph-find-recordings-btn",
                n_clicks=0,
            )
        )

        return html.Div(className="node-detail-card", children=children), clicked_keyword

    # ── Find recordings containing the clicked node's keyword ─────
    @app.callback(
        Output("graph-node-recordings", "children"),
        Input("graph-find-recordings-btn", "n_clicks"),
        State("graph-clicked-keyword", "data"),
        prevent_initial_call=True,
    )
    def find_node_recordings(n_clicks, keyword):
        """Search for recordings matching the clicked graph node's keyword."""
        if not n_clicks or not keyword:
            raise PreventUpdate

        from app_v2.services.xray import xray_log
        from app_v2.services.data_service import get_data_service

        xray_log("graph", "search", f"Finding recordings for '{keyword}'")

        service = get_data_service()
        results = service._text_search(keyword, limit=10)

        if not results:
            return html.Div(
                className="graph-recordings-empty",
                children=html.P(f"No recordings found for '{keyword}'"),
            )

        # Group by recording
        seen_recordings = {}
        for r in results:
            rid = r.event.recording_id
            if rid not in seen_recordings:
                seen_recordings[rid] = r

        cards = []
        for rid, result in list(seen_recordings.items())[:6]:
            ev = result.event
            date_str = ev.start_ts.strftime("%b %d, %Y") if ev.start_ts else ""
            cards.append(
                html.Div(
                    className="graph-rec-card",
                    id={"type": "graph-rec-click", "recording_id": rid},
                    n_clicks=0,
                    children=[
                        html.Div(className="graph-rec-title", children=ev.title[:60] or "Untitled"),
                        html.Div(
                            className="graph-rec-meta",
                            children=[
                                html.Span(date_str, className="graph-rec-date"),
                                html.Span(
                                    ev.category.replace("_", " ").title(),
                                    className="graph-rec-cat",
                                ),
                            ],
                        ),
                    ],
                )
            )

        return html.Div(
            className="graph-recordings-list",
            children=[
                html.H5(f"Recordings with '{keyword}'", className="graph-recordings-heading"),
                *cards,
            ],
        )

    # ── Click a recording from graph results → open detail ────────
    @app.callback(
        Output("selected-recording", "data", allow_duplicate=True),
        Input({"type": "graph-rec-click", "recording_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_graph_recording(n_clicks_list):
        """Open recording detail when clicking a graph result card."""
        if not any(n_clicks_list):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            raise PreventUpdate

        recording_id = triggered.get("recording_id")
        if not recording_id:
            raise PreventUpdate

        from app_v2.services.xray import xray_log
        xray_log("graph", "select", f"Opening recording from knowledge graph")

        return recording_id
