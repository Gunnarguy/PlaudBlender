"""Graph callbacks — node interaction, layout switching."""

from dash import Input, Output, State, callback, ctx, html, no_update
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

        return layout_configs.get(layout_name, {"name": layout_name, "fit": True})

    @app.callback(
        Output("graph-node-detail", "children"),
        Input("knowledge-graph", "tapNodeData"),
        prevent_initial_call=True,
    )
    def show_node_detail(node_data):
        """Show details when a graph node is clicked."""
        if not node_data:
            raise PreventUpdate

        node_type = node_data.get("type", "unknown")
        label = node_data.get("full_label") or node_data.get("label", "Unknown")
        count = node_data.get("count", node_data.get("mention_count", 0))
        categories = node_data.get("categories", "")
        sentiment = node_data.get("sentiment")

        type_icons = {
            "topic": "●",
            "category": "■",
            "date": "◼",
            "person": "●",
            "project": "◆",
            "organization": "■",
            "location": "⬠",
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
                        style={"color": color, "fontSize": "1.2rem"},
                    ),
                    html.H4(label, className="node-detail-name"),
                    html.Span(
                        node_type.capitalize(),
                        className="node-detail-type",
                    ),
                ],
            ),
        ]

        if count:
            unit = "event" if node_type == "category" else "mention"
            children.append(
                html.P(
                    f"{count} {unit}{'s' if count != 1 else ''}",
                    className="node-detail-count",
                )
            )

        if categories:
            children.append(
                html.P(
                    f"Categories: {categories}",
                    className="node-detail-desc",
                    style={"color": "#94a3b8", "fontSize": "0.8rem"},
                )
            )

        if sentiment is not None and sentiment != 0:
            sentiment_label = (
                "positive"
                if sentiment > 0.1
                else ("negative" if sentiment < -0.1 else "neutral")
            )
            children.append(
                html.P(
                    f"Avg. sentiment: {sentiment} ({sentiment_label})",
                    className="node-detail-desc",
                    style={"color": "#94a3b8", "fontSize": "0.8rem"},
                )
            )

        return html.Div(className="node-detail-card", children=children)
