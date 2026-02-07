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

        # Layout-specific configs
        layout_configs = {
            "cose-bilkent": {
                "name": "cose-bilkent",
                "animate": True,
                "animationDuration": 500,
                "nodeDimensionsIncludeLabels": True,
                "nodeRepulsion": 4500,
                "idealEdgeLength": 50,
                "edgeElasticity": 0.45,
                "gravity": 0.25,
                "numIter": 2500,
                "fit": True,
                "padding": 30,
            },
            "cose": {
                "name": "cose",
                "animate": True,
                "animationDuration": 500,
                "nodeRepulsion": 400000,
                "idealEdgeLength": 100,
                "gravity": 80,
                "fit": True,
                "padding": 30,
            },
            "dagre": {
                "name": "dagre",
                "animate": True,
                "animationDuration": 500,
                "rankDir": "TB",
                "rankerFunction": "tight-tree",
                "fit": True,
                "padding": 30,
            },
            "circle": {
                "name": "circle",
                "animate": True,
                "animationDuration": 500,
                "fit": True,
                "padding": 30,
            },
            "grid": {
                "name": "grid",
                "animate": True,
                "animationDuration": 500,
                "fit": True,
                "padding": 30,
            },
            "concentric": {
                "name": "concentric",
                "animate": True,
                "animationDuration": 500,
                "fit": True,
                "padding": 30,
                "minNodeSpacing": 50,
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
        description = node_data.get("description", "")
        count = node_data.get("count", node_data.get("mention_count", 0))

        type_icons = {
            "person": "👤",
            "project": "📁",
            "organization": "🏢",
            "topic": "💬",
            "category": "🏷️",
            "recording": "🎙️",
            "location": "📍",
            "action": "⚡",
            "date": "📅",
            "metric": "📊",
        }

        icon = type_icons.get(node_type.lower(), "●")

        children = [
            html.Div(
                className="node-detail-header",
                children=[
                    html.Span(icon, className="node-detail-icon"),
                    html.H4(label, className="node-detail-name"),
                    html.Span(
                        node_type.capitalize(),
                        className="node-detail-type",
                    ),
                ],
            ),
        ]

        if description:
            children.append(html.P(description, className="node-detail-desc"))

        if count:
            children.append(
                html.P(
                    f"Mentioned {count} time{'s' if count != 1 else ''}",
                    className="node-detail-count",
                )
            )

        return html.Div(className="node-detail-card", children=children)
