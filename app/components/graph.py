"""Cytoscape Knowledge Graph Component."""

from dash import html, dcc
import dash_cytoscape as cyto

# Load extra layouts
cyto.load_extra_layouts()


# Cytoscape stylesheet for beautiful graph visualization
GRAPH_STYLESHEET = [
    # Base node style
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "#6366f1",
            "color": "#ffffff",
            "font-size": "12px",
            "font-weight": "600",
            "text-outline-color": "#1e1b4b",
            "text-outline-width": "2px",
            "width": "data(size)",
            "height": "data(size)",
            "border-width": "2px",
            "border-color": "#4f46e5",
        },
    },
    # Category nodes (purple)
    {
        "selector": ".category",
        "style": {
            "background-color": "#8b5cf6",
            "border-color": "#7c3aed",
            "shape": "round-rectangle",
        },
    },
    # Person nodes (blue)
    {
        "selector": ".person",
        "style": {
            "background-color": "#3b82f6",
            "border-color": "#2563eb",
            "shape": "ellipse",
        },
    },
    # Organization nodes (green)
    {
        "selector": ".organization",
        "style": {
            "background-color": "#10b981",
            "border-color": "#059669",
            "shape": "round-rectangle",
        },
    },
    # Project nodes (orange)
    {
        "selector": ".project",
        "style": {
            "background-color": "#f59e0b",
            "border-color": "#d97706",
            "shape": "diamond",
        },
    },
    # Recording nodes (teal)
    {
        "selector": ".recording",
        "style": {
            "background-color": "#14b8a6",
            "border-color": "#0d9488",
            "shape": "rectangle",
        },
    },
    # Topic nodes (pink)
    {
        "selector": ".topic",
        "style": {
            "background-color": "#ec4899",
            "border-color": "#db2777",
            "shape": "ellipse",
        },
    },
    # Selected node
    {
        "selector": ":selected",
        "style": {
            "border-width": "4px",
            "border-color": "#fbbf24",
            "background-color": "#fef3c7",
            "color": "#1e1b4b",
            "text-outline-color": "#fef3c7",
        },
    },
    # Hovered node
    {
        "selector": "node:active",
        "style": {
            "overlay-opacity": 0.2,
            "overlay-color": "#fbbf24",
        },
    },
    # Edge styles
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "line-color": "#94a3b8",
            "target-arrow-color": "#94a3b8",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "opacity": 0.7,
        },
    },
    # Edge with label
    {
        "selector": "edge[label]",
        "style": {
            "label": "data(label)",
            "font-size": "10px",
            "color": "#64748b",
            "text-rotation": "autorotate",
            "text-margin-y": "-10px",
        },
    },
    # Selected edge
    {
        "selector": "edge:selected",
        "style": {
            "line-color": "#fbbf24",
            "target-arrow-color": "#fbbf24",
            "width": 3,
            "opacity": 1,
        },
    },
    # Highlighted nodes (from search)
    {
        "selector": ".highlighted",
        "style": {
            "border-width": "4px",
            "border-color": "#22c55e",
            "background-color": "#bbf7d0",
            "color": "#14532d",
            "text-outline-color": "#bbf7d0",
            "z-index": 999,
        },
    },
    # Dimmed nodes (when others are highlighted)
    {
        "selector": ".dimmed",
        "style": {
            "opacity": 0.3,
        },
    },
]


def create_graph_component(graph_id: str = "knowledge-graph") -> html.Div:
    """Create the Cytoscape knowledge graph component.

    Args:
        graph_id: DOM ID for the graph element

    Returns:
        Dash HTML Div containing the graph
    """
    return html.Div(
        className="graph-container",
        children=[
            # Graph header with controls
            html.Div(
                className="graph-header",
                children=[
                    html.H3("🕸️ Knowledge Graph", className="graph-title"),
                    html.Div(
                        className="graph-controls",
                        children=[
                            html.Button(
                                "Reset View",
                                id="graph-reset-btn",
                                className="btn btn-secondary btn-sm",
                            ),
                            dcc.Dropdown(
                                id="graph-layout-select",
                                className="layout-select",
                                value="cose-bilkent",
                                clearable=False,
                                searchable=False,
                                options=[
                                    {
                                        "label": "CoSE-Bilkent (Best)",
                                        "value": "cose-bilkent",
                                    },
                                    {"label": "Force-Directed", "value": "cose"},
                                    {"label": "Hierarchical", "value": "dagre"},
                                    {"label": "Circular", "value": "circle"},
                                    {"label": "Grid", "value": "grid"},
                                    {"label": "Concentric", "value": "concentric"},
                                ],
                                style={"width": "180px", "fontSize": "12px"},
                            ),
                        ],
                    ),
                ],
            ),
            # The graph itself
            cyto.Cytoscape(
                id=graph_id,
                elements=[],  # Populated via callback
                stylesheet=GRAPH_STYLESHEET,
                layout={
                    # cose-bilkent handles large graphs much better
                    "name": "cose-bilkent",
                    "animate": False,
                    "randomize": False,
                    "nodeDimensionsIncludeLabels": True,
                    "nodeRepulsion": 4500,
                    "idealEdgeLength": 50,
                    "edgeElasticity": 0.45,
                    "nestingFactor": 0.1,
                    "gravity": 0.25,
                    "numIter": 2500,
                    "tile": True,
                    "tilingPaddingVertical": 10,
                    "tilingPaddingHorizontal": 10,
                    "fit": True,
                    "padding": 30,
                },
                style={"width": "100%", "height": "100%", "minHeight": "600px"},
                responsive=True,
                boxSelectionEnabled=True,
                userZoomingEnabled=True,
                userPanningEnabled=True,
                autoungrabify=False,
                autounselectify=False,
                minZoom=0.1,
                maxZoom=5,
            ),
            # Legend
            html.Div(
                className="graph-legend",
                children=[
                    html.Span("Legend:", className="legend-label"),
                    html.Span("● Person", className="legend-item person"),
                    html.Span("■ Project", className="legend-item project"),
                    html.Span("◆ Category", className="legend-item category"),
                    html.Span("▪ Recording", className="legend-item recording"),
                ],
            ),
        ],
    )
