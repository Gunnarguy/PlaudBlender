"""Knowledge Graph visualization component using Cytoscape."""

from dash import html, dcc
import dash_cytoscape as cyto

# Load extra layouts (cose-bilkent, dagre, etc.)
cyto.load_extra_layouts()

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

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
            "font-size": "11px",
            "font-weight": "600",
            "text-outline-color": "#1e1b4b",
            "text-outline-width": "2px",
            "width": "data(size)",
            "height": "data(size)",
            "border-width": "2px",
            "border-color": "#4f46e5",
        },
    },
    # Person nodes (blue circles)
    {
        "selector": ".person",
        "style": {
            "background-color": "#3b82f6",
            "border-color": "#2563eb",
            "shape": "ellipse",
        },
    },
    # Organization nodes (green rounded rectangles)
    {
        "selector": ".organization",
        "style": {
            "background-color": "#10b981",
            "border-color": "#059669",
            "shape": "round-rectangle",
        },
    },
    # Project nodes (orange diamonds)
    {
        "selector": ".project",
        "style": {
            "background-color": "#f59e0b",
            "border-color": "#d97706",
            "shape": "diamond",
        },
    },
    # Topic nodes (pink ellipses)
    {
        "selector": ".topic",
        "style": {
            "background-color": "#ec4899",
            "border-color": "#db2777",
            "shape": "ellipse",
        },
    },
    # Category nodes (purple rounded rectangles)
    {
        "selector": ".category",
        "style": {
            "background-color": "#8b5cf6",
            "border-color": "#7c3aed",
            "shape": "round-rectangle",
        },
    },
    # Recording nodes (teal rectangles)
    {
        "selector": ".recording",
        "style": {
            "background-color": "#14b8a6",
            "border-color": "#0d9488",
            "shape": "rectangle",
        },
    },
    # Location nodes (amber)
    {
        "selector": ".location",
        "style": {
            "background-color": "#f97316",
            "border-color": "#ea580c",
            "shape": "round-pentagon",
        },
    },
    # Action nodes (sky blue)
    {
        "selector": ".action",
        "style": {
            "background-color": "#0ea5e9",
            "border-color": "#0284c7",
            "shape": "round-tag",
        },
    },
    # Date nodes
    {
        "selector": ".date",
        "style": {
            "background-color": "#64748b",
            "border-color": "#475569",
            "shape": "round-rectangle",
        },
    },
    # Metric nodes
    {
        "selector": ".metric",
        "style": {
            "background-color": "#a855f7",
            "border-color": "#9333ea",
            "shape": "round-hexagon",
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
    # Dimmed nodes
    {
        "selector": ".dimmed",
        "style": {
            "opacity": 0.3,
        },
    },
    # Edge styles
    {
        "selector": "edge",
        "style": {
            "width": 2,
            "line-color": "#475569",
            "target-arrow-color": "#475569",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "opacity": 0.6,
        },
    },
    # Edge with label
    {
        "selector": "edge[label]",
        "style": {
            "label": "data(label)",
            "font-size": "9px",
            "color": "#94a3b8",
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
]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH VIEW COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════


def create_graph_view(graph_data=None) -> html.Div:
    """Create the full knowledge graph view with controls and legend.

    Args:
        graph_data: GraphData with nodes and edges lists (Cytoscape format)

    Returns:
        Dash component for the graph view
    """
    elements = []
    node_count = 0
    edge_count = 0

    if graph_data:
        elements = graph_data.nodes + graph_data.edges
        node_count = len(graph_data.nodes)
        edge_count = len(graph_data.edges)

    # Count entity types for legend
    type_counts = {}
    if graph_data:
        for node in graph_data.nodes:
            ntype = node.get("classes", "unknown")
            type_counts[ntype] = type_counts.get(ntype, 0) + 1

    return html.Div(
        className="graph-view",
        children=[
            # Header
            html.Div(
                className="view-header",
                children=[
                    html.H2("🕸️ Knowledge Graph", className="view-title"),
                    html.P(
                        f"{node_count} entities, {edge_count} connections",
                        className="view-subtitle",
                    ),
                ],
            ),
            # Controls bar
            html.Div(
                className="graph-controls",
                children=[
                    dcc.Dropdown(
                        id="graph-layout-select",
                        className="graph-layout-dropdown",
                        value="cose-bilkent",
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Force-Directed (Best)", "value": "cose-bilkent"},
                            {"label": "Force Layout", "value": "cose"},
                            {"label": "Hierarchical", "value": "dagre"},
                            {"label": "Circular", "value": "circle"},
                            {"label": "Grid", "value": "grid"},
                            {"label": "Concentric", "value": "concentric"},
                        ],
                        style={"width": "200px"},
                    ),
                    html.Button(
                        "🔄 Reset View",
                        id="graph-reset-btn",
                        className="graph-btn",
                    ),
                    html.Button(
                        "🔍 Fit All",
                        id="graph-fit-btn",
                        className="graph-btn",
                    ),
                ],
            ),
            # Graph container
            html.Div(
                className="graph-canvas",
                children=(
                    [
                        cyto.Cytoscape(
                            id="knowledge-graph",
                            elements=elements,
                            stylesheet=GRAPH_STYLESHEET,
                            layout={
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
                                "fit": True,
                                "padding": 30,
                            },
                            style={
                                "width": "100%",
                                "height": "100%",
                                "minHeight": "600px",
                            },
                            responsive=True,
                            boxSelectionEnabled=True,
                            userZoomingEnabled=True,
                            userPanningEnabled=True,
                            minZoom=0.1,
                            maxZoom=5,
                        ),
                    ]
                    if elements
                    else [
                        html.Div(
                            className="graph-empty",
                            children=[
                                html.P(
                                    "🕸️", style={"fontSize": "3rem", "opacity": "0.3"}
                                ),
                                html.P(
                                    "No knowledge graph data yet",
                                    style={"color": "#94a3b8"},
                                ),
                                html.P(
                                    "Run the pipeline to extract entities and build the graph.",
                                    style={"color": "#64748b", "fontSize": "0.875rem"},
                                ),
                                html.Code(
                                    "python scripts/chronos_pipeline.py --graph",
                                    style={"color": "#8b5cf6"},
                                ),
                            ],
                        )
                    ]
                ),
            ),
            # Legend
            html.Div(
                className="graph-legend",
                children=[
                    html.Span("Legend:", className="legend-title"),
                    _legend_item(
                        "●", "Person", "#3b82f6", type_counts.get("person", 0)
                    ),
                    _legend_item(
                        "◆", "Project", "#f59e0b", type_counts.get("project", 0)
                    ),
                    _legend_item(
                        "■",
                        "Organization",
                        "#10b981",
                        type_counts.get("organization", 0),
                    ),
                    _legend_item("●", "Topic", "#ec4899", type_counts.get("topic", 0)),
                    _legend_item(
                        "■", "Category", "#8b5cf6", type_counts.get("category", 0)
                    ),
                    _legend_item(
                        "▪", "Recording", "#14b8a6", type_counts.get("recording", 0)
                    ),
                    _legend_item(
                        "⬠", "Location", "#f97316", type_counts.get("location", 0)
                    ),
                ],
            ),
            # Node detail panel (populated by callback)
            html.Div(id="graph-node-detail", className="graph-node-detail"),
        ],
    )


def _legend_item(symbol: str, label: str, color: str, count: int = 0) -> html.Span:
    """Create a legend item."""
    if count == 0:
        return html.Span()  # Hide if no items

    return html.Span(
        className="legend-item",
        children=[
            html.Span(symbol, style={"color": color, "marginRight": "4px"}),
            html.Span(f"{label} ({count})", style={"marginRight": "12px"}),
        ],
    )
