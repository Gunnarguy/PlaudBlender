"""Knowledge Graph visualization component using Cytoscape."""

from dash import html, dcc
import dash_cytoscape as cyto

# Load extra layouts (cose-bilkent, dagre, etc.)
cyto.load_extra_layouts()

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

GRAPH_STYLESHEET = [
    # ── Base node ─────────────────────────────────────────────────────────────
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "background-color": "#6366f1",
            "color": "#e2e8f0",
            "font-size": "10px",
            "font-weight": "500",
            "text-outline-color": "#0f172a",
            "text-outline-width": "2px",
            "width": "data(size)",
            "height": "data(size)",
            "border-width": "2px",
            "border-color": "#4f46e5",
            "transition-property": "background-color, border-color, width, height, opacity",
            "transition-duration": "0.2s",
        },
    },
    # ── Category hub nodes (large, bold) ──────────────────────────────────────
    {
        "selector": ".category",
        "style": {
            "background-color": "data(color)",
            "border-color": "data(color)",
            "shape": "round-rectangle",
            "font-size": "13px",
            "font-weight": "700",
            "text-outline-width": "3px",
            "text-outline-color": "#0f172a",
            "color": "#ffffff",
            "border-width": "3px",
            "border-opacity": 0.6,
        },
    },
    # ── Topic/keyword nodes (circles) ─────────────────────────────────────────
    {
        "selector": ".topic",
        "style": {
            "background-color": "#475569",
            "border-color": "#64748b",
            "shape": "ellipse",
            "font-size": "9px",
        },
    },
    # ── Person nodes ──────────────────────────────────────────────────────────
    {
        "selector": ".person",
        "style": {
            "background-color": "#3b82f6",
            "border-color": "#2563eb",
            "shape": "ellipse",
        },
    },
    # ── Project nodes ─────────────────────────────────────────────────────────
    {
        "selector": ".project",
        "style": {
            "background-color": "#f59e0b",
            "border-color": "#d97706",
            "shape": "diamond",
        },
    },
    # ── Organization nodes ────────────────────────────────────────────────────
    {
        "selector": ".organization",
        "style": {
            "background-color": "#10b981",
            "border-color": "#059669",
            "shape": "round-rectangle",
        },
    },
    # ── Location nodes ────────────────────────────────────────────────────────
    {
        "selector": ".location",
        "style": {
            "background-color": "#f97316",
            "border-color": "#ea580c",
            "shape": "round-pentagon",
        },
    },
    # ── Date nodes ────────────────────────────────────────────────────────────
    {
        "selector": ".date",
        "style": {
            "background-color": "#1e293b",
            "border-color": "#475569",
            "shape": "round-rectangle",
            "font-size": "8px",
            "color": "#94a3b8",
        },
    },
    # ── Selected node ─────────────────────────────────────────────────────────
    {
        "selector": ":selected",
        "style": {
            "border-width": "4px",
            "border-color": "#fbbf24",
            "background-opacity": 1,
            "z-index": 999,
        },
    },
    # ── Hovered node ──────────────────────────────────────────────────────────
    {
        "selector": "node:active",
        "style": {
            "overlay-opacity": 0.15,
            "overlay-color": "#fbbf24",
        },
    },
    # ── Edge base ─────────────────────────────────────────────────────────────
    {
        "selector": "edge",
        "style": {
            "width": 1.5,
            "line-color": "#334155",
            "curve-style": "bezier",
            "opacity": 0.4,
        },
    },
    # ── Edge connecting keyword ↔ keyword (co-occurrence) ─────────────────────
    {
        "selector": "edge[label]",
        "style": {
            "label": "data(label)",
            "font-size": "7px",
            "color": "#64748b",
            "text-rotation": "autorotate",
            "text-margin-y": "-8px",
            "line-style": "dashed",
            "opacity": 0.3,
        },
    },
    # ── Edge connecting to category (solid, slightly brighter) ────────────────
    {
        "selector": "edge[target ^= 'cat:']",
        "style": {
            "opacity": 0.5,
            "line-style": "solid",
        },
    },
    # ── Selected edge ─────────────────────────────────────────────────────────
    {
        "selector": "edge:selected",
        "style": {
            "line-color": "#fbbf24",
            "width": 3,
            "opacity": 1,
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH VIEW COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════


def create_graph_view(graph_data=None) -> html.Div:
    """Create the knowledge graph view with Cytoscape and controls.

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
                    html.H2("Knowledge Graph", className="view-title"),
                    html.P(
                        f"{node_count} entities • {edge_count} connections",
                        className="view-subtitle",
                    ),
                ],
            ),
            # Controls bar
            html.Div(
                className="graph-controls",
                children=[
                    html.Div(
                        id="graph-layout-controls",
                        className="graph-layout-controls",
                        style={"display": "flex"},
                        children=[
                            dcc.Dropdown(
                                id="graph-layout-select",
                                className="graph-layout-dropdown",
                                value="cose-bilkent",
                                clearable=False,
                                searchable=False,
                                options=[
                                    {
                                        "label": "Force-Directed",
                                        "value": "cose-bilkent",
                                    },
                                    {"label": "Hierarchical", "value": "dagre"},
                                    {"label": "Circular", "value": "circle"},
                                    {"label": "Concentric", "value": "concentric"},
                                ],
                                style={"width": "180px"},
                            ),
                        ],
                    ),
                ],
            ),
            # Graph container
            html.Div(
                id="graph-canvas-container",
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
                                    "No knowledge graph data yet",
                                    style={"color": "#e6edf3", "fontSize": "1.1rem"},
                                ),
                                html.P(
                                    "Run the pipeline to extract entities and build the graph.",
                                    style={"color": "#8b949e", "fontSize": "0.875rem"},
                                ),
                                html.Code(
                                    "python scripts/chronos_pipeline.py --full",
                                    style={"color": "#a371f7"},
                                ),
                            ],
                        )
                    ]
                ),
            ),
            # Legend
            html.Div(
                id="graph-legend-container",
                className="graph-legend",
                style={"display": "flex"},
                children=[
                    html.Span("Legend:", className="legend-title"),
                    _legend_item(
                        "■", "Category", "#8b5cf6", type_counts.get("category", 0)
                    ),
                    _legend_item("●", "Topic", "#475569", type_counts.get("topic", 0)),
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
                    _legend_item("■", "Date", "#475569", type_counts.get("date", 0)),
                ],
            ),
            # Node detail panel (populated by callback)
            html.Div(
                id="graph-node-detail",
                className="graph-node-detail",
            ),
            # Recording results for clicked node (populated by callback)
            html.Div(
                id="graph-node-recordings",
                className="graph-node-recordings",
            ),
            # Hidden store for the clicked node's keyword
            dcc.Store(id="graph-clicked-keyword", data=None),
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
