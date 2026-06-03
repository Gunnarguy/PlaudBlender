"""Knowledge Graph visualization component using Plotly 3D Scatter plots."""

from dash import html, dcc
import networkx as nx
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional

# Keep this for backward compatibility if other modules import it
GRAPH_STYLESHEET: List[Dict[str, Any]] = []

def create_graph_view(graph_data=None, layout_name: str = "lanes") -> html.Div:
    """Create the knowledge graph view with a Plotly 3D force-directed graph.

    Args:
        graph_data: GraphData with nodes and edges lists
        layout_name: The layout type (lanes, levels, orbit, timeline, force)

    Returns:
        Dash component for the graph view
    """
    node_count = 0
    edge_count = 0
    fig = None

    if graph_data and graph_data.nodes:
        node_count = len(graph_data.nodes)
        edge_count = len(graph_data.edges)
        fig = _build_plotly_3d_figure(graph_data, layout_name)

    # Count entity types for legend (optional secondary display)
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
                    html.H2("Knowledge Graph (3D Space)", className="view-title"),
                    html.P(
                        f"{node_count} entities • {edge_count} connections • Rotate, zoom, and click nodes to explore",
                        className="view-subtitle",
                    ),
                ],
            ),
            # Layout control dropdown
            html.Div(
                className="graph-controls",
                style={"marginBottom": "20px", "padding": "12px", "backgroundColor": "rgba(15, 23, 42, 0.4)", "borderRadius": "12px", "border": "1px solid rgba(255,255,255,0.05)"},
                children=[
                    html.Span("Visualization Mode: ", style={"color": "#94a3b8", "fontWeight": "600", "marginRight": "12px"}),
                    dcc.Dropdown(
                        id="web-graph-layout-select",
                        options=[
                            {"label": "Lanes (Category Columns)", "value": "lanes"},
                            {"label": "Levels (Hierarchical Layers)", "value": "levels"},
                            {"label": "Orbit (Concentric Shells)", "value": "orbit"},
                            {"label": "Timeline (Chronological Helix)", "value": "timeline"},
                            {"label": "Force (Standard physics)", "value": "force"},
                        ],
                        value=layout_name,
                        clearable=False,
                        searchable=False,
                        style={"width": "260px", "display": "inline-block", "verticalAlign": "middle", "color": "#0f172a"},
                    ),
                ]
            ),
            # Graph container
            html.Div(
                id="graph-canvas-container",
                className="graph-canvas",
                style={"minHeight": "650px", "position": "relative"},
                children=(
                    [
                        dcc.Graph(
                            id="knowledge-graph",
                            figure=fig,
                            style={
                                "width": "100%",
                                "height": "650px",
                            },
                            config={
                                "scrollZoom": True,
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            }
                        ),
                    ]
                    if fig
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


def _compute_layout_positions(G, node_props, layout_type="lanes"):
    import math
    pos = {}
    nodes = list(G.nodes())

    if layout_type == "lanes":
        categories = [n for n in nodes if node_props.get(n, {}).get("type") == "category"]
        others = [n for n in nodes if node_props.get(n, {}).get("type") != "category"]

        num_cats = max(1, len(categories))
        lane_radius = 2.4

        category_map = {}
        for idx, cat in enumerate(categories):
            theta = (idx / num_cats) * 2 * math.pi
            pos[cat] = (lane_radius * math.cos(theta), 0.5, lane_radius * math.sin(theta))
            category_map[cat] = idx

        lane_groups = {}
        for node in others:
            props = node_props.get(node, {})
            cats_str = props.get("categories", "")
            primary_cat = "default"
            if cats_str:
                cats_list = [c.strip() for c in cats_str.split(",") if c.strip()]
                if cats_list:
                    primary_cat = "cat:" + cats_list[0]
            lane_groups.setdefault(primary_cat, []).append(node)

        for cat_id, group_nodes in lane_groups.items():
            cat_idx = category_map.get(cat_id, num_cats)
            theta = 0.0 if cat_idx == num_cats else (cat_idx / num_cats) * 2 * math.pi
            base_x = 0.0 if cat_idx == num_cats else lane_radius * math.cos(theta)
            base_z = 0.0 if cat_idx == num_cats else lane_radius * math.sin(theta)

            group_nodes.sort(key=lambda n: node_props.get(n, {}).get("avg_ts", 0.0))

            for idx, node in enumerate(group_nodes):
                y_offset = (idx - (len(group_nodes) - 1) / 2) * 0.14
                pos[node] = (
                    base_x + math.sin(idx) * 0.06,
                    y_offset - 0.4,
                    base_z + math.cos(idx) * 0.06
                )
    elif layout_type in ("levels", "breadthfirst"):
        layer0 = [n for n in nodes if node_props.get(n, {}).get("type") == "category"]
        layer1 = [n for n in nodes if node_props.get(n, {}).get("type") == "topic"]
        layer2 = [n for n in nodes if node_props.get(n, {}).get("type") not in ("category", "topic")]

        def layout_layer(layer_nodes, y_val, radius):
            count = max(1, len(layer_nodes))
            layer_nodes.sort(key=lambda n: node_props.get(n, {}).get("avg_ts", 0.0))
            for idx, node in enumerate(layer_nodes):
                theta = (idx / count) * 2 * math.pi
                pos[node] = (radius * math.cos(theta), y_val, radius * math.sin(theta))

        layout_layer(layer0, 1.2, 0.7)
        layout_layer(layer1, 0.0, 1.6)
        layout_layer(layer2, -1.2, 2.4)
    elif layout_type in ("orbit", "concentric"):
        sorted_nodes = sorted(nodes, key=lambda n: node_props.get(n, {}).get("count", 0), reverse=True)
        categories = [n for n in sorted_nodes if node_props.get(n, {}).get("type") == "category"]
        others = [n for n in sorted_nodes if node_props.get(n, {}).get("type") != "category"]

        num_cats = max(1, len(categories))
        for idx, cat in enumerate(categories):
            theta = (idx / num_cats) * 2 * math.pi
            phi = math.acos(-1.0 + (2.0 * idx) / num_cats)
            r = 0.4
            pos[cat] = (
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi)
            )

        others.sort(key=lambda n: node_props.get(n, {}).get("avg_ts", 0.0), reverse=True)
        for idx, node in enumerate(others):
            shell = idx % 3
            r = 0.9 + shell * 0.7
            count_in_shell = math.ceil(len(others) / 3)
            idx_in_shell = idx // 3

            theta = (idx_in_shell / count_in_shell) * 2 * math.pi
            phi = math.acos(-1.0 + (2.0 * idx_in_shell) / count_in_shell)
            pos[node] = (
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi)
            )
    elif layout_type in ("timeline", "circle"):
        sorted_nodes = sorted(nodes, key=lambda n: node_props.get(n, {}).get("avg_ts", 0.0))
        N = len(sorted_nodes)
        for idx, node in enumerate(sorted_nodes):
            theta = (idx / max(1, N)) * 2 * math.pi * 3.5
            radius = 1.5 - (idx / max(1, N)) * 0.4
            pos[node] = (
                radius * math.cos(theta),
                (idx / max(1, N)) * 3.6 - 1.8,
                radius * math.sin(theta)
            )
    else:
        import networkx as nx
        pos = nx.spring_layout(G, dim=3, k=0.25, iterations=60, seed=42)

    for n in nodes:
        if n not in pos:
            pos[n] = (0.0, 0.0, 0.0)

    return pos


def _build_plotly_3d_figure(graph_data, layout_name: str = "lanes") -> go.Figure:
    """Build a 3D Graph Figure using NetworkX and Plotly."""
    # 1. Construct NetworkX graph
    G = nx.Graph()

    # Store node properties for reference
    node_props = {}
    for node in graph_data.nodes:
        data = node.get("data", {})
        node_id = data.get("id")
        G.add_node(node_id)
        node_props[node_id] = data

    for edge in graph_data.edges:
        data = edge.get("data", {})
        G.add_edge(data.get("source"), data.get("target"), weight=data.get("weight", 1))

    # 2. Compute Layout coordinates
    pos = _compute_layout_positions(G, node_props, layout_name)

    # 3. Create Edge Line segments
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    # 4. Group Nodes by Class for styled traces
    type_groups: Dict[str, List[str]] = {}
    for node_id in G.nodes():
        props = node_props.get(node_id, {})
        ntype = props.get("type", "topic")
        type_groups.setdefault(ntype, []).append(node_id)

    # Color system matching Cytoscape views
    type_colors = {
        "category": "#a855f7", # Violet
        "topic": "#64748b",     # Slate
        "person": "#3b82f6",    # Blue
        "project": "#eab308",   # Amber
        "organization": "#10b981", # Emerald
        "location": "#f97316",  # Orange
        "date": "#475569"       # Muted slate
    }

    type_labels = {
        "category": "📁 Categories",
        "topic": "💬 Topics",
        "person": "👤 People",
        "project": "📌 Projects",
        "organization": "🏢 Organizations",
        "location": "📍 Locations",
        "date": "📅 Dates"
    }

    traces = []

    # A. Draw Edges Trace (thin, semi-transparent grey lines)
    edges_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.18)", width=1.5),
        hoverinfo="none",
        showlegend=False,
        name="Connections"
    )
    traces.append(edges_trace)

    # B. Draw Nodes Traces (grouped by type)
    for ntype, ids in type_groups.items():
        x_coords = []
        y_coords = []
        z_coords = []
        sizes = []
        hover_texts = []
        custom_data_list = []

        color = type_colors.get(ntype, "#6366f1")
        name = type_labels.get(ntype, ntype.capitalize())

        for node_id in ids:
            x, y, z = pos[node_id]
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

            props = node_props.get(node_id, {})
            # Determine size dynamically
            size_val = props.get("size", 20)
            base_size = max(8, min(24, size_val / 2.8))
            if ntype == "category":
                sizes.append(base_size * 1.5)
            else:
                sizes.append(base_size)

            # Build detailed hover text
            label = props.get("label", node_id)
            count = props.get("count", props.get("mention_count", 0))
            sentiment = props.get("sentiment", 0)

            tooltip = f"<b>{label}</b> [{ntype.upper()}]<br/>"
            if count:
                tooltip += f"Mentions: {count}<br/>"
            if sentiment:
                tooltip += f"Sentiment: {'+' if sentiment > 0 else ''}{sentiment:.2f}<br/>"
            hover_texts.append(tooltip)

            custom_data_list.append(props)

        node_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            name=name,
            marker=dict(
                symbol="circle",
                size=sizes,
                color=color,
                line=dict(color="#0f172a", width=1.5),
                opacity=0.92
            ),
            text=hover_texts,
            hoverinfo="text",
            customdata=custom_data_list,
            showlegend=True
        )
        traces.append(node_trace)

    # 5. Build Layout
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            font=dict(color="#e2e8f0", size=10),
            bgcolor="rgba(9, 13, 22, 0.7)"
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title=""),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="rgba(99, 102, 241, 0.4)",
            font=dict(color="#f8fafc", size=11, family="sans-serif")
        )
    )

    return go.Figure(data=traces, layout=layout)
