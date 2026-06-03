"""Knowledge Graph visualization component using Plotly 3D Scatter plots."""

from dash import html, dcc
import networkx as nx
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional

# Keep this for backward compatibility if other modules import it
GRAPH_STYLESHEET: List[Dict[str, Any]] = []

def create_graph_view(graph_data=None) -> html.Div:
    """Create the knowledge graph view with a Plotly 3D force-directed graph.

    Args:
        graph_data: GraphData with nodes and edges lists

    Returns:
        Dash component for the graph view
    """
    node_count = 0
    edge_count = 0
    fig = None

    if graph_data and graph_data.nodes:
        node_count = len(graph_data.nodes)
        edge_count = len(graph_data.edges)
        fig = _build_plotly_3d_figure(graph_data)

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


def _build_plotly_3d_figure(graph_data) -> go.Figure:
    """Build a 3D Force-Directed Graph Figure using NetworkX and Plotly."""
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

    # 2. Compute 3D Spring Layout coordinates
    # dim=3 forces 3D coordinate vectors (x, y, z)
    pos = nx.spring_layout(G, dim=3, k=0.25, iterations=60, seed=42)

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
    # Grouping allows interactive toggling of classes in the Plotly legend
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
        line=dict(color="rgba(148, 163, 184, 0.22)", width=1.5),
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
            sizes.append(max(6, min(24, size_val / 2.8)))

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

            # Store the original Cytoscape-formatted dict in customdata
            # this makes it accessible in Dash click callbacks!
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
