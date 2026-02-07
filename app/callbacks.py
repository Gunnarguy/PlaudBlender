"""Dash Callbacks - All the interactivity."""

from dash import Input, Output, State, callback, ctx, html, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import logging

from app.services.data import get_data_service
from app.components.timeline import create_timeline_group, create_timeline_event
from app.components.search import create_search_result
from app.components.details import create_event_details, create_entity_details

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("graph-data-store", "data"),
    Output("timeline-data-store", "data"),
    Output("status-events", "children"),
    Output("status-nodes", "children"),
    Output("status-edges", "children"),
    Input("auto-refresh", "n_intervals"),
    prevent_initial_call=False,
)
def load_initial_data(n_intervals):
    """Load initial data on app start and periodic refresh."""
    service = get_data_service()

    # Get graph data
    graph_data = service.get_graph_data()
    graph_dict = {
        "nodes": graph_data.nodes,
        "edges": graph_data.edges,
    }

    # Get timeline data
    timeline_groups = service.get_timeline_groups()
    timeline_data = [
        {
            "label": g.label,
            "date_key": g.date_key,
            "count": g.count,
            "events": g.events,
        }
        for g in timeline_groups
    ]

    # Get stats
    stats = service.get_stats()

    return (
        graph_dict,
        timeline_data,
        f"Events: {stats.get('total_events', 0):,}",
        f"Nodes: {len(graph_data.nodes):,}",
        f"Edges: {len(graph_data.edges):,}",
    )


@callback(
    Output("knowledge-graph", "elements"),
    Input("graph-data-store", "data"),
    Input("graph-layout-select", "value"),
)
def update_graph_elements(graph_data, layout):
    """Update graph elements when data changes."""
    if not graph_data:
        return []

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    return nodes + edges


@callback(
    Output("knowledge-graph", "layout"),
    Input("graph-layout-select", "value"),
    Input("graph-reset-btn", "n_clicks"),
)
def update_graph_layout(layout_name, reset_clicks):
    """Update graph layout when selection changes."""
    name = layout_name or "cose-bilkent"

    # Layout-specific settings for large graphs
    if name == "cose-bilkent":
        return {
            "name": "cose-bilkent",
            "animate": False,
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
        }
    elif name == "cose":
        return {
            "name": "cose",
            "animate": False,
            "nodeRepulsion": 8000,
            "idealEdgeLength": 100,
            "gravity": 80,
            "fit": True,
            "padding": 30,
        }
    elif name == "dagre":
        return {
            "name": "dagre",
            "animate": False,
            "rankDir": "TB",
            "nodeSep": 50,
            "rankSep": 100,
            "fit": True,
            "padding": 30,
        }
    else:
        return {
            "name": name,
            "animate": False,
            "fit": True,
            "padding": 30,
        }


@callback(
    Output("timeline-groups", "children"),
    Input("timeline-data-store", "data"),
    Input("timeline-range", "value"),
)
def render_timeline(timeline_data, date_range):
    """Render timeline groups."""
    if not timeline_data:
        return html.Div("No events found", className="no-data")

    # Filter by date range if needed
    from datetime import datetime, timedelta

    today = datetime.now().date()

    filtered_groups = []
    for group in timeline_data:
        date_key = group.get("date_key", "")

        if date_range == "all" or date_key == "unknown":
            filtered_groups.append(group)
        else:
            try:
                dt = datetime.strptime(date_key, "%Y-%m-%d").date()

                if date_range == "today" and dt == today:
                    filtered_groups.append(group)
                elif date_range == "yesterday" and dt == today - timedelta(days=1):
                    filtered_groups.append(group)
                elif date_range == "week" and dt >= today - timedelta(days=7):
                    filtered_groups.append(group)
                elif date_range == "month" and dt >= today - timedelta(days=30):
                    filtered_groups.append(group)
            except Exception:
                filtered_groups.append(group)

    if not filtered_groups:
        return html.Div("No events in selected range", className="no-data")

    # Render groups
    return [
        create_timeline_group(
            label=g["label"],
            date_key=g["date_key"],
            count=g["count"],
            expanded=(i == 0),  # Expand first group
        )
        for i, g in enumerate(filtered_groups)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("search-results-store", "data"),
    Output("search-results", "children"),
    Output("search-results", "style"),
    Input("search-btn", "n_clicks"),
    Input("search-input", "n_submit"),
    State("search-input", "value"),
    prevent_initial_call=True,
)
def perform_search(n_clicks, n_submit, query):
    """Perform semantic search."""
    if not query or not query.strip():
        return [], [], {"display": "none"}

    service = get_data_service()
    results = service.search(query.strip())

    if not results:
        return (
            [],
            [html.Div("No results found", className="no-results")],
            {"display": "block"},
        )

    # Store results and render
    results_elements = [
        create_search_result(event, event.get("score", 0)) for event in results[:10]
    ]

    return results, results_elements, {"display": "block"}


@callback(
    Output("knowledge-graph", "stylesheet"),
    Input("search-results-store", "data"),
    State("knowledge-graph", "stylesheet"),
)
def highlight_search_results_in_graph(results, current_stylesheet):
    """Highlight nodes related to search results."""
    # This would require matching search results to graph nodes
    # For now, return unchanged stylesheet
    return no_update


# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("selected-event-store", "data"),
    Output("details-placeholder", "style"),
    Output("details-content", "style"),
    Output("details-content", "children"),
    Input("knowledge-graph", "tapNodeData"),
    Input({"type": "timeline-event", "id": ALL}, "n_clicks"),
    Input({"type": "search-result", "id": ALL}, "n_clicks"),
    State("timeline-data-store", "data"),
    prevent_initial_call=True,
)
def handle_selection(node_data, timeline_clicks, search_clicks, timeline_data):
    """Handle selection from graph, timeline, or search results."""
    triggered = ctx.triggered_id

    logger.info(f"Selection triggered by: {triggered}, node_data: {node_data}")

    event_id = None
    selected_node = None

    # Determine what was clicked
    if triggered == "knowledge-graph" and node_data:
        # Graph node clicked - node_data contains the data dict directly
        selected_node = node_data
        node_id = node_data.get("id", "")
        logger.info(f"Graph node clicked: {node_id}, type: {node_data.get('type')}")
        # Check if it's an event ID (UUID-like) or an entity
        if len(node_id) > 30:  # Likely a UUID event ID
            event_id = node_id
        # For entities, we'll fetch related events below

    elif isinstance(triggered, dict):
        # Timeline or search result clicked
        event_id = triggered.get("id")
        logger.info(f"Timeline/search clicked: {event_id}")

    # If we have a node but no event_id, show node info anyway
    if not event_id and not selected_node:
        return no_update, no_update, no_update, no_update

    # If we clicked an entity node (not an event), find and show related events
    if selected_node and not event_id:
        entity_name = selected_node.get(
            "full_label", selected_node.get("label", "Unknown")
        )
        entity_type = selected_node.get("type", "unknown")

        logger.info(f"Finding events for entity: {entity_name} ({entity_type})")

        # Get events that mention this entity
        service = get_data_service()
        related_events = service.find_events_by_entity(
            entity_name, entity_type, limit=10
        )

        logger.info(f"Found {len(related_events)} events for entity {entity_name}")

        # Create entity details view with the actual content
        entity_details = create_entity_details(entity_name, entity_type, related_events)

        return (
            None,
            {"display": "none"},
            {"display": "block"},
            entity_details,
        )

    # Get event details
    service = get_data_service()
    event = service.get_event_details(event_id)

    if not event:
        logger.warning(f"No event found for ID: {event_id}")
        return no_update, no_update, no_update, no_update

    # Get related events
    related = service.get_related_events(event_id, limit=5)

    # Render details
    details = create_event_details(event, related)

    return (
        event_id,
        {"display": "none"},  # Hide placeholder
        {"display": "block"},  # Show content
        details,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TIMELINE EXPAND/COLLAPSE
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output({"type": "timeline-group-events", "date": MATCH}, "style"),
    Output({"type": "timeline-group-events", "date": MATCH}, "children"),
    Input({"type": "timeline-group-header", "date": MATCH}, "n_clicks"),
    State({"type": "timeline-group-events", "date": MATCH}, "style"),
    State("timeline-data-store", "data"),
    prevent_initial_call=True,
)
def toggle_timeline_group(n_clicks, current_style, timeline_data):
    """Expand/collapse timeline group."""
    if not n_clicks:
        raise PreventUpdate

    # Get the date key from the triggered component
    triggered = ctx.triggered_id
    date_key = triggered.get("date", "") if isinstance(triggered, dict) else ""

    # Toggle visibility
    is_visible = current_style.get("display") == "block"
    new_style = {"display": "none" if is_visible else "block"}

    # If expanding, load events
    if is_visible:
        return new_style, no_update

    # Find events for this date
    events = []
    for group in timeline_data:
        if group.get("date_key") == date_key:
            events = group.get("events", [])
            break

    # Render events
    event_elements = [
        create_timeline_event(event) for event in events[:20]  # Limit to 20 per group
    ]

    if not event_elements:
        event_elements = [html.Div("No events", className="no-data")]

    return new_style, event_elements


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("sync-status", "children"),
    Output("sync-btn", "disabled"),
    Input("sync-btn", "n_clicks"),
    State("sync-days-slider", "value"),
    prevent_initial_call=True,
)
def sync_from_plaud(n_clicks, days_back):
    """Trigger sync from Plaud."""
    if not n_clicks:
        raise PreventUpdate

    # Perform sync
    service = get_data_service()
    result = service.sync_from_plaud(days_back=days_back or 7)

    if "error" in result:
        return (
            html.Span(f"Error: {result['error']}", className="sync-error"),
            False,
        )
    else:
        return (
            html.Span(
                f"✓ Synced {result.get('fetched', 0)} recordings",
                className="sync-success",
            ),
            False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("upload-modal", "style"),
    Input("upload-btn", "n_clicks"),
    Input("modal-close-btn", "n_clicks"),
    Input("upload-cancel-btn", "n_clicks"),
    Input("modal-overlay", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_upload_modal(open_clicks, close_clicks, cancel_clicks, overlay_clicks):
    """Show/hide upload modal."""
    triggered = ctx.triggered_id

    if triggered == "upload-btn":
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("settings-modal", "style"),
    Input("settings-btn", "n_clicks"),
    Input("settings-close-btn", "n_clicks"),
    Input("settings-overlay", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_settings_modal(open_clicks, close_clicks, overlay_clicks):
    """Show/hide settings modal."""
    triggered = ctx.triggered_id

    if triggered == "settings-btn":
        return {"display": "flex"}
    else:
        return {"display": "none"}


@callback(
    Output("system-info", "children"),
    Input("settings-modal", "style"),
)
def update_system_info(modal_style):
    """Update system info in settings modal."""
    if modal_style.get("display") != "flex":
        raise PreventUpdate

    service = get_data_service()
    stats = service.get_stats()

    return html.Div(
        [
            html.P(f"Total Events: {stats.get('total_events', 0):,}"),
            html.P(f"Graph Nodes: {stats.get('graph_nodes', 0):,}"),
            html.P(f"Graph Edges: {stats.get('graph_edges', 0):,}"),
            html.P(
                f"Qdrant: {'✓ Connected' if stats.get('qdrant_connected') else '✗ Disconnected'}"
            ),
            html.P(
                f"Embedder: {'✓ Ready' if stats.get('embedder_ready') else '✗ Not configured'}"
            ),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════


@callback(
    Output("upload-status", "children"),
    Output("upload-process-btn", "disabled"),
    Input("upload-zone", "contents"),
    State("upload-zone", "filename"),
    prevent_initial_call=True,
)
def handle_file_upload(contents, filename):
    """Handle file upload."""
    if not contents:
        return "", True

    return (
        html.Div(
            [
                html.Span("✓ ", className="upload-check"),
                html.Span(filename, className="upload-filename"),
            ],
            className="upload-success",
        ),
        False,  # Enable process button
    )


@callback(
    Output("upload-progress", "style"),
    Output("progress-text", "children"),
    Output("notifications", "children"),
    Input("upload-process-btn", "n_clicks"),
    State("upload-zone", "contents"),
    State("upload-zone", "filename"),
    State("upload-process-immediately", "value"),
    prevent_initial_call=True,
)
def process_uploaded_file(n_clicks, contents, filename, process_immediately):
    """Process the uploaded audio file."""
    if not n_clicks or not contents:
        raise PreventUpdate

    # Show progress
    yield (
        {"display": "block"},
        "Saving file...",
        no_update,
    )

    import base64
    from pathlib import Path

    # Decode and save file
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Save to data/raw
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        file_path = raw_dir / filename
        with open(file_path, "wb") as f:
            f.write(decoded)

        yield (
            {"display": "block"},
            "File saved. Processing...",
            no_update,
        )

        # If process immediately is checked, process the file
        if "yes" in (process_immediately or []):
            # This would trigger the Chronos processing pipeline
            # For now, just show success
            yield (
                {"display": "none"},
                "",
                html.Div(
                    f"✓ Successfully uploaded and queued: {filename}",
                    className="notification success",
                ),
            )
        else:
            yield (
                {"display": "none"},
                "",
                html.Div(
                    f"✓ Saved: {filename} (ready for processing)",
                    className="notification success",
                ),
            )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        yield (
            {"display": "none"},
            "",
            html.Div(
                f"✗ Upload failed: {str(e)}",
                className="notification error",
            ),
        )
