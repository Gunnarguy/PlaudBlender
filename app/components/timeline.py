"""Timeline Sidebar Component."""

from dash import html, dcc


def create_timeline_component() -> html.Div:
    """Create the timeline sidebar component.

    Returns:
        Dash HTML Div containing the timeline
    """
    return html.Div(
        className="timeline-sidebar",
        children=[
            # Timeline header
            html.Div(
                className="timeline-header",
                children=[
                    html.H3("📅 Timeline", className="timeline-title"),
                    html.Button(
                        "↻ Sync",
                        id="sync-btn",
                        className="btn btn-primary btn-sm",
                        title="Sync from Plaud",
                    ),
                ],
            ),
            # Sync status
            html.Div(id="sync-status", className="sync-status"),
            # Date range filter
            html.Div(
                className="timeline-filter",
                children=[
                    dcc.Dropdown(
                        id="timeline-range",
                        options=[
                            {"label": "Today", "value": "today"},
                            {"label": "Yesterday", "value": "yesterday"},
                            {"label": "This Week", "value": "week"},
                            {"label": "This Month", "value": "month"},
                            {"label": "All Time", "value": "all"},
                        ],
                        value="all",
                        clearable=False,
                        className="timeline-dropdown",
                    ),
                ],
            ),
            # Timeline groups container (populated via callback)
            html.Div(
                id="timeline-groups",
                className="timeline-groups",
                children=[
                    # Placeholder - replaced by callback
                    html.Div(
                        className="timeline-loading",
                        children=[
                            html.Span("Loading timeline..."),
                        ],
                    ),
                ],
            ),
            # Filters section
            html.Div(
                className="timeline-filters",
                children=[
                    html.H4("🏷️ Categories", className="filter-title"),
                    html.Div(
                        id="category-filters",
                        className="category-filters",
                        # Populated via callback
                    ),
                ],
            ),
            # Stats
            html.Div(
                id="timeline-stats",
                className="timeline-stats",
            ),
        ],
    )


def create_timeline_group(
    label: str, date_key: str, count: int, expanded: bool = False
) -> html.Div:
    """Create a single timeline group (date header + expandable events).

    Args:
        label: Display label (e.g., "Today", "Jan 15, 2026")
        date_key: Date key for filtering
        count: Number of events
        expanded: Whether to show expanded by default

    Returns:
        Timeline group element
    """
    return html.Div(
        className=f"timeline-group {'expanded' if expanded else ''}",
        children=[
            html.Div(
                className="timeline-group-header",
                id={"type": "timeline-group-header", "date": date_key},
                children=[
                    html.Span(label, className="group-label"),
                    html.Span(f"({count})", className="group-count"),
                    html.Span("▼" if expanded else "▶", className="group-chevron"),
                ],
            ),
            html.Div(
                className="timeline-group-events",
                id={"type": "timeline-group-events", "date": date_key},
                style={"display": "block" if expanded else "none"},
            ),
        ],
    )


def create_timeline_event(event: dict) -> html.Div:
    """Create a single timeline event item.

    Args:
        event: Event dictionary from Qdrant

    Returns:
        Timeline event element
    """
    event_id = event.get("id", "")
    narrative = event.get("narrative", "")[:100]
    category = event.get("category", "general")
    timestamp = event.get("event_timestamp", "")

    # Format time if available
    time_str = ""
    if timestamp:
        try:
            from datetime import datetime

            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%I:%M %p")
        except Exception:
            pass

    return html.Div(
        className="timeline-event",
        id={"type": "timeline-event", "id": event_id},
        children=[
            html.Div(
                className="event-header",
                children=[
                    html.Span(time_str, className="event-time") if time_str else None,
                    html.Span(category, className=f"event-category cat-{category}"),
                ],
            ),
            html.P(
                narrative + ("..." if len(event.get("narrative", "")) > 100 else ""),
                className="event-preview",
            ),
        ],
    )
