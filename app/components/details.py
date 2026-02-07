"""Event Details Panel Component."""

from dash import html


def create_details_component() -> html.Div:
    """Create the event details panel component.

    Returns:
        Dash HTML Div containing the details panel
    """
    return html.Div(
        className="details-panel",
        id="details-panel",
        children=[
            # Placeholder shown when no event is selected
            html.Div(
                id="details-placeholder",
                className="details-placeholder",
                children=[
                    html.Div(className="placeholder-icon", children="🎯"),
                    html.H4("Select an Event"),
                    html.P(
                        "Click on a node in the graph or an event in the timeline to see details."
                    ),
                ],
            ),
            # Actual details (hidden by default)
            html.Div(
                id="details-content",
                className="details-content",
                style={"display": "none"},
            ),
        ],
    )


def create_event_details(event: dict, related_events: list = None) -> html.Div:
    """Create the event details view.

    Args:
        event: Event dictionary from Qdrant
        related_events: List of related events

    Returns:
        Event details element
    """
    narrative = event.get("narrative", "No narrative available")
    category = event.get("category", "general")
    timestamp = event.get("event_timestamp", "")
    recording_id = event.get("recording_id", "")
    duration = event.get("duration_seconds", 0)
    actors = event.get("actors", [])
    emotions = event.get("emotional_tone", [])
    importance = event.get("importance", 5)

    # Format timestamp
    date_str = ""
    time_str = ""
    if timestamp:
        try:
            from datetime import datetime

            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                dt = datetime.fromtimestamp(timestamp)
            date_str = dt.strftime("%B %d, %Y")
            time_str = dt.strftime("%I:%M %p")
        except Exception:
            pass

    # Format duration
    duration_str = ""
    if duration:
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    return html.Div(
        className="event-details",
        children=[
            # Header
            html.Div(
                className="details-header",
                children=[
                    html.Span(
                        category.title(), className=f"category-badge cat-{category}"
                    ),
                    html.Div(
                        className="importance-stars",
                        children=[
                            html.Span("★" if i < importance else "☆") for i in range(10)
                        ],
                        title=f"Importance: {importance}/10",
                    ),
                ],
            ),
            # Date & Time
            html.Div(
                className="details-datetime",
                children=[
                    html.Span(f"📅 {date_str}", className="date") if date_str else None,
                    html.Span(f"🕐 {time_str}", className="time") if time_str else None,
                    (
                        html.Span(f"⏱️ {duration_str}", className="duration")
                        if duration_str
                        else None
                    ),
                ],
            ),
            # Narrative
            html.Div(
                className="details-narrative",
                children=[
                    html.H4("What Happened"),
                    html.P(narrative),
                ],
            ),
            # Actors
            (
                html.Div(
                    className="details-actors",
                    children=[
                        html.H4("People Involved"),
                        html.Div(
                            className="actor-tags",
                            children=(
                                [
                                    html.Span(actor, className="actor-tag")
                                    for actor in (
                                        actors if isinstance(actors, list) else []
                                    )
                                ]
                                if actors
                                else [
                                    html.Span(
                                        "No people identified", className="no-data"
                                    )
                                ]
                            ),
                        ),
                    ],
                )
                if actors
                else None
            ),
            # Emotional Tone
            html.Div(
                className="details-emotions",
                children=[
                    html.H4("Emotional Tone"),
                    html.Div(
                        className="emotion-tags",
                        children=(
                            [
                                html.Span(emotion, className="emotion-tag")
                                for emotion in (
                                    emotions if isinstance(emotions, list) else []
                                )
                            ]
                            if emotions
                            else [html.Span("Neutral", className="emotion-tag neutral")]
                        ),
                    ),
                ],
            ),
            # Related Events
            html.Div(
                className="details-related",
                children=[
                    html.H4(f"📎 Related Events ({len(related_events or [])})"),
                    html.Div(
                        className="related-list",
                        children=(
                            [
                                html.Div(
                                    className="related-item",
                                    id={
                                        "type": "related-event",
                                        "id": rev.get("id", ""),
                                    },
                                    children=[
                                        html.Span(
                                            rev.get("narrative", "")[:80] + "...",
                                            className="related-text",
                                        ),
                                        html.Span(
                                            f"{rev.get('score', 0):.0%}",
                                            className="related-score",
                                        ),
                                    ],
                                )
                                for rev in (related_events or [])[:5]
                            ]
                            if related_events
                            else [
                                html.Span(
                                    "No related events found", className="no-data"
                                )
                            ]
                        ),
                    ),
                ],
            ),
            # Metadata
            html.Details(
                className="details-meta",
                children=[
                    html.Summary("Technical Details"),
                    html.Div(
                        className="meta-grid",
                        children=[
                            html.Div(
                                [
                                    html.Strong("Event ID:"),
                                    html.Code(event.get("id", "")[:12]),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Strong("Recording:"),
                                    html.Code(
                                        recording_id[:12] if recording_id else "N/A"
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_entity_details(
    entity_name: str, entity_type: str, related_events: list
) -> html.Div:
    """Create the entity details view showing events that mention this entity.

    Args:
        entity_name: Name of the entity (topic, action, person, etc.)
        entity_type: Type of entity
        related_events: List of events that mention this entity

    Returns:
        Entity details element showing related content
    """
    # Type icons
    type_icons = {
        "topic": "💡",
        "action": "⚡",
        "person": "👤",
        "location": "📍",
        "organization": "🏢",
        "concept": "🎯",
        "event": "📅",
        "unknown": "❓",
    }

    icon = type_icons.get(entity_type.lower(), "🔍")

    # Build event list
    event_items = []
    for event in (related_events or [])[:10]:
        narrative = event.get("narrative", "")[:500]
        if len(event.get("narrative", "")) > 500:
            narrative += "..."

        timestamp = event.get("event_timestamp", "")
        date_str = ""
        if timestamp:
            try:
                from datetime import datetime

                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%b %d, %Y")
            except:
                pass

        category = event.get("category", "general")
        score = event.get("score", 0)

        event_items.append(
            html.Div(
                className="entity-event-item",
                id={"type": "entity-event", "id": event.get("id", "")},
                children=[
                    html.Div(
                        className="entity-event-header",
                        children=[
                            html.Span(
                                category.title(),
                                className=f"category-badge cat-{category}",
                            ),
                            (
                                html.Span(date_str, className="event-date")
                                if date_str
                                else None
                            ),
                            (
                                html.Span(f"{score:.0%}", className="relevance-score")
                                if score
                                else None
                            ),
                        ],
                    ),
                    html.P(narrative, className="entity-event-narrative"),
                ],
            )
        )

    if not event_items:
        event_items = [
            html.Div(
                "No related events found. Try searching for this topic.",
                className="no-data",
            )
        ]

    return html.Div(
        className="entity-details",
        children=[
            # Header
            html.Div(
                className="entity-header",
                children=[
                    html.Span(
                        icon, className="entity-icon", style={"fontSize": "2rem"}
                    ),
                    html.Div(
                        [
                            html.H3(
                                entity_name, style={"margin": "0", "color": "#f1f5f9"}
                            ),
                            html.Span(
                                entity_type.title(),
                                className="entity-type-badge",
                                style={
                                    "background": "#374151",
                                    "padding": "2px 8px",
                                    "borderRadius": "4px",
                                    "fontSize": "0.75rem",
                                    "color": "#9ca3af",
                                },
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                    "marginBottom": "16px",
                },
            ),
            # Related content header
            html.H4(
                f"📝 Related Content ({len(related_events or [])} events)",
                style={"marginBottom": "12px", "color": "#94a3b8"},
            ),
            # Event list
            html.Div(
                className="entity-events-list",
                children=event_items,
                style={
                    "maxHeight": "calc(100vh - 300px)",
                    "overflowY": "auto",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "12px",
                },
            ),
        ],
    )
