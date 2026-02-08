"""Topic timeline view component."""

from dash import html
from typing import List, Tuple

from app_v2.services.data_service import TopicTimeline, TopicOccurrence


def create_topic_card(topic: str, count: int) -> html.Div:
    """Create a clickable topic card."""
    return html.Div(
        id={"type": "topic-card", "topic": topic},
        className="topic-card",
        children=[
            html.Span(topic, className="topic-name"),
            html.Span(f"{count}", className="topic-count"),
        ],
    )


def create_topics_grid(topics: List[Tuple[str, int]]) -> html.Div:
    """Create a grid of all topics."""
    if not topics:
        return html.Div(
            className="topics-empty",
            children=[
                html.Span("💡", className="empty-icon"),
                html.P("No topics found yet."),
            ],
        )

    return html.Div(
        className="topics-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("Topics", className="view-title"),
                    html.P(
                        f"{len(topics)} unique topics from your recordings",
                        className="view-subtitle",
                    ),
                ],
            ),
            html.Div(
                className="topics-grid",
                children=[
                    create_topic_card(topic, count)
                    for topic, count in topics[:50]  # Limit to top 50
                ],
            ),
        ],
    )


def create_occurrence_card(occurrence: TopicOccurrence) -> html.Div:
    """Create a card for a topic occurrence."""
    date_str = occurrence.timestamp.strftime("%b %d")
    time_str = occurrence.timestamp.strftime("%I:%M %p")

    return html.Div(
        id={
            "type": "occurrence-card",
            "id": occurrence.event_id,
            "recording_id": occurrence.recording_id,
        },
        className="occurrence-card",
        children=[
            html.Div(
                className="occurrence-header",
                children=[
                    html.Span(f"📅 {date_str}", className="occ-date"),
                    html.Span(f"🕐 {time_str}", className="occ-time"),
                    html.Span(
                        occurrence.category,
                        className=f"occ-category cat-{occurrence.category}",
                    ),
                ],
            ),
            html.P(occurrence.text_snippet, className="occurrence-text"),
        ],
    )


def create_topic_timeline_view(timeline: TopicTimeline) -> html.Div:
    """Create the topic timeline view."""
    if not timeline.occurrences:
        return html.Div(
            className="topic-timeline empty",
            children=[
                html.Span("💡", className="empty-icon"),
                html.P(f'No occurrences found for "{timeline.topic}"'),
            ],
        )

    return html.Div(
        className="topic-timeline",
        children=[
            # Back button
            html.Button(
                id="back-to-topics-btn",
                className="back-btn",
                children=[
                    html.Span("←", className="back-icon"),
                    html.Span("Back to Topics", className="back-text"),
                ],
            ),
            # Header
            html.Div(
                className="topic-header",
                children=[
                    html.H2(f'💡 Topic: "{timeline.topic}"', className="topic-title"),
                    html.Div(
                        className="topic-stats",
                        children=[
                            html.Span(
                                f"{timeline.total_occurrences} occurrences",
                                className="stat",
                            ),
                            html.Span("•", className="stat-sep"),
                            html.Span(
                                f"{timeline.recording_count} recordings",
                                className="stat",
                            ),
                            html.Span("•", className="stat-sep"),
                            html.Span(f"{timeline.day_count} days", className="stat"),
                        ],
                    ),
                ],
            ),
            # Timeline visual (simplified)
            html.Div(
                className="topic-timeline-visual",
                children=[
                    html.H4("Timeline", className="section-title"),
                    html.Div(
                        className="timeline-dots",
                        children=[
                            html.Span(
                                occurrence.timestamp.strftime("%b %d"),
                                className="timeline-dot",
                                title=occurrence.text_snippet[:100],
                            )
                            for occurrence in timeline.occurrences[:10]
                        ],
                    ),
                ],
            ),
            # Occurrences list
            html.Div(
                className="occurrences-section",
                children=[
                    html.H4("Occurrences", className="section-title"),
                    html.Div(
                        className="occurrences-list",
                        children=[
                            create_occurrence_card(occ) for occ in timeline.occurrences
                        ],
                    ),
                ],
            ),
        ],
    )
