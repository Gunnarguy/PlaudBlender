"""Recording detail view component."""

from dash import html, dcc
from typing import Optional
from datetime import datetime

from app_v2.services.data_service import RecordingDetail, Event


def create_event_card(event: Event, index: int) -> html.Div:
    """Create a card for a single event."""
    # Format time
    time_str = event.start_ts.strftime("%I:%M %p")
    end_time = event.end_ts.strftime("%I:%M %p")

    # Sentiment indicator
    if event.sentiment > 0.3:
        sentiment_icon = "😊"
        sentiment_class = "positive"
    elif event.sentiment < -0.3:
        sentiment_icon = "😔"
        sentiment_class = "negative"
    else:
        sentiment_icon = "😐"
        sentiment_class = "neutral"

    # Category colors
    category_colors = {
        "work": "#3b82f6",
        "personal": "#8b5cf6",
        "meeting": "#f59e0b",
        "reflection": "#10b981",
        "idea": "#ec4899",
        "deep_work": "#6366f1",
        "break": "#64748b",
        "unknown": "#374151",
    }
    cat_color = category_colors.get(event.category, "#374151")

    return html.Div(
        id={"type": "event-card", "id": event.id},
        className="event-card",
        children=[
            # Left timeline marker
            html.Div(
                className="event-marker",
                style={"backgroundColor": cat_color},
            ),
            # Main content
            html.Div(
                className="event-content",
                children=[
                    # Header
                    html.Div(
                        className="event-header",
                        children=[
                            html.Span(
                                f"{time_str} - {end_time}", className="event-time"
                            ),
                            html.Span(
                                event.category,
                                className="event-category",
                                style={"backgroundColor": cat_color},
                            ),
                            html.Span(
                                f"{sentiment_icon} {event.sentiment:.1f}",
                                className=f"event-sentiment {sentiment_class}",
                            ),
                        ],
                    ),
                    # Text content
                    html.P(event.clean_text, className="event-text"),
                    # Keywords
                    html.Div(
                        className="event-keywords",
                        children=(
                            [
                                html.Span(kw, className="keyword-tag small")
                                for kw in event.keywords[:5]
                            ]
                            if event.keywords
                            else []
                        ),
                    ),
                    # Meta info
                    html.Div(
                        className="event-meta",
                        children=[
                            html.Span(
                                f"Speaker: {event.speaker}", className="meta-item"
                            ),
                            html.Span(
                                f"Duration: {int(event.duration_seconds)}s",
                                className="meta-item",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_recording_detail(
    detail: RecordingDetail, back_date: str, transcript: Optional[str] = None
) -> html.Div:
    """Create the full recording detail view."""
    summary = detail.summary
    events = detail.events

    # Format date
    date_str = summary.start_time.strftime("%B %d, %Y")

    # Calculate category percentages
    cat_pcts = detail.category_percentages

    # Category colors
    category_colors = {
        "work": "#3b82f6",
        "personal": "#8b5cf6",
        "meeting": "#f59e0b",
        "reflection": "#10b981",
        "idea": "#ec4899",
        "deep_work": "#6366f1",
        "break": "#64748b",
        "unknown": "#374151",
    }

    return html.Div(
        className="recording-detail",
        children=[
            # Back button
            html.Button(
                id={"type": "back-btn", "date": back_date},
                className="back-btn",
                children=[
                    html.Span("←", className="back-icon"),
                    html.Span(f"Back to {back_date}", className="back-text"),
                ],
            ),
            # Header
            html.Div(
                className="recording-detail-header",
                children=[
                    html.H2(f"🎙️ Recording: {date_str}", className="detail-title"),
                    html.Div(
                        className="detail-time",
                        children=[
                            html.Span(
                                summary.time_range_formatted, className="time-range"
                            ),
                            html.Span(
                                f"({summary.duration_formatted})", className="duration"
                            ),
                        ],
                    ),
                ],
            ),
            # Category breakdown
            html.Div(
                className="category-breakdown",
                children=[
                    html.H4("Category Breakdown", className="section-title"),
                    html.Div(
                        className="category-bars-detailed",
                        children=[
                            html.Div(
                                className="category-row",
                                children=[
                                    html.Span(cat, className="category-label"),
                                    html.Div(
                                        className="category-bar-wrapper",
                                        children=[
                                            html.Div(
                                                className="category-bar-fill",
                                                style={
                                                    "width": f"{pct}%",
                                                    "backgroundColor": category_colors.get(
                                                        cat, "#374151"
                                                    ),
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Span(f"{pct:.0f}%", className="category-pct"),
                                ],
                            )
                            for cat, pct in sorted(
                                cat_pcts.items(), key=lambda x: -x[1]
                            )
                        ],
                    ),
                ],
            ),
            # Keywords
            html.Div(
                className="recording-keywords-section",
                children=[
                    html.H4("Keywords", className="section-title"),
                    html.Div(
                        className="keywords-list",
                        children=(
                            [
                                html.Span(kw, className="keyword-tag")
                                for kw in summary.keywords
                            ]
                            if summary.keywords
                            else [html.Span("No keywords", className="no-keywords")]
                        ),
                    ),
                ],
            ),
            # Timeline visual
            html.Div(
                className="timeline-visual",
                children=[
                    html.H4(
                        f"Timeline ({len(events)} events)", className="section-title"
                    ),
                    html.Div(
                        className="timeline-bar",
                        children=[
                            html.Div(
                                className="timeline-event-marker",
                                style={
                                    "left": f"{((e.start_ts - summary.start_time).total_seconds() / max(summary.duration_seconds, 1)) * 100}%",
                                    "backgroundColor": category_colors.get(
                                        e.category, "#374151"
                                    ),
                                },
                                title=f"{e.start_ts.strftime('%I:%M %p')}: {e.category}",
                            )
                            for e in events
                        ],
                    ),
                    html.Div(
                        className="timeline-labels",
                        children=[
                            html.Span(
                                summary.start_time.strftime("%I:%M %p"),
                                className="timeline-start",
                            ),
                            html.Span(
                                summary.end_time.strftime("%I:%M %p"),
                                className="timeline-end",
                            ),
                        ],
                    ),
                ],
            ),
            # Transcript viewer (collapsible)
            *(
                [
                    html.Details(
                        className="transcript-section",
                        children=[
                            html.Summary(
                                children=[
                                    html.H4(
                                        "📝 Raw Transcript",
                                        className="section-title",
                                        style={"display": "inline"},
                                    ),
                                    html.Span(
                                        f" ({len(transcript.split())} words, {len(transcript):,} chars)",
                                        className="transcript-meta",
                                    ),
                                ],
                            ),
                            html.Pre(
                                transcript,
                                className="transcript-text",
                            ),
                        ],
                    )
                ]
                if transcript
                else []
            ),
            # Events list
            html.Div(
                className="events-section",
                children=[
                    html.H4("Events", className="section-title"),
                    html.Div(
                        className="events-list",
                        children=[
                            create_event_card(event, i)
                            for i, event in enumerate(events)
                        ],
                    ),
                ],
            ),
        ],
    )


def create_recording_placeholder() -> html.Div:
    """Create placeholder when no recording is selected."""
    return html.Div(
        className="recording-placeholder",
        children=[
            html.Span("🎙️", className="placeholder-icon"),
            html.H3("Select a Recording"),
            html.P("Click on a recording in the day view to see details."),
        ],
    )
