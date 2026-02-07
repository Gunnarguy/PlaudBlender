"""Day view components - showing recordings grouped by day."""

from dash import html
from typing import List, Optional

from app_v2.services.data_service import DaySummary, RecordingSummary


def create_category_bar(categories: dict, height: int = 8) -> html.Div:
    """Create a stacked bar showing category distribution."""
    if not categories:
        return html.Div(className="category-bar empty")

    total = sum(categories.values())
    if total == 0:
        return html.Div(className="category-bar empty")

    # Category colors
    colors = {
        "work": "#3b82f6",  # blue
        "personal": "#8b5cf6",  # purple
        "meeting": "#f59e0b",  # amber
        "reflection": "#10b981",  # emerald
        "idea": "#ec4899",  # pink
        "deep_work": "#6366f1",  # indigo
        "break": "#64748b",  # slate
        "unknown": "#374151",  # gray
    }

    segments = []
    for cat, count in categories.items():
        pct = (count / total) * 100
        color = colors.get(cat, "#374151")
        segments.append(
            html.Div(
                className="category-segment",
                style={
                    "width": f"{pct}%",
                    "backgroundColor": color,
                    "height": f"{height}px",
                },
                title=f"{cat}: {count} ({pct:.0f}%)",
            )
        )

    return html.Div(
        className="category-bar",
        children=segments,
        style={"display": "flex", "borderRadius": "4px", "overflow": "hidden"},
    )


def create_recording_card(recording: RecordingSummary, day_date: str) -> html.Div:
    """Create a card for a single recording."""
    # Format short ID for display
    short_id = recording.recording_id[:8]

    # Top keywords (limit to 3)
    keywords = recording.keywords[:3]

    return html.Div(
        id={"type": "recording-card", "id": recording.recording_id, "date": day_date},
        className="recording-card",
        children=[
            # Header row
            html.Div(
                className="recording-header",
                children=[
                    html.Span(
                        recording.time_range_formatted, className="recording-time"
                    ),
                    html.Span(
                        recording.duration_formatted, className="recording-duration"
                    ),
                ],
            ),
            # Category distribution bar
            create_category_bar(recording.categories),
            # Stats row
            html.Div(
                className="recording-stats",
                children=[
                    html.Span(f"{recording.event_count} events", className="stat"),
                    html.Span(f"• {recording.top_category}", className="stat category"),
                ],
            ),
            # Keywords
            html.Div(
                className="recording-keywords",
                children=(
                    [html.Span(kw, className="keyword-tag") for kw in keywords]
                    if keywords
                    else [html.Span("No keywords", className="no-keywords")]
                ),
            ),
        ],
    )


def create_day_card(day: DaySummary, expanded: bool = False) -> html.Div:
    """Create a card for a day with collapsible recording list."""
    return html.Div(
        className=f"day-card {'expanded' if expanded else ''}",
        children=[
            # Day header (clickable to expand/collapse)
            html.Div(
                id={"type": "day-header", "date": day.date},
                className="day-header",
                children=[
                    # Left side - date and stats
                    html.Div(
                        className="day-info",
                        children=[
                            html.H3(f"📅 {day.date_display}", className="day-title"),
                            html.Div(
                                className="day-stats",
                                children=[
                                    html.Span(
                                        f"{day.recording_count} recording{'s' if day.recording_count != 1 else ''}",
                                        className="stat",
                                    ),
                                    html.Span("•", className="stat-sep"),
                                    html.Span(
                                        f"{day.event_count} events", className="stat"
                                    ),
                                    html.Span("•", className="stat-sep"),
                                    html.Span(
                                        day.duration_formatted,
                                        className="stat duration",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Right side - expand indicator
                    html.Span(
                        "▼" if expanded else "▶",
                        className="expand-icon",
                    ),
                ],
            ),
            # Category bar for the whole day
            html.Div(
                className="day-category-bar",
                children=[create_category_bar(day.categories, height=6)],
            ),
            # Top keywords for the day
            html.Div(
                className="day-keywords",
                children=(
                    [
                        html.Span(kw, className="keyword-tag")
                        for kw in day.top_keywords[:5]
                    ]
                    if day.top_keywords
                    else []
                ),
            ),
            # Collapsible recordings section
            html.Div(
                id={"type": "day-recordings", "date": day.date},
                className="day-recordings",
                style={"display": "block" if expanded else "none"},
                children=[
                    create_recording_card(rec, day.date) for rec in day.recordings
                ],
            ),
        ],
    )


def create_day_view(days: List[DaySummary]) -> html.Div:
    """Create the full day view with all days."""
    if not days:
        return html.Div(
            className="empty-state",
            children=[
                html.Span("📭", className="empty-icon"),
                html.H3("No recordings yet"),
                html.P("Sync from Plaud to see your recordings here."),
            ],
        )

    return html.Div(
        className="day-view",
        children=[
            # Header
            html.Div(
                className="view-header",
                children=[
                    html.H2("Your Recordings", className="view-title"),
                    html.Div(
                        className="view-meta",
                        children=[
                            html.Span(
                                f"{sum(d.recording_count for d in days)} recordings",
                                className="meta-stat",
                            ),
                            html.Span("•", className="meta-sep"),
                            html.Span(
                                f"{sum(d.event_count for d in days)} events",
                                className="meta-stat",
                            ),
                            html.Span("•", className="meta-sep"),
                            html.Span(f"{len(days)} days", className="meta-stat"),
                        ],
                    ),
                ],
            ),
            # Day cards (first day expanded by default)
            html.Div(
                className="days-list",
                id="days-list",
                children=[
                    create_day_card(day, expanded=(i == 0))
                    for i, day in enumerate(days)
                ],
            ),
        ],
    )
