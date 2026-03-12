"""Recording detail view component."""

from dash import html, dcc
from typing import Optional, List
from datetime import datetime

from app_v2.services.data_service import RecordingDetail, Event

# Canonical category list + colors
CATEGORIES = [
    "work",
    "personal",
    "meeting",
    "reflection",
    "idea",
    "deep_work",
    "break",
    "unknown",
]
CATEGORY_COLORS = {
    "work": "#3b82f6",
    "personal": "#8b5cf6",
    "meeting": "#f59e0b",
    "reflection": "#10b981",
    "idea": "#ec4899",
    "deep_work": "#6366f1",
    "break": "#64748b",
    "unknown": "#374151",
}


def _time_of_day_label(hour: int) -> str:
    """Human-friendly label for an hour."""
    if hour < 6:
        return "🌙 Early Morning"
    if hour < 9:
        return "🌅 Morning"
    if hour < 12:
        return "☀️ Mid-Morning"
    if hour < 14:
        return "🌤️ Early Afternoon"
    if hour < 17:
        return "⛅ Afternoon"
    if hour < 20:
        return "🌇 Evening"
    return "🌙 Night"


def _duration_context(seconds: float) -> str:
    """Human-friendly duration context."""
    mins = seconds / 60
    if mins < 2:
        return "Quick note"
    if mins < 5:
        return "Brief capture"
    if mins < 15:
        return "Short session"
    if mins < 30:
        return "Session"
    if mins < 60:
        return "Long session"
    if mins < 120:
        return "Extended session"
    return "Marathon session"


def _find_key_moments(events: List[Event], max_moments: int = 3) -> List[Event]:
    """Pick the most notable events as key moments.

    Selects events with: high |sentiment|, many keywords, unique speakers.
    """
    if not events:
        return []

    scored = []
    for e in events:
        score = abs(e.sentiment) * 2
        score += min(len(e.keywords), 5) * 0.3
        score += 0.5 if e.speaker and e.speaker != "unknown" else 0
        score += 0.3 if e.category in ("idea", "meeting", "deep_work") else 0
        scored.append((score, e))

    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:max_moments]]


def create_event_card(event: Event, index: int, highlighted: bool = False) -> html.Div:
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

    cat_color = CATEGORY_COLORS.get(event.category, "#374151")

    card_class = "event-card highlighted" if highlighted else "event-card"

    return html.Div(
        id={"type": "event-card", "id": event.id},
        className=card_class,
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
                            # Editable category dropdown
                            dcc.Dropdown(
                                id={"type": "event-category-edit", "id": event.id},
                                options=[
                                    {
                                        "label": cat.replace("_", " ").title(),
                                        "value": cat,
                                    }
                                    for cat in CATEGORIES
                                ],
                                value=event.category,
                                clearable=False,
                                className="event-category-dropdown",
                                style={"borderColor": cat_color},
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
    detail: RecordingDetail,
    back_date: str,
    transcript: Optional[str] = None,
    highlight_event_id: Optional[str] = None,
    ai_summary: Optional[str] = None,
) -> html.Div:
    """Create the full recording detail view."""
    summary = detail.summary
    events = detail.events

    # Format date
    date_str = summary.start_time.strftime("%B %d, %Y")
    day_of_week = summary.start_time.strftime("%A")
    hour = summary.start_time.hour
    ambient = _time_of_day_label(hour)
    dur_ctx = _duration_context(summary.duration_seconds)

    # Calculate category percentages
    cat_pcts = detail.category_percentages

    # Key moments
    key_moments = _find_key_moments(events)

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
            # Breadcrumb
            html.Div(
                className="detail-breadcrumb",
                children=[
                    html.Span("Timeline", className="breadcrumb-item"),
                    html.Span("›", className="breadcrumb-sep"),
                    html.Span(back_date, className="breadcrumb-item"),
                    html.Span("›", className="breadcrumb-sep"),
                    html.Span(
                        summary.time_range_formatted,
                        className="breadcrumb-item current",
                    ),
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
                    # Ambient context tags
                    html.Div(
                        className="detail-ambient-tags",
                        children=[
                            html.Span(day_of_week, className="ambient-pill"),
                            html.Span(ambient, className="ambient-pill"),
                            html.Span(dur_ctx, className="ambient-pill"),
                        ],
                    ),
                ],
            ),
            # AI Summary (if available)
            *(
                [
                    html.Div(
                        className="ai-summary-section",
                        children=[
                            html.H4("✨ Summary", className="section-title"),
                            html.P(ai_summary, className="ai-summary-text"),
                        ],
                    )
                ]
                if ai_summary
                else []
            ),
            # Key Moments (top 3 notable events)
            *(
                [
                    html.Div(
                        className="key-moments-section",
                        children=[
                            html.H4(
                                f"⚡ Key Moments ({len(key_moments)})",
                                className="section-title",
                            ),
                            html.Div(
                                className="key-moments-list",
                                children=[
                                    html.Div(
                                        className="key-moment",
                                        children=[
                                            html.Span(
                                                km.start_ts.strftime("%I:%M %p"),
                                                className="moment-time",
                                            ),
                                            html.Span(
                                                km.category,
                                                className="moment-category",
                                                style={
                                                    "backgroundColor": CATEGORY_COLORS.get(
                                                        km.category, "#374151"
                                                    )
                                                },
                                            ),
                                            html.Span(
                                                km.clean_text[:120]
                                                + (
                                                    "…"
                                                    if len(km.clean_text) > 120
                                                    else ""
                                                ),
                                                className="moment-text",
                                            ),
                                        ],
                                    )
                                    for km in key_moments
                                ],
                            ),
                        ],
                    )
                ]
                if key_moments
                else []
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
                                                    "backgroundColor": CATEGORY_COLORS.get(
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
                                id={"type": "timeline-marker", "event": e.id},
                                className="timeline-event-marker",
                                style={
                                    "left": f"{((e.start_ts - summary.start_time).total_seconds() / max(summary.duration_seconds, 1)) * 100}%",
                                    "backgroundColor": CATEGORY_COLORS.get(
                                        e.category, "#374151"
                                    ),
                                },
                                title=f"{e.start_ts.strftime('%I:%M %p')}: {e.category} — {e.clean_text[:60]}",
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
                    # Header with count + inline filter
                    html.Div(
                        className="events-section-header",
                        children=[
                            html.H4(
                                f"Events ({len(events)})",
                                className="section-title",
                            ),
                            dcc.Input(
                                id="event-filter-input",
                                type="text",
                                placeholder="Filter events…",
                                className="event-filter-input",
                                debounce=False,
                            ),
                        ],
                    ),
                    html.Div(
                        id="events-list-container",
                        className="events-list",
                        children=[
                            create_event_card(
                                event,
                                i,
                                highlighted=(event.id == highlight_event_id),
                            )
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
            html.P("Click on a recording in the timeline to see details."),
            html.Div(
                className="keyboard-hints",
                children=[
                    html.Span("Press ", className="hint-text"),
                    html.Kbd("/", className="kbd"),
                    html.Span(" to search • ", className="hint-text"),
                    html.Kbd("Esc", className="kbd"),
                    html.Span(" to close panel", className="hint-text"),
                ],
            ),
        ],
    )
