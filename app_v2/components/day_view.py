"""Timeline view components — recordings as a stream of consciousness."""

from datetime import datetime, timedelta
from dash import html, dcc
from typing import List, Optional

from app_v2.services.data_service import DaySummary, RecordingSummary
from app_v2.components import CATEGORY_COLORS

# ── Intensity ramp for heat-map ───────────────────────────────────────────────
_HEAT_LEVELS = [
    "#1e293b",  # 0 events — dark slate (empty)
    "#1e3a5f",  # 1-2 events — subtle blue
    "#1d4ed8",  # 3-5
    "#2563eb",  # 6-10
    "#3b82f6",  # 11-20
    "#60a5fa",  # 21+
]


def _heat_color(event_count: int) -> str:
    """Map event count to a heat-map color."""
    if event_count == 0:
        return _HEAT_LEVELS[0]
    if event_count <= 2:
        return _HEAT_LEVELS[1]
    if event_count <= 5:
        return _HEAT_LEVELS[2]
    if event_count <= 10:
        return _HEAT_LEVELS[3]
    if event_count <= 20:
        return _HEAT_LEVELS[4]
    return _HEAT_LEVELS[5]


def create_category_bar(categories: dict, height: int = 8) -> html.Div:
    """Create a stacked bar showing category distribution."""
    if not categories:
        return html.Div(className="category-bar empty")

    total = sum(categories.values())
    if total == 0:
        return html.Div(className="category-bar empty")

    segments = []
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        color = CATEGORY_COLORS.get(cat, "#374151")
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


def create_day_timeline_strip(
    recordings: List[RecordingSummary], day_date: str
) -> html.Div:
    """Visual time-of-day strip: recordings as colored blocks.

    The axis auto-fits to the actual recordings for that day (±30 min padding,
    snapped to clean hour boundaries).
    """
    if not recordings:
        return html.Div()

    # ── Compute dynamic axis bounds ──────────────────────────────────────────
    PADDING_SECS = 30 * 60  # 30-minute padding each side

    def _to_secs(dt: datetime) -> int:
        return dt.hour * 3600 + dt.minute * 60 + dt.second

    earliest = min(_to_secs(r.start_time) for r in recordings)
    latest = max(_to_secs(r.end_time) for r in recordings)

    axis_start_s = max(0, (earliest - PADDING_SECS) // 3600 * 3600)  # floor to hour
    axis_end_s = min(
        86400, ((latest + PADDING_SECS + 3599) // 3600) * 3600
    )  # ceil to hour
    axios_span = max(axis_end_s - axis_start_s, 3600)  # at least 1h

    def to_pct(dt: datetime) -> float:
        offset = _to_secs(dt) - axis_start_s
        return max(0.0, min(100.0, offset / axios_span * 100))

    def dur_pct(secs: float) -> float:
        return max(0.5, secs / axios_span * 100)

    # ── Recording blocks ─────────────────────────────────────────────────────
    blocks = []
    for rec in sorted(recordings, key=lambda r: r.start_time):
        left = to_pct(rec.start_time)
        width = dur_pct(rec.duration_seconds)
        if left + width > 100:
            width = 100.0 - left
        color = CATEGORY_COLORS.get(rec.top_category, "#374151")
        label = rec.start_time.strftime("%-I:%M%p").lower() if width > 8 else ""
        blocks.append(
            html.Div(
                className="day-timeline-block",
                style={
                    "left": f"{left:.2f}%",
                    "width": f"{width:.2f}%",
                    "backgroundColor": color,
                },
                title=(
                    f"{rec.time_range_formatted}  •  "
                    f"{rec.duration_formatted}  •  "
                    f"{rec.top_category}  •  "
                    f"{rec.event_count} events"
                ),
                children=[html.Span(label, className="block-label")],
            )
        )

    # ── Hour tick marks — only hours within the visible window ───────────────
    ALL_TICK_LABELS = {
        0: "midnight",
        1: "1am",
        2: "2am",
        3: "3am",
        4: "4am",
        5: "5am",
        6: "6am",
        7: "7am",
        8: "8am",
        9: "9am",
        10: "10am",
        11: "11am",
        12: "noon",
        13: "1pm",
        14: "2pm",
        15: "3pm",
        16: "4pm",
        17: "5pm",
        18: "6pm",
        19: "7pm",
        20: "8pm",
        21: "9pm",
        22: "10pm",
        23: "11pm",
    }
    # Include every hour that falls inside the visible window
    axis_start_h = axis_start_s // 3600
    axis_end_h = axis_end_s // 3600
    span_hours = axis_end_h - axis_start_h

    # Thin out ticks when window is large (>10h → every 2h; >16h → every 3h)
    step = 1 if span_hours <= 6 else (2 if span_hours <= 12 else 3)

    hour_marks = []
    for h in range(axis_start_h, axis_end_h + 1, step):
        if h > 23:
            break
        pct = (h * 3600 - axis_start_s) / axios_span * 100
        if pct < 0 or pct > 100:
            continue
        hour_marks.append(
            html.Div(
                className="hour-mark",
                style={"left": f"{pct:.2f}%"},
                children=[
                    html.Span(ALL_TICK_LABELS.get(h, f"{h}h"), className="hour-label")
                ],
            )
        )

    return html.Div(
        className="day-timeline-strip",
        children=[
            html.Div(className="day-timeline-track", children=blocks),
            html.Div(className="day-timeline-hours", children=hour_marks),
        ],
    )


def create_recording_card(recording: RecordingSummary, day_date: str) -> html.Div:
    """Create a card for a single recording — time-first design."""
    top_cat = recording.top_category
    cat_color = CATEGORY_COLORS.get(top_cat, "#374151")
    keywords = recording.keywords[:4]

    # Sentiment indicator
    s = recording.avg_sentiment
    if s > 0.2:
        sentiment_icon, sentiment_cls = "↑", "sentiment-pos"
    elif s < -0.2:
        sentiment_icon, sentiment_cls = "↓", "sentiment-neg"
    else:
        sentiment_icon, sentiment_cls = "–", "sentiment-neu"

    # Ambient context — time-of-day label
    hour = recording.start_time.hour
    ambient = _time_of_day_label(hour)

    # Duration context
    mins = recording.duration_seconds / 60
    if mins < 5:
        dur_ctx = "quick note"
    elif mins < 15:
        dur_ctx = "short session"
    elif mins < 45:
        dur_ctx = "session"
    elif mins < 90:
        dur_ctx = "long session"
    else:
        dur_ctx = "extended session"

    return html.Div(
        id={"type": "recording-card", "id": recording.recording_id, "date": day_date},
        className=f"recording-card recording-cat-{top_cat}",
        style={"borderLeft": f"3px solid {cat_color}"},
        children=[
            # ── Time row ───────────────────────────────────────────────
            html.Div(
                className="recording-header",
                children=[
                    html.Span(
                        recording.time_range_formatted,
                        className="recording-time",
                    ),
                    html.Div(
                        className="recording-header-right",
                        children=[
                            # Cloud/Local indicator
                            *(
                                [
                                    html.Span(
                                        "☁️",
                                        className="source-badge cloud",
                                        title="Plaud Cloud + AI",
                                    )
                                ]
                                if recording.has_plaud_ai
                                else (
                                    [
                                        html.Span(
                                            "☁️",
                                            className="source-badge cloud-only",
                                            title="Plaud Cloud",
                                        )
                                    ]
                                    if recording.source == "plaud_cloud"
                                    else [
                                        html.Span(
                                            "💾",
                                            className="source-badge local",
                                            title="Local only (USB import)",
                                        )
                                    ]
                                )
                            ),
                            html.Span(
                                recording.duration_formatted,
                                className="recording-duration",
                            ),
                            html.Span(
                                sentiment_icon,
                                className=f"sentiment-badge {sentiment_cls}",
                            ),
                        ],
                    ),
                ],
            ),
            # ── Ambient row ─────────────────────────────────────────────
            html.Div(
                className="recording-ambient",
                children=[
                    html.Span(ambient, className="ambient-tag"),
                    html.Span("•", className="ambient-sep"),
                    html.Span(dur_ctx, className="ambient-tag"),
                ],
            ),
            # ── Category bar ────────────────────────────────────────────
            create_category_bar(recording.categories, height=4),
            # ── Stats row ───────────────────────────────────────────────
            html.Div(
                className="recording-stats",
                children=[
                    html.Span(
                        top_cat,
                        className="category-pill",
                        style={
                            "background": f"{cat_color}22",
                            "color": cat_color,
                            "borderColor": f"{cat_color}44",
                        },
                    ),
                    html.Span(
                        f"{recording.event_count} events",
                        className="stat",
                    ),
                ],
            ),
            # ── Keywords ────────────────────────────────────────────────
            html.Div(
                className="recording-keywords",
                children=(
                    [html.Span(kw, className="keyword-tag small") for kw in keywords]
                    if keywords
                    else []
                ),
            ),
        ],
    )


def _time_of_day_label(hour: int) -> str:
    """Human-friendly label for an hour."""
    if hour < 6:
        return "🌙 early morning"
    if hour < 9:
        return "🌅 morning"
    if hour < 12:
        return "☀️ mid-morning"
    if hour < 14:
        return "🌤️ afternoon"
    if hour < 17:
        return "⛅ mid-afternoon"
    if hour < 20:
        return "🌇 evening"
    return "🌙 night"


def create_day_card(day: DaySummary, expanded: bool = False) -> html.Div:
    """Create a card for a day with collapsible recording list."""
    # Build a quick day summary line from top categories + time span
    if day.recordings:
        first = min(r.start_time for r in day.recordings)
        last = max(r.end_time for r in day.recordings)
        span_label = f"{first.strftime('%-I:%M%p').lower()} – {last.strftime('%-I:%M%p').lower()}"
        top_cats = sorted(day.categories.items(), key=lambda x: -x[1])[:3]
        cat_labels = ", ".join(c for c, _ in top_cats)
        day_summary_text = f"{span_label}  •  {cat_labels}"
    else:
        day_summary_text = "No recordings"

    # One-line AI summary (from recording-level Plaud summaries)
    ai_summary_line = getattr(day, "ai_summary", None)

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
                            html.H3(day.date_display, className="day-title"),
                            html.Div(
                                className="day-summary-line",
                                children=[
                                    html.Span(
                                        day_summary_text, className="day-summary-text"
                                    ),
                                ],
                            ),
                            *(
                                [
                                    html.Div(
                                        className="day-ai-summary-line",
                                        children=[
                                            html.Span(
                                                "✨ ", className="ai-summary-icon"
                                            ),
                                            html.Span(
                                                ai_summary_line,
                                                className="day-ai-summary-text",
                                            ),
                                        ],
                                    )
                                ]
                                if ai_summary_line
                                else []
                            ),
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
            # ── Time-of-day timeline strip ───────────────────────────────
            create_day_timeline_strip(day.recordings, day.date),
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
                    create_recording_card(rec, day.date)
                    for rec in sorted(day.recordings, key=lambda r: r.start_time)
                ],
            ),
        ],
    )


def create_heat_map_strip(
    days: List[DaySummary], num_calendar_days: int = 30
) -> html.Div:
    """Create a 30-day heat-map strip showing recording density.

    Each cell = one calendar day. Intensity = event count. Click to scroll.
    """
    if not days:
        return html.Div()

    # Build a {YYYY-MM-DD: DaySummary} lookup
    day_lookup = {d.date: d for d in days}

    today = datetime.now().date()
    cells = []
    for offset in range(num_calendar_days - 1, -1, -1):
        d = today - timedelta(days=offset)
        key = d.strftime("%Y-%m-%d")
        day_data = day_lookup.get(key)
        count = day_data.event_count if day_data else 0
        rec_count = day_data.recording_count if day_data else 0
        color = _heat_color(count)

        # Day-of-week label for first row
        dow = d.strftime("%a")[0]  # M, T, W, ...
        day_num = d.strftime("%-d")
        is_today = offset == 0

        tooltip = f"{d.strftime('%b %-d')}: {count} events, {rec_count} recordings"

        cells.append(
            html.Div(
                id={"type": "heatmap-cell", "date": key},
                className=f"heatmap-cell {'heatmap-today' if is_today else ''}",
                style={"backgroundColor": color},
                title=tooltip,
                children=[
                    html.Span(day_num, className="heatmap-day-num"),
                ],
            )
        )

    return html.Div(
        className="heatmap-strip",
        children=[
            html.Div(
                className="heatmap-header",
                children=[
                    html.Span("Last 30 days", className="heatmap-title"),
                    html.Div(
                        className="heatmap-legend",
                        children=[
                            html.Span("Less", className="heatmap-legend-label"),
                            *[
                                html.Div(
                                    className="heatmap-legend-cell",
                                    style={"backgroundColor": c},
                                )
                                for c in _HEAT_LEVELS
                            ],
                            html.Span("More", className="heatmap-legend-label"),
                        ],
                    ),
                ],
            ),
            html.Div(className="heatmap-cells", children=cells),
        ],
    )


def create_day_view(days: List[DaySummary]) -> html.Div:
    """Create the full timeline view with heat-map, date controls, and day cards."""
    if not days:
        return html.Div(
            className="empty-state",
            children=[
                html.Span("📭", className="empty-icon"),
                html.H3("No recordings yet"),
                html.P("Sync from Plaud to see your recordings here."),
            ],
        )

    total_recs = sum(d.recording_count for d in days)
    total_events = sum(d.event_count for d in days)
    total_hours = sum(d.total_duration_seconds for d in days) / 3600

    return html.Div(
        className="day-view timeline-view",
        children=[
            # Header
            html.Div(
                className="view-header",
                children=[
                    html.H2("⏱️ Timeline", className="view-title"),
                    html.Div(
                        className="view-meta",
                        children=[
                            html.Span(
                                f"{total_recs} recordings",
                                className="meta-stat",
                            ),
                            html.Span("•", className="meta-sep"),
                            html.Span(
                                f"{total_events} events",
                                className="meta-stat",
                            ),
                            html.Span("•", className="meta-sep"),
                            html.Span(
                                f"{total_hours:.1f} hours",
                                className="meta-stat",
                            ),
                            html.Span("•", className="meta-sep"),
                            html.Span(f"{len(days)} days", className="meta-stat"),
                        ],
                    ),
                ],
            ),
            # Heat-map strip (30 days at a glance)
            create_heat_map_strip(days),
            # Date range filter
            html.Div(
                className="timeline-range-controls",
                children=[
                    dcc.Dropdown(
                        id="timeline-range-select",
                        className="timeline-range-dropdown",
                        value=0,
                        clearable=False,
                        searchable=False,
                        options=[
                            {"label": "Last 7 days", "value": 7},
                            {"label": "Last 14 days", "value": 14},
                            {"label": "Last 30 days", "value": 30},
                            {"label": "All time", "value": 0},
                        ],
                        style={"width": "160px"},
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
