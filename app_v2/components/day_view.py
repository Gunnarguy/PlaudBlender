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
                    + (
                        f"  •  estimated time ({rec.time_estimate_reason})"
                        if rec.time_is_estimated
                        else ""
                    )
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


def _sentiment_sparkline(arc: list, width: int = 80, height: int = 20):
    """Render a tiny inline SVG sparkline from sentiment values."""
    if not arc or len(arc) < 2:
        return html.Span()
    n = len(arc)
    # Normalize sentiment [-1, 1] → [height, 0] (inverted Y for SVG)
    points = []
    for i, val in enumerate(arc):
        x = (i / (n - 1)) * width
        y = height - ((val + 1) / 2) * height  # -1→bottom, +1→top
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    # Color: green if mostly positive, red if mostly negative, blue if mixed
    avg = sum(arc) / len(arc)
    color = "#10b981" if avg > 0.15 else "#ef4444" if avg < -0.15 else "#60a5fa"
    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" '
        f'stroke="#ffffff15" stroke-width="0.5"/>'
        f"</svg>"
    )
    return html.Div(
        className="sentiment-sparkline",
        # Dash doesn't support raw SVG in html.Div directly; use Iframe srcdoc
        # or dangerously_allow_html. Simplest: use a series of colored dots.
    )


def _sentiment_dots(arc: list, max_dots: int = 12):
    """Render sentiment as a row of colored dots — one per event (sampled)."""
    if not arc:
        return html.Span()
    # Sample down to max_dots if needed
    if len(arc) > max_dots:
        step = len(arc) / max_dots
        sampled = [arc[int(i * step)] for i in range(max_dots)]
    else:
        sampled = arc
    dots = []
    for val in sampled:
        if val > 0.3:
            color, title = "#10b981", "positive"
        elif val > 0.1:
            color, title = "#6ee7b7", "slightly positive"
        elif val < -0.3:
            color, title = "#ef4444", "negative"
        elif val < -0.1:
            color, title = "#fca5a5", "slightly negative"
        else:
            color, title = "#94a3b8", "neutral"
        dots.append(
            html.Span(
                className="mood-dot",
                style={"backgroundColor": color},
                title=f"{val:+.1f} ({title})",
            )
        )
    return html.Div(className="mood-dots", children=dots)


def create_recording_card(recording: RecordingSummary, day_date: str) -> html.Div:
    """Create a rich card for a single recording — shows what happened, not just metadata."""

    # ── Pending/processing placeholder card ─────────────────────────
    proc_status = getattr(recording, "processing_status", "completed")
    if proc_status in ("pending", "processing") and recording.event_count == 0:
        hour = recording.start_time.hour
        ambient = _time_of_day_label(hour)
        status_label = "⏳ Waiting to process" if proc_status == "pending" else "🔄 Processing…"
        status_color = "#f59e0b" if proc_status == "pending" else "#3b82f6"
        mins = int(recording.duration_seconds // 60)
        secs = int(recording.duration_seconds % 60)
        dur_text = f"{mins}:{secs:02d}" if mins else f"{secs}s"
        return html.Div(
            id={"type": "recording-card", "id": recording.recording_id, "date": day_date},
            className="recording-card recording-cat-unknown recording-pending",
            style={"borderLeft": "3px solid #f59e0b", "opacity": "0.8"},
            children=[
                html.Div(
                    className="recording-header",
                    children=[
                        html.Div(
                            className="recording-header-left",
                            children=[
                                html.Span(
                                    recording.time_range_formatted,
                                    className="recording-time",
                                ),
                                html.Span(ambient, className="ambient-inline"),
                            ],
                        ),
                        html.Div(
                            className="recording-header-right",
                            children=[
                                html.Span(dur_text, className="recording-duration"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="recording-pending-notice",
                    children=[
                        html.Span(status_label, style={"color": status_color, "fontWeight": "600"}),
                        html.Span(
                            " — Run Full Sync to extract events from this recording.",
                            className="stat muted",
                        ),
                    ],
                    style={"padding": "8px 0", "fontSize": "0.85rem"},
                ),
            ],
        )

    top_cat = recording.top_category
    cat_color = CATEGORY_COLORS.get(top_cat, "#374151")
    keywords = recording.keywords[:5]

    # Sentiment indicator
    s = recording.avg_sentiment
    if s > 0.2:
        sentiment_icon, sentiment_cls = "😊", "sentiment-pos"
    elif s < -0.2:
        sentiment_icon, sentiment_cls = "😔", "sentiment-neg"
    else:
        sentiment_icon, sentiment_cls = "😐", "sentiment-neu"

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

    # Category breakdown text
    cat_parts = []
    for cat, count in sorted(recording.categories.items(), key=lambda x: -x[1])[:3]:
        cat_parts.append(f"{count} {cat}")
    cat_breakdown = ", ".join(cat_parts)

    children = []

    # ── Time + meta header ─────────────────────────────────────────
    children.append(
        html.Div(
            className="recording-header",
            children=[
                html.Div(
                    className="recording-header-left",
                    children=[
                        html.Span(
                            recording.time_range_formatted,
                            className="recording-time",
                        ),
                        html.Span(ambient, className="ambient-inline"),
                    ],
                ),
                html.Div(
                    className="recording-header-right",
                    children=[
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
                                else (
                                    [
                                        html.Span(
                                            "📝",
                                            className="source-badge notion",
                                            title="Notion Import",
                                        )
                                    ]
                                    if recording.source == "notion"
                                    else [
                                        html.Span(
                                            "💾",
                                            className="source-badge local",
                                            title="Local (USB)",
                                        )
                                    ]
                                )
                            )
                        ),
                        *(
                            [
                                html.Span(
                                    "≈",
                                    className="source-badge estimated",
                                    title=(
                                        recording.time_estimate_reason
                                        or "Time estimated from Notion import defaults"
                                    ),
                                )
                            ]
                            if recording.time_is_estimated
                            else []
                        ),
                        html.Span(
                            recording.duration_formatted, className="recording-duration"
                        ),
                        html.Span(
                            sentiment_icon,
                            className=f"sentiment-badge {sentiment_cls}",
                            title=f"Avg sentiment: {s:+.2f}",
                        ),
                    ],
                ),
            ],
        )
    )

    # ── Preview text — the most important part ─────────────────────
    preview = getattr(recording, "preview_text", "") or ""
    if preview:
        children.append(
            html.Div(
                className="recording-preview",
                children=[
                    html.P(
                        preview + ("…" if len(preview) >= 148 else ""),
                        className="preview-text",
                    ),
                ],
            )
        )

    # ── Category bar ────────────────────────────────────────────────
    children.append(create_category_bar(recording.categories, height=4))

    # ── Stats + category breakdown ──────────────────────────────────
    children.append(
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
                html.Span(f"{recording.event_count} moments", className="stat"),
                html.Span(f"({dur_ctx})", className="stat muted"),
            ],
        )
    )

    # ── Mood arc (sentiment dots) ───────────────────────────────────
    arc = getattr(recording, "sentiment_arc", []) or []
    if len(arc) >= 2:
        children.append(
            html.Div(
                className="recording-mood-row",
                children=[
                    html.Span("Mood:", className="mood-label"),
                    _sentiment_dots(arc),
                ],
            )
        )

    # ── Inline event previews (expandable) ──────────────────────────
    event_previews = getattr(recording, "event_previews", []) or []
    if event_previews:
        preview_items = []
        for i, snippet in enumerate(event_previews):
            preview_items.append(
                html.Li(
                    snippet + ("…" if len(snippet) >= 118 else ""),
                    className="event-preview-item",
                )
            )
        remaining = recording.event_count - len(event_previews)
        children.append(
            html.Div(
                className="recording-event-previews",
                children=[
                    html.Ul(className="event-preview-list", children=preview_items),
                    *(
                        [html.Span(
                            f"+{remaining} more moment{'s' if remaining != 1 else ''} — click to explore",
                            className="event-preview-more",
                        )]
                        if remaining > 0
                        else []
                    ),
                ],
            )
        )

    # ── Keywords ────────────────────────────────────────────────────
    if keywords:
        children.append(
            html.Div(
                className="recording-keywords",
                children=[html.Span(kw, className="keyword-tag small") for kw in keywords],
            )
        )

    return html.Div(
        id={"type": "recording-card", "id": recording.recording_id, "date": day_date},
        className=f"recording-card recording-cat-{top_cat}",
        style={"borderLeft": f"3px solid {cat_color}"},
        children=children,
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


def _build_day_flow_narrative(recordings: list) -> str:
    """Generate a human-readable narrative of how the day flowed.

    E.g. 'Morning deep work → afternoon meetings → evening reflection'
    """
    if not recordings:
        return ""
    sorted_recs = sorted(recordings, key=lambda r: r.start_time)

    segments = []
    for rec in sorted_recs:
        hour = rec.start_time.hour
        if hour < 6:
            period = "early morning"
        elif hour < 9:
            period = "morning"
        elif hour < 12:
            period = "mid-morning"
        elif hour < 14:
            period = "afternoon"
        elif hour < 17:
            period = "mid-afternoon"
        elif hour < 20:
            period = "evening"
        else:
            period = "night"
        cat = rec.top_category
        segments.append((period, cat))

    # Deduplicate consecutive identical segments
    flow_parts = []
    prev = None
    for period, cat in segments:
        label = f"{period} {cat}"
        if label != prev:
            flow_parts.append(label)
            prev = label

    return " → ".join(flow_parts[:6])


def _day_mood_summary(recordings: list) -> str:
    """One-word mood summary for the day based on average sentiment."""
    if not recordings:
        return ""
    all_sentiments = []
    for r in recordings:
        arc = getattr(r, "sentiment_arc", []) or []
        all_sentiments.extend(arc)
    if not all_sentiments:
        return ""
    avg = sum(all_sentiments) / len(all_sentiments)
    if avg > 0.4:
        return "😊 Great day"
    if avg > 0.15:
        return "🙂 Good day"
    if avg > -0.15:
        return "😐 Mixed day"
    if avg > -0.4:
        return "😕 Tough day"
    return "😔 Hard day"


def create_day_card(day: DaySummary, expanded: bool = False) -> html.Div:
    """Create a rich card for a day with flow narrative, mood, and collapsible recordings."""
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

    # Flow narrative
    flow = _build_day_flow_narrative(day.recordings) if day.recordings else ""

    # Mood
    mood = _day_mood_summary(day.recordings) if day.recordings else ""

    # One-line AI summary (from recording-level Plaud summaries)
    ai_summary_line = getattr(day, "ai_summary", None)

    # Build header children
    header_info_children = [
        html.H3(day.date_display, className="day-title"),
        html.Div(
            className="day-summary-line",
            children=[
                html.Span(day_summary_text, className="day-summary-text"),
                *(
                    [html.Span(f"  •  {mood}", className="day-mood-badge")]
                    if mood
                    else []
                ),
            ],
        ),
    ]

    # Flow narrative line
    if flow:
        header_info_children.append(
            html.Div(
                className="day-flow-narrative",
                children=[
                    html.Span("📖 ", className="flow-icon"),
                    html.Span(flow, className="flow-text"),
                ],
            )
        )

    # AI summary line
    if ai_summary_line:
        header_info_children.append(
            html.Div(
                className="day-ai-summary-line",
                children=[
                    html.Span("✨ ", className="ai-summary-icon"),
                    html.Span(ai_summary_line, className="day-ai-summary-text"),
                ],
            )
        )

    # Stats line
    header_info_children.append(
        html.Div(
            className="day-stats",
            children=[
                html.Span(
                    f"{day.recording_count} recording{'s' if day.recording_count != 1 else ''}",
                    className="stat",
                ),
                html.Span("•", className="stat-sep"),
                html.Span(f"{day.event_count} moments", className="stat"),
                html.Span("•", className="stat-sep"),
                html.Span(day.duration_formatted, className="stat duration"),
            ],
        )
    )

    return html.Div(
        className=f"day-card {'expanded' if expanded else ''}",
        children=[
            # Day header (clickable to expand/collapse)
            html.Div(
                id={"type": "day-header", "date": day.date},
                className="day-header",
                children=[
                    html.Div(className="day-info", children=header_info_children),
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
                        for kw in day.top_keywords[:6]
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
