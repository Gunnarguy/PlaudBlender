"""Recording detail view component."""

from dash import html, dcc
from typing import Optional, List
from datetime import datetime, timedelta

from app_v2.services.data_service import RecordingDetail, Event
from app_v2.components import CATEGORIES, CATEGORY_COLORS, CATEGORY_LABELS
from app_v2.components.recording_status import build_recording_system_strip


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
                                        "label": CATEGORY_LABELS.get(cat, cat),
                                        "value": cat,
                                    }
                                    for cat in CATEGORIES
                                ],
                                value=event.category,
                                clearable=False,
                                className="event-category-dropdown",
                                style={"borderColor": cat_color},
                            ),
                            *(
                                [
                                    html.Span(
                                        f"{int(event.category_confidence * 100)}%",
                                        className=f"confidence-badge {'high' if event.category_confidence >= 0.7 else 'medium' if event.category_confidence >= 0.4 else 'low'}",
                                        title=f"Category confidence: {event.category_confidence:.0%}",
                                    )
                                ]
                                if event.category_confidence is not None
                                else []
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
                            *(
                                [
                                    html.Span(
                                        "⚠️ duration capped",
                                        className="meta-item capped-badge",
                                        title="Original duration exceeded 4 hours — likely a Gemini hallucination. Capped to 4h.",
                                    )
                                ]
                                if event.duration_capped
                                else []
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _build_narrative_sections(events: List[Event]) -> List[html.Div]:
    """Group events into flowing narrative sections based on time gaps and category shifts."""
    if not events:
        return [html.Div(html.P("No events to narrate.", className="empty-narrative"))]

    sorted_events = sorted(events, key=lambda e: e.start_ts)
    sections: List[html.Div] = []
    current_group: List[Event] = [sorted_events[0]]

    for prev, cur in zip(sorted_events, sorted_events[1:]):
        gap = (cur.start_ts - prev.end_ts).total_seconds()
        category_shift = cur.category != prev.category
        # Break on 10+ minute gap or category change
        if gap > 600 or category_shift:
            sections.append(_render_narrative_group(current_group))
            current_group = [cur]
        else:
            current_group.append(cur)

    if current_group:
        sections.append(_render_narrative_group(current_group))

    return sections


def _render_narrative_group(events: List[Event]) -> html.Div:
    """Render a group of related events as a narrative block."""
    first = events[0]
    last = events[-1]
    start = first.start_ts.strftime("%I:%M %p")
    end = last.end_ts.strftime("%I:%M %p")
    category = first.category
    cat_color = CATEGORY_COLORS.get(category, "#374151")
    cat_label = CATEGORY_LABELS.get(category, category)

    # Merge event texts into flowing paragraphs
    paragraphs = " ".join(e.clean_text for e in events)

    # Gather unique keywords
    all_kw = []
    seen = set()
    for e in events:
        for kw in e.keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                all_kw.append(kw)

    return html.Div(
        className="narrative-block",
        children=[
            html.Div(
                className="narrative-header",
                children=[
                    html.Span(
                        cat_label,
                        className="narrative-category-pill",
                        style={
                            "background": f"{cat_color}22",
                            "color": cat_color,
                            "borderColor": f"{cat_color}44",
                        },
                    ),
                    html.Span(f"{start} – {end}", className="narrative-time"),
                    html.Span(
                        f"{len(events)} event{'s' if len(events) != 1 else ''}",
                        className="narrative-count",
                    ),
                ],
            ),
            html.P(paragraphs, className="narrative-text"),
            *(
                [
                    html.Div(
                        className="narrative-keywords",
                        children=[
                            html.Span(kw, className="keyword-tag small")
                            for kw in all_kw[:8]
                        ],
                    )
                ]
                if all_kw
                else []
            ),
        ],
    )


def create_recording_detail(
    detail: RecordingDetail,
    back_date: str,
    transcript: Optional[str] = None,
    highlight_event_id: Optional[str] = None,
    ai_summary: Optional[str] = None,
    extracted_data: Optional[dict] = None,
    workflow_status: Optional[dict] = None,
    plaud_transcript: Optional[str] = None,
) -> html.Div:
    """Create the full recording detail view with tabbed layout."""
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
    system_strip = build_recording_system_strip(
        summary, workflow_status=workflow_status, detail=True
    )

    # ── Build tab content ─────────────────────────────────────────────────────

    # Overview tab
    overview_children = []

    # Per-recording workflow actions
    wf_running = False
    if workflow_status:
        wf_st = str(workflow_status.get("status", "")).upper()
        wf_running = wf_st in ("PENDING", "PROCESSING", "RUNNING")
    show_run_btn = not wf_running  # Show "Run Plaud AI" when no workflow is active

    if show_run_btn:
        overview_children.append(
            html.Div(
                className="recording-actions-bar",
                children=[
                    html.Button(
                        "☁️ Run Plaud AI",
                        id="run-single-workflow-btn",
                        className="action-btn plaud-ai-btn",
                        n_clicks=0,
                    ),
                    dcc.Dropdown(
                        id="single-workflow-template",
                        options=[
                            {"label": "Summary Only", "value": ""},
                            {"label": "📋 General Summary", "value": "general"},
                            {"label": "📝 Meeting Notes", "value": "meeting"},
                            {"label": "💡 Brainstorm / Ideas", "value": "brainstorm"},
                            {"label": "📅 Daily Log", "value": "daily_log"},
                            {"label": "🎤 Interview", "value": "interview"},
                        ],
                        value="",
                        clearable=False,
                        className="action-dropdown",
                        style={"width": "180px"},
                    ),
                    dcc.Dropdown(
                        id="single-workflow-model",
                        options=[
                            {"label": "OpenAI", "value": "openai"},
                            {"label": "Gemini", "value": "gemini"},
                            {"label": "Claude", "value": "claude"},
                        ],
                        value="openai",
                        clearable=False,
                        className="action-dropdown",
                        style={"width": "120px"},
                    ),
                    html.Div(
                        id="single-workflow-result", className="workflow-inline-result"
                    ),
                ],
            )
        )
    else:
        # Hidden placeholders so callbacks don't break
        overview_children.append(
            html.Div(
                style={"display": "none"},
                children=[
                    html.Button(
                        id="run-single-workflow-btn",
                        n_clicks=0,
                        style={"display": "none"},
                    ),
                    dcc.Dropdown(
                        id="single-workflow-template",
                        value="",
                        style={"display": "none"},
                    ),
                    dcc.Dropdown(
                        id="single-workflow-model",
                        value="openai",
                        style={"display": "none"},
                    ),
                    html.Div(id="single-workflow-result"),
                ],
            )
        )

    # Workflow status badge
    if workflow_status:
        wf_st = str(workflow_status.get("status", "")).upper()
        wf_color = {
            "SUCCESS": "#10b981",
            "COMPLETED": "#10b981",
            "FAILED": "#ef4444",
            "ERROR": "#ef4444",
            "PROCESSING": "#3b82f6",
            "RUNNING": "#3b82f6",
            "PENDING": "#f59e0b",
        }.get(wf_st, "#94a3b8")
        wf_template = workflow_status.get("template_id")
        wf_submitted = workflow_status.get("submitted_at", "")

        overview_children.append(
            html.Div(
                className="workflow-status-section",
                children=[
                    html.Div(
                        className="workflow-status-header",
                        children=[
                            html.Span(
                                f"☁️ Plaud Workflow: {wf_st}",
                                className="workflow-status-badge",
                                style={"backgroundColor": wf_color},
                            ),
                            *(
                                [
                                    html.Span(
                                        f"Template: {wf_template}",
                                        className="workflow-template-tag",
                                    ),
                                ]
                                if wf_template
                                else []
                            ),
                            *(
                                [
                                    html.Span(
                                        f"Submitted: {wf_submitted[:16]}",
                                        className="workflow-submitted-text",
                                    ),
                                ]
                                if wf_submitted
                                else []
                            ),
                        ],
                    ),
                    *(
                        [
                            html.Span(
                                f"Error: {workflow_status.get('error', '')}",
                                className="workflow-error-text",
                            ),
                        ]
                        if wf_st in ("FAILED", "ERROR") and workflow_status.get("error")
                        else []
                    ),
                ],
            )
        )

    # AI Summary
    if ai_summary:
        overview_children.append(
            html.Div(
                className="ai-summary-section",
                children=[
                    html.H4("✨ Summary", className="section-title"),
                    html.P(ai_summary, className="ai-summary-text"),
                ],
            )
        )

    # Extracted Data (AI_ETL output)
    if extracted_data and isinstance(extracted_data, dict):
        etl_children = []
        for key, value in extracted_data.items():
            if isinstance(value, list):
                etl_children.append(
                    html.Div(
                        className="etl-field",
                        children=[
                            html.Span(
                                key.replace("_", " ").title(),
                                className="etl-field-label",
                            ),
                            html.Ul(
                                className="etl-field-list",
                                children=[
                                    html.Li(str(item)[:200]) for item in value[:20]
                                ],
                            ),
                        ],
                    )
                )
            elif isinstance(value, dict):
                etl_children.append(
                    html.Div(
                        className="etl-field",
                        children=[
                            html.Span(
                                key.replace("_", " ").title(),
                                className="etl-field-label",
                            ),
                            html.Div(
                                className="etl-nested",
                                children=[
                                    html.Div(
                                        [
                                            html.Span(
                                                f"{k}: ",
                                                className="etl-nested-key",
                                            ),
                                            html.Span(
                                                str(v)[:200],
                                                className="etl-nested-value",
                                            ),
                                        ],
                                        className="etl-nested-row",
                                    )
                                    for k, v in value.items()
                                ],
                            ),
                        ],
                    )
                )
            else:
                etl_children.append(
                    html.Div(
                        className="etl-field",
                        children=[
                            html.Span(
                                key.replace("_", " ").title(),
                                className="etl-field-label",
                            ),
                            html.Span(str(value)[:500], className="etl-field-value"),
                        ],
                    )
                )

        if etl_children:
            overview_children.append(
                html.Div(
                    className="extracted-data-section",
                    children=[
                        html.H4(
                            "🔬 Extracted Data (AI ETL)",
                            className="section-title",
                        ),
                        html.Div(
                            className="etl-fields-grid",
                            children=etl_children,
                        ),
                    ],
                )
            )

    # Key Moments
    if key_moments:
        overview_children.append(
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
                                        CATEGORY_LABELS.get(km.category, km.category),
                                        className="moment-category",
                                        style={
                                            "backgroundColor": CATEGORY_COLORS.get(
                                                km.category, "#374151"
                                            )
                                        },
                                    ),
                                    html.Span(
                                        km.clean_text[:120]
                                        + ("…" if len(km.clean_text) > 120 else ""),
                                        className="moment-text",
                                    ),
                                ],
                            )
                            for km in key_moments
                        ],
                    ),
                ],
            )
        )

    # Timeline visual
    overview_children.append(
        html.Div(
            className="timeline-visual",
            children=[
                html.H4(f"Timeline ({len(events)} events)", className="section-title"),
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
        )
    )

    # Category breakdown
    overview_children.append(
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
                                html.Span(
                                    CATEGORY_LABELS.get(cat, cat),
                                    className="category-label",
                                ),
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
                        for cat, pct in sorted(cat_pcts.items(), key=lambda x: -x[1])
                    ],
                ),
            ],
        )
    )

    # Keywords
    overview_children.append(
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
        )
    )

    # Events tab
    events_tab = html.Div(
        className="events-section",
        children=[
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
    )

    # Transcript tab
    if transcript:
        word_count = len(transcript.split())
        char_count = len(transcript)
        transcript_tab = html.Div(
            className="transcript-tab-content",
            children=[
                html.Div(
                    className="transcript-stats",
                    children=[
                        html.Span(f"{word_count:,} words", className="transcript-stat"),
                        html.Span("·", className="stat-sep"),
                        html.Span(f"{char_count:,} chars", className="transcript-stat"),
                    ],
                ),
                html.Pre(transcript, className="transcript-text"),
            ],
        )
    else:
        transcript_tab = html.Div(
            className="transcript-tab-content empty",
            children=[
                html.P("No transcript available", className="empty-transcript"),
            ],
        )

    # Narrative tab — flowing paragraphs grouped by time & category
    narrative_sections = _build_narrative_sections(events)
    narrative_tab = html.Div(
        className="narrative-tab-content",
        children=narrative_sections,
    )

    # Tab labels with counts
    tab_labels = {
        "overview": "Overview",
        "events": f"Events ({len(events)})",
        "narrative": "Narrative",
        "transcript": f"Transcript{'  ✓' if transcript else ''}",
    }

    # Build optional comparison tab (when Plaud workflow transcript exists)
    comparison_tab = None
    if plaud_transcript and transcript:
        local_words = len(transcript.split())
        plaud_words = len(plaud_transcript.split())
        word_diff = plaud_words - local_words
        diff_sign = "+" if word_diff > 0 else ""
        comparison_tab = html.Div(
            className="comparison-tab-content",
            children=[
                html.Div(
                    className="comparison-stats",
                    children=[
                        html.Span(
                            f"Local: {local_words:,} words",
                            className="comparison-stat local",
                        ),
                        html.Span("vs", className="comparison-sep"),
                        html.Span(
                            f"Plaud AI: {plaud_words:,} words ({diff_sign}{word_diff:,})",
                            className="comparison-stat plaud",
                        ),
                    ],
                ),
                html.Div(
                    className="comparison-panels",
                    children=[
                        html.Div(
                            className="comparison-panel",
                            children=[
                                html.H5(
                                    "📝 Plaud-Fetched Transcript",
                                    className="comparison-panel-title",
                                ),
                                html.Pre(
                                    transcript,
                                    className="transcript-text comparison-local",
                                ),
                            ],
                        ),
                        html.Div(
                            className="comparison-panel",
                            children=[
                                html.H5(
                                    "☁️ Plaud AI Transcript",
                                    className="comparison-panel-title",
                                ),
                                html.Pre(
                                    plaud_transcript,
                                    className="transcript-text comparison-plaud",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
        tab_labels["comparison"] = "Compare ↔"

    return html.Div(
        className="recording-detail",
        children=[
            # Store recording ID for per-recording callbacks
            dcc.Store(id="detail-recording-id", data=summary.recording_id),
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
            # Header (always visible)
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
                            *(
                                [
                                    html.Span(
                                        "estimated",
                                        className="source-badge estimated detail-estimated-badge",
                                        title=(
                                            summary.time_estimate_reason
                                            or "Time estimated from Notion import defaults"
                                        ),
                                    )
                                ]
                                if summary.time_is_estimated
                                else []
                            ),
                            html.Span(
                                f"({summary.duration_formatted})", className="duration"
                            ),
                        ],
                    ),
                    html.Div(
                        className="detail-ambient-tags",
                        children=[
                            html.Span(day_of_week, className="ambient-pill"),
                            html.Span(ambient, className="ambient-pill"),
                            html.Span(dur_ctx, className="ambient-pill"),
                        ],
                    ),
                    *([system_strip] if system_strip is not None else []),
                ],
            ),
            # Tab navigation
            dcc.Tabs(
                id="detail-tabs",
                value="overview",
                className="detail-tabs",
                children=[
                    dcc.Tab(
                        label=tab_labels["overview"],
                        value="overview",
                        className="detail-tab",
                        selected_className="detail-tab--selected",
                        children=[
                            html.Div(
                                className="detail-tab-content",
                                children=overview_children,
                            )
                        ],
                    ),
                    dcc.Tab(
                        label=tab_labels["events"],
                        value="events",
                        className="detail-tab",
                        selected_className="detail-tab--selected",
                        children=[
                            html.Div(
                                className="detail-tab-content",
                                children=[events_tab],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label=tab_labels["narrative"],
                        value="narrative",
                        className="detail-tab",
                        selected_className="detail-tab--selected",
                        children=[
                            html.Div(
                                className="detail-tab-content",
                                children=[narrative_tab],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label=tab_labels["transcript"],
                        value="transcript",
                        className="detail-tab",
                        selected_className="detail-tab--selected",
                        children=[
                            html.Div(
                                className="detail-tab-content",
                                children=[transcript_tab],
                            )
                        ],
                    ),
                    *(
                        [
                            dcc.Tab(
                                label=tab_labels["comparison"],
                                value="comparison",
                                className="detail-tab",
                                selected_className="detail-tab--selected",
                                children=[
                                    html.Div(
                                        className="detail-tab-content",
                                        children=[comparison_tab],
                                    )
                                ],
                            )
                        ]
                        if comparison_tab
                        else []
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
