"""Notion sync & browse component.

Shows Notion connection status, recordings from the Notion database,
page content preview, the interplay between Notion and Chronos/Plaud data,
a coverage calendar heatmap, import-to-Chronos controls, and write-back features.
"""

from dash import html, dcc
from typing import List, Optional, Dict


def create_notion_view(
    status=None,
    recordings=None,
    chronos_recording_ids=None,
    match_map=None,
    coverage_calendar=None,
) -> html.Div:
    """Create the Notion integration view.

    Args:
        status: NotionSyncStatus dict (or None if not yet fetched)
        recordings: List of NotionRecording dicts
        chronos_recording_ids: Set of recording IDs already in Chronos
        match_map: {notion_page_id → chronos_recording_id or None}
        coverage_calendar: List of day dicts from get_coverage_calendar()
    """
    recordings = recordings or []
    chronos_recording_ids = chronos_recording_ids or set()
    match_map = match_map or {}

    # Count unmatched for the banner
    unmatched_count = sum(1 for pid, cid in match_map.items() if cid is None)

    return html.Div(
        className="notion-view",
        children=[
            # Header
            html.Div(
                className="notion-header",
                children=[
                    html.Div(
                        className="notion-title-row",
                        children=[
                            html.H2("📔 Notion Integration"),
                            html.Div(
                                className="notion-header-actions",
                                children=[
                                    html.Button(
                                        "🔄 Fetch from Notion",
                                        id="notion-fetch-btn",
                                        className="sync-action-btn",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        f"🚀 Import All to Chronos ({unmatched_count})",
                                        id="notion-import-all-btn",
                                        className="sync-action-btn notion-import-btn",
                                        n_clicks=0,
                                        disabled=unmatched_count == 0,
                                    ) if recordings else None,
                                ],
                            ),
                        ],
                    ),
                    html.P(
                        "Pull recordings from Notion, see the overlap with Chronos, "
                        "and import missing recordings for full AI processing.",
                        className="notion-subtitle",
                    ),
                ],
            ),

            # Import progress area
            html.Div(id="notion-import-progress", className="notion-import-progress"),

            # Connection Status Card
            _build_connection_card(status),

            # Coverage Calendar Heatmap
            _build_coverage_calendar(coverage_calendar),

            # Interplay Overview (Notion vs Chronos)
            _build_interplay_card(recordings, chronos_recording_ids, match_map),

            # Schema card (shows detected properties)
            _build_schema_card(status),

            # Recordings list (with import/writeback buttons per row)
            _build_recordings_list(recordings, chronos_recording_ids, match_map),

            # Hidden stores for callbacks
            dcc.Store(id="notion-recordings-store", data=[]),
            dcc.Store(id="notion-status-store", data=None),
            dcc.Store(id="notion-selected-page", data=None),
            dcc.Store(id="notion-match-map-store", data={}),
            dcc.Store(id="notion-coverage-store", data=[]),

            # Page content modal / detail panel
            html.Div(id="notion-page-detail", className="notion-page-detail"),
        ],
    )


def _build_connection_card(status) -> html.Div:
    """Build the connection status card."""
    if status is None:
        return html.Div(
            className="notion-card notion-connection-card",
            children=[
                html.H3("🔌 Connection"),
                html.P(
                    "Click 'Fetch Recordings' to connect to Notion and pull your data.",
                    className="notion-muted",
                ),
                html.Div(
                    className="notion-config-hint",
                    children=[
                        html.Span("Required: ", style={"fontWeight": "600"}),
                        html.Code("NOTION_TOKEN"),
                        html.Span(" and "),
                        html.Code("NOTION_DATABASE_ID"),
                        html.Span(" in your .env file. "),
                        html.Span(
                            "Create an integration at notion.so/profile/integrations",
                            className="notion-muted",
                        ),
                    ],
                ),
            ],
        )

    connected_icon = "🟢" if status.get("connected") else "🔴"
    db_icon = "📁" if status.get("database_found") else "❌"

    error_children = []
    if status.get("error"):
        error_children = [
            html.Div(
                className="notion-error",
                children=[
                    html.Span("⚠️ ", className="error-icon"),
                    html.Span(status["error"]),
                ],
            )
        ]

    return html.Div(
        className="notion-card notion-connection-card",
        children=[
            html.H3("🔌 Connection"),
            html.Div(
                className="notion-status-row",
                children=[
                    html.Div(
                        className="notion-status-item",
                        children=[
                            html.Span(connected_icon, className="status-icon"),
                            html.Span("API Connected" if status.get("connected") else "Disconnected"),
                        ],
                    ),
                    html.Div(
                        className="notion-status-item",
                        children=[
                            html.Span(db_icon, className="status-icon"),
                            html.Span(
                                status.get("database_title", "No database")
                                if status.get("database_found")
                                else "Database not found"
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-status-item",
                        children=[
                            html.Span("📄"),
                            html.Span(f"{status.get('total_pages', 0)} pages in Notion"),
                        ],
                    ),
                ],
            ),
            *error_children,
        ],
    )


def _build_coverage_calendar(coverage_calendar) -> html.Div:
    """Build a visual coverage calendar heatmap.

    Shows last 90 days as a grid:
    - 🟣 Purple = Chronos only
    - 🟠 Amber = Notion only
    - 🟢 Green = Both systems
    - ⬛ Grey = No data
    """
    if not coverage_calendar:
        return html.Div(
            className="notion-card notion-calendar-card",
            children=[
                html.H3("📅 Knowledge Coverage"),
                html.P(
                    "Fetch recordings to see your coverage calendar — "
                    "which days have data in which system.",
                    className="notion-muted",
                ),
            ],
        )

    # Stats summary
    days_both = sum(1 for d in coverage_calendar if d.get("has_both"))
    days_chronos = sum(1 for d in coverage_calendar if d.get("has_chronos") and not d.get("has_notion"))
    days_notion = sum(1 for d in coverage_calendar if d.get("has_notion") and not d.get("has_chronos"))
    days_empty = sum(1 for d in coverage_calendar if not d.get("has_chronos") and not d.get("has_notion"))
    total_days = len(coverage_calendar)
    coverage_pct = round(((total_days - days_empty) / total_days) * 100) if total_days else 0

    # Build day cells
    day_cells = []
    for day in coverage_calendar:
        has_c = day.get("has_chronos", False)
        has_n = day.get("has_notion", False)
        imported = day.get("imported", False)

        if has_c and has_n:
            cell_class = "cal-day cal-both"
            tooltip = f"{day['date']}: {day.get('chronos_count', 0)} Chronos + {day.get('notion_count', 0)} Notion"
        elif has_c:
            cell_class = "cal-day cal-chronos"
            tooltip = f"{day['date']}: {day.get('chronos_count', 0)} Chronos recordings"
        elif has_n:
            cell_class = "cal-day cal-notion" + (" cal-imported" if imported else "")
            tooltip = f"{day['date']}: {day.get('notion_count', 0)} Notion recordings" + (" (imported)" if imported else "")
        else:
            cell_class = "cal-day cal-empty"
            tooltip = f"{day['date']}: no recordings"

        day_cells.append(
            html.Div(
                className=cell_class,
                title=tooltip,
                children=[html.Span(day["date"][-2:])],  # Day number
            )
        )

    return html.Div(
        className="notion-card notion-calendar-card",
        children=[
            html.H3("📅 Knowledge Coverage"),
            html.Div(
                className="calendar-stats-row",
                children=[
                    html.Span(f"{coverage_pct}% coverage", className="calendar-coverage-pct"),
                    html.Span(f"{days_both} both", className="cal-legend-both"),
                    html.Span(f"{days_chronos} Chronos only", className="cal-legend-chronos"),
                    html.Span(f"{days_notion} Notion only", className="cal-legend-notion"),
                    html.Span(f"{days_empty} gaps", className="cal-legend-empty"),
                ],
            ),
            html.Div(
                className="calendar-grid",
                children=day_cells,
            ),
            html.Div(
                className="calendar-legend",
                children=[
                    html.Span("■ Both", className="cal-legend-both"),
                    html.Span("■ Chronos", className="cal-legend-chronos"),
                    html.Span("■ Notion", className="cal-legend-notion"),
                    html.Span("■ No data", className="cal-legend-empty"),
                ],
            ),
        ],
    )


def _build_interplay_card(recordings, chronos_ids, match_map=None) -> html.Div:
    """Build the Notion ↔ Chronos interplay overview card."""
    match_map = match_map or {}
    total_notion = len(recordings)
    if total_notion == 0:
        return html.Div(
            className="notion-card notion-interplay-card",
            children=[
                html.H3("🔀 Notion ↔ Chronos Interplay"),
                html.P(
                    "Fetch recordings to see the overlap and gaps between systems.",
                    className="notion-muted",
                ),
            ],
        )

    # Categorize recordings using smart matching
    notion_only = []
    in_both = []
    for rec in recordings:
        if isinstance(rec, dict):
            page_id = rec.get("page_id", "")
        else:
            page_id = getattr(rec, "page_id", "")

        if match_map.get(page_id):
            in_both.append(rec)
        else:
            notion_only.append(rec)

    notion_only_count = len(notion_only)
    both_count = len(in_both)
    total_chronos = len(chronos_ids)

    return html.Div(
        className="notion-card notion-interplay-card",
        children=[
            html.H3("🔀 Notion ↔ Chronos Interplay"),
            html.Div(
                className="interplay-stats",
                children=[
                    # Notion Only
                    html.Div(
                        className="interplay-stat notion-only",
                        children=[
                            html.Span(str(notion_only_count), className="interplay-number"),
                            html.Span("Notion Only", className="interplay-label"),
                            html.Span(
                                "Recordings only in Notion — not yet in Chronos",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                    # Overlap
                    html.Div(
                        className="interplay-stat both-systems",
                        children=[
                            html.Span(str(both_count), className="interplay-number"),
                            html.Span("In Both", className="interplay-label"),
                            html.Span(
                                "Recordings that exist in both systems",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                    # Chronos Total
                    html.Div(
                        className="interplay-stat chronos-total",
                        children=[
                            html.Span(str(total_chronos), className="interplay-number"),
                            html.Span("Chronos Total", className="interplay-label"),
                            html.Span(
                                "All recordings currently in Chronos/Qdrant",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                    # Notion Total
                    html.Div(
                        className="interplay-stat notion-total",
                        children=[
                            html.Span(str(total_notion), className="interplay-number"),
                            html.Span("Notion Total", className="interplay-label"),
                            html.Span(
                                "All recordings found in your Notion database",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                ],
            ),
            # Visual bar showing proportions
            _build_interplay_bar(notion_only_count, both_count, total_chronos),
        ],
    )


def _build_interplay_bar(notion_only: int, both: int, chronos_only: int) -> html.Div:
    """Visual proportion bar showing data distribution."""
    total = notion_only + both + chronos_only
    if total == 0:
        return html.Div()

    notion_pct = (notion_only / total) * 100
    both_pct = (both / total) * 100
    chronos_pct = (chronos_only / total) * 100

    return html.Div(
        className="interplay-bar-container",
        children=[
            html.Div(
                className="interplay-bar",
                children=[
                    html.Div(
                        className="bar-segment notion-only-seg",
                        style={"width": f"{notion_pct}%"},
                        title=f"Notion only: {notion_only}",
                    ) if notion_pct > 0 else None,
                    html.Div(
                        className="bar-segment both-seg",
                        style={"width": f"{both_pct}%"},
                        title=f"In both: {both}",
                    ) if both_pct > 0 else None,
                    html.Div(
                        className="bar-segment chronos-only-seg",
                        style={"width": f"{chronos_pct}%"},
                        title=f"Chronos only: {chronos_only}",
                    ) if chronos_pct > 0 else None,
                ],
            ),
            html.Div(
                className="interplay-bar-legend",
                children=[
                    html.Span("■ Notion only", className="legend-notion"),
                    html.Span("■ In both", className="legend-both"),
                    html.Span("■ Chronos only", className="legend-chronos"),
                ],
            ),
        ],
    )


def _build_schema_card(status) -> html.Div:
    """Show the detected Notion database schema / properties."""
    if status is None or not status.get("schema"):
        return html.Div()

    schema = status.get("schema", {})
    if not schema:
        return html.Div()

    return html.Div(
        className="notion-card notion-schema-card",
        children=[
            html.H3("🗂️ Database Schema"),
            html.P(
                f"Detected {len(schema)} properties in your Notion database:",
                className="notion-muted",
            ),
            html.Div(
                className="schema-grid",
                children=[
                    html.Div(
                        className="schema-item",
                        children=[
                            html.Span(prop_name, className="schema-name"),
                            html.Span(
                                prop_type,
                                className=f"schema-type schema-type-{prop_type}",
                            ),
                        ],
                    )
                    for prop_name, prop_type in sorted(schema.items())
                ],
            ),
        ],
    )


def _build_recordings_list(recordings, chronos_ids, match_map=None) -> html.Div:
    """Build the scrollable recordings list."""
    if not recordings:
        return html.Div(
            className="notion-card notion-recordings-card",
            children=[
                html.H3("📝 Recordings"),
                html.P(
                    "No recordings fetched yet. Click 'Fetch Recordings' above.",
                    className="notion-muted",
                ),
            ],
        )

    # Group by date
    by_date = {}
    for rec in recordings:
        if isinstance(rec, dict):
            date = rec.get("date", "Unknown")
            if not date:
                date = rec.get("created_time", "")[:10] or "Unknown"
        else:
            date = getattr(rec, "date", "Unknown") or "Unknown"
        by_date.setdefault(date, []).append(rec)

    date_groups = []
    for date in sorted(by_date.keys(), reverse=True):
        recs = by_date[date]
        date_groups.append(
            html.Div(
                className="notion-date-group",
                children=[
                    html.Div(
                        className="notion-date-header",
                        children=[
                            html.Span(_format_date(date), className="notion-date-label"),
                            html.Span(
                                f"{len(recs)} recording{'s' if len(recs) != 1 else ''}",
                                className="notion-date-count",
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-recordings-in-date",
                        children=[
                            _build_recording_row(rec, chronos_ids, match_map)
                            for rec in recs
                        ],
                    ),
                ],
            )
        )

    return html.Div(
        className="notion-card notion-recordings-card",
        children=[
            html.Div(
                className="notion-recordings-header",
                children=[
                    html.H3(f"📝 Recordings ({len(recordings)})"),
                ],
            ),
            html.Div(
                className="notion-recordings-scroll",
                children=date_groups,
            ),
        ],
    )


def _build_recording_row(rec, chronos_ids, match_map=None) -> html.Div:
    """Build a single recording row with action buttons."""
    match_map = match_map or {}
    if isinstance(rec, dict):
        page_id = rec.get("page_id", "")
        title = rec.get("title", "Untitled")
        created = rec.get("created_time", "")
        url = rec.get("url", "")
        tags = rec.get("tags", [])
        category = rec.get("category", "")
        transcript = rec.get("transcript", "")
        summary = rec.get("summary", "")
        duration = rec.get("duration", "")
    else:
        page_id = getattr(rec, "page_id", "")
        title = getattr(rec, "title", "Untitled")
        created = getattr(rec, "created_time", "")
        url = getattr(rec, "url", "")
        tags = getattr(rec, "tags", [])
        category = getattr(rec, "category", "")
        transcript = getattr(rec, "transcript", "")
        summary = getattr(rec, "summary", "")
        duration = getattr(rec, "duration", "")

    # Use smart match_map for matching
    in_chronos = bool(match_map.get(page_id))

    source_badge = (
        html.Span("✅ In Chronos", className="notion-badge badge-both")
        if in_chronos
        else html.Span("👻 Not in Chronos", className="notion-badge badge-notion-only")
    )

    # Action buttons based on state
    action_buttons = []
    if not in_chronos:
        # Ghost recording — offer to import
        action_buttons.append(
            html.Button(
                "⚡ Import to Chronos",
                id={"type": "notion-import-one", "page_id": page_id},
                className="notion-action-btn notion-import-small",
                n_clicks=0,
            )
        )
    else:
        # In Chronos — offer write-back
        action_buttons.append(
            html.Button(
                "📤 Enrich in Notion",
                id={"type": "notion-writeback", "page_id": page_id},
                className="notion-action-btn notion-writeback-btn",
                n_clicks=0,
            )
        )

    # Preview text (transcript or summary snippet)
    preview = ""
    if transcript:
        preview = transcript[:200] + ("…" if len(transcript) > 200 else "")
    elif summary:
        preview = summary[:200] + ("…" if len(summary) > 200 else "")

    # Time from created_time
    time_str = ""
    if created:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            time_str = dt.strftime("%I:%M %p")
        except Exception:
            time_str = created[11:16] if len(created) > 16 else ""

    children = [
        # Row header
        html.Div(
            className="notion-rec-header",
            children=[
                html.Div(
                    className="notion-rec-title-row",
                    children=[
                        html.Span(title, className="notion-rec-title"),
                        source_badge,
                    ],
                ),
                html.Div(
                    className="notion-rec-meta",
                    children=[
                        html.Span(time_str, className="notion-rec-time") if time_str else None,
                        html.Span(f" · {duration}", className="notion-rec-duration") if duration else None,
                        html.Span(f" · {category}", className="notion-rec-category") if category else None,
                    ],
                ),
            ],
        ),
    ]

    # Preview text
    if preview:
        children.append(
            html.P(preview, className="notion-rec-preview")
        )

    # Tags
    if tags:
        children.append(
            html.Div(
                className="notion-rec-tags",
                children=[
                    html.Span(tag, className="notion-tag") for tag in tags[:6]
                ],
            )
        )

    # Action buttons row
    if action_buttons:
        children.append(
            html.Div(
                className="notion-rec-actions",
                children=action_buttons,
            )
        )

    return html.Div(
        className=f"notion-rec-row {'in-chronos' if in_chronos else 'notion-only-row'}",
        id={"type": "notion-rec-click", "page_id": page_id},
        children=children,
        **{"data-url": url},
    )


def _format_date(date_str: str) -> str:
    """Format a date string for display."""
    if not date_str or date_str == "Unknown":
        return "Unknown Date"
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%A, %B %d, %Y")
    except Exception:
        return date_str


def create_notion_page_detail(rec, body_text: str = "", in_chronos: bool = False) -> html.Div:
    """Create a detail panel for a selected Notion page/recording."""
    if isinstance(rec, dict):
        page_id = rec.get("page_id", "")
        title = rec.get("title", "Untitled")
        url = rec.get("url", "")
        transcript = rec.get("transcript", "")
        summary = rec.get("summary", "")
        tags = rec.get("tags", [])
        category = rec.get("category", "")
        created = rec.get("created_time", "")
        properties = rec.get("properties", {})
    else:
        page_id = getattr(rec, "page_id", "")
        title = getattr(rec, "title", "Untitled")
        url = getattr(rec, "url", "")
        transcript = getattr(rec, "transcript", "")
        summary = getattr(rec, "summary", "")
        tags = getattr(rec, "tags", [])
        category = getattr(rec, "category", "")
        created = getattr(rec, "created_time", "")
        properties = getattr(rec, "properties", {})

    children = [
        html.Div(
            className="notion-detail-header",
            children=[
                html.H3(title),
                html.Div(
                    className="notion-detail-actions",
                    children=[
                        html.A(
                            "Open in Notion ↗",
                            href=url,
                            target="_blank",
                            rel="noopener noreferrer",
                            className="notion-open-link",
                        ) if url else None,
                        html.Button(
                            "⚡ Import to Chronos",
                            id={"type": "notion-import-one", "page_id": page_id},
                            className="notion-action-btn notion-import-small",
                            n_clicks=0,
                        ) if not in_chronos else html.Button(
                            "📤 Enrich in Notion",
                            id={"type": "notion-writeback", "page_id": page_id},
                            className="notion-action-btn notion-writeback-btn",
                            n_clicks=0,
                        ),
                    ],
                ),
            ],
        ),
    ]

    # Metadata row
    meta_items = []
    if created:
        meta_items.append(html.Span(f"Created: {created[:10]}"))
    if category:
        meta_items.append(html.Span(f"Category: {category}"))
    if tags:
        meta_items.append(html.Span(f"Tags: {', '.join(tags)}"))

    if meta_items:
        children.append(
            html.Div(
                className="notion-detail-meta",
                children=[
                    item for pair in zip(meta_items, [html.Span(" · ")] * len(meta_items))
                    for item in pair
                ][:-1],  # Remove trailing separator
            )
        )

    # Summary
    if summary:
        children.append(
            html.Div(
                className="notion-detail-section",
                children=[
                    html.H4("Summary"),
                    html.P(summary, className="notion-detail-text"),
                ],
            )
        )

    # Transcript
    if transcript:
        children.append(
            html.Div(
                className="notion-detail-section",
                children=[
                    html.H4("Transcript"),
                    html.Div(
                        transcript,
                        className="notion-detail-transcript",
                    ),
                ],
            )
        )

    # Page body content
    if body_text:
        children.append(
            html.Div(
                className="notion-detail-section",
                children=[
                    html.H4("Page Content"),
                    html.Pre(body_text, className="notion-detail-body"),
                ],
            )
        )

    # Raw properties (collapsible)
    if properties:
        prop_rows = []
        for name, value in sorted(properties.items()):
            if value:  # Only show non-empty
                prop_rows.append(
                    html.Tr([
                        html.Td(name, className="prop-name"),
                        html.Td(str(value)[:300], className="prop-value"),
                    ])
                )

        if prop_rows:
            children.append(
                html.Details(
                    className="notion-detail-props",
                    children=[
                        html.Summary(f"All Properties ({len(prop_rows)})"),
                        html.Table(
                            className="notion-props-table",
                            children=[html.Tbody(prop_rows)],
                        ),
                    ],
                )
            )

    return html.Div(
        className="notion-detail-panel",
        children=children,
    )
