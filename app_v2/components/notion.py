"""Notion sync & browse component.

Shows Notion connection status, recordings from the Notion database,
page content preview, and the interplay between Notion and Chronos/Plaud data.
"""

from dash import html, dcc
from typing import List, Optional, Dict


def create_notion_view(
    status=None,
    recordings=None,
    chronos_recording_ids=None,
) -> html.Div:
    """Create the Notion integration view.

    Args:
        status: NotionSyncStatus object (or None if not yet fetched)
        recordings: List of NotionRecording objects
        chronos_recording_ids: Set of recording IDs already in Chronos (for matching)
    """
    recordings = recordings or []
    chronos_recording_ids = chronos_recording_ids or set()

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
                                        "🔄 Fetch Recordings",
                                        id="notion-fetch-btn",
                                        className="sync-action-btn",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.P(
                        "Pull recordings and transcripts from your Notion database — "
                        "the missing pieces that Plaud's API couldn't deliver.",
                        className="notion-subtitle",
                    ),
                ],
            ),

            # Connection Status Card
            _build_connection_card(status),

            # Interplay Overview (Notion vs Chronos)
            _build_interplay_card(recordings, chronos_recording_ids),

            # Schema card (shows detected properties)
            _build_schema_card(status),

            # Recordings list
            _build_recordings_list(recordings, chronos_recording_ids),

            # Hidden stores for callbacks
            dcc.Store(id="notion-recordings-store", data=[]),
            dcc.Store(id="notion-status-store", data=None),
            dcc.Store(id="notion-selected-page", data=None),

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


def _build_interplay_card(recordings, chronos_ids) -> html.Div:
    """Build the Notion ↔ Chronos interplay overview card."""
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

    # Categorize recordings
    notion_only = []
    in_both = []
    for rec in recordings:
        if isinstance(rec, dict):
            title = rec.get("title", "")
        else:
            title = getattr(rec, "title", "")

        # Simple title-based matching (can be enhanced later)
        matched = False
        title_lower = title.lower() if title else ""
        for cid in chronos_ids:
            if isinstance(cid, str) and cid.lower() in title_lower:
                matched = True
                break

        if matched:
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


def _build_recordings_list(recordings, chronos_ids) -> html.Div:
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
                            _build_recording_row(rec, chronos_ids)
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


def _build_recording_row(rec, chronos_ids) -> html.Div:
    """Build a single recording row in the Notion list."""
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

    # Determine if this recording is also in Chronos
    in_chronos = False
    title_lower = title.lower() if title else ""
    for cid in chronos_ids:
        if isinstance(cid, str) and cid.lower() in title_lower:
            in_chronos = True
            break

    source_badge = (
        html.Span("✅ Also in Chronos", className="notion-badge badge-both")
        if in_chronos
        else html.Span("📔 Notion only", className="notion-badge badge-notion-only")
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


def create_notion_page_detail(rec, body_text: str = "") -> html.Div:
    """Create a detail panel for a selected Notion page/recording."""
    if isinstance(rec, dict):
        title = rec.get("title", "Untitled")
        url = rec.get("url", "")
        transcript = rec.get("transcript", "")
        summary = rec.get("summary", "")
        tags = rec.get("tags", [])
        category = rec.get("category", "")
        created = rec.get("created_time", "")
        properties = rec.get("properties", {})
    else:
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
                html.A(
                    "Open in Notion ↗",
                    href=url,
                    target="_blank",
                    rel="noopener noreferrer",
                    className="notion-open-link",
                ) if url else None,
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
