"""Notion sync & browse component.

Shows Notion connection status, recordings from the Notion database,
page content preview, the interplay between Notion and Chronos/Plaud data,
a coverage calendar heatmap, import-to-Chronos controls, and write-back features.
"""

from dash import html, dcc
from typing import List, Optional, Dict


def _build_notion_hero(
    status, recordings, matched_count: int, unmatched_count: int, has_db_id: bool
) -> html.Div:
    """Build the hero section for the Notion workspace."""
    connected = bool(status and status.get("connected"))
    database_title = (
        status.get("database_title")
        if status and status.get("database_found")
        else ("Configured data source" if has_db_id else "No data source selected")
    )
    total_pages = status.get("total_pages", 0) if status else 0
    total_recordings = len(recordings or [])

    return html.Div(
        className="notion-card notion-hero-card",
        children=[
            html.Div(
                className="notion-hero-content",
                children=[
                    html.Div(
                        className="notion-hero-copy",
                        children=[
                            html.Span(
                                "Knowledge Bridge", className="notion-hero-eyebrow"
                            ),
                            html.H2("Notion Workspace"),
                            html.P(
                                "Review what exists in Notion, see what Chronos already knows, and act on the gaps without leaving the tab.",
                                className="notion-subtitle",
                            ),
                            html.Div(
                                className="notion-hero-meta",
                                children=[
                                    html.Span(
                                        (
                                            "Live connection"
                                            if connected
                                            else "Connection pending"
                                        ),
                                        className="notion-hero-meta-item",
                                    ),
                                    html.Span(
                                        database_title,
                                        className="notion-hero-meta-item",
                                    ),
                                    html.Span(
                                        f"{total_pages or total_recordings} pages indexed",
                                        className="notion-hero-meta-item",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-header-actions notion-hero-actions",
                        children=[
                            html.Button(
                                "Refresh Notion",
                                id="notion-fetch-btn",
                                className="sync-action-btn",
                                n_clicks=0,
                            ),
                            (
                                html.Button(
                                    f"Import Missing ({unmatched_count})",
                                    id="notion-import-all-btn",
                                    className="sync-action-btn notion-import-btn",
                                    n_clicks=0,
                                    disabled=unmatched_count == 0,
                                )
                                if recordings
                                else None
                            ),
                            (
                                html.Button(
                                    f"📤 Write Back All ({matched_count})",
                                    id="notion-writeback-all-btn",
                                    className="sync-action-btn notion-writeback-btn",
                                    n_clicks=0,
                                    disabled=matched_count == 0,
                                )
                                if recordings and matched_count > 0
                                else None
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="notion-hero-stats",
                children=[
                    html.Div(
                        className="notion-hero-stat",
                        children=[
                            html.Span("Status", className="notion-hero-stat-label"),
                            html.Span(
                                "Connected" if connected else "Offline",
                                className="notion-hero-stat-value",
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-hero-stat",
                        children=[
                            html.Span("Pages", className="notion-hero-stat-label"),
                            html.Span(
                                str(total_pages or total_recordings),
                                className="notion-hero-stat-value",
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-hero-stat",
                        children=[
                            html.Span("Matched", className="notion-hero-stat-label"),
                            html.Span(
                                str(matched_count), className="notion-hero-stat-value"
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-hero-stat",
                        children=[
                            html.Span("Missing", className="notion-hero-stat-label"),
                            html.Span(
                                str(unmatched_count), className="notion-hero-stat-value"
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _build_empty_detail_panel() -> html.Div:
    """Default detail placeholder before a page is selected."""
    return html.Div(
        className="notion-card notion-detail-panel notion-detail-empty",
        children=[
            html.Span("Selected Page", className="notion-detail-kicker"),
            html.H3("Open a recording"),
            html.P(
                "Select any Notion recording to inspect transcript, page body, tags, raw properties, and import or write-back actions.",
                className="notion-muted",
            ),
        ],
    )


def create_notion_view(
    status=None,
    recordings=None,
    chronos_recording_ids=None,
    match_map=None,
    coverage_calendar=None,
    databases=None,
    active_category=None,
    stale_map=None,
) -> html.Div:
    """Create the Notion integration view.

    Args:
        status: NotionSyncStatus dict (or None if not yet fetched)
        recordings: List of NotionRecording dicts
        chronos_recording_ids: Set of recording IDs already in Chronos
        match_map: {notion_page_id → chronos_recording_id or None}
        coverage_calendar: List of day dicts from get_coverage_calendar()
        databases: List of available Notion databases for selection
        active_category: Currently selected category filter (None = All)
        stale_map: {notion_page_id → True} for pages edited after import
    """
    recordings = recordings or []
    chronos_recording_ids = chronos_recording_ids or set()
    match_map = match_map or {}
    stale_map = stale_map or {}
    databases = databases or []

    # Count unmatched for the banner
    unmatched_count = sum(1 for pid, cid in match_map.items() if cid is None)
    matched_count = sum(1 for cid in match_map.values() if cid)

    # Determine if we need the database picker
    from src.config import get_settings
    has_db_id = bool(get_settings().notion_database_id)

    top_cards = [_build_connection_card(status)]
    if not (has_db_id and not databases):
        top_cards.append(_build_database_picker(databases, has_db_id))

    analytics_cards = [
        _build_interplay_card(recordings, chronos_recording_ids, match_map),
        _build_coverage_calendar(coverage_calendar),
    ]
    if status and status.get("schema"):
        analytics_cards.append(_build_schema_card(status))

    return html.Div(
        className="notion-view",
        children=[
            _build_notion_hero(
                status=status,
                recordings=recordings,
                matched_count=matched_count,
                unmatched_count=unmatched_count,
                has_db_id=has_db_id,
            ),
            # Import progress area
            html.Div(id="notion-import-progress", className="notion-import-progress"),
            # Poll interval for batch import progress (disabled by default)
            dcc.Interval(
                id="notion-import-poll",
                interval=2000,
                disabled=True,
            ),
            html.Div(className="notion-top-grid", children=top_cards),
            # Auto-load trigger (fires once on mount — disabled when data is pre-fetched)
            dcc.Store(id="notion-auto-loaded", data=bool(recordings)),
            dcc.Interval(
                id="notion-auto-fetch-trigger",
                interval=500,  # 500ms after mount
                max_intervals=1,
                disabled=(not has_db_id)
                or bool(recordings),  # Skip if data already loaded
            ),
            html.Div(className="notion-analytics-grid", children=analytics_cards),
            html.Div(
                className="notion-workspace-grid",
                children=[
                    html.Div(
                        className="notion-list-column",
                        children=[
                            _build_search_toolbar(recordings, active_category),
                            _build_recordings_list(
                                recordings, chronos_recording_ids, match_map, stale_map
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-side-column",
                        children=[
                            html.Div(
                                id="notion-page-detail",
                                className="notion-page-detail",
                                children=_build_empty_detail_panel(),
                            ),
                        ],
                    ),
                ],
            ),
            # Hidden stores for callbacks (pre-populated when data is fetched during navigation)
            dcc.Store(id="notion-recordings-store", data=recordings or []),
            dcc.Store(id="notion-status-store", data=status),
            dcc.Store(id="notion-selected-page", data=None),
            dcc.Store(id="notion-match-map-store", data=match_map or {}),
            dcc.Store(id="notion-coverage-store", data=coverage_calendar or []),
            dcc.Store(id="notion-databases-store", data=databases),
        ],
    )


def _build_database_picker(databases, has_db_id) -> html.Div:
    """Build the database picker card — shown when no DB configured or for switching."""
    if has_db_id and not databases:
        # DB is configured and we haven't listed alternatives — nothing to show
        return html.Div()

    if not databases:
        return html.Div(
            className="notion-card notion-db-picker-card",
            children=[
                html.H3("📂 Select a Data Source"),
                html.P(
                    "No data source configured. Click 'Fetch' to discover your Notion data sources, "
                    "or set NOTION_DATABASE_ID in .env.",
                    className="notion-muted",
                ),
                html.Button(
                    "🔍 Discover Data Sources",
                    id="notion-discover-dbs-btn",
                    className="sync-action-btn",
                    n_clicks=0,
                ),
            ],
        )

    # Show database cards for selection
    db_cards = []
    for db in databases:
        icon = db.get("icon", "📁")
        title = db.get("title", "Untitled")
        desc = db.get("description", "")
        prop_count = db.get("property_count", 0)
        last_edited = db.get("last_edited", "")[:10]
        db_id = db.get("id", "")

        is_selected = False
        try:
            from src.config import get_settings
            is_selected = get_settings().notion_database_id == db_id
        except Exception:
            pass

        db_cards.append(
            html.Div(
                className=f"notion-db-card {'notion-db-selected' if is_selected else ''}",
                children=[
                    html.Div(
                        className="notion-db-card-header",
                        children=[
                            html.Span(icon, className="notion-db-icon"),
                            html.Div(
                                className="notion-db-card-info",
                                children=[
                                    html.Span(title, className="notion-db-title"),
                                    html.Span(
                                        f"{prop_count} properties · edited {last_edited}",
                                        className="notion-db-meta",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.P(desc, className="notion-db-desc") if desc else None,
                    html.Button(
                        "✅ Selected" if is_selected else "Use This Database",
                        id={"type": "notion-select-db", "db_id": db_id},
                        className="notion-action-btn notion-import-small" + (" notion-db-active-btn" if is_selected else ""),
                        n_clicks=0,
                        disabled=is_selected,
                    ),
                ],
            )
        )

    # If a DB is already selected, wrap grid in collapsible details
    if has_db_id:
        current_id = None
        try:
            from src.config import get_settings

            current_id = get_settings().notion_database_id
        except Exception:
            pass
        selected_name = next(
            (
                db.get("title", "Untitled")
                for db in databases
                if db.get("id") == current_id
            ),
            "selected",
        )
        return html.Div(
            className="notion-card notion-db-picker-card",
            children=[
                html.H3(f"📂 Data Source: {selected_name}"),
                html.Details(
                    children=[
                        html.Summary(
                            f"Switch data source ({len(databases)} available)"
                        ),
                        html.Div(className="notion-db-grid", children=db_cards),
                    ],
                ),
            ],
        )

    return html.Div(
        className="notion-card notion-db-picker-card",
        children=[
            html.H3("📂 Select a Data Source"),
            html.Div(className="notion-db-grid", children=db_cards),
        ],
    )


def _build_search_toolbar(recordings, active_category=None) -> html.Div:
    """Build search/filter/sort toolbar for the recordings list."""
    if not recordings:
        return html.Div()

    # Count categories for filter buttons
    categories = {}
    for rec in recordings:
        cat = (rec.get("category", "") if isinstance(rec, dict)
               else getattr(rec, "category", "")) or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1

    all_active = "notion-filter-active" if not active_category else ""
    filter_buttons = [
        html.Button(
            f"All ({len(recordings)})",
            id="notion-filter-all",
            className=f"notion-filter-btn {all_active}",
            n_clicks=0,
        )
    ]
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        if cat:
            cat_active = "notion-filter-active" if active_category == cat else ""
            filter_buttons.append(
                html.Button(
                    f"{cat} ({count})",
                    id={"type": "notion-filter-cat", "category": cat},
                    className=f"notion-filter-btn {cat_active}",
                    n_clicks=0,
                )
            )

    return html.Div(
        className="notion-card notion-toolbar-card",
        children=[
            html.Div(
                className="notion-toolbar",
                children=[
                    # Search input
                    html.Div(
                        className="notion-search-box",
                        children=[
                            html.Span("🔍", className="notion-search-icon"),
                            dcc.Input(
                                id="notion-search-input",
                                type="text",
                                placeholder="Search recordings by title, transcript, tags...",
                                className="notion-search-input",
                                debounce=True,
                            ),
                        ],
                    ),
                    # Sort dropdown
                    html.Div(
                        className="notion-sort-box",
                        children=[
                            html.Label("Sort: ", className="notion-sort-label"),
                            dcc.Dropdown(
                                id="notion-sort-dropdown",
                                options=[
                                    {"label": "Date (newest)", "value": "date-desc"},
                                    {"label": "Date (oldest)", "value": "date-asc"},
                                    {"label": "Title A-Z", "value": "title-asc"},
                                    {"label": "Title Z-A", "value": "title-desc"},
                                ],
                                value="date-desc",
                                clearable=False,
                                className="notion-sort-dropdown",
                            ),
                        ],
                    ),
                ],
            ),
            # Category filters
            html.Div(
                className="notion-filter-row",
                children=filter_buttons,
            ) if len(categories) > 1 else None,
        ],
    )


def _build_connection_card(status) -> html.Div:
    """Build the connection status card with OAuth or token info."""
    # Check OAuth status
    oauth_status = _get_notion_auth_status()
    has_oauth = oauth_status.get("is_authenticated", False)
    has_credentials = oauth_status.get("has_credentials", False)
    workspace = oauth_status.get("workspace_name", "")

    if status is None:
        # Not yet connected — show setup instructions
        auth_children = []
        if has_oauth:
            auth_children = [
                html.Div(
                    className="notion-auth-status notion-auth-connected",
                    children=[
                        html.Span("🟢 "),
                        html.Span(f"OAuth connected to {workspace}"),
                    ],
                ),
            ]
        elif has_credentials:
            auth_children = [
                html.A(
                    "🔗 Connect Notion Account",
                    href="/auth/notion",
                    className="sync-action-btn notion-import-btn",
                    style={"display": "inline-block", "textDecoration": "none", "marginBottom": "10px"},
                ),
            ]
        else:
            auth_children = [
                html.Div(
                    className="notion-config-hint",
                    children=[
                        html.Span("Option A: ", style={"fontWeight": "600"}),
                        html.Span("Set "),
                        html.Code("NOTION_CLIENT_ID"),
                        html.Span(" + "),
                        html.Code("NOTION_CLIENT_SECRET"),
                        html.Span(" in .env, then use the Connect button. "),
                    ],
                ),
                html.Div(
                    className="notion-config-hint",
                    children=[
                        html.Span("Option B: ", style={"fontWeight": "600"}),
                        html.Span("Set "),
                        html.Code("NOTION_TOKEN"),
                        html.Span(" directly (internal integration). "),
                    ],
                ),
            ]

        return html.Div(
            className="notion-card notion-connection-card",
            children=[
                html.H3("🔌 Connection"),
                *auth_children,
                html.P(
                    "Click 'Fetch Recordings' to connect to Notion and pull your data.",
                    className="notion-muted",
                ),
                html.Div(
                    className="notion-config-hint",
                    children=[
                        html.Span("Data source auto-detected on first visit. "),
                        html.A(
                            "Manage integrations →",
                            href="https://www.notion.so/my-integrations",
                            target="_blank",
                            className="notion-muted",
                            style={"textDecoration": "underline"},
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

    # Auth row in the connected state
    auth_row_children = []
    if has_oauth:
        auth_row_children = [
            html.Div(
                className="notion-status-item",
                children=[
                    html.Span("🔑"),
                    html.Span(f"OAuth: {workspace}"),
                ],
            ),
        ]
    elif has_credentials and not has_oauth:
        auth_row_children = [
            html.Div(
                className="notion-status-item",
                children=[
                    html.A(
                        "🔗 Connect ",
                        href="/auth/notion",
                        style={"color": "#6ee7b7", "textDecoration": "underline", "fontSize": "0.8rem"},
                    ),
                ],
            ),
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
                    *auth_row_children,
                ],
            ),
            *error_children,
        ],
    )


def _get_notion_auth_status() -> dict:
    """Check Notion OAuth status (safe, returns empty dict on failure)."""
    try:
        from src.notion_oauth import NotionOAuthClient
        client = NotionOAuthClient()
        return client.token_status
    except Exception:
        return {"is_authenticated": False, "has_credentials": False}


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
    chronos_only_count = max(0, total_chronos - both_count)
    total_unique = notion_only_count + both_count + chronos_only_count

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
                            html.Span(
                                str(notion_only_count), className="interplay-number"
                            ),
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
                                "Recordings matched across both systems",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                    # Chronos Only
                    html.Div(
                        className="interplay-stat chronos-only",
                        children=[
                            html.Span(
                                str(chronos_only_count), className="interplay-number"
                            ),
                            html.Span("Chronos Only", className="interplay-label"),
                            html.Span(
                                "Recordings only in Chronos — not in Notion",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                    # Total Unique
                    html.Div(
                        className="interplay-stat total-unique",
                        children=[
                            html.Span(str(total_unique), className="interplay-number"),
                            html.Span("Total Unique", className="interplay-label"),
                            html.Span(
                                "Combined unique recordings across both systems",
                                className="interplay-desc",
                            ),
                        ],
                    ),
                ],
            ),
            # Visual bar showing proportions
            _build_interplay_bar(notion_only_count, both_count, chronos_only_count),
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


def _build_recordings_list(
    recordings, chronos_ids, match_map=None, stale_map=None
) -> html.Div:
    """Build the scrollable recordings list."""
    stale_map = stale_map or {}
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
                            html.Span(
                                _format_date(date), className="notion-date-label"
                            ),
                            html.Span(
                                f"{len(recs)} recording{'s' if len(recs) != 1 else ''}",
                                className="notion-date-count",
                            ),
                        ],
                    ),
                    html.Div(
                        className="notion-recordings-in-date",
                        children=[
                            _build_recording_row(rec, chronos_ids, match_map, stale_map)
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


def _build_recording_row(rec, chronos_ids, match_map=None, stale_map=None) -> html.Div:
    """Build a single recording row with action buttons."""
    match_map = match_map or {}
    stale_map = stale_map or {}
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
    is_stale = stale_map.get(page_id, False)

    # Build badges
    badges = []
    if in_chronos:
        badges.append(html.Span("✅ In Chronos", className="notion-badge badge-both"))
    else:
        badges.append(
            html.Span("👻 Not in Chronos", className="notion-badge badge-notion-only")
        )
    if is_stale:
        badges.append(
            html.Span("🔄 Updated in Notion", className="notion-badge badge-stale")
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
                        *badges,
                    ],
                ),
                html.Div(
                    className="notion-rec-meta",
                    children=[
                        (
                            html.Span(time_str, className="notion-rec-time")
                            if time_str
                            else None
                        ),
                        (
                            html.Span(f" · {duration}", className="notion-rec-duration")
                            if duration
                            else None
                        ),
                        (
                            html.Span(f" · {category}", className="notion-rec-category")
                            if category
                            else None
                        ),
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


def create_notion_page_detail(
    rec, body_text: str = "", in_chronos: bool = False, matched_recording_id: str = ""
) -> html.Div:
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

    # Build action buttons list
    detail_actions = []
    if url:
        detail_actions.append(
            html.A(
                "Open in Notion ↗",
                href=url,
                target="_blank",
                rel="noopener noreferrer",
                className="notion-open-link",
            )
        )
    if not in_chronos:
        detail_actions.append(
            html.Button(
                "⚡ Import to Chronos",
                id={"type": "notion-import-one", "page_id": page_id},
                className="notion-action-btn notion-import-small",
                n_clicks=0,
            )
        )
    else:
        detail_actions.append(
            html.Button(
                "📤 Enrich in Notion",
                id={"type": "notion-writeback", "page_id": page_id},
                className="notion-action-btn notion-writeback-btn",
                n_clicks=0,
            )
        )
        if matched_recording_id:
            detail_actions.append(
                html.Button(
                    "📍 View in Timeline",
                    id="notion-detail-goto-timeline",
                    className="notion-action-btn notion-goto-timeline-btn",
                    n_clicks=0,
                    **{"data-recording-id": matched_recording_id},
                )
            )

    children = [
        html.Span("Selected Page", className="notion-detail-kicker"),
        html.Div(
            className="notion-detail-header",
            children=[
                html.H3(title),
                html.Div(
                    className="notion-detail-actions",
                    children=detail_actions,
                ),
            ],
        ),
        # Store matched recording id for the deep-link callback
        dcc.Store(id="notion-detail-matched-rec-id", data=matched_recording_id or ""),
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
        className="notion-card notion-detail-panel",
        children=children,
    )
