"""Stats view component with enhanced analytics."""

from dash import html, dcc
from typing import Dict

from app_v2.services.data_service import Stats
from app_v2.components import CATEGORY_COLORS


def _hour_label(h: int) -> str:
    """Format hour for heatmap labels."""
    if h == 0:
        return "12a"
    if h < 12:
        return f"{h}a"
    if h == 12:
        return "12p"
    return f"{h - 12}p"


def create_hour_category_heatmap(
    categories_by_hour: dict, categories: dict
) -> html.Div:
    """Create a 24h × category heatmap showing when each category appears."""
    if not categories_by_hour:
        return html.Div(className="chart-empty", children=["No data"])

    # Get active categories (sorted by total count)
    active_cats = sorted(categories.keys(), key=lambda c: -categories.get(c, 0))
    # Limit to top 6 categories to keep compact
    active_cats = [c for c in active_cats if c != "unknown"][:6]

    # Find max value for color scaling
    max_val = 1
    for h_data in categories_by_hour.values():
        for count in h_data.values():
            max_val = max(max_val, count)

    rows = []
    for cat in active_cats:
        cat_label = cat.replace("_", " ").title()
        cat_color = CATEGORY_COLORS.get(cat, "#374151")

        cells = []
        for hour in range(24):
            count = categories_by_hour.get(hour, {}).get(cat, 0)
            intensity = count / max_val if max_val else 0
            # Scale from transparent to full category color
            bg = (
                f"rgba({_hex_to_rgb(cat_color)}, {max(0.05, intensity)})"
                if count
                else "transparent"
            )

            cells.append(
                html.Div(
                    className="heatmap-cell",
                    style={"backgroundColor": bg},
                    title=f"{cat_label} at {_hour_label(hour)}: {count} events",
                    children=(
                        [html.Span(str(count), className="heatmap-cell-val")]
                        if count
                        else []
                    ),
                )
            )

        rows.append(
            html.Div(
                className="heatmap-row",
                children=[
                    html.Span(cat_label, className="heatmap-row-label"),
                    html.Div(className="heatmap-cells", children=cells),
                ],
            )
        )

    # Hour labels along bottom
    hour_labels = html.Div(
        className="heatmap-hour-labels",
        children=[
            html.Span("", className="heatmap-row-label"),  # spacer
            html.Div(
                className="heatmap-cells",
                children=[
                    html.Span(
                        _hour_label(h) if h % 3 == 0 else "",
                        className="heatmap-hour-label",
                    )
                    for h in range(24)
                ],
            ),
        ],
    )

    return html.Div(
        className="heatmap-container",
        children=[*rows, hour_labels],
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R, G, B' string."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def create_stat_card(icon: str, label: str, value: str) -> html.Div:
    """Create a stat card."""
    return html.Div(
        className="stat-card",
        children=[
            html.Span(icon, className="stat-icon"),
            html.Div(
                className="stat-content",
                children=[
                    html.Span(value, className="stat-value"),
                    html.Span(label, className="stat-label"),
                ],
            ),
        ],
    )


def create_category_chart(categories: Dict[str, int]) -> html.Div:
    """Create a horizontal bar chart for categories."""
    if not categories:
        return html.Div(className="chart-empty", children=["No data"])

    total = sum(categories.values())

    return html.Div(
        className="category-chart",
        children=[
            html.Div(
                className="chart-row",
                children=[
                    html.Span(cat, className="chart-label"),
                    html.Div(
                        className="chart-bar-wrapper",
                        children=[
                            html.Div(
                                className="chart-bar",
                                style={
                                    "width": f"{(count / total) * 100}%",
                                    "backgroundColor": CATEGORY_COLORS.get(
                                        cat, "#374151"
                                    ),
                                },
                            ),
                        ],
                    ),
                    html.Span(str(count), className="chart-value"),
                ],
            )
            for cat, count in sorted(categories.items(), key=lambda x: -x[1])
        ],
    )


def create_day_of_week_chart(data: Dict[str, int]) -> html.Div:
    """Create a chart showing activity by day of week."""
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    max_val = max(data.values()) if data else 1

    return html.Div(
        className="day-chart",
        children=[
            html.Div(
                className="day-bar",
                children=[
                    html.Div(
                        className="bar-fill",
                        style={"height": f"{(data.get(day, 0) / max_val) * 100}%"},
                    ),
                    html.Span(day[:3], className="day-label"),
                ],
            )
            for day in days_order
        ],
    )


def create_hour_chart(data: Dict[int, int]) -> html.Div:
    """Create a chart showing activity by hour."""
    max_val = max(data.values()) if data else 1

    return html.Div(
        className="hour-chart",
        children=[
            html.Div(
                className="hour-bar",
                children=[
                    html.Div(
                        className="bar-fill",
                        style={"height": f"{(data.get(hour, 0) / max_val) * 100}%"},
                        title=f"{hour}:00 - {data.get(hour, 0)} events",
                    ),
                ],
            )
            for hour in range(24)
        ],
    )


def create_sentiment_chart(dist: Dict[str, int]) -> html.Div:
    """Create a sentiment distribution bar."""
    total = sum(dist.values()) or 1
    pos = dist.get("positive", 0)
    neu = dist.get("neutral", 0)
    neg = dist.get("negative", 0)

    return html.Div(
        className="sentiment-chart",
        children=[
            html.Div(
                className="sentiment-bar-stack",
                children=[
                    html.Div(
                        className="sentiment-segment",
                        style={
                            "width": f"{(pos / total) * 100}%",
                            "backgroundColor": "#10b981",
                        },
                        title=f"Positive: {pos}",
                    ),
                    html.Div(
                        className="sentiment-segment",
                        style={
                            "width": f"{(neu / total) * 100}%",
                            "backgroundColor": "#64748b",
                        },
                        title=f"Neutral: {neu}",
                    ),
                    html.Div(
                        className="sentiment-segment",
                        style={
                            "width": f"{(neg / total) * 100}%",
                            "backgroundColor": "#ef4444",
                        },
                        title=f"Negative: {neg}",
                    ),
                ],
            ),
            html.Div(
                className="sentiment-labels",
                children=[
                    html.Span(
                        f"😊 {pos} positive",
                        style={"color": "#10b981"},
                    ),
                    html.Span(
                        f"😐 {neu} neutral",
                        style={"color": "#94a3b8"},
                    ),
                    html.Span(
                        f"😟 {neg} negative",
                        style={"color": "#ef4444"},
                    ),
                ],
            ),
        ],
    )


def create_stats_view(stats: Stats) -> html.Div:
    """Create the full stats view with enhanced analytics."""

    # Determine sentiment indicator
    sentiment_emoji = "😐"
    if stats.avg_sentiment > 0.15:
        sentiment_emoji = "😊"
    elif stats.avg_sentiment < -0.15:
        sentiment_emoji = "😟"

    # Build Plaud cloud section if stats are available
    plaud_section_children = []
    if stats.plaud_cloud_stats:
        cs = stats.plaud_cloud_stats
        cloud_total = cs.get("total_count", 0)
        cloud_hours = cs.get("total_duration_hours", 0)
        cloud_avg_min = cs.get("avg_duration_minutes", 0)
        date_range = cs.get("date_range", {})
        earliest = date_range.get("earliest", "—")
        latest = date_range.get("latest", "—")
        local_recs = stats.total_recordings
        # Cap sync % at 100 — local DB accumulates old recordings that
        # Plaud cloud may have pruned, so local > cloud is normal.
        synced_pct = min((local_recs / cloud_total * 100), 100) if cloud_total else 0

        plaud_section_children = [
            html.Div(
                className="stats-section plaud-cloud-section",
                children=[
                    html.H3("☁️ Plaud Cloud", className="section-title"),
                    html.Div(
                        className="stats-grid",
                        children=[
                            create_stat_card(
                                "🌐", "Cloud Recordings", str(cloud_total)
                            ),
                            create_stat_card("⏱️", "Cloud Hours", f"{cloud_hours:.1f}"),
                            create_stat_card(
                                "📊", "Avg Duration", f"{cloud_avg_min:.0f}m"
                            ),
                            create_stat_card("🔄", "Synced", f"{synced_pct:.0f}%"),
                        ],
                    ),
                    html.Div(
                        className="plaud-date-range",
                        children=[
                            html.Span(f"Recording range: {earliest} → {latest}"),
                            html.Span(
                                f" · {local_recs} local / {cloud_total} in cloud"
                                + (" (local includes older recordings pruned from cloud)" if local_recs > cloud_total else ""),
                                className="plaud-sync-detail",
                            ),
                        ],
                    ),
                ],
            ),
        ]

    return html.Div(
        className="stats-view",
        children=[
            # Header
            html.Div(
                className="view-header",
                children=[
                    html.H2("📊 Statistics & Analytics", className="view-title"),
                    html.P(
                        "Deep insights into your recording activity",
                        className="view-subtitle",
                    ),
                ],
            ),
            # Summary cards — row 1
            html.Div(
                className="stats-grid",
                children=[
                    create_stat_card("🎙️", "Recordings", str(stats.total_recordings)),
                    create_stat_card("📝", "Events", str(stats.total_events)),
                    create_stat_card("📅", "Days", str(stats.total_days)),
                    create_stat_card(
                        "⏱️", "Total Hours", f"{stats.total_duration_hours:.1f}"
                    ),
                ],
            ),
            # Summary cards — row 2 (new)
            html.Div(
                className="stats-grid",
                style={"marginTop": "8px"},
                children=[
                    create_stat_card(
                        "📈",
                        "Avg Events/Rec",
                        f"{stats.avg_events_per_recording:.1f}",
                    ),
                    create_stat_card(
                        "⏳",
                        "Avg Duration",
                        f"{stats.avg_recording_duration_min:.0f}m",
                    ),
                    create_stat_card(
                        sentiment_emoji,
                        "Avg Sentiment",
                        f"{stats.avg_sentiment:+.2f}",
                    ),
                    create_stat_card(
                        "✅",
                        "Pipeline",
                        f"{stats.pipeline_completion_rate:.0f}%",
                    ),
                    *(
                        [
                            create_stat_card(
                                "⚠️",
                                "Duration Capped",
                                str(stats.events_duration_capped),
                            )
                        ]
                        if stats.events_duration_capped
                        else []
                    ),
                ],
            ),
            # Plaud cloud stats (if available)
            *plaud_section_children,
            # API Cost tracker
            create_cost_section(),
            # Insights callout
            html.Div(
                className="stats-insights",
                children=[
                    html.H3("💡 Insights", className="section-title"),
                    html.Div(
                        className="insights-grid",
                        children=[
                            _insight_card(
                                "Most productive day",
                                stats.most_productive_day or "—",
                                f"{stats.events_by_day_of_week.get(stats.most_productive_day, 0)} events",
                            ),
                            _insight_card(
                                "Peak hour",
                                (
                                    f"{stats.most_productive_hour}:00"
                                    if stats.most_productive_hour
                                    else "—"
                                ),
                                f"{stats.events_by_hour.get(stats.most_productive_hour, 0)} events",
                            ),
                            _insight_card(
                                "Longest recording",
                                f"{stats.longest_recording_min:.0f} min",
                                "",
                            ),
                        ],
                    ),
                ],
            ),
            # Sentiment analysis
            html.Div(
                className="stats-section",
                children=[
                    html.H3("🎭 Sentiment Analysis", className="section-title"),
                    create_sentiment_chart(stats.sentiment_distribution),
                ],
            ),
            # Category breakdown
            html.Div(
                className="stats-section",
                children=[
                    html.H3("Categories", className="section-title"),
                    create_category_chart(stats.categories),
                ],
            ),
            # Hour × Category heatmap
            html.Div(
                className="stats-section",
                children=[
                    html.H3("🕐 When You Do What", className="section-title"),
                    html.P(
                        "Activity by hour and category — brighter = more events",
                        className="section-subtitle",
                    ),
                    create_hour_category_heatmap(
                        stats.categories_by_hour, stats.categories
                    ),
                ],
            ),
            # Activity by day of week
            html.Div(
                className="stats-section",
                children=[
                    html.H3("Activity by Day", className="section-title"),
                    create_day_of_week_chart(stats.events_by_day_of_week),
                ],
            ),
            # Activity by hour
            html.Div(
                className="stats-section",
                children=[
                    html.H3("Activity by Hour", className="section-title"),
                    create_hour_chart(stats.events_by_hour),
                    html.Div(
                        className="hour-labels",
                        children=[
                            html.Span("12am"),
                            html.Span("6am"),
                            html.Span("12pm"),
                            html.Span("6pm"),
                            html.Span("12am"),
                        ],
                    ),
                ],
            ),
            # Top keywords
            html.Div(
                className="stats-section",
                children=[
                    html.H3("Top Keywords", className="section-title"),
                    html.Div(
                        className="keywords-cloud",
                        children=[
                            html.Span(
                                kw,
                                className="keyword-tag",
                                style={
                                    "fontSize": f"{max(0.75, min(1.5, count / 3))}rem"
                                },
                            )
                            for kw, count in stats.top_keywords[:20]
                        ],
                    ),
                ],
            ),
        ],
    )


def _insight_card(label: str, value: str, detail: str) -> html.Div:
    """Create a small insight card."""
    children = [
        html.Span(label, className="insight-label"),
        html.Span(value, className="insight-value"),
    ]
    if detail:
        children.append(html.Span(detail, className="insight-detail"))

    return html.Div(className="insight-card", children=children)


# ════════════════════════════════════════════════════════════
# API Cost Section
# ════════════════════════════════════════════════════════════


def _fmt_cost(usd: float) -> str:
    """Format a USD cost to appropriate precision."""
    if usd == 0:
        return "FREE"
    if usd < 0.01:
        return f"${usd:.4f}"
    if usd < 1:
        return f"${usd:.3f}"
    return f"${usd:.2f}"


def _fmt_tokens(n: int) -> str:
    """Format token count for display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def create_cost_section() -> html.Div:
    """Create the API cost tracking section for Stats view.

    Renders a live-updating cost dashboard via a client-side interval
    that polls /xray/api/costs every 15 seconds.
    """
    from src.chronos.cost_tracker import (
        get_session_cost,
        get_cost_summary,
        get_model_pricing_table,
    )

    session = get_session_cost()
    historical = get_cost_summary(days=30)
    pricing = get_model_pricing_table()

    # --- Session summary cards ---
    session_cards = html.Div(
        className="stats-grid",
        id="cost-session-cards",
        children=[
            create_stat_card(
                "💰", "Session Cost", _fmt_cost(session["total_cost_usd"])
            ),
            create_stat_card("📞", "API Calls", str(session["total_calls"])),
            create_stat_card(
                "📥",
                "Input Tokens",
                _fmt_tokens(session["total_input_tokens"]),
            ),
            create_stat_card(
                "📤",
                "Output Tokens",
                _fmt_tokens(session["total_output_tokens"]),
            ),
        ],
    )

    # --- Historical summary (30 days) ---
    hist_cards = html.Div(
        className="stats-grid",
        style={"marginTop": "8px"},
        children=[
            create_stat_card(
                "📅", "30-Day Cost", _fmt_cost(historical["total_cost_usd"])
            ),
            create_stat_card("📊", "30-Day Calls", str(historical["total_calls"])),
            create_stat_card(
                "📥",
                "30-Day Input",
                _fmt_tokens(historical["total_input_tokens"]),
            ),
            create_stat_card(
                "📤",
                "30-Day Output",
                _fmt_tokens(historical["total_output_tokens"]),
            ),
        ],
    )

    # --- Per-model breakdown (session) ---
    model_rows = []
    for model, data in sorted(
        session["by_model"].items(),
        key=lambda x: -x[1]["cost_usd"],
    ):
        model_rows.append(
            html.Div(
                className="cost-model-row",
                children=[
                    html.Span(model, className="cost-model-name"),
                    html.Span(f'{data["calls"]} calls', className="cost-model-calls"),
                    html.Span(
                        f'{_fmt_tokens(data["input_tokens"])} → {_fmt_tokens(data["output_tokens"])}',
                        className="cost-model-tokens",
                    ),
                    html.Span(
                        _fmt_cost(data["cost_usd"]),
                        className="cost-model-amount"
                        + (" cost-free" if data["cost_usd"] == 0 else " cost-paid"),
                    ),
                ],
            )
        )

    # Add historical models not in session
    for model, data in sorted(
        historical["by_model"].items(),
        key=lambda x: -x[1]["cost_usd"],
    ):
        if model not in session["by_model"]:
            model_rows.append(
                html.Div(
                    className="cost-model-row cost-historical",
                    children=[
                        html.Span(model, className="cost-model-name"),
                        html.Span(
                            f'{data["calls"]} calls (30d)',
                            className="cost-model-calls",
                        ),
                        html.Span(
                            f'{_fmt_tokens(data["input_tokens"])} → {_fmt_tokens(data["output_tokens"])}',
                            className="cost-model-tokens",
                        ),
                        html.Span(
                            _fmt_cost(data["cost_usd"]),
                            className="cost-model-amount"
                            + (" cost-free" if data["cost_usd"] == 0 else " cost-paid"),
                        ),
                    ],
                )
            )

    model_breakdown = html.Div(
        className="cost-model-breakdown",
        children=(
            model_rows
            if model_rows
            else [html.Div("No API calls recorded yet", className="cost-empty")]
        ),
    )

    # --- Daily cost chart (last 30 days) ---
    daily_bars = []
    by_day = historical.get("by_day", [])
    max_day_cost = max((d["cost_usd"] for d in by_day), default=0.01) or 0.01
    for day_data in reversed(by_day[-14:]):  # Last 14 days
        pct = (day_data["cost_usd"] / max_day_cost) * 100 if max_day_cost else 0
        date_short = day_data["date"][-5:] if day_data["date"] else "?"  # MM-DD
        daily_bars.append(
            html.Div(
                className="cost-day-bar",
                children=[
                    html.Div(
                        className="bar-fill"
                        + (
                            " cost-bar-free"
                            if day_data["cost_usd"] == 0
                            else " cost-bar-paid"
                        ),
                        style={"height": f"{max(2, pct)}%"},
                        title=f'{day_data["date"]}: {_fmt_cost(day_data["cost_usd"])} ({day_data["calls"]} calls)',
                    ),
                    html.Span(date_short, className="cost-day-label"),
                ],
            )
        )

    daily_chart = html.Div(
        className="cost-daily-chart",
        children=(
            daily_bars
            if daily_bars
            else [html.Div("No daily data yet", className="cost-empty")]
        ),
    )

    # --- Model pricing reference table ---
    pricing_rows = []
    for p in pricing:
        inp_str = "FREE" if p["input_per_mtok"] == 0 else f'${p["input_per_mtok"]:.2f}'
        out_str = (
            "FREE" if p["output_per_mtok"] == 0 else f'${p["output_per_mtok"]:.2f}'
        )
        pricing_rows.append(
            html.Tr(
                className="pricing-row"
                + (" pricing-free" if p["tier"] == "free" else ""),
                children=[
                    html.Td(p["label"], className="pricing-label"),
                    html.Td(p["provider"].title(), className="pricing-provider"),
                    html.Td(inp_str, className="pricing-input"),
                    html.Td(out_str, className="pricing-output"),
                    html.Td(
                        "FREE" if p["tier"] == "free" else "Paid",
                        className="pricing-tier"
                        + (" tier-free" if p["tier"] == "free" else " tier-paid"),
                    ),
                ],
            )
        )

    pricing_table = html.Table(
        className="pricing-table",
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("Model"),
                        html.Th("Provider"),
                        html.Th("Input/1M"),
                        html.Th("Output/1M"),
                        html.Th("Tier"),
                    ]
                )
            ),
            html.Tbody(pricing_rows),
        ],
    )

    # --- Recent calls (last 20) ---
    recent_rows = []
    for r in session.get("recent", [])[:10]:
        recent_rows.append(
            html.Div(
                className="cost-recent-row",
                children=[
                    html.Span(
                        f'{r["ago_s"]:.0f}s ago',
                        className="cost-recent-ago",
                    ),
                    html.Span(r["model"], className="cost-recent-model"),
                    html.Span(r["type"], className="cost-recent-type"),
                    html.Span(
                        f'{_fmt_tokens(r["input_tokens"])}→{_fmt_tokens(r["output_tokens"])}',
                        className="cost-recent-tokens",
                    ),
                    html.Span(
                        _fmt_cost(r["cost_usd"]),
                        className="cost-recent-cost"
                        + (" cost-free" if r["cost_usd"] == 0 else " cost-paid"),
                    ),
                ],
            )
        )

    recent_section = html.Div(
        className="cost-recent-list",
        children=(
            recent_rows
            if recent_rows
            else [html.Div("No recent calls", className="cost-empty")]
        ),
    )

    # --- Auto-refresh interval ---
    interval = dcc.Interval(
        id="cost-refresh-interval",
        interval=15_000,  # 15 seconds
        n_intervals=0,
    )

    return html.Div(
        className="stats-section cost-section",
        children=[
            html.H3("💰 API Cost Tracker", className="section-title"),
            html.P(
                "Real-time spending across all AI models — session + 30-day history",
                className="section-subtitle",
            ),
            interval,
            html.Div(
                id="cost-live-container",
                children=[
                    session_cards,
                    hist_cards,
                    # Model breakdown
                    html.Div(
                        className="cost-subsection",
                        children=[
                            html.H4("Per Model", className="cost-subsection-title"),
                            model_breakdown,
                        ],
                    ),
                    # Daily chart
                    html.Div(
                        className="cost-subsection",
                        children=[
                            html.H4(
                                "Daily Cost (14 days)",
                                className="cost-subsection-title",
                            ),
                            daily_chart,
                        ],
                    ),
                    # Recent calls
                    html.Div(
                        className="cost-subsection",
                        children=[
                            html.H4("Recent Calls", className="cost-subsection-title"),
                            recent_section,
                        ],
                    ),
                ],
            ),
            # Pricing reference
            html.Details(
                className="cost-pricing-details",
                children=[
                    html.Summary("Model Pricing Reference (per 1M tokens)"),
                    pricing_table,
                ],
            ),
        ],
    )
