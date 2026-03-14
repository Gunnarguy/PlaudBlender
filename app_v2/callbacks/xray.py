"""X-ray activity monitor callbacks — toggle, poll, filter, clear."""

import time
from dash import Input, Output, State, html, no_update, callback
from dash.exceptions import PreventUpdate

# Source → icon mapping
_SOURCE_ICONS = {
    "search": "🔍",
    "nav": "🧭",
    "graph": "🕸",
    "pipeline": "⚙️",
    "detail": "📋",
    "day": "📅",
}


def _source_class(source: str) -> str:
    """Return the CSS class for a source badge."""
    known = {"search", "nav", "graph", "pipeline", "detail", "day"}
    return f"xray-source xray-source-{source}" if source in known else "xray-source xray-source-default"


def _dur_tier(ms: float) -> str:
    """Classify duration into fast/med/slow."""
    if ms < 100:
        return "fast"
    return "med" if ms < 500 else "slow"


def _dur_bar_pct(ms: float) -> int:
    """Duration → bar width % (log-scale, 1ms=5%, 2000ms=100%)."""
    if ms <= 0:
        return 5
    import math
    pct = int(min(100, max(5, (math.log10(ms) / math.log10(2000)) * 100)))
    return pct


def _build_event_row(e: dict) -> html.Div:
    """Build a single event row with visual elements."""
    ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
    source = e.get("source", "?")
    icon = _SOURCE_ICONS.get(source, "●")
    level = e.get("level", "info")
    level_class = f"xray-level-{level}"

    # Duration bar
    dur_widget = ""
    if e.get("duration_ms") is not None:
        ms = e["duration_ms"]
        tier = _dur_tier(ms)
        pct = _dur_bar_pct(ms)
        dur_widget = html.Div(
            className="xray-dur-wrap",
            children=[
                html.Div(
                    className="xray-dur-bar",
                    children=[
                        html.Div(
                            className=f"xray-dur-fill {tier}",
                            style={"width": f"{pct}%"},
                        )
                    ],
                ),
                html.Span(
                    f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.1f}s",
                    className=f"xray-dur-label {tier}",
                ),
            ],
        )

    # Detail tag
    detail = ""
    if e.get("detail"):
        detail = html.Span(e["detail"], className="xray-detail")

    return html.Div(
        className=f"xray-row {level_class}",
        **{"data-source": source, "data-level": level},
        children=[
            html.Span(ts, className="xray-ts"),
            html.Span(
                children=[html.Span(icon, style={"marginRight": "3px"}), source],
                className=_source_class(source),
            ),
            html.Span(e.get("op", ""), className="xray-op"),
            html.Span(e.get("message", ""), className="xray-msg"),
            dur_widget,
            detail,
        ],
    )


def _build_live_stats(events: list) -> list:
    """Build the collapsed-bar stat chips."""
    if not events:
        return []

    total = len(events)
    timed = [e["duration_ms"] for e in events if e.get("duration_ms") is not None]
    avg_ms = sum(timed) / len(timed) if timed else 0
    errors = sum(1 for e in events if e.get("level") == "error")
    slowest = max(timed) if timed else 0

    avg_tier = "good" if avg_ms < 100 else ("warn" if avg_ms < 500 else "bad")
    slow_tier = "good" if slowest < 200 else ("warn" if slowest < 1000 else "bad")

    stats = [
        html.Span(className="xray-stat", children=[
            html.Span(f"{total}", className=f"xray-stat-value good"),
            " ops",
        ]),
    ]
    if timed:
        stats.append(
            html.Span(className="xray-stat", children=[
                html.Span("avg ", style={"color": "#475569"}),
                html.Span(f"{avg_ms:.0f}ms", className=f"xray-stat-value {avg_tier}"),
            ])
        )
        stats.append(
            html.Span(className="xray-stat", children=[
                html.Span("peak ", style={"color": "#475569"}),
                html.Span(
                    f"{slowest:.0f}ms" if slowest < 1000 else f"{slowest / 1000:.1f}s",
                    className=f"xray-stat-value {slow_tier}",
                ),
            ])
        )
    if errors:
        stats.append(
            html.Span(className="xray-stat", children=[
                html.Span(f"{errors}", className="xray-stat-value bad"),
                " errors",
            ])
        )

    return stats


def register_xray_callbacks(app):
    """Register X-ray panel callbacks."""

    @app.callback(
        Output("xray-drawer", "className"),
        Output("xray-poll", "disabled"),
        Output("xray-chevron", "children"),
        Input("xray-toggle", "n_clicks"),
        State("xray-drawer", "className"),
        prevent_initial_call=True,
    )
    def toggle_xray(n_clicks, current_class):
        if not n_clicks:
            raise PreventUpdate
        is_collapsed = "collapsed" in (current_class or "")
        if is_collapsed:
            return "xray-drawer expanded", False, "▼"
        return "xray-drawer collapsed", True, "▲"

    @app.callback(
        Output("xray-log", "children"),
        Output("xray-badge", "children"),
        Output("xray-count", "children"),
        Output("xray-live-stats", "children"),
        Input("xray-poll", "n_intervals"),
        Input("xray-clear-btn", "n_clicks"),
        Input("xray-filter-all", "n_clicks"),
        Input("xray-filter-search", "n_clicks"),
        Input("xray-filter-nav", "n_clicks"),
        Input("xray-filter-graph", "n_clicks"),
        Input("xray-filter-errors", "n_clicks"),
        State("xray-filter-all", "className"),
        State("xray-filter-search", "className"),
        State("xray-filter-nav", "className"),
        State("xray-filter-graph", "className"),
        State("xray-filter-errors", "className"),
        prevent_initial_call=True,
    )
    def update_xray_log(
        n_intervals, clear_clicks,
        f_all, f_search, f_nav, f_graph, f_errors,
        cls_all, cls_search, cls_nav, cls_graph, cls_errors,
    ):
        from dash import ctx
        from app_v2.services.xray import get_recent_events, clear_events

        if ctx.triggered_id == "xray-clear-btn":
            clear_events()
            empty = html.Div(
                className="xray-empty",
                children=[
                    html.Span("📡", className="xray-empty-icon"),
                    "Listening for activity...",
                ],
            )
            return [empty], "", "0 events", []

        # Determine active filter from which button was clicked
        filter_map = {
            "xray-filter-all": "all",
            "xray-filter-search": "search",
            "xray-filter-nav": "nav",
            "xray-filter-graph": "graph",
            "xray-filter-errors": "error",
        }

        # If a filter button was clicked, use it; otherwise detect from classes
        active_filter = "all"
        if ctx.triggered_id in filter_map:
            active_filter = filter_map[ctx.triggered_id]
        else:
            # Detect from current active class
            for btn_id, filt in filter_map.items():
                cls = {
                    "xray-filter-all": cls_all,
                    "xray-filter-search": cls_search,
                    "xray-filter-nav": cls_nav,
                    "xray-filter-graph": cls_graph,
                    "xray-filter-errors": cls_errors,
                }.get(btn_id, "")
                if "active" in (cls or ""):
                    active_filter = filt
                    break

        all_events = get_recent_events(80)
        if not all_events:
            empty = html.Div(
                className="xray-empty",
                children=[
                    html.Span("📡", className="xray-empty-icon"),
                    "No activity yet — interact with the app to see events.",
                ],
            )
            return [empty], "", "0 events", []

        # Filter
        if active_filter == "error":
            events = [e for e in all_events if e.get("level") in ("error", "warn")]
        elif active_filter != "all":
            events = [e for e in all_events if e.get("source") == active_filter]
        else:
            events = all_events

        rows = [_build_event_row(e) for e in events]

        if not rows:
            rows = [html.Div(
                className="xray-empty",
                children=f"No {active_filter} events yet.",
            )]

        badge = str(len(all_events)) if all_events else ""
        showing = f"{len(events)}" if active_filter != "all" else f"{len(all_events)}"
        count_label = f"{showing} events"
        live_stats = _build_live_stats(all_events)

        return rows, badge, count_label, live_stats

    # Filter button active state management (client-side)
    for btn_id in ["xray-filter-all", "xray-filter-search", "xray-filter-nav", "xray-filter-graph", "xray-filter-errors"]:
        app.clientside_callback(
            """
            function(n) {
                // Deactivate all filter buttons, activate the clicked one
                var btns = document.querySelectorAll('.xray-filter-btn');
                btns.forEach(function(b) { b.classList.remove('active'); });
                var me = document.getElementById('""" + btn_id + """');
                if (me) me.classList.add('active');
                return window.dash_clientside.no_update;
            }
            """,
            Output(btn_id, "id"),  # dummy output
            Input(btn_id, "n_clicks"),
            prevent_initial_call=True,
        )
