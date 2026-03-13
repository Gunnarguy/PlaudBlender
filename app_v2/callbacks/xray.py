"""X-ray drawer callbacks — toggle, poll, clear."""

import time
from dash import Input, Output, State, html, no_update, callback
from dash.exceptions import PreventUpdate


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
        Input("xray-poll", "n_intervals"),
        Input("xray-clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_xray_log(n_intervals, clear_clicks):
        from dash import ctx
        from app_v2.services.xray import get_recent_events, clear_events

        if ctx.triggered_id == "xray-clear-btn":
            clear_events()
            return [], "", "0 events"

        events = get_recent_events(80)
        if not events:
            return (
                [
                    html.Div(
                        "No telemetry yet — interact with the app.",
                        className="xray-empty",
                    )
                ],
                "",
                "0 events",
            )

        rows = []
        for e in events:
            ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            dur = ""
            if e.get("duration_ms") is not None:
                ms = e["duration_ms"]
                dur_class = (
                    "xray-dur fast"
                    if ms < 100
                    else ("xray-dur med" if ms < 500 else "xray-dur slow")
                )
                dur = html.Span(f"{ms:.0f}ms", className=dur_class)

            detail = ""
            if e.get("detail"):
                detail = html.Span(e["detail"], className="xray-detail")

            level_class = f"xray-level-{e.get('level', 'info')}"

            rows.append(
                html.Div(
                    className=f"xray-row {level_class}",
                    children=[
                        html.Span(ts, className="xray-ts"),
                        html.Span(e["source"], className="xray-source"),
                        html.Span(e["op"], className="xray-op"),
                        html.Span(e["message"], className="xray-msg"),
                        dur,
                        detail,
                    ],
                )
            )

        badge = str(len(events)) if events else ""
        count_label = f"{len(events)} events"
        return rows, badge, count_label
