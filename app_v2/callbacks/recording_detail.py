"""Callbacks for recording detail interactions."""

import logging

from dash import (
    Input,
    Output,
    State,
    callback_context,
    ALL,
    no_update,
    ClientsideFunction,
    html,
)

logger = logging.getLogger(__name__)


def register_recording_detail_callbacks(app):
    """Register recording detail callbacks."""

    @app.callback(
        Output("category-save-status", "children"),
        Input({"type": "event-category-edit", "id": ALL}, "value"),
        State({"type": "event-category-edit", "id": ALL}, "id"),
        prevent_initial_call=True,
    )
    def save_category_override(values, ids):
        """Save a user category override when the dropdown changes."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        # Find which dropdown triggered
        trigger = ctx.triggered[0]
        prop_id = trigger["prop_id"]
        new_value = trigger["value"]

        if not new_value:
            return no_update

        # Extract the event ID from the trigger prop_id
        # prop_id format: '{"id":"<event_id>","type":"event-category-edit"}.value'
        import json

        try:
            id_part = prop_id.rsplit(".", 1)[0]
            parsed = json.loads(id_part)
            event_id = parsed["id"]
        except (json.JSONDecodeError, KeyError):
            logger.error(f"Failed to parse trigger prop_id: {prop_id}")
            return no_update

        from app_v2.services.data_service import get_data_service
        from app_v2.services.xray import xray_log

        svc = get_data_service()
        ok = svc.save_category_override(event_id, new_value)
        is_success = ok.get("success", False) if isinstance(ok, dict) else bool(ok)

        if is_success:
            logger.info(f"Category override saved: event={event_id} → {new_value}")
            xray_log("detail", "category", f"You re-categorized this as '{new_value}'")
            return f"✓ Category updated to {new_value}"
        else:
            logger.warning(f"Failed to save category override: event={event_id}")
            xray_log(
                "detail", "category", f"Couldn't save that category change", level="error"
            )
            return "⚠ Could not save override"

    # Clientside callback for instant event filtering (no server round-trip)
    app.clientside_callback(
        """
        function(filterText) {
            if (!filterText) {
                // Show all event cards
                var cards = document.querySelectorAll('.event-card');
                cards.forEach(function(c) { c.style.display = ''; });
                return window.dash_clientside.no_update;
            }
            var term = filterText.toLowerCase();
            var cards = document.querySelectorAll('.event-card');
            cards.forEach(function(c) {
                var text = c.textContent.toLowerCase();
                c.style.display = text.includes(term) ? '' : 'none';
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("events-list-container", "className"),  # dummy output
        Input("event-filter-input", "value"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("single-workflow-result", "children"),
        Input("run-single-workflow-btn", "n_clicks"),
        State("detail-recording-id", "data"),
        State("single-workflow-template", "value"),
        State("single-workflow-model", "value"),
        prevent_initial_call=True,
    )
    def submit_single_workflow(n_clicks, recording_id, template_id, model):
        """Submit a Plaud AI workflow for the currently viewed recording."""
        if not n_clicks or not recording_id:
            return no_update

        from app_v2.services.data_service import get_data_service

        svc = get_data_service()
        result = svc.submit_single_recording_workflow(
            recording_id=recording_id,
            template_id=template_id or None,
            model=model or "openai",
        )

        if result.get("error"):
            return html.Span(
                f"❌ {result['error']}",
                className="workflow-inline-error",
            )

        return html.Span(
            f"☁️ Submitted → {result.get('workflow_id', '')[:12]}…",
            className="workflow-inline-success",
        )
