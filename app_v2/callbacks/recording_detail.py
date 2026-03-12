"""Callbacks for recording detail interactions."""

import logging

from dash import Input, Output, State, callback_context, ALL, no_update, ClientsideFunction

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

        svc = get_data_service()
        ok = svc.save_category_override(event_id, new_value)

        if ok:
            logger.info(f"Category override saved: event={event_id} → {new_value}")
            return f"✓ Category updated to {new_value}"
        else:
            logger.warning(f"Failed to save category override: event={event_id}")
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
