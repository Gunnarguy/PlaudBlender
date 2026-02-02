"""Streamlit component: Webhook Management Panel

Full-featured webhook management UI:
- View incoming webhook events
- Configure webhook endpoints
- Test webhook connectivity
- Process webhook events through Chronos pipeline
- Real-time event handling
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import hmac
import hashlib

import streamlit as st

from src.plaud_webhook import PlaudWebhookHandler, PlaudEventType, get_webhook_handler
from src.plaud_admin import PlaudAdminClient
from src.database import SessionLocal, init_db
from src.database.chronos_repository import (
    list_chronos_webhook_events,
    mark_webhook_event_processed,
    add_chronos_webhook_event,
)
from src.config import get_settings


def _event_type_emoji(event_type: str) -> str:
    """Get emoji for event type."""
    event_type_lower = event_type.lower() if event_type else ""
    if "transcribe" in event_type_lower or "transcript" in event_type_lower:
        return "📝"
    elif "workflow" in event_type_lower:
        return "⚡"
    elif "recording" in event_type_lower:
        return "🎙️"
    elif "device" in event_type_lower:
        return "📱"
    elif "upload" in event_type_lower:
        return "📤"
    elif "error" in event_type_lower or "fail" in event_type_lower:
        return "❌"
    else:
        return "📨"


def _status_badge(status: str) -> str:
    """Get status badge HTML."""
    colors = {
        "pending": ("orange", "⏳"),
        "processing": ("blue", "⚙️"),
        "processed": ("green", "✅"),
        "failed": ("red", "❌"),
    }
    color, emoji = colors.get(status, ("gray", "❓"))
    return f"{emoji} {status.title()}"


def render_webhook_events() -> None:
    """Render incoming webhook events viewer."""
    st.subheader("📨 Incoming Events")
    st.caption("View and process webhook events received from Plaud.")

    init_db()
    session = SessionLocal()

    try:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect(
                "Status",
                options=["pending", "processing", "processed", "failed"],
                default=["pending", "processing"],
                key="webhook_status_filter",
            )
        with col2:
            days_back = st.number_input(
                "Days back",
                min_value=1,
                max_value=30,
                value=7,
                key="webhook_days_back",
            )
        with col3:
            limit = st.number_input(
                "Max events",
                min_value=10,
                max_value=500,
                value=100,
                key="webhook_limit",
            )

        # Refresh button
        if st.button("🔄 Refresh Events", width="content"):
            st.rerun()

        # Get events
        events = list_chronos_webhook_events(session, limit=int(limit))

        # Filter by status
        if status_filter:
            events = [e for e in events if e.processing_status in status_filter]

        if not events:
            st.info(
                "No webhook events found. Configure a webhook endpoint and wait for events from Plaud."
            )
            _show_webhook_setup_hint()
            return

        st.caption(f"Showing {len(events)} event(s)")

        # Summary metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Total", len(events))
        with metric_cols[1]:
            pending = sum(1 for e in events if e.processing_status == "pending")
            st.metric("Pending", pending)
        with metric_cols[2]:
            processed = sum(1 for e in events if e.processing_status == "processed")
            st.metric("Processed", processed)
        with metric_cols[3]:
            failed = sum(1 for e in events if e.processing_status == "failed")
            st.metric("Failed", failed)

        st.divider()

        # Bulk actions
        bulk_cols = st.columns(4)
        with bulk_cols[0]:
            if st.button(
                "✅ Mark All Pending as Processed",
                width="stretch",
                disabled=pending == 0,
            ):
                for e in events:
                    if e.processing_status == "pending":
                        mark_webhook_event_processed(
                            session, str(e.event_id), status="processed"
                        )
                st.success(f"Marked {pending} events as processed")
                st.rerun()

        with bulk_cols[1]:
            if st.button(
                "⚡ Process All Pending",
                width="stretch",
                disabled=pending == 0,
            ):
                _process_pending_events(session, events)

        # Event list
        for event in events:
            event_id = str(event.event_id)
            event_type = event.event_type or "unknown"

            with st.expander(
                f"{_event_type_emoji(event_type)} {event_type} · "
                f"{_status_badge(event.processing_status)} · "
                f"{event.received_at.strftime('%Y-%m-%d %H:%M:%S') if event.received_at else 'N/A'}",
                expanded=event.processing_status == "pending",
            ):
                # Event details
                detail_cols = st.columns([2, 1, 1])
                with detail_cols[0]:
                    st.write(f"**Event ID:** `{event_id}`")
                    st.write(f"**Webhook ID:** `{event.webhook_id or 'N/A'}`")
                with detail_cols[1]:
                    st.write(f"**Recording ID:** `{event.recording_id or 'N/A'}`")
                with detail_cols[2]:
                    st.write(f"**Status:** {_status_badge(event.processing_status)}")

                # Payload viewer
                if event.payload:
                    with st.expander("📋 Payload", expanded=False):
                        st.json(event.payload)

                # Headers viewer
                if event.headers:
                    with st.expander("🔧 Headers", expanded=False):
                        st.json(event.headers)

                # Action buttons
                action_cols = st.columns(4)
                with action_cols[0]:
                    if st.button(
                        "✅ Mark Processed",
                        key=f"processed_{event_id}",
                        width="stretch",
                        disabled=event.processing_status == "processed",
                    ):
                        mark_webhook_event_processed(
                            session, event_id, status="processed"
                        )
                        st.rerun()

                with action_cols[1]:
                    if st.button(
                        "❌ Mark Failed",
                        key=f"failed_{event_id}",
                        width="stretch",
                    ):
                        mark_webhook_event_processed(session, event_id, status="failed")
                        st.rerun()

                with action_cols[2]:
                    if st.button(
                        "🔄 Replay to Pipeline",
                        key=f"replay_{event_id}",
                        width="stretch",
                    ):
                        _replay_event_to_pipeline(event)

                with action_cols[3]:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{event_id}",
                        width="stretch",
                    ):
                        # Would need a delete function
                        st.warning("Delete not implemented yet")

    finally:
        session.close()


def _show_webhook_setup_hint() -> None:
    """Show hint for webhook setup."""
    settings = get_settings()
    with st.expander("ℹ️ How to set up webhooks"):
        st.markdown(
            """
            1. **Configure webhook URL** in your `.env`:
               ```
               PLAUD_WEBHOOK_URL=https://your-server.com/api/plaud-webhook
               PLAUD_WEBHOOK_SECRET=your_secret_key
               ```

            2. **Create a webhook** in the Webhooks tab

            3. **Test the connection** using the Ping button

            4. Events will appear here as Plaud sends them
            """
        )

        if settings.plaud_webhook_url:
            st.success(f"Webhook URL configured: `{settings.plaud_webhook_url}`")
        else:
            st.warning("PLAUD_WEBHOOK_URL not set in .env")


def _process_pending_events(session, events) -> None:
    """Process all pending webhook events through the pipeline."""
    pending = [e for e in events if e.processing_status == "pending"]
    if not pending:
        st.info("No pending events to process")
        return

    progress = st.progress(0, text="Processing events...")

    for i, event in enumerate(pending):
        progress.progress(
            (i + 1) / len(pending),
            text=f"Processing {i + 1}/{len(pending)}: {event.event_type}",
        )

        try:
            _replay_event_to_pipeline(event, quiet=True)
            mark_webhook_event_processed(
                session, str(event.event_id), status="processed"
            )
        except Exception as e:
            mark_webhook_event_processed(
                session, str(event.event_id), status="failed", error=str(e)
            )

    progress.empty()
    st.success(f"Processed {len(pending)} events")
    st.rerun()


def _replay_event_to_pipeline(event, quiet: bool = False) -> None:
    """Replay a webhook event through the Chronos pipeline."""
    rec_id = event.recording_id
    if not rec_id and event.payload:
        rec_id = event.payload.get("recording_id") or event.payload.get("file_id")

    if not rec_id:
        if not quiet:
            st.error("No recording_id found in event payload")
        return

    import subprocess
    import sys

    if not quiet:
        st.info(f"Triggering ingest for {rec_id}...")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/chronos_pipeline.py",
            "--ingest",
            "--limit",
            "1",
            "--recording-id",
            str(rec_id),
        ],
        capture_output=True,
        text=True,
    )

    if not quiet:
        if result.returncode == 0:
            st.success(f"✅ Ingest triggered for {rec_id}")
            st.code(result.stdout)
        else:
            st.error(f"❌ Ingest failed (exit {result.returncode})")
            st.code(result.stderr or result.stdout)


def render_webhook_config() -> None:
    """Render webhook configuration UI."""
    st.subheader("⚙️ Webhook Configuration")
    st.caption("Manage webhook endpoints for receiving Plaud events.")

    try:
        admin = PlaudAdminClient()
    except Exception as e:
        st.error(f"Failed to initialize Plaud client: {e}")
        return

    # Current webhooks
    st.markdown("**Registered Webhooks**")

    with st.spinner("Loading webhooks..."):
        webhooks = admin.list_webhooks()

    if not webhooks:
        st.info("No webhooks configured yet.")
    else:
        for webhook in webhooks:
            webhook_id = webhook.get("id", "unknown")
            with st.expander(
                f"🔗 {webhook.get('url', 'Unknown URL')}",
                expanded=False,
            ):
                st.write(f"**ID:** `{webhook_id}`")
                st.write(f"**Events:** {', '.join(webhook.get('events', ['all']))}")
                st.write(f"**Created:** {webhook.get('created_at', 'N/A')}")

                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button(
                        "🔔 Ping",
                        key=f"ping_{webhook_id}",
                        width="stretch",
                    ):
                        if admin.ping_webhook(webhook_id):
                            st.success("Ping sent successfully!")
                        else:
                            st.error("Ping failed")

                with btn_cols[1]:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_wh_{webhook_id}",
                        width="stretch",
                    ):
                        if admin.delete_webhook(webhook_id):
                            st.success("Webhook deleted")
                            st.rerun()
                        else:
                            st.error("Failed to delete webhook")

    st.divider()

    # Create new webhook
    st.markdown("**Create New Webhook**")

    settings = get_settings()
    default_url = settings.plaud_webhook_url or ""

    with st.form("create_webhook_form"):
        url = st.text_input(
            "Callback URL",
            value=default_url,
            placeholder="https://your-server.com/api/plaud-webhook",
            help="The URL that will receive webhook events",
        )

        events = st.multiselect(
            "Events to receive",
            options=[
                "recording.created",
                "recording.updated",
                "recording.deleted",
                "transcription.completed",
                "workflow.completed",
                "workflow.failed",
                "device.connected",
                "device.disconnected",
            ],
            default=[
                "recording.created",
                "transcription.completed",
                "workflow.completed",
            ],
            help="Select which events to receive (empty = all events)",
        )

        submitted = st.form_submit_button("Create Webhook", type="primary")

        if submitted and url.strip():
            try:
                result = admin.create_webhook(url.strip(), events=events or None)
                st.success(f"✅ Webhook created: `{result.get('id')}`")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create webhook: {e}")


def render_webhook_test() -> None:
    """Render webhook testing UI."""
    st.subheader("🧪 Test Webhooks")
    st.caption("Simulate webhook events for testing the integration.")

    # Simulate incoming event
    st.markdown("**Simulate Incoming Event**")

    with st.form("simulate_event"):
        event_type = st.selectbox(
            "Event Type",
            options=[
                "recording.created",
                "transcription.completed",
                "workflow.completed",
                "custom",
            ],
        )

        if event_type == "custom":
            custom_type = st.text_input("Custom event type")
            event_type = custom_type or "custom.event"

        recording_id = st.text_input(
            "Recording ID (optional)",
            placeholder="plaud_recording_123",
        )

        payload_json = st.text_area(
            "Custom payload (JSON)",
            value="{}",
            height=100,
            help="Additional payload data in JSON format",
        )

        submitted = st.form_submit_button("Simulate Event")

        if submitted:
            try:
                payload = json.loads(payload_json)
                if recording_id:
                    payload["recording_id"] = recording_id

                init_db()
                session = SessionLocal()
                try:
                    add_chronos_webhook_event(
                        session,
                        webhook_id=None,
                        event_type=event_type,
                        payload=payload,
                        headers={"X-Simulated": "true"},
                        recording_id=recording_id or None,
                    )
                    st.success(f"✅ Simulated {event_type} event created")
                    st.rerun()
                finally:
                    session.close()

            except json.JSONDecodeError:
                st.error("Invalid JSON in payload")
            except Exception as e:
                st.error(f"Failed to create event: {e}")

    st.divider()

    # Webhook signature verification test
    st.markdown("**Test Signature Verification**")

    settings = get_settings()
    secret = settings.plaud_webhook_secret

    if not secret:
        st.warning("PLAUD_WEBHOOK_SECRET not set. Signature verification is disabled.")
    else:
        st.success("Webhook secret is configured ✓")

        with st.expander("Test signature verification"):
            test_payload = st.text_area(
                "Test payload",
                value='{"event": "test", "timestamp": 1234567890}',
            )
            test_signature = st.text_input("Signature header value")

            if st.button("Verify Signature"):
                handler = get_webhook_handler()
                is_valid = handler.verify_signature(
                    test_payload.encode(), test_signature
                )
                if is_valid:
                    st.success("✅ Signature is valid")
                else:
                    st.error("❌ Signature is invalid")

                # Show expected signature
                expected = hmac.new(
                    secret.encode(),
                    test_payload.encode(),
                    hashlib.sha256,
                ).hexdigest()
                st.code(f"Expected: sha256={expected}")


def render_webhook_panel() -> None:
    """Main webhook panel component."""
    st.header("🔔 Webhook Management")
    st.caption(
        "Receive and process real-time events from Plaud. "
        "Events trigger automatic updates to your Chronos knowledge base."
    )

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📨 Events", "⚙️ Configuration", "🧪 Testing"])

    with tab1:
        render_webhook_events()

    with tab2:
        render_webhook_config()

    with tab3:
        render_webhook_test()


if __name__ == "__main__":
    st.set_page_config(page_title="Webhook Management", layout="wide")
    render_webhook_panel()
