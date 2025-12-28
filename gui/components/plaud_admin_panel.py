"""Streamlit sub-component: Plaud Admin Panel

Comprehensive Plaud integration hub that provides:
- Device management (battery, storage, sync status)
- AI workflow orchestration (transcription + ETL + summary)
- Webhook management (incoming events, configuration, testing)
- Quick actions and system overview
"""

import streamlit as st

from src.plaud_admin import PlaudAdminClient
from src.plaud_client import PlaudClient
from src.config import get_settings

# Import the sub-panels
from .device_panel import render_device_panel
from .workflow_panel import render_workflow_panel
from .webhook_panel import render_webhook_panel


def _check_plaud_connection() -> tuple[bool, str]:
    """Check if Plaud OAuth is configured and working."""
    settings = get_settings()

    if not settings.plaud_client_id or not settings.plaud_client_secret:
        return False, "PLAUD_CLIENT_ID or PLAUD_CLIENT_SECRET not set in .env"

    try:
        client = PlaudClient()
        # Try to make a basic request
        user = client.get_user_info()
        return True, f"Connected as {user.get('email', user.get('id', 'Unknown'))}"
    except Exception as e:
        return False, str(e)


def render_plaud_overview() -> None:
    """Render the Plaud overview/dashboard tab."""
    st.subheader("🏠 Plaud Integration Overview")

    # Connection status
    connected, status_msg = _check_plaud_connection()

    if connected:
        st.success(f"✅ {status_msg}")
    else:
        st.error(f"❌ Connection failed: {status_msg}")
        st.info("Run `python plaud_setup.py` to configure OAuth.")
        return

    # Quick stats
    st.divider()
    st.markdown("### 📊 Quick Stats")

    col1, col2, col3, col4 = st.columns(4)

    # Get stats from various sources
    try:
        client = PlaudClient()
        admin = PlaudAdminClient(plaud_client=client)

        # Devices
        devices = admin.list_devices()
        with col1:
            st.metric("Devices", len(devices))

        # Webhooks
        webhooks = admin.list_webhooks()
        with col2:
            st.metric("Webhooks", len(webhooks))

        # Recordings from DB
        from src.database import SessionLocal, init_db
        from src.database.models import ChronosRecording

        init_db()
        session = SessionLocal()
        try:
            rec_count = session.query(ChronosRecording).count()
            with col3:
                st.metric("Recordings", rec_count)

            pending = (
                session.query(ChronosRecording)
                .filter_by(processing_status="pending")
                .count()
            )
            with col4:
                st.metric("Pending", pending)
        finally:
            session.close()

    except Exception as e:
        st.warning(f"Could not load all stats: {e}")

    # Quick actions
    st.divider()
    st.markdown("### ⚡ Quick Actions")

    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button(
            "🔄 Sync Recordings",
            use_container_width=True,
            help="Fetch latest recordings from Plaud",
        ):
            import subprocess
            import sys

            with st.spinner("Syncing..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/chronos_pipeline.py",
                        "--ingest",
                        "--limit",
                        "25",
                    ],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("Sync complete!")
            else:
                st.error(f"Sync failed: {result.stderr}")

    with action_cols[1]:
        if st.button(
            "⚡ Process Pending",
            use_container_width=True,
            help="Process pending recordings through Gemini",
        ):
            import subprocess
            import sys

            with st.spinner("Processing..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/chronos_pipeline.py",
                        "--process",
                        "--limit",
                        "10",
                    ],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("Processing complete!")
            else:
                st.error(f"Processing failed: {result.stderr}")

    with action_cols[2]:
        if st.button(
            "📊 Index to Qdrant",
            use_container_width=True,
            help="Index processed events to vector store",
        ):
            import subprocess
            import sys

            with st.spinner("Indexing..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/chronos_pipeline.py",
                        "--index",
                        "--limit",
                        "50",
                    ],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("Indexing complete!")
            else:
                st.error(f"Indexing failed: {result.stderr}")

    with action_cols[3]:
        if st.button(
            "🔗 Full Pipeline",
            use_container_width=True,
            help="Run complete ingest → process → index",
        ):
            import subprocess
            import sys

            with st.spinner("Running full pipeline..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/chronos_pipeline.py",
                        "--full",
                        "--limit",
                        "10",
                    ],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("Pipeline complete!")
                st.code(
                    result.stdout[-2000:]
                    if len(result.stdout) > 2000
                    else result.stdout
                )
            else:
                st.error(f"Pipeline failed: {result.stderr}")

    # Configuration summary
    st.divider()
    st.markdown("### 🔧 Configuration")

    settings = get_settings()
    config_cols = st.columns(2)

    with config_cols[0]:
        st.markdown("**Plaud API**")
        st.write(
            {
                "Client ID": "✅ Set" if settings.plaud_client_id else "❌ Missing",
                "Client Secret": (
                    "✅ Set" if settings.plaud_client_secret else "❌ Missing"
                ),
                "Webhook Secret": (
                    "✅ Set" if settings.plaud_webhook_secret else "⚠️ Optional"
                ),
                "Webhook URL": settings.plaud_webhook_url or "Not configured",
            }
        )

    with config_cols[1]:
        st.markdown("**Processing**")
        st.write(
            {
                "Default Language": settings.plaud_default_language,
                "Diarization": (
                    "Enabled" if settings.plaud_enable_diarization else "Disabled"
                ),
                "Workflow Timeout": f"{settings.plaud_workflow_timeout}s",
            }
        )


def render_plaud_admin_panel():
    """Main Plaud Admin panel with tabbed interface."""
    st.header("🎙️ Plaud Integration Hub")
    st.caption(
        "Complete Plaud API integration: devices, workflows, webhooks, and more. "
        "Everything you need to manage your voice recording pipeline."
    )

    # Main tabs for different Plaud features
    tab_overview, tab_devices, tab_workflows, tab_webhooks, tab_legacy = st.tabs(
        [
            "🏠 Overview",
            "📱 Devices",
            "⚡ Workflows",
            "🔔 Webhooks",
            "🔧 Legacy Admin",
        ]
    )

    with tab_overview:
        render_plaud_overview()

    with tab_devices:
        render_device_panel()

    with tab_workflows:
        render_workflow_panel()

    with tab_webhooks:
        render_webhook_panel()

    with tab_legacy:
        _render_legacy_admin()


def _render_legacy_admin():
    """Legacy admin panel (original simple version)."""
    st.subheader("🔧 Legacy Admin")
    st.caption("Basic device and webhook management (original implementation).")

    try:
        client = PlaudAdminClient()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return

    st.markdown("**Devices**")
    devices = client.list_devices()
    if not devices:
        st.info("No devices found or insufficient permissions.")
    else:
        for d in devices:
            st.markdown(
                f"**{d.get('name', d.get('id', 'device'))}** — {d.get('serial_number', '')}"
            )

    st.markdown("**Webhooks**")
    webhooks = client.list_webhooks()
    if not webhooks:
        st.info("No webhooks configured.")
    else:
        for w in webhooks:
            st.markdown(
                f"- `{w.get('id')}` → {w.get('url')} (events: {w.get('events')})"
            )

    with st.expander("Create webhook (legacy)"):
        url = st.text_input(
            "Callback URL",
            placeholder="https://example.com/plaud-webhook",
            key="legacy_webhook_url",
        )
        events = st.text_input(
            "Events (comma-separated)",
            placeholder="recording.created, recording.updated",
            key="legacy_webhook_events",
        )
        if st.button("Create webhook", key="legacy_create_webhook"):
            evs = [e.strip() for e in events.split(",") if e.strip()]
            try:
                res = client.create_webhook(url, events=evs if evs else None)
                st.success(f"Created webhook: {res.get('id')}")
            except Exception as e:
                st.error(f"Failed to create webhook: {e}")
