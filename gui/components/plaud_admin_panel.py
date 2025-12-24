"""Streamlit sub-component: Plaud Admin Panel

Provides UI pages for device management and webhooks using the new PlaudAdminClient.
"""
import streamlit as st
from src.plaud_admin import PlaudAdminClient


def render_plaud_admin_panel():
    st.header("🔧 Plaud Admin")
    client = PlaudAdminClient()

    st.subheader("Devices")
    devices = client.list_devices()
    if not devices:
        st.info("No devices found or insufficient permissions to list devices.")
    else:
        for d in devices:
            st.markdown(f"**{d.get('name', d.get('id', 'device'))}** — {d.get('serial_number', '')}")

    st.subheader("Webhooks")
    webhooks = client.list_webhooks()
    if not webhooks:
        st.info("No webhooks configured.")
    else:
        for w in webhooks:
            st.markdown(f"- `{w.get('id')}` → {w.get('url')} (events: {w.get('events')})")

    with st.expander("Create webhook"):
        url = st.text_input("Callback URL", placeholder="https://example.com/plaud-webhook")
        events = st.text_input("Events (comma-separated)", placeholder="recording.created, recording.updated")
        if st.button("Create webhook"):
            evs = [e.strip() for e in events.split(",") if e.strip()]
            try:
                res = client.create_webhook(url, events=evs if evs else None)
                st.success(f"Created webhook: {res.get('id')}")
            except Exception as e:
                st.error(f"Failed to create webhook: {e}")
