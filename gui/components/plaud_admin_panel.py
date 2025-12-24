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
            st.markdown(
                f"**{d.get('name', d.get('id', 'device'))}** — {d.get('serial_number', '')}"
            )

    st.subheader("Webhooks")
    webhooks = client.list_webhooks()
    if not webhooks:
        st.info("No webhooks configured.")
    else:
        for w in webhooks:
            st.markdown(
                f"- `{w.get('id')}` → {w.get('url')} (events: {w.get('events')})"
            )

    with st.expander("Create webhook"):
        url = st.text_input(
            "Callback URL", placeholder="https://example.com/plaud-webhook"
        )
        events = st.text_input(
            "Events (comma-separated)",
            placeholder="recording.created, recording.updated",
        )
        if st.button("Create webhook"):
            evs = [e.strip() for e in events.split(",") if e.strip()]
            try:
                res = client.create_webhook(url, events=evs if evs else None)
                st.success(f"Created webhook: {res.get('id')}")
            except Exception as e:
                st.error(f"Failed to create webhook: {e}")

    st.divider()
    st.subheader("Webhook events (received)")
    from src.database import SessionLocal, init_db
    from src.database.chronos_repository import list_chronos_webhook_events, mark_webhook_event_processed

    init_db()
    session = SessionLocal()
    try:
        events = list_chronos_webhook_events(session, limit=200)
        if not events:
            st.info("No webhook events recorded yet. Configure a webhook to forward events here.")
        else:
            for ev in events:
                ev_id = str(ev.event_id)
                with st.expander(f"{ev.event_type} · {ev.received_at}"):
                    st.markdown(f"**Webhook ID:** {ev.webhook_id}  \n**Event ID:** {ev_id}")
                    st.json({"payload": ev.payload, "headers": ev.headers})
                    cols = st.columns([1, 1, 2])
                    with cols[0]:
                        if st.button("Mark processed", key=f"mark_{ev_id}"):
                            mark_webhook_event_processed(session, ev_id, status="processed")
                            st.success("Marked processed")
                            st.experimental_rerun()
                    with cols[1]:
                        if st.button("Mark failed", key=f"fail_{ev_id}"):
                            mark_webhook_event_processed(session, ev_id, status="failed")
                            st.success("Marked failed")
                            st.experimental_rerun()
                    with cols[2]:
                        if st.button("Replay into ingest", key=f"replay_{ev_id}"):
                            # Lightweight replay: if event references a recording, enqueue ingest via CLI
                            rec_id = ev.recording_id or (ev.payload or {}).get("recording_id")
                            if not rec_id:
                                st.error("No recording_id found in event payload to replay.")
                            else:
                                rec_id = str(rec_id)
                                st.info(f"Triggering ingest for {rec_id}…")
                                import subprocess, sys

                                code = subprocess.run(
                                    [
                                        sys.executable,
                                        "scripts/chronos_pipeline.py",
                                        "--ingest",
                                        "--limit",
                                        "1",
                                        "--recording-id",
                                        rec_id,
                                    ],
                                    capture_output=True,
                                    text=True,
                                )
                                st.code((code.stdout or "") + (code.stderr or ""))
                                if code.returncode == 0:
                                    st.success("Ingest triggered")
                                else:
                                    st.error(f"Ingest CLI failed (exit {code.returncode})")
    finally:
        session.close()
