"""
Enhanced Plaud Device Panel - Deep device integration with auto-sync.

This component provides:
- USB device detection (when Plaud is plugged in via USB)
- Real-time device monitoring with auto-refresh
- Auto-sync controls and status
- Device stats, battery, storage, and recording info
- One-click sync to Chronos pipeline

Usage:
    from gui.components.device_integration import render_device_integration
    render_device_integration()
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st

from src.plaud_device import PlaudDeviceManager, PlaudDevice, DeviceState, DeviceType
from src.plaud_usb_watcher import (
    PlaudUSBWatcher,
    USBPlaudDevice,
    get_usb_watcher,
)
from src.plaud_auto_sync import (
    PlaudAutoSync,
    SyncConfig,
    SyncJob,
    get_auto_sync,
)

logger = logging.getLogger(__name__)


def _battery_emoji(level: int) -> str:
    """Get battery emoji based on level."""
    if level >= 80:
        return "🔋"
    elif level >= 50:
        return "🔋"
    elif level >= 20:
        return "🪫"
    else:
        return "🪫"


def _device_type_emoji(device_type: DeviceType) -> str:
    """Get emoji for device type."""
    return {
        DeviceType.NOTE_PIN: "📌",
        DeviceType.NOTE: "📓",
        DeviceType.NOTE_PRO: "📔",
    }.get(device_type, "📱")


def _state_emoji(state: DeviceState) -> str:
    """Get emoji for device state."""
    return {
        DeviceState.IDLE: "💤",
        DeviceState.RECORDING: "🔴",
        DeviceState.SYNCING: "🔄",
        DeviceState.UPLOADING: "📤",
        DeviceState.PAUSED: "⏸️",
        DeviceState.UNKNOWN: "❓",
    }.get(state, "❓")


def _format_time_ago(dt: Optional[datetime]) -> str:
    """Format datetime as time ago."""
    if not dt:
        return "Never"
    delta = datetime.now(dt.tzinfo if dt.tzinfo else None) - dt
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = delta.seconds // 60
    if minutes > 0:
        return f"{minutes}m ago"
    return "Just now"


def render_usb_devices_section() -> None:
    """Render USB-connected devices section."""
    st.subheader("🔌 USB Connected Devices")
    st.caption("Plaud devices plugged in via USB are detected automatically.")

    # Initialize or get USB watcher
    if "usb_watcher" not in st.session_state:
        st.session_state.usb_watcher = get_usb_watcher()

    watcher = st.session_state.usb_watcher

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if watcher.is_running:
            if st.button("⏹️ Stop Watching", use_container_width=True):
                watcher.stop()
                st.rerun()
        else:
            if st.button("▶️ Start Watching", use_container_width=True):
                watcher.start()
                st.rerun()
    with col2:
        if st.button("🔍 Scan Now", use_container_width=True):
            watcher.scan_now()
            st.rerun()
    with col3:
        status_text = "🟢 Watching" if watcher.is_running else "⚪ Stopped"
        st.markdown(f"**Status:** {status_text}")

    # Display connected USB devices
    usb_devices = watcher.connected_devices

    if not usb_devices:
        st.info(
            "No Plaud devices detected via USB. "
            "Plug in your Plaud Note, NotePin, or Note Pro to see it here."
        )
        return

    st.success(f"✅ {len(usb_devices)} device(s) connected via USB")

    for path, device in usb_devices.items():
        with st.expander(
            f"💾 {device.volume_name} — {device.device_type.value} — "
            f"{device.audio_file_count} files ({device.total_audio_size_mb:.1f} MB)",
            expanded=True,
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Audio Files", device.audio_file_count)
            with c2:
                st.metric("Total Size", f"{device.total_audio_size_mb:.1f} MB")
            with c3:
                st.metric("Connected", _format_time_ago(device.connected_at))

            st.caption(f"**Path:** `{device.volume_path}`")
            st.caption(f"**Folders:** {', '.join(device.recording_folders) or 'None'}")

            # Sync button
            if device.has_recordings:
                if st.button(
                    "⬇️ Sync All to Chronos",
                    key=f"sync_usb_{path}",
                    use_container_width=True,
                ):
                    st.info("Syncing USB recordings to Chronos...")
                    # This would trigger local file ingestion
                    # For now, show the command that would run
                    st.code(
                        f"chronos_pipeline.py --ingest-local '{device.volume_path}'",
                        language="bash",
                    )

            # Show recent files
            if st.checkbox("Show recent files", key=f"files_{path}"):
                files = device.list_audio_files()[:10]
                for f in files:
                    st.text(f"  📄 {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


def render_auto_sync_section() -> None:
    """Render auto-sync configuration and status."""
    st.subheader("🔄 Auto-Sync")
    st.caption("Automatically sync recordings when devices connect or events occur.")

    # Initialize auto-sync
    if "auto_sync" not in st.session_state:
        st.session_state.auto_sync = get_auto_sync()

    sync = st.session_state.auto_sync

    # Status and controls
    col1, col2 = st.columns([1, 1])

    with col1:
        if sync.is_running:
            st.markdown("**Status:** 🟢 Running")
            if st.button("⏹️ Stop Auto-Sync", use_container_width=True):
                sync.stop()
                st.rerun()
        else:
            st.markdown("**Status:** ⚪ Stopped")
            if st.button("▶️ Start Auto-Sync", use_container_width=True):
                sync.start()
                st.rerun()

    with col2:
        status = sync.get_status()
        st.markdown(f"**Pending Jobs:** {status['pending_jobs']}")
        st.markdown(f"**Last Sync:** {status.get('last_sync', 'Never')}")

    # Configuration
    with st.expander("⚙️ Auto-Sync Settings", expanded=False):
        sync.config.sync_on_usb_connect = st.checkbox(
            "Sync when USB device connects",
            value=sync.config.sync_on_usb_connect,
            help="Automatically ingest recordings when a Plaud device is plugged in",
        )
        sync.config.sync_on_webhook = st.checkbox(
            "Sync on webhook events",
            value=sync.config.sync_on_webhook,
            help="Sync when Plaud sends a webhook (transcription complete, etc.)",
        )
        sync.config.process_after_ingest = st.checkbox(
            "Process after ingest",
            value=sync.config.process_after_ingest,
            help="Run Gemini processing after ingesting new recordings",
        )
        sync.config.index_after_process = st.checkbox(
            "Index after process",
            value=sync.config.index_after_process,
            help="Index to Qdrant after processing",
        )
        sync.config.min_sync_interval_seconds = st.slider(
            "Minimum interval between syncs (seconds)",
            min_value=10,
            max_value=300,
            value=sync.config.min_sync_interval_seconds,
        )

    # Manual trigger
    if st.button("🚀 Trigger Manual Sync", use_container_width=True):
        job = sync.trigger_manual_sync(full=True)
        st.info(f"Manual sync queued: {job.trigger.value}")

    # Sync history
    if sync.sync_history:
        st.markdown("**Recent Syncs:**")
        for job in reversed(sync.sync_history[-5:]):
            status_icon = "✅" if job.status == "completed" else "❌"
            st.caption(
                f"{status_icon} {job.trigger.value} — {job.status} — "
                f"{job.timestamp.strftime('%H:%M:%S')}"
            )


def render_api_devices_section() -> None:
    """Render devices from Plaud API."""
    st.subheader("📡 API Connected Devices")
    st.caption("Devices registered with your Plaud account via WiFi/Bluetooth.")

    # Check for Plaud credentials
    from dotenv import load_dotenv

    load_dotenv()
    client_id = os.getenv("PLAUD_CLIENT_ID")
    client_secret = os.getenv("PLAUD_CLIENT_SECRET")

    if not client_id or not client_secret:
        st.warning(
            "Plaud OAuth not configured. "
            "Run `python plaud_setup.py` to authenticate."
        )
        return

    try:
        manager = PlaudDeviceManager()
    except Exception as e:
        st.error(f"Failed to initialize device manager: {e}")
        return

    # Refresh button
    if st.button("🔄 Refresh from API", use_container_width=True):
        st.rerun()

    # Get devices
    with st.spinner("Fetching devices from Plaud API..."):
        devices = manager.list_devices()

    if not devices:
        st.info(
            "No devices found via API. Make sure your devices are "
            "connected to WiFi and synced with your Plaud account."
        )
        return

    # Summary metrics
    summary = manager.get_devices_summary()
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Devices", summary["total_devices"])
    with cols[1]:
        avg_battery = summary["battery"]["average_level"]
        st.metric("Avg Battery", f"{avg_battery:.0f}%")
    with cols[2]:
        used_mb = summary["storage"]["used_mb"]
        total_mb = summary["storage"]["total_mb"]
        if total_mb > 0:
            pct = (used_mb / total_mb) * 100
            st.metric("Storage Used", f"{pct:.0f}%")
        else:
            st.metric("Storage Used", "N/A")
    with cols[3]:
        recording = summary["state"]["recording"]
        st.metric("Recording Now", recording)

    st.markdown("---")

    # Device cards
    for device in devices:
        emoji = _device_type_emoji(device.device_type)
        bat_emoji = _battery_emoji(device.battery_level)
        state_emoji = _state_emoji(device.state)

        with st.expander(
            f"{emoji} {device.name} — {bat_emoji} {device.battery_level}% — "
            f"{state_emoji} {device.state.value}",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Battery",
                    f"{device.battery_level}%",
                    delta="⚡ Charging" if device.is_charging else None,
                )

            with c2:
                if device.storage_total_mb > 0:
                    used_pct = device.storage_percent_used
                    st.metric(
                        "Storage",
                        f"{device.storage_used_mb:.0f}/{device.storage_total_mb:.0f} MB",
                    )
                    st.progress(min(used_pct / 100, 1.0))
                else:
                    st.metric("Storage", "N/A")

            with c3:
                wifi_icon = "📶" if device.wifi_connected else "📴"
                st.metric(
                    "WiFi",
                    f"{wifi_icon} {'Connected' if device.wifi_connected else 'Offline'}",
                )

            with c4:
                st.metric("Last Sync", _format_time_ago(device.last_sync))

            # Device details
            st.caption(f"**Serial:** `{device.serial_number}`")
            st.caption(f"**Firmware:** {device.firmware_version}")

            # Recordings from this device
            if st.checkbox(f"Show recordings", key=f"api_rec_{device.id}"):
                with st.spinner("Loading recordings..."):
                    recordings = manager.get_device_recordings(device.id, limit=10)

                if recordings:
                    for rec in recordings:
                        st.text(
                            f"  📄 {rec.filename} — "
                            f"{rec.duration_minutes:.1f} min — "
                            f"{rec.file_size_mb:.1f} MB"
                        )
                else:
                    st.info("No recordings found on device.")


def render_device_integration() -> None:
    """Main device integration panel with all sections."""
    st.header("📱 Device Integration")
    st.caption(
        "Deep Plaud device integration with USB detection, auto-sync, and real-time monitoring."
    )

    # Tabs for different views
    tab_usb, tab_api, tab_sync, tab_webhooks = st.tabs(
        ["🔌 USB Devices", "📡 API Devices", "🔄 Auto-Sync", "🔔 Webhook Status"]
    )

    with tab_usb:
        render_usb_devices_section()

    with tab_api:
        render_api_devices_section()

    with tab_sync:
        render_auto_sync_section()

    with tab_webhooks:
        render_webhook_status_section()


def render_webhook_status_section() -> None:
    """Render webhook server status and recent events."""
    st.subheader("🔔 Webhook Server")
    st.caption("Local webhook receiver for Plaud event notifications.")

    # Try to import webhook server
    try:
        from src.plaud_webhook_server import get_webhook_server
    except ImportError as e:
        st.warning(f"Webhook server not available: {e}")
        st.info("Install Flask with: `pip install flask`")
        return

    # Initialize webhook server (but don't start yet)
    if "webhook_server" not in st.session_state:
        st.session_state.webhook_server = get_webhook_server()

    server = st.session_state.webhook_server

    # Status and controls
    col1, col2 = st.columns([1, 2])

    with col1:
        if server.is_running:
            st.markdown("**Status:** 🟢 Running")
            if st.button("⏹️ Stop Server"):
                server.stop()
                st.rerun()
        else:
            st.markdown("**Status:** ⚪ Stopped")
            if st.button("▶️ Start Server"):
                server.start()
                st.info(f"Webhook server started on port {server.port}")
                st.rerun()

    with col2:
        if server.is_running:
            st.code(server.webhook_url, language=None)
            st.caption(
                "Configure this URL in your Plaud developer portal. "
                "For external access, use ngrok."
            )

    # Recent events
    if server.event_log:
        st.markdown("**Recent Events:**")
        for entry in reversed(server.event_log[-10:]):
            icon = "✅" if entry.processed else "⏳"
            st.caption(
                f"{icon} {entry.event.event_type.value} — "
                f"{entry.received_at.strftime('%H:%M:%S')}"
            )
    else:
        st.info("No webhook events received yet.")


if __name__ == "__main__":
    # Test run as standalone
    st.set_page_config(page_title="Device Integration Test", layout="wide")
    render_device_integration()
