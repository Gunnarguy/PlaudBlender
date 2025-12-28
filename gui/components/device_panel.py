"""Streamlit component: Device Management Panel

Full-featured device management UI:
- Real-time device status (battery, storage, state)
- Device recordings browser
- Sync recordings to Chronos pipeline
- Device metrics visualization
"""

from typing import Optional
from datetime import datetime, timedelta

import streamlit as st

from src.plaud_device import PlaudDeviceManager, DeviceState, DeviceType, PlaudDevice


def _battery_emoji(level: int) -> str:
    """Get battery emoji based on level."""
    if level >= 80:
        return "🔋"
    elif level >= 50:
        return "🔋"
    elif level >= 20:
        return "🪫"
    else:
        return "🪫⚠️"


def _state_emoji(state: DeviceState) -> str:
    """Get state emoji."""
    return {
        DeviceState.RECORDING: "🔴",
        DeviceState.PAUSED: "⏸️",
        DeviceState.UPLOADING: "📤",
        DeviceState.SYNCING: "🔄",
        DeviceState.IDLE: "💤",
        DeviceState.UNKNOWN: "❓",
    }.get(state, "❓")


def _device_type_emoji(device_type: DeviceType) -> str:
    """Get device type emoji."""
    return {
        DeviceType.NOTE_PIN: "📌",
        DeviceType.NOTE: "📝",
        DeviceType.NOTE_PRO: "🎙️",
        DeviceType.UNKNOWN: "📱",
    }.get(device_type, "📱")


def render_device_card(device: PlaudDevice) -> None:
    """Render a single device card with rich status info."""
    with st.container():
        # Header with device name and type
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"### {_device_type_emoji(device.device_type)} {device.name}")
            st.caption(
                f"Serial: `{device.serial_number}` · FW: `{device.firmware_version}`"
            )
        with col2:
            st.metric(
                "Battery",
                f"{device.battery_level}%",
                delta="Charging" if device.is_charging else None,
            )
        with col3:
            st.metric(
                "State",
                f"{_state_emoji(device.state)} {device.state.value.title()}",
            )

        # Storage bar
        storage_pct = device.storage_percent_used
        st.progress(
            min(storage_pct / 100, 1.0),
            text=f"Storage: {device.storage_used_mb:,} MB / {device.storage_total_mb:,} MB ({storage_pct:.1f}% used)",
        )

        # Connectivity info
        sync_info = (
            "Never"
            if not device.last_sync
            else device.last_sync.strftime("%Y-%m-%d %H:%M")
        )
        wifi_status = "✅ Connected" if device.wifi_connected else "❌ Disconnected"
        st.caption(f"WiFi: {wifi_status} · Last sync: {sync_info}")


def render_device_summary(manager: PlaudDeviceManager) -> None:
    """Render summary statistics for all devices."""
    with st.spinner("Loading device summary..."):
        summary = manager.get_devices_summary()

    if summary.get("total_devices", 0) == 0:
        st.info(
            "No devices found. Make sure your Plaud devices are bound to your account."
        )
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Devices", summary["total_devices"])
    with col2:
        st.metric(
            "Avg Battery",
            f"{summary['battery']['average_level']:.0f}%",
            delta=f"{summary['battery']['charging_count']} charging",
        )
    with col3:
        storage = summary["storage"]
        st.metric(
            "Storage Used",
            f"{storage['used_mb']:,} MB",
            delta=f"{storage['free_mb']:,} MB free",
            delta_color="inverse",
        )
    with col4:
        state = summary["state"]
        st.metric(
            "Status",
            f"{state['recording']} recording",
            delta=f"{state['syncing']} syncing",
        )

    # Device type breakdown
    st.divider()
    by_type = summary.get("by_type", {})
    type_cols = st.columns(3)
    with type_cols[0]:
        st.write(f"📌 NotePin: **{by_type.get('notepin', 0)}**")
    with type_cols[1]:
        st.write(f"📝 Note: **{by_type.get('note', 0)}**")
    with type_cols[2]:
        st.write(f"🎙️ NotePro: **{by_type.get('notepro', 0)}**")


def render_device_recordings(manager: PlaudDeviceManager, device: PlaudDevice) -> None:
    """Render recordings browser for a specific device."""
    st.subheader(f"📼 Recordings on {device.name}")

    # Pagination controls
    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.number_input(
            "Recordings per page",
            min_value=10,
            max_value=100,
            value=25,
            step=5,
            key=f"rec_limit_{device.id}",
        )

    with st.spinner("Loading recordings..."):
        recordings = manager.get_device_recordings(device.id, limit=int(limit))

    if not recordings:
        st.info("No recordings found on this device.")
        return

    st.caption(f"Found {len(recordings)} recording(s)")

    # Recording list
    for rec in recordings:
        with st.expander(
            f"📄 {rec.filename} · {rec.duration_minutes:.1f} min · {rec.file_size_mb:.1f} MB",
            expanded=False,
        ):
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.write(f"**ID:** `{rec.id}`")
                st.write(f"**Created:** {rec.created_at.strftime('%Y-%m-%d %H:%M')}")
            with rc2:
                st.write(f"**Duration:** {rec.duration_seconds:.1f}s")
                st.write(f"**Size:** {rec.file_size_bytes:,} bytes")
            with rc3:
                st.write(f"**Uploaded:** {'✅' if rec.uploaded else '❌'}")
                st.write(f"**Synced:** {'✅' if rec.synced else '❌'}")

            # Action buttons
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button(
                    "⬇️ Sync to Chronos",
                    key=f"sync_{rec.id}",
                    use_container_width=True,
                    help="Download this recording and add to Chronos pipeline",
                ):
                    st.session_state[f"sync_recording_{rec.id}"] = True
                    st.info(f"Queueing {rec.filename} for sync...")
                    # This would trigger the ingest pipeline
                    import subprocess
                    import sys

                    result = subprocess.run(
                        [
                            sys.executable,
                            "scripts/chronos_pipeline.py",
                            "--ingest",
                            "--recording-id",
                            rec.id,
                            "--limit",
                            "1",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        st.success(f"✅ {rec.filename} synced to Chronos")
                    else:
                        st.error(f"Sync failed: {result.stderr}")


def render_device_panel() -> None:
    """Main device panel component."""
    st.header("📱 Device Manager")
    st.caption(
        "Monitor and manage your Plaud devices. See battery levels, storage, "
        "and sync recordings directly to Chronos."
    )

    try:
        manager = PlaudDeviceManager()
    except Exception as e:
        st.error(f"Failed to initialize device manager: {e}")
        st.info(
            "Make sure your Plaud OAuth is configured (run `python plaud_setup.py`)."
        )
        return

    # Tab layout for different views
    tab1, tab2 = st.tabs(["📊 Overview", "📋 Device Details"])

    with tab1:
        render_device_summary(manager)

        st.divider()
        st.subheader("All Devices")

        if st.button("🔄 Refresh Devices", use_container_width=False):
            st.rerun()

        with st.spinner("Loading devices..."):
            devices = manager.list_devices()

        if not devices:
            st.warning("No devices found.")
            return

        for device in devices:
            with st.expander(
                f"{_device_type_emoji(device.device_type)} {device.name} · "
                f"{_battery_emoji(device.battery_level)} {device.battery_level}% · "
                f"{_state_emoji(device.state)} {device.state.value}",
                expanded=False,
            ):
                render_device_card(device)

    with tab2:
        with st.spinner("Loading devices..."):
            devices = manager.list_devices()

        if not devices:
            st.warning("No devices found.")
            return

        device_options = {
            f"{_device_type_emoji(d.device_type)} {d.name} ({d.serial_number})": d
            for d in devices
        }

        selected_label = st.selectbox(
            "Select device",
            options=list(device_options.keys()),
            key="device_select",
        )

        if selected_label:
            selected_device = device_options[selected_label]

            render_device_card(selected_device)

            st.divider()

            # Device actions
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button(
                    "🔄 Refresh Status",
                    use_container_width=True,
                    key="refresh_device",
                ):
                    st.rerun()
            with action_cols[1]:
                if st.button(
                    "📊 Full Status JSON",
                    use_container_width=True,
                    key="show_status",
                ):
                    status = manager.get_device_status(selected_device.id)
                    st.json(status)
            with action_cols[2]:
                if st.button(
                    "📼 View Recordings",
                    use_container_width=True,
                    key="view_recordings",
                ):
                    st.session_state.show_device_recordings = selected_device.id

            # Show recordings if requested
            if st.session_state.get("show_device_recordings") == selected_device.id:
                st.divider()
                render_device_recordings(manager, selected_device)


if __name__ == "__main__":
    st.set_page_config(page_title="Device Manager", layout="wide")
    render_device_panel()
