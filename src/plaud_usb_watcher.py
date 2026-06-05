"""
Plaud USB Device Watcher - Automatic device detection for macOS.

Monitors /Volumes/ for Plaud device connections and triggers callbacks
when devices are plugged in or removed.

Plaud devices typically mount as:
- PLAUD_NOTE, PLAUD_PIN, PLAUD_PRO (USB mass storage mode)
- Contains audio files in /RECORD/ or /VOICE/ folders

Usage:
    from src.plaud_usb_watcher import PlaudUSBWatcher, start_watcher

    watcher = PlaudUSBWatcher()
    watcher.on_device_connected(lambda path, info: print(f"Connected: {path}"))
    watcher.start()
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Common Plaud volume name patterns
PLAUD_VOLUME_PATTERNS = [
    "PLAUD",
    "PLAUD_NOTE",
    "PLAUD_PIN",
    "PLAUD_PRO",
    "NOTE_PIN",
    "NOTEPIN",
    "NOTE PRO",
]

# Folders that indicate a Plaud device (includes actual Plaud folder names)
PLAUD_SIGNATURE_FOLDERS = ["RECORD", "VOICE", "AUDIO", "REC", "NOTES", "CALLS"]

# Audio file extensions to look for (lowercase only - we compare with .lower())
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".asr"}


class PlaudDeviceType(Enum):
    """Type of Plaud device detected via USB."""

    NOTE_PIN = "NotePin"
    NOTE = "Note"
    NOTE_PRO = "NotePro"
    UNKNOWN = "Unknown"


@dataclass
class USBPlaudDevice:
    """Represents a Plaud device connected via USB."""

    volume_path: Path
    volume_name: str
    device_type: PlaudDeviceType
    connected_at: datetime = field(default_factory=datetime.now)

    # Cached stats
    audio_file_count: int = 0
    total_audio_size_mb: float = 0.0
    recording_folders: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Scan device after initialization."""
        self._scan_device()

    def _scan_device(self) -> None:
        """Scan the device for audio files and stats."""
        self.audio_file_count = 0
        self.total_audio_size_mb = 0.0
        self.recording_folders = []

        try:
            # Look for recording folders
            for folder_name in PLAUD_SIGNATURE_FOLDERS:
                folder_path = self.volume_path / folder_name
                if folder_path.exists() and folder_path.is_dir():
                    self.recording_folders.append(folder_name)
                    # Count and size audio files
                    for audio_file in folder_path.rglob("*"):
                        if (
                            audio_file.is_file()
                            and audio_file.suffix.lower() in AUDIO_EXTENSIONS
                        ):
                            self.audio_file_count += 1
                            self.total_audio_size_mb += (
                                audio_file.stat().st_size / 1024 / 1024
                            )
        except Exception as e:
            logger.error(f"Error scanning device {self.volume_name}: {e}")

    def refresh(self) -> None:
        """Re-scan the device for updated stats."""
        self._scan_device()

    @property
    def has_recordings(self) -> bool:
        """Check if device has audio recordings."""
        return self.audio_file_count > 0

    def list_audio_files(self) -> List[Path]:
        """Get all audio files on the device."""
        files = []
        for folder_name in self.recording_folders:
            folder_path = self.volume_path / folder_name
            if folder_path.exists():
                for audio_file in folder_path.rglob("*"):
                    if (
                        audio_file.is_file()
                        and audio_file.suffix.lower() in AUDIO_EXTENSIONS
                    ):
                        files.append(audio_file)
        return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "volume_path": str(self.volume_path),
            "volume_name": self.volume_name,
            "device_type": self.device_type.value,
            "connected_at": self.connected_at.isoformat(),
            "audio_file_count": self.audio_file_count,
            "total_audio_size_mb": round(self.total_audio_size_mb, 2),
            "recording_folders": self.recording_folders,
            "has_recordings": self.has_recordings,
        }


# Type aliases for callbacks
DeviceCallback = Callable[[USBPlaudDevice], None]
DisconnectCallback = Callable[[str], None]  # volume_path


class PlaudUSBWatcher:
    """
    Watches for Plaud USB device connections on macOS.

    Uses polling of /Volumes/ directory to detect new mounted devices.
    For more advanced detection, could use fsevents or IOKit.
    """

    def __init__(
        self,
        volumes_path: str = "",
        poll_interval: float = 2.0,
    ):
        """
        Initialize the USB watcher.

        Args:
            volumes_path: Path to monitor for mounted volumes (auto-detected if empty)
            poll_interval: How often to check for new devices (seconds)
        """
        if not volumes_path:
            import sys as _sys
            if _sys.platform == "darwin":
                volumes_path = "/Volumes"
            else:
                import getpass
                volumes_path = f"/media/{getpass.getuser()}"
        self.volumes_path = Path(volumes_path)
        self.poll_interval = poll_interval

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._known_volumes: Set[str] = set()
        self._connected_devices: Dict[str, USBPlaudDevice] = {}

        # Callbacks
        self._on_connect_callbacks: List[DeviceCallback] = []
        self._on_disconnect_callbacks: List[DisconnectCallback] = []

        # Initialize known volumes lazily in _run_loop to avoid blocking main thread
        pass

    def _update_known_volumes(self) -> None:
        """Get current list of mounted volumes."""
        try:
            if self.volumes_path.exists():
                self._known_volumes = {
                    v.name
                    for v in self.volumes_path.iterdir()
                    if v.is_dir() and not v.name.startswith(".")
                }
        except Exception as e:
            logger.error(f"Error reading volumes: {e}")
            self._known_volumes = set()

    def _is_plaud_device(self, volume_path: Path) -> bool:
        """Check if a volume looks like a Plaud device."""
        volume_name = volume_path.name.upper()

        # Check by name pattern
        for pattern in PLAUD_VOLUME_PATTERNS:
            if pattern.upper() in volume_name:
                return True

        # Check for signature folders (RECORD, VOICE, etc.)
        for folder in PLAUD_SIGNATURE_FOLDERS:
            if (volume_path / folder).exists():
                return True

        return False

    def _detect_device_type(self, volume_path: Path) -> PlaudDeviceType:
        """Detect the type of Plaud device based on volume name/contents."""
        volume_name = volume_path.name.upper()

        if "PRO" in volume_name:
            return PlaudDeviceType.NOTE_PRO
        elif "PIN" in volume_name:
            return PlaudDeviceType.NOTE_PIN
        elif "NOTE" in volume_name:
            return PlaudDeviceType.NOTE

        # Try to detect from file structure
        # Note Pro typically has more storage
        # NotePin has minimal storage
        return PlaudDeviceType.UNKNOWN

    def _check_for_changes(self) -> None:
        """Check for new or removed volumes."""
        try:
            current_volumes = {
                v.name
                for v in self.volumes_path.iterdir()
                if v.is_dir() and not v.name.startswith(".")
            }
        except Exception as e:
            logger.error(f"Error checking volumes: {e}")
            return

        # Check for new volumes
        new_volumes = current_volumes - self._known_volumes
        for vol_name in new_volumes:
            vol_path = self.volumes_path / vol_name
            if self._is_plaud_device(vol_path):
                logger.info(f"🔌 Plaud device connected: {vol_name}")
                device = USBPlaudDevice(
                    volume_path=vol_path,
                    volume_name=vol_name,
                    device_type=self._detect_device_type(vol_path),
                )
                self._connected_devices[str(vol_path)] = device

                # Fire callbacks
                for callback in self._on_connect_callbacks:
                    try:
                        callback(device)
                    except Exception as e:
                        logger.error(f"Connect callback error: {e}")

        # Check for removed volumes
        removed_volumes = self._known_volumes - current_volumes
        for vol_name in removed_volumes:
            vol_path = str(self.volumes_path / vol_name)
            if vol_path in self._connected_devices:
                logger.info(f"🔌 Plaud device disconnected: {vol_name}")
                del self._connected_devices[vol_path]

                # Fire callbacks
                for callback in self._on_disconnect_callbacks:
                    try:
                        callback(vol_path)
                    except Exception as e:
                        logger.error(f"Disconnect callback error: {e}")

        self._known_volumes = current_volumes

    def _run_loop(self) -> None:
        """Main polling loop."""
        # Initialize known volumes on the background thread to avoid blocking Dash startup
        self._update_known_volumes()
        logger.info(f"USB watcher started, monitoring {self.volumes_path}")
        while self._running:
            self._check_for_changes()
            time.sleep(self.poll_interval)
        logger.info("USB watcher stopped")

    # Public API

    def on_device_connected(self, callback: DeviceCallback) -> None:
        """
        Register a callback for device connections.

        Args:
            callback: Function that receives USBPlaudDevice
        """
        self._on_connect_callbacks.append(callback)

    def on_device_disconnected(self, callback: DisconnectCallback) -> None:
        """
        Register a callback for device disconnections.

        Args:
            callback: Function that receives volume path string
        """
        self._on_disconnect_callbacks.append(callback)

    def start(self) -> None:
        """Start watching for USB devices in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the USB watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def scan_now(self) -> List[USBPlaudDevice]:
        """
        Immediately scan for connected Plaud devices.

        Returns:
            List of currently connected devices
        """
        # First, scan all current volumes for Plaud devices (not just new ones)
        self._scan_all_volumes()
        return list(self._connected_devices.values())

    def _scan_all_volumes(self) -> None:
        """Scan all mounted volumes for Plaud devices."""
        try:
            for vol in self.volumes_path.iterdir():
                if not vol.is_dir() or vol.name.startswith("."):
                    continue
                vol_path_str = str(vol)
                # Skip if already tracked
                if vol_path_str in self._connected_devices:
                    continue
                # Check if it's a Plaud device
                if self._is_plaud_device(vol):
                    logger.info(f"🔌 Found Plaud device: {vol.name}")
                    device = USBPlaudDevice(
                        volume_path=vol,
                        volume_name=vol.name,
                        device_type=self._detect_device_type(vol),
                    )
                    self._connected_devices[vol_path_str] = device
        except Exception as e:
            logger.error(f"Error scanning volumes: {e}")

    @property
    def connected_devices(self) -> Dict[str, USBPlaudDevice]:
        """Get all currently connected Plaud devices."""
        return self._connected_devices.copy()

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running


# Singleton instance
_usb_watcher: Optional[PlaudUSBWatcher] = None


def get_usb_watcher() -> PlaudUSBWatcher:
    """Get the singleton USB watcher instance."""
    global _usb_watcher
    if _usb_watcher is None:
        _usb_watcher = PlaudUSBWatcher()
    return _usb_watcher


def start_watcher() -> PlaudUSBWatcher:
    """Start the USB watcher if not already running."""
    watcher = get_usb_watcher()
    if not watcher.is_running:
        watcher.start()
    return watcher


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("🔍 Starting Plaud USB Device Watcher...")
    print("   Press Ctrl+C to stop\n")

    watcher = PlaudUSBWatcher()

    def on_connect(device: USBPlaudDevice):
        print(f"\n✅ Device Connected!")
        print(f"   Volume: {device.volume_name}")
        print(f"   Type: {device.device_type.value}")
        print(f"   Audio files: {device.audio_file_count}")
        print(f"   Total size: {device.total_audio_size_mb:.1f} MB")
        print(f"   Folders: {device.recording_folders}")

    def on_disconnect(path: str):
        print(f"\n❌ Device Disconnected: {path}")

    watcher.on_device_connected(on_connect)
    watcher.on_device_disconnected(on_disconnect)

    # Initial scan
    print("🔍 Scanning for connected devices...")
    devices = watcher.scan_now()
    if devices:
        print(f"   Found {len(devices)} Plaud device(s)")
        for d in devices:
            print(f"   - {d.volume_name}: {d.audio_file_count} files")
    else:
        print("   No Plaud devices currently connected")

    watcher.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping watcher...")
        watcher.stop()
        sys.exit(0)
