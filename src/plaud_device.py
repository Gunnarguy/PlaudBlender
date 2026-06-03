"""
Plaud Device Manager - Device operations and management.

Based on Plaud API documentation:
- List bound devices (NotePin, Note, NotePro)
- Get device status (battery, storage, recording state)
- Manage device settings
- Handle device recordings

Usage:
    device_manager = PlaudDeviceManager()

    # List all devices
    devices = device_manager.list_devices()

    # Get device status
    status = device_manager.get_device_status(device_id)

    # Get device recordings
    recordings = device_manager.get_device_recordings(device_id)
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


import requests
from .plaud_oauth import PlaudOAuthClient
from .plaud_api_token import PlaudAPITokenClient
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Plaud API base URL
PLAUD_API_BASE = "https://platform.plaud.ai/developer/api/open/third-party"


_LOGGED_DEVICES_404 = False
_LOGGED_DEVICE_FILES_404 = False


class DeviceType(Enum):
    """Plaud device types."""

    NOTE_PIN = "notepin"  # Lightweight wearable (30g)
    NOTE = "note"  # AI recorder for daily notes
    NOTE_PRO = "notepro"  # Professional-grade
    UNKNOWN = "unknown"


class DeviceState(Enum):
    """Device recording state."""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    UPLOADING = "uploading"
    SYNCING = "syncing"
    UNKNOWN = "unknown"


@dataclass
class PlaudDevice:
    """Representation of a Plaud device."""

    id: str
    name: str
    device_type: DeviceType
    serial_number: str
    firmware_version: str
    battery_level: int  # 0-100
    is_charging: bool
    storage_total_mb: int
    storage_free_mb: int
    state: DeviceState
    last_sync: Optional[datetime] = None
    wifi_connected: bool = False
    owner_id: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def storage_used_mb(self) -> int:
        """Calculate used storage."""
        return self.storage_total_mb - self.storage_free_mb

    @property
    def storage_percent_used(self) -> float:
        """Calculate storage usage percentage."""
        if self.storage_total_mb == 0:
            return 0.0
        return (self.storage_used_mb / self.storage_total_mb) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type.value,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
            "battery_level": self.battery_level,
            "is_charging": self.is_charging,
            "storage_total_mb": self.storage_total_mb,
            "storage_free_mb": self.storage_free_mb,
            "storage_percent_used": self.storage_percent_used,
            "state": self.state.value,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "wifi_connected": self.wifi_connected,
        }


@dataclass
class DeviceRecording:
    """Recording on a device."""

    id: str
    device_id: str
    filename: str
    duration_ms: int
    file_size_bytes: int
    created_at: datetime
    uploaded: bool = False
    synced: bool = False

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


class PlaudDeviceManager:
    """
    Manager for Plaud devices.

    Provides:
    - Device discovery and listing
    - Status monitoring (battery, storage, state)
    - Recording management
    - Settings configuration
    """

    def __init__(
        self,
        oauth_client: Optional[PlaudOAuthClient] = None,
        api_token_client: Optional[PlaudAPITokenClient] = None,
    ):
        """
        Initialize device manager.

        Args:
            oauth_client: PlaudOAuthClient instance (auto-created if not provided)
            api_token_client: PlaudAPITokenClient instance (auto-created if not provided)
        """
        self.oauth = oauth_client or PlaudOAuthClient()
        self.api_token_client = api_token_client or PlaudAPITokenClient()

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests."""
        return {
            "Authorization": f"Bearer {self.oauth.get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request."""
        url = f"{PLAUD_API_BASE}{endpoint}"
        headers = self._get_headers()

        response = requests.request(method, url, headers=headers, **kwargs)

        # Handle token refresh
        if response.status_code in (401, 422):
            logger.info("Token rejected, refreshing...")
            try:
                self.oauth.refresh_access_token()
            except Exception as exc:
                logger.error(f"Refresh failed: {exc}")
                raise
            headers = self._get_headers()
            response = requests.request(method, url, headers=headers, **kwargs)

        response.raise_for_status()
        return response.json()

    def _parse_device_type(self, type_str: str) -> DeviceType:
        """Parse device type string."""
        type_lower = (type_str or "").lower()
        if "notepin" in type_lower or "pin" in type_lower:
            return DeviceType.NOTE_PIN
        elif "notepro" in type_lower or "pro" in type_lower:
            return DeviceType.NOTE_PRO
        elif "note" in type_lower:
            return DeviceType.NOTE
        return DeviceType.UNKNOWN

    def _parse_device_state(self, state_str: str) -> DeviceState:
        """Parse device state string."""
        state_lower = (state_str or "").lower()
        if "recording" in state_lower:
            return DeviceState.RECORDING
        elif "paused" in state_lower:
            return DeviceState.PAUSED
        elif "upload" in state_lower:
            return DeviceState.UPLOADING
        elif "sync" in state_lower:
            return DeviceState.SYNCING
        elif "idle" in state_lower:
            return DeviceState.IDLE
        return DeviceState.UNKNOWN

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse datetime string."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def list_devices(self) -> List[PlaudDevice]:
        """
        List all bound devices using API token if available, else fallback to OAuth/dev endpoint.

        Returns:
            List of PlaudDevice objects
        """
        # Try API-token method first
        try:
            api_token = self.api_token_client.get_api_token()
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            url = "https://api.plaud.ai/devices/"
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Devices may be under 'devices' or root list
            devices_data: list[dict[str, Any]] = data.get("devices", []) if isinstance(data, dict) else data  # type: ignore[assignment]
            if devices_data is None and isinstance(data, list):
                devices_data = data
            if not devices_data:
                devices_data = []
            if not isinstance(devices_data, list):
                devices_data = []
            devices = []
            for d in devices_data:
                device = PlaudDevice(
                    id=d.get("id", ""),
                    name=d.get("name") or d.get("device_name", "Unknown Device"),
                    device_type=self._parse_device_type(
                        d.get("type") or d.get("device_type", "")
                    ),
                    serial_number=d.get("serial_number") or d.get("sn", ""),
                    firmware_version=d.get("firmware_version")
                    or d.get("fw_version", ""),
                    battery_level=d.get("battery_level") or d.get("power", 0),
                    is_charging=d.get("is_charging", False),
                    storage_total_mb=d.get("storage_total")
                    or d.get("total_storage", 0),
                    storage_free_mb=d.get("storage_free") or d.get("free_storage", 0),
                    state=self._parse_device_state(
                        d.get("state") or d.get("status", "")
                    ),
                    last_sync=self._parse_datetime(
                        d.get("last_sync") or d.get("last_upload")
                    ),
                    wifi_connected=d.get("wifi_connected", False),
                    owner_id=d.get("owner_id") or d.get("user_id"),
                    raw_data=d,
                )
                devices.append(device)
            logger.info(f"[API-token] Found {len(devices)} device(s)")
            return devices
        except Exception as api_exc:
            logger.warning(f"API-token device listing failed: {api_exc}")
            # Fallback to old OAuth/dev endpoint
            try:
                response = self._request("GET", "/devices")
                devices_data_any = (
                    response
                    if isinstance(response, list)
                    else response.get("devices", [])
                )
                devices_data: List[Dict[str, Any]] = (
                    devices_data_any if isinstance(devices_data_any, list) else []
                )
                devices = []
                for item in devices_data:
                    if not isinstance(item, dict):
                        continue
                    d = item
                    device = PlaudDevice(
                        id=d.get("id", ""),
                        name=d.get("name") or d.get("device_name", "Unknown Device"),
                        device_type=self._parse_device_type(
                            d.get("type") or d.get("device_type", "")
                        ),
                        serial_number=d.get("serial_number") or d.get("sn", ""),
                        firmware_version=d.get("firmware_version")
                        or d.get("fw_version", ""),
                        battery_level=d.get("battery_level") or d.get("power", 0),
                        is_charging=d.get("is_charging", False),
                        storage_total_mb=d.get("storage_total")
                        or d.get("total_storage", 0),
                        storage_free_mb=d.get("storage_free")
                        or d.get("free_storage", 0),
                        state=self._parse_device_state(
                            d.get("state") or d.get("status", "")
                        ),
                        last_sync=self._parse_datetime(
                            d.get("last_sync") or d.get("last_upload")
                        ),
                        wifi_connected=d.get("wifi_connected", False),
                        owner_id=d.get("owner_id") or d.get("user_id"),
                        raw_data=d,
                    )
                    devices.append(device)
                logger.info(f"[OAuth fallback] Found {len(devices)} device(s)")
                return devices
            except Exception as e:
                global _LOGGED_DEVICES_404
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 404 and not _LOGGED_DEVICES_404:
                    _LOGGED_DEVICES_404 = True
                    logger.info(
                        "Plaud devices endpoint is not available for this auth/API base (404). "
                        "Device management may require the App API-token API (api.plaud.ai)."
                    )
                elif status_code != 404:
                    logger.error(f"Error listing devices: {e}")
                return []

    def get_device(self, device_id: str) -> Optional[PlaudDevice]:
        """
        Get a specific device by ID.

        Args:
            device_id: Device ID

        Returns:
            PlaudDevice or None if not found
        """
        try:
            response = self._request("GET", f"/devices/{device_id}")
            if not isinstance(response, dict):
                return None
            d = response

            return PlaudDevice(
                id=d.get("id", device_id),
                name=d.get("name") or d.get("device_name", "Unknown Device"),
                device_type=self._parse_device_type(
                    d.get("type") or d.get("device_type", "")
                ),
                serial_number=d.get("serial_number") or d.get("sn", ""),
                firmware_version=d.get("firmware_version") or d.get("fw_version", ""),
                battery_level=d.get("battery_level") or d.get("power", 0),
                is_charging=d.get("is_charging", False),
                storage_total_mb=d.get("storage_total") or d.get("total_storage", 0),
                storage_free_mb=d.get("storage_free") or d.get("free_storage", 0),
                state=self._parse_device_state(d.get("state") or d.get("status", "")),
                last_sync=self._parse_datetime(
                    d.get("last_sync") or d.get("last_upload")
                ),
                wifi_connected=d.get("wifi_connected", False),
                owner_id=d.get("owner_id") or d.get("user_id"),
                raw_data=d,
            )
        except Exception as e:
            global _LOGGED_DEVICES_404
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 404 and not _LOGGED_DEVICES_404:
                _LOGGED_DEVICES_404 = True
                logger.info(
                    "Plaud devices endpoint is not available for this auth/API base (404). "
                    "Device management may require the App API-token API (api.plaud.ai)."
                )
            elif status_code != 404:
                logger.error(f"Error getting device {device_id}: {e}")
            return None

    def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """
        Get detailed status for a device.

        Args:
            device_id: Device ID

        Returns:
            Status dictionary
        """
        device = self.get_device(device_id)
        if not device:
            return {"error": "Device not found"}

        return {
            "id": device.id,
            "name": device.name,
            "type": device.device_type.value,
            "state": device.state.value,
            "battery": {
                "level": device.battery_level,
                "is_charging": device.is_charging,
            },
            "storage": {
                "total_mb": device.storage_total_mb,
                "free_mb": device.storage_free_mb,
                "used_mb": device.storage_used_mb,
                "percent_used": round(device.storage_percent_used, 1),
            },
            "connectivity": {
                "wifi_connected": device.wifi_connected,
                "last_sync": device.last_sync.isoformat() if device.last_sync else None,
            },
            "firmware": device.firmware_version,
        }

    def get_device_recordings(
        self, device_id: str, limit: int = 50, offset: int = 0
    ) -> List[DeviceRecording]:
        """
        Get recordings from a specific device.

        Args:
            device_id: Device ID
            limit: Maximum recordings to return
            offset: Pagination offset

        Returns:
            List of DeviceRecording objects
        """
        try:
            params = {"limit": limit, "offset": offset}
            response = self._request(
                "GET", f"/devices/{device_id}/files", params=params
            )

            files_data_any = (
                response if isinstance(response, list) else response.get("files", [])
            )
            files_data: List[Dict[str, Any]] = (
                files_data_any if isinstance(files_data_any, list) else []
            )

            recordings = []
            for item in files_data:
                if not isinstance(item, dict):
                    continue
                f = item
                recording = DeviceRecording(
                    id=f.get("id", ""),
                    device_id=device_id,
                    filename=f.get("filename") or f.get("name", ""),
                    duration_ms=f.get("duration") or f.get("duration_ms", 0),
                    file_size_bytes=f.get("file_size") or f.get("size", 0),
                    created_at=self._parse_datetime(
                        f.get("created_at") or f.get("start_at")
                    )
                    or datetime.utcnow(),
                    uploaded=f.get("uploaded", False),
                    synced=f.get("synced", False),
                )
                recordings.append(recording)

            logger.info(f"Found {len(recordings)} recording(s) on device {device_id}")
            return recordings

        except Exception as e:
            global _LOGGED_DEVICE_FILES_404
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 404 and not _LOGGED_DEVICE_FILES_404:
                _LOGGED_DEVICE_FILES_404 = True
                logger.info(
                    "Plaud device files endpoint is not available for this auth/API base (404). "
                    "Fetching recordings directly from devices may require a different Plaud API surface."
                )
            elif status_code != 404:
                logger.error(f"Error getting device recordings: {e}")
            return []

    def get_devices_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all devices.

        Returns:
            Summary with device counts, storage, battery info
        """
        devices = self.list_devices()

        if not devices:
            return {"total_devices": 0, "devices": []}

        total_storage = sum(d.storage_total_mb for d in devices)
        total_free = sum(d.storage_free_mb for d in devices)
        avg_battery = sum(d.battery_level for d in devices) / len(devices)

        recording_count = sum(1 for d in devices if d.state == DeviceState.RECORDING)

        return {
            "total_devices": len(devices),
            "by_type": {
                DeviceType.NOTE_PIN.value: sum(
                    1 for d in devices if d.device_type == DeviceType.NOTE_PIN
                ),
                DeviceType.NOTE.value: sum(
                    1 for d in devices if d.device_type == DeviceType.NOTE
                ),
                DeviceType.NOTE_PRO.value: sum(
                    1 for d in devices if d.device_type == DeviceType.NOTE_PRO
                ),
            },
            "storage": {
                "total_mb": total_storage,
                "free_mb": total_free,
                "used_mb": total_storage - total_free,
            },
            "battery": {
                "average_level": round(avg_battery, 1),
                "charging_count": sum(1 for d in devices if d.is_charging),
                "low_battery_count": sum(1 for d in devices if d.battery_level < 20),
            },
            "state": {
                "recording": recording_count,
                "idle": sum(1 for d in devices if d.state == DeviceState.IDLE),
                "syncing": sum(1 for d in devices if d.state == DeviceState.SYNCING),
            },
            "devices": [d.to_dict() for d in devices],
        }


def get_device_manager() -> PlaudDeviceManager:
    """Get a PlaudDeviceManager instance."""
    return PlaudDeviceManager()


if __name__ == "__main__":
    # Quick test
    manager = get_device_manager()
    print("PlaudDeviceManager initialized successfully")

    # List devices
    devices = manager.list_devices()
    print(f"Found {len(devices)} device(s)")

    for device in devices:
        print(
            f"  - {device.name} ({device.device_type.value}): {device.battery_level}% battery"
        )
