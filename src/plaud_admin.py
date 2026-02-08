"""
Administrative Plaud API helpers not covered by the main PlaudClient.

Provides device and webhook management helpers and other convenience methods
used by the Streamlit UI for deeper Plaud integration.
"""

from typing import Any, Dict, List, Optional

from .plaud_client import PlaudClient
from .utils.logger import get_logger

logger = get_logger(__name__)


_LOGGED_MISSING_DEVICES_ENDPOINT = False
_LOGGED_MISSING_WEBHOOKS_ENDPOINT = False


class PlaudAdminClient:
    """Convenience wrapper for administrative Plaud API operations.

    This module intentionally keeps methods high-level and tolerant of missing
    endpoints so the UI can present graceful messaging even if an endpoint
    doesn't exist or the currently authenticated app lacks permissions.
    """

    def __init__(self, plaud_client: Optional[PlaudClient] = None):
        self.plaud = plaud_client or PlaudClient()

    def list_devices(self) -> List[Dict[str, Any]]:
        """List devices associated with the current Plaud account.

        Falls back to an empty list if the API doesn't expose a devices endpoint.
        """
        try:
            result = self.plaud._request("GET", "/devices")
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                devices = result.get("devices")
                return devices if isinstance(devices, list) else []
            return []
        except Exception as e:
            global _LOGGED_MISSING_DEVICES_ENDPOINT
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 404 and not _LOGGED_MISSING_DEVICES_ENDPOINT:
                _LOGGED_MISSING_DEVICES_ENDPOINT = True
                logger.info(
                    "Plaud devices endpoint is not available for this auth/API base (404). "
                    "Device management may require the App API-token API (api.plaud.ai) instead of third-party OAuth."
                )
            elif status_code != 404:
                logger.warning("Could not list devices: %s", e)
            return []

    def get_device(self, device_id: str) -> Dict[str, Any]:
        try:
            return self.plaud._request("GET", f"/devices/{device_id}")
        except Exception as e:
            logger.warning("Could not get device %s: %s", device_id, e)
            return {}

    def list_webhooks(self) -> List[Dict[str, Any]]:
        try:
            result = self.plaud._request("GET", "/webhooks")
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                webhooks = result.get("webhooks")
                return webhooks if isinstance(webhooks, list) else []
            return []
        except Exception as e:
            global _LOGGED_MISSING_WEBHOOKS_ENDPOINT
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 404 and not _LOGGED_MISSING_WEBHOOKS_ENDPOINT:
                _LOGGED_MISSING_WEBHOOKS_ENDPOINT = True
                logger.info(
                    "Plaud webhooks endpoint is not available for this auth/API base (404). "
                    "Webhook management may require a different Plaud API surface than third-party OAuth."
                )
            elif status_code != 404:
                logger.warning("Could not list webhooks: %s", e)
            return []

    def create_webhook(
        self, url: str, events: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"url": url}
        if events:
            payload["events"] = events
        try:
            return self.plaud._request("POST", "/webhooks", json=payload)
        except Exception as e:
            logger.error("Failed to create webhook: %s", e)
            raise

    def delete_webhook(self, webhook_id: str) -> bool:
        try:
            self.plaud._request("DELETE", f"/webhooks/{webhook_id}")
            return True
        except Exception as e:
            logger.error("Failed to delete webhook %s: %s", webhook_id, e)
            return False

    def ping_webhook(self, webhook_id: str) -> bool:
        try:
            self.plaud._request("POST", f"/webhooks/{webhook_id}/ping")
            return True
        except Exception as e:
            logger.warning("Webhook ping failed: %s", e)
            return False
