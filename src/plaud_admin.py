"""
Administrative Plaud API helpers not covered by the main PlaudClient.

Provides device and webhook management helpers and other convenience methods
used by the Streamlit UI for deeper Plaud integration.
"""

from typing import Any, Dict, List, Optional

from .plaud_client import PlaudClient
from .utils.logger import get_logger

logger = get_logger(__name__)


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
            return self.plaud._request("GET", "/devices")
        except Exception as e:
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
            return self.plaud._request("GET", "/webhooks")
        except Exception as e:
            logger.warning("Could not list webhooks: %s", e)
            return []

    def create_webhook(
        self, url: str, events: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        payload = {"url": url}
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
