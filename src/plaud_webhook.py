"""
Plaud Webhook Handler - Receive async notifications from Plaud API.

Based on Plaud API documentation:
- Receive real-time notifications for transcription completion
- Verify webhook signatures for security
- Handle various event types (file uploads, transcription complete, workflow complete)

Usage:
    # Flask example
    from src.plaud_webhook import PlaudWebhookHandler

    handler = PlaudWebhookHandler()

    @app.route('/webhook', methods=['POST'])
    def webhook():
        signature = request.headers.get('Plaud-Signature')
        if not handler.verify_signature(request.data, signature):
            return {'error': 'Invalid signature'}, 400

        event = handler.parse_event(request.get_json())
        handler.handle_event(event)
        return {'status': 'success'}, 200
"""

import os
import hmac
import hashlib
import logging
import uuid
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PlaudEventType(Enum):
    """Plaud webhook event types."""

    # File events
    FILE_UPLOADED = "file.uploaded"
    FILE_DELETED = "file.deleted"

    # Transcription events
    AUDIO_TRANSCRIBE_STARTED = "audio_transcribe.started"
    AUDIO_TRANSCRIBE_COMPLETED = "audio_transcribe.completed"
    AUDIO_TRANSCRIBE_FAILED = "audio_transcribe.failed"

    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_TASK_COMPLETED = "workflow.task_completed"

    # AI Summary events
    AI_SUMMARY_COMPLETED = "ai_summary.completed"
    AI_SUMMARY_FAILED = "ai_summary.failed"

    # ETL events
    AI_ETL_COMPLETED = "ai_etl.completed"
    AI_ETL_FAILED = "ai_etl.failed"

    # Device events
    DEVICE_CONNECTED = "device.connected"
    DEVICE_DISCONNECTED = "device.disconnected"
    RECORDING_UPLOADED = "recording.uploaded"

    # Unknown/custom
    UNKNOWN = "unknown"


@dataclass
class PlaudEvent:
    """Parsed Plaud webhook event."""

    event_type: PlaudEventType
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_id(self) -> Optional[str]:
        """Get file ID from event data."""
        return self.data.get("file_id") or self.data.get("id")

    @property
    def workflow_id(self) -> Optional[str]:
        """Get workflow ID from event data."""
        return self.data.get("workflow_id")

    @property
    def recording_id(self) -> Optional[str]:
        """Get recording ID from event data."""
        return self.data.get("recording_id") or self.file_id

    def __repr__(self) -> str:
        return f"PlaudEvent({self.event_type.value}, id={self.event_id[:8]}...)"


# Type alias for event handlers
EventHandler = Callable[[PlaudEvent], None]


class PlaudWebhookHandler:
    """
    Handler for Plaud webhook events.

    Provides:
    - Signature verification for security
    - Event parsing and type detection
    - Extensible event handling with callbacks
    """

    def __init__(self, webhook_secret: Optional[str] = None):
        """
        Initialize webhook handler.

        Args:
            webhook_secret: Plaud webhook signing secret (from env if not provided)
        """
        self.webhook_secret = webhook_secret or os.getenv("PLAUD_WEBHOOK_SECRET")
        self._handlers: Dict[PlaudEventType, List[EventHandler]] = {}

        if not self.webhook_secret:
            logger.warning(
                "PLAUD_WEBHOOK_SECRET not set - signature verification is disabled"
            )

    @staticmethod
    def _normalize_signature(signature_header: str) -> str:
        """Normalize Plaud signature header value.

        Supports both raw hex and prefixed formats like `sha256=<hex>`.
        """
        value = signature_header.strip()
        if "=" in value:
            prefix, maybe_sig = value.split("=", 1)
            if prefix.lower() in {"sha256", "hmac-sha256", "v1"} and maybe_sig:
                return maybe_sig.strip()
        return value

    def verify_signature(
        self, payload_body: bytes, signature_header: Optional[str]
    ) -> bool:
        """
        Verify that the payload was sent from Plaud.

        Args:
            payload_body: Raw request body bytes
            signature_header: Plaud-Signature header value

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.webhook_secret:
            logger.warning("Skipping signature verification - no secret configured")
            return True

        if not signature_header:
            logger.warning("Missing Plaud-Signature header")
            return False

        try:
            hash_object = hmac.new(
                self.webhook_secret.encode("utf-8"),
                msg=payload_body,
                digestmod=hashlib.sha256,
            )
            expected_signature = hash_object.hexdigest()
            provided_signature = self._normalize_signature(signature_header)

            if hmac.compare_digest(expected_signature, provided_signature):
                return True
            else:
                logger.warning("Signature mismatch")
                return False
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False

    def parse_event(self, payload: Dict[str, Any]) -> PlaudEvent:
        """
        Parse a webhook payload into a PlaudEvent.

        Args:
            payload: JSON payload from webhook

        Returns:
            Parsed PlaudEvent
        """
        event_type_str = (
            payload.get("event_type")
            or payload.get("type")
            or payload.get("event")
            or "unknown"
        )

        # Try to match event type
        try:
            event_type = PlaudEventType(event_type_str)
        except ValueError:
            logger.warning(f"Unknown event type: {event_type_str}")
            event_type = PlaudEventType.UNKNOWN

        # Parse timestamp
        timestamp_str = payload.get("timestamp") or payload.get("created_at")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        return PlaudEvent(
            event_type=event_type,
            event_id=(
                payload.get("event_id")
                or payload.get("id")
                or payload.get("request_id")
                or str(uuid.uuid4())
            ),
            timestamp=timestamp,
            data=payload.get("data") or payload.get("payload") or {},
            raw_payload=payload,
        )

    def register_handler(
        self, event_type: PlaudEventType, handler: EventHandler
    ) -> None:
        """
        Register a handler for a specific event type.

        Args:
            event_type: Event type to handle
            handler: Callback function to invoke
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.value}")

    def on_transcribe_complete(self, handler: EventHandler) -> EventHandler:
        """Decorator to register a transcription complete handler."""
        self.register_handler(PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED, handler)
        return handler

    def on_workflow_complete(self, handler: EventHandler) -> EventHandler:
        """Decorator to register a workflow complete handler."""
        self.register_handler(PlaudEventType.WORKFLOW_COMPLETED, handler)
        return handler

    def on_recording_uploaded(self, handler: EventHandler) -> EventHandler:
        """Decorator to register a recording upload handler."""
        self.register_handler(PlaudEventType.RECORDING_UPLOADED, handler)
        return handler

    def handle_event(self, event: PlaudEvent) -> None:
        """
        Handle a parsed event by invoking registered handlers.

        Args:
            event: Parsed PlaudEvent
        """
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            logger.info(f"No handlers for event type: {event.event_type.value}")
            return

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.event_type.value}: {e}")

    def process_webhook(
        self,
        payload_body: bytes,
        signature_header: Optional[str],
        payload_json: Dict[str, Any],
    ) -> bool:
        """
        Full webhook processing pipeline.

        Args:
            payload_body: Raw request body for signature verification
            signature_header: Plaud-Signature header
            payload_json: Parsed JSON payload

        Returns:
            True if successfully processed, False if verification failed
        """
        # Verify signature
        if not self.verify_signature(payload_body, signature_header):
            return False

        # Parse and handle event
        event = self.parse_event(payload_json)
        logger.info(f"Processing webhook event: {event}")

        self.handle_event(event)
        return True


# Default handler instance
_webhook_handler: Optional[PlaudWebhookHandler] = None


def get_webhook_handler() -> PlaudWebhookHandler:
    """Get the singleton webhook handler instance."""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = PlaudWebhookHandler()
    return _webhook_handler


def create_flask_webhook_endpoint(app, path: str = "/webhook/plaud"):
    """
    Create a Flask webhook endpoint for Plaud events.

    Args:
        app: Flask application instance
        path: URL path for the webhook endpoint

    Returns:
        The webhook handler instance
    """
    from flask import request, jsonify

    handler = get_webhook_handler()

    @app.route(path, methods=["POST"])
    def plaud_webhook():
        signature = request.headers.get("Plaud-Signature")

        if not handler.verify_signature(request.data, signature):
            return jsonify({"error": "Invalid signature"}), 400

        event = handler.parse_event(request.get_json())
        handler.handle_event(event)

        return jsonify({"status": "success"}), 200

    logger.info(f"Registered Plaud webhook endpoint at {path}")
    return handler


if __name__ == "__main__":
    # Test event parsing
    handler = PlaudWebhookHandler()

    test_payload = {
        "event_type": "audio_transcribe.completed",
        "event_id": "evt_123456",
        "timestamp": "2024-12-26T10:00:00Z",
        "data": {"file_id": "file_abc123", "transcript": "This is a test transcript."},
    }

    event = handler.parse_event(test_payload)
    print(f"Parsed event: {event}")
    print(f"File ID: {event.file_id}")
