"""
Plaud Webhook Server - Local HTTP server for receiving Plaud webhooks.

Runs a lightweight Flask server to receive webhook events from Plaud API.
Can be started alongside the Streamlit app to enable real-time event handling.

Usage:
    # Start as standalone server
    python -m src.plaud_webhook_server

    # Or import and start in background
    from src.plaud_webhook_server import start_webhook_server
    server = start_webhook_server(port=8090)
"""

import os
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from queue import Queue

from flask import Flask, request, jsonify

from .plaud_webhook import (
    PlaudWebhookHandler,
    PlaudEvent,
    PlaudEventType,
    get_webhook_handler,
)
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class EventLogEntry:
    """A logged webhook event."""

    event: PlaudEvent
    received_at: datetime = field(default_factory=datetime.now)
    processed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event.event_type.value,
            "event_id": self.event.event_id,
            "timestamp": self.event.timestamp.isoformat(),
            "received_at": self.received_at.isoformat(),
            "processed": self.processed,
            "data": self.event.data,
        }


class PlaudWebhookServer:
    """
    Local webhook server for receiving Plaud events.

    Features:
    - Runs Flask in a background thread
    - Queues events for processing
    - Maintains event log for UI display
    - Integrates with PlaudWebhookHandler for signature verification
    """

    def __init__(self, port: int = 8090, host: str = "0.0.0.0"):
        """
        Initialize webhook server.

        Args:
            port: Port to listen on
            host: Host to bind to
        """
        self.port = port
        self.host = host

        # Flask app
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Quiet Flask logs

        # Event handling
        self.handler = get_webhook_handler()
        self.event_queue: Queue[PlaudEvent] = Queue()
        self.event_log: List[EventLogEntry] = []
        self.max_log_size = 100

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Custom callbacks
        self._event_callbacks: List[Callable[[PlaudEvent], None]] = []

        # Register routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up Flask routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify(
                {
                    "status": "healthy",
                    "server": "plaud-webhook-server",
                    "events_received": len(self.event_log),
                    "queue_size": self.event_queue.qsize(),
                }
            )

        @self.app.route("/webhook/plaud", methods=["POST"])
        def plaud_webhook():
            """Receive Plaud webhook events."""
            try:
                signature = request.headers.get("Plaud-Signature")
                payload_body = request.data
                payload_json = request.get_json() or {}

                # Verify signature
                if not self.handler.verify_signature(payload_body, signature):
                    logger.warning("Invalid webhook signature")
                    return jsonify({"error": "Invalid signature"}), 401

                # Parse event
                event = self.handler.parse_event(payload_json)
                logger.info(f"Received webhook event: {event.event_type.value}")

                # Log event
                entry = EventLogEntry(event=event)
                self.event_log.append(entry)

                # Trim log if too large
                if len(self.event_log) > self.max_log_size:
                    self.event_log = self.event_log[-self.max_log_size :]

                # Queue for processing
                self.event_queue.put(event)

                # Fire custom callbacks
                for callback in self._event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                # Let handler process registered handlers
                self.handler.handle_event(event)
                entry.processed = True

                return jsonify({"status": "success", "event_id": event.event_id}), 200

            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/events", methods=["GET"])
        def list_events():
            """List recent webhook events."""
            limit = request.args.get("limit", 20, type=int)
            events = [e.to_dict() for e in self.event_log[-limit:]]
            return jsonify({"events": events, "total": len(self.event_log)})

        @self.app.route("/events/clear", methods=["POST"])
        def clear_events():
            """Clear event log."""
            self.event_log.clear()
            return jsonify({"status": "cleared"})

    def on_event(self, callback: Callable[[PlaudEvent], None]) -> None:
        """
        Register a callback for any webhook event.

        Args:
            callback: Function that receives PlaudEvent
        """
        self._event_callbacks.append(callback)

    def on_transcribe_complete(
        self, callback: Callable[[PlaudEvent], None]
    ) -> Callable[[PlaudEvent], None]:
        """
        Decorator to register a transcription complete handler.

        Usage:
            @server.on_transcribe_complete
            def handle_transcription(event):
                print(f"Transcription complete: {event.file_id}")
        """
        self.handler.register_handler(
            PlaudEventType.AUDIO_TRANSCRIBE_COMPLETED, callback
        )
        return callback

    def on_recording_uploaded(
        self, callback: Callable[[PlaudEvent], None]
    ) -> Callable[[PlaudEvent], None]:
        """Decorator to register a recording upload handler."""
        self.handler.register_handler(PlaudEventType.RECORDING_UPLOADED, callback)
        return callback

    def on_device_connected(
        self, callback: Callable[[PlaudEvent], None]
    ) -> Callable[[PlaudEvent], None]:
        """Decorator to register a device connected handler."""
        self.handler.register_handler(PlaudEventType.DEVICE_CONNECTED, callback)
        return callback

    def start(self) -> None:
        """Start the webhook server in a background thread."""
        if self._running:
            logger.warning("Webhook server already running")
            return

        def run_server():
            # Disable Flask's reloader and debugger
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )

        self._running = True
        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        logger.info(f"Webhook server started on http://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the webhook server (note: Flask doesn't stop gracefully)."""
        self._running = False
        logger.info("Webhook server stop requested (will stop on next request)")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    @property
    def webhook_url(self) -> str:
        """Get the webhook URL to configure in Plaud."""
        # For local development, this would need ngrok or similar
        return f"http://localhost:{self.port}/webhook/plaud"

    def get_recent_events(self, limit: int = 10) -> List[EventLogEntry]:
        """Get recent events from the log."""
        return self.event_log[-limit:]

    def pop_event(self, timeout: Optional[float] = None) -> Optional[PlaudEvent]:
        """
        Pop an event from the queue.

        Args:
            timeout: How long to wait for an event (None = don't block)

        Returns:
            PlaudEvent or None if queue is empty
        """
        try:
            if timeout is not None:
                return self.event_queue.get(timeout=timeout)
            else:
                return self.event_queue.get_nowait()
        except:
            return None


# Singleton instance
_webhook_server: Optional[PlaudWebhookServer] = None


def get_webhook_server() -> PlaudWebhookServer:
    """Get the singleton webhook server instance."""
    global _webhook_server
    if _webhook_server is None:
        _webhook_server = PlaudWebhookServer()
    return _webhook_server


def start_webhook_server(port: int = 8090) -> PlaudWebhookServer:
    """
    Start the webhook server if not already running.

    Args:
        port: Port to listen on

    Returns:
        The webhook server instance
    """
    global _webhook_server
    if _webhook_server is None:
        _webhook_server = PlaudWebhookServer(port=port)
    if not _webhook_server.is_running:
        _webhook_server.start()
    return _webhook_server


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090

    print(f"🌐 Starting Plaud Webhook Server on port {port}...")
    print(f"   Webhook URL: http://localhost:{port}/webhook/plaud")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   Event log: http://localhost:{port}/events")
    print("\n   Configure this URL in your Plaud developer portal.")
    print("   For external access, use ngrok: ngrok http {port}")
    print("\n   Press Ctrl+C to stop\n")

    server = PlaudWebhookServer(port=port)

    @server.on_transcribe_complete
    def handle_transcription(event: PlaudEvent):
        print(f"📝 Transcription complete: {event.file_id}")

    @server.on_recording_uploaded
    def handle_recording(event: PlaudEvent):
        print(f"🎙️ Recording uploaded: {event.recording_id}")

    @server.on_device_connected
    def handle_device(event: PlaudEvent):
        print(f"📱 Device connected: {event.data}")

    # Start in foreground (not as daemon)
    server.app.run(host="0.0.0.0", port=port, debug=True)
