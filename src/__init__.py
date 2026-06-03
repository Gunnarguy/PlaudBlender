"""
PlaudBlender - Voice Transcript Knowledge Graph

Transform your Plaud recordings into a searchable, visual knowledge graph.
"""

__version__ = "2.1.0"

# Only import core Plaud clients by default
# Other modules have heavy dependencies (qdrant, google-ai, etc.)
from .plaud_oauth import PlaudOAuthClient
from .plaud_client import PlaudClient
from .plaud_workflow import PlaudWorkflowClient, get_workflow_client
from .plaud_device import PlaudDeviceManager, get_device_manager
from .plaud_webhook import PlaudWebhookHandler, get_webhook_handler

__all__ = [
    "PlaudOAuthClient",
    "PlaudClient",
    "PlaudWorkflowClient",
    "get_workflow_client",
    "PlaudDeviceManager",
    "get_device_manager",
    "PlaudWebhookHandler",
    "get_webhook_handler",
]
