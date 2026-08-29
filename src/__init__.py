"""
PlaudBlender - Voice Transcript Knowledge Graph

Transform your Plaud recordings into a searchable, visual knowledge graph.
"""

__version__ = "2.1.0"

__all__ = [
    "PlaudOAuthClient",
    "PlaudClient",
    "PlaudWorkflowClient",
    "get_workflow_client",
    "PlaudWebhookHandler",
    "get_webhook_handler",
]


def __getattr__(name):
    """Preserve the public imports without eagerly loading every integration."""
    if name == "PlaudOAuthClient":
        from .plaud_oauth import PlaudOAuthClient

        return PlaudOAuthClient
    if name == "PlaudClient":
        from .plaud_client import PlaudClient

        return PlaudClient
    if name in {"PlaudWorkflowClient", "get_workflow_client"}:
        from .plaud_workflow import PlaudWorkflowClient, get_workflow_client

        return {"PlaudWorkflowClient": PlaudWorkflowClient, "get_workflow_client": get_workflow_client}[name]
    if name in {"PlaudWebhookHandler", "get_webhook_handler"}:
        from .plaud_webhook import PlaudWebhookHandler, get_webhook_handler

        return {"PlaudWebhookHandler": PlaudWebhookHandler, "get_webhook_handler": get_webhook_handler}[name]
    raise AttributeError(name)
