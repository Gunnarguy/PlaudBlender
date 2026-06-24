
import src.plaud_webhook_server as plaud_webhook_server_module
from src.plaud_webhook_server import get_webhook_server, PlaudWebhookServer


def test_get_webhook_server_singleton():
    """Test that get_webhook_server implements the singleton pattern correctly."""
    # Ensure a clean state before testing
    original_server = plaud_webhook_server_module._webhook_server
    plaud_webhook_server_module._webhook_server = None

    try:
        # First call should create a new instance
        server1 = get_webhook_server()
        assert isinstance(server1, PlaudWebhookServer)

        # Second call should return the exact same instance
        server2 = get_webhook_server()
        assert server1 is server2

        # Resetting the global should cause a new instance to be created
        plaud_webhook_server_module._webhook_server = None
        server3 = get_webhook_server()
        assert isinstance(server3, PlaudWebhookServer)
        assert server3 is not server1
    finally:
        # Restore the original state to not affect other tests
        plaud_webhook_server_module._webhook_server = original_server
