import pytest

import src.plaud_webhook as plaud_webhook_module
from src.plaud_webhook import get_webhook_handler, PlaudWebhookHandler


def test_get_webhook_handler_singleton():
    """Test that get_webhook_handler implements the singleton pattern correctly."""
    # Ensure a clean state before testing
    original_handler = plaud_webhook_module._webhook_handler
    plaud_webhook_module._webhook_handler = None

    try:
        # First call should create a new instance
        handler1 = get_webhook_handler()
        assert isinstance(handler1, PlaudWebhookHandler)

        # Second call should return the exact same instance
        handler2 = get_webhook_handler()
        assert handler1 is handler2

        # Resetting the global should cause a new instance to be created
        plaud_webhook_module._webhook_handler = None
        handler3 = get_webhook_handler()
        assert isinstance(handler3, PlaudWebhookHandler)
        assert handler3 is not handler1
    finally:
        # Restore the original state to not affect other tests
        plaud_webhook_module._webhook_handler = original_handler
