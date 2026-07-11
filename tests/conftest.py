import sys
import os
import pytest

# Ensure project root is on path for test imports
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def test_settings_factory():
    """Factory fixture to create Settings instances with overrides for testing."""
    from src.config import Settings

    def _create_settings(**kwargs):
        settings = Settings()
        # Mock credentials/API status to prevent loading real keys
        settings.gemini_api_key = "test-gemini"
        settings.openai_api_key_configured = False
        settings.qdrant_api_key = None
        settings.notion_token = None

        for k, v in kwargs.items():
            setattr(settings, k, v)
        return settings

    return _create_settings

