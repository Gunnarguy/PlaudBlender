import pytest
from unittest.mock import patch, MagicMock

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from api.dependencies import get_service, get_config
from app_v2.services.data_service import ChronosDataService
from src.config import Settings

def test_get_service_caching():
    """Verify that get_service is cached."""
    with patch("api.dependencies.get_data_service") as mock_get_data_service:
        mock_service = MagicMock(spec=ChronosDataService)
        mock_get_data_service.return_value = mock_service

        # Clear cache before testing to ensure clean state
        get_service.cache_clear()

        result1 = get_service()
        result2 = get_service()

        # get_data_service should only be called once because of lru_cache
        mock_get_data_service.assert_called_once()

        assert result1 is mock_service
        assert result2 is mock_service

def test_get_config_no_caching():
    """Verify that get_config is not cached and returns fresh settings."""
    with patch("api.dependencies.get_settings") as mock_get_settings:
        mock_settings = MagicMock(spec=Settings)
        mock_get_settings.return_value = mock_settings

        result1 = get_config()
        result2 = get_config()

        # get_settings should be called twice (no cache)
        assert mock_get_settings.call_count == 2

        assert result1 is mock_settings
        assert result2 is mock_settings

def test_fastapi_dependency_overrides():
    """Verify that FastAPI dependency overrides work correctly with our dependencies."""
    app = FastAPI()

    @app.get("/service")
    def route_service(service: ChronosDataService = Depends(get_service)):
        return {"type": type(service).__name__}

    @app.get("/config")
    def route_config(config: Settings = Depends(get_config)):
        return {"has_config": config is not None}

    # Test without overrides
    # Note: we need to mock the underlying functions because they might fail without real env
    with patch("api.dependencies.get_data_service") as mock_get_data_service, \
         patch("api.dependencies.get_settings") as mock_get_settings:

        mock_get_data_service.return_value = MagicMock(spec=ChronosDataService)
        mock_get_settings.return_value = MagicMock(spec=Settings)

        # Clear cache for get_service
        get_service.cache_clear()

        client = TestClient(app)

        # Verify default behavior
        response_svc = client.get("/service")
        assert response_svc.status_code == 200
        assert response_svc.json() == {"type": "MagicMock"}

        response_cfg = client.get("/config")
        assert response_cfg.status_code == 200
        assert response_cfg.json() == {"has_config": True}

    # Now test with FastAPI dependency overrides
    class DummyService:
        pass

    def override_get_service():
        return DummyService()

    def override_get_config():
        return None  # Just return None for testing

    app.dependency_overrides[get_service] = override_get_service
    app.dependency_overrides[get_config] = override_get_config

    client = TestClient(app)

    # Verify overridden behavior
    response_svc = client.get("/service")
    assert response_svc.status_code == 200
    assert response_svc.json() == {"type": "DummyService"}

    response_cfg = client.get("/config")
    assert response_cfg.status_code == 200
    assert response_cfg.json() == {"has_config": False}

    # Cleanup overrides
    app.dependency_overrides = {}

def test_get_database():
    """Verify that get_database yields from get_db correctly."""
    from api.dependencies import get_database
    with patch("api.dependencies.get_db") as mock_get_db:
        # Mock get_db to return a generator
        def mock_generator():
            yield "mock_session"
        mock_get_db.return_value = mock_generator()

        generator = get_database()
        result = next(generator)

        mock_get_db.assert_called_once()
        assert result == "mock_session"
