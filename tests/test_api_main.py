from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app


def test_cors_allowed_origin():
    client = TestClient(app)
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    )


def test_cors_regex_allowed():
    client = TestClient(app)
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://192.168.1.100",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://192.168.1.100"


def test_cors_rejected_origin():
    client = TestClient(app)
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 400


@patch("api.main.init_db")
@patch("api.main.logger")
@patch("src.plaud_oauth.PlaudOAuthClient")
def test_lifespan_startup_success(mock_plaud_cls, mock_logger, mock_init_db):
    mock_plaud_instance = MagicMock()
    mock_plaud_cls.return_value = mock_plaud_instance
    mock_plaud_instance.token_status_with_recovery.return_value = {
        "is_authenticated": True
    }

    with TestClient(app):
        pass  # just entering the context will trigger the lifespan events

    mock_init_db.assert_called_once()
    mock_plaud_instance.token_status_with_recovery.assert_called_once_with(
        attempt_recovery=True
    )
    mock_logger.info.assert_called_with("Plaud auth is ready")


@patch("api.main.init_db")
@patch("api.main.logger")
@patch("src.plaud_oauth.PlaudOAuthClient")
def test_lifespan_startup_not_authenticated(mock_plaud_cls, mock_logger, mock_init_db):
    mock_plaud_instance = MagicMock()
    mock_plaud_cls.return_value = mock_plaud_instance
    mock_plaud_instance.token_status_with_recovery.return_value = {
        "is_authenticated": False,
        "has_refresh_token": False,
    }

    with TestClient(app):
        pass

    mock_init_db.assert_called_once()
    mock_plaud_instance.token_status_with_recovery.assert_called_once_with(
        attempt_recovery=True
    )
    mock_logger.warning.assert_called_with(
        "Plaud auth not connected on startup (has_refresh_token=%s)", False
    )


@patch("api.main.init_db")
@patch("api.main.logger")
@patch("src.plaud_oauth.PlaudOAuthClient")
def test_lifespan_startup_exception(mock_plaud_cls, mock_logger, mock_init_db):
    mock_plaud_instance = MagicMock()
    mock_plaud_cls.return_value = mock_plaud_instance
    mock_plaud_instance.token_status_with_recovery.side_effect = Exception("Test Error")

    with TestClient(app):
        pass

    mock_init_db.assert_called_once()
    mock_plaud_instance.token_status_with_recovery.assert_called_once_with(
        attempt_recovery=True
    )
    mock_logger.warning.assert_called_once()
    assert "Plaud auth warmup failed" in mock_logger.warning.call_args[0][0]
