import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app


def _wait_for(predicate, timeout=10.0):
    """Block until predicate() holds, or the timeout expires.

    api.main's lifespan fires run_startup_tasks() with asyncio.create_task and
    yields without awaiting it, so startup work keeps running on the portal
    thread after TestClient hands control back. Asserting the moment the context
    opens races that task, which is why these tests passed or failed depending
    on how warm the imports happened to be. Waiting for the task's own last
    observable effect removes the race without changing the deliberate
    fire-and-forget behavior of startup.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _logged(mock_method, needle):
    """True once mock_method has been called with a message containing needle."""
    return any(
        call.args and needle in str(call.args[0])
        for call in mock_method.call_args_list
    )


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
        # Entering the context triggers the lifespan; the startup work it spawns
        # finishes afterwards, so wait for it rather than racing it.
        assert _wait_for(
            lambda: _logged(mock_logger.info, "Plaud auth is ready")
        ), "startup task never completed the Plaud warmup"

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
        assert _wait_for(
            lambda: _logged(mock_logger.warning, "Plaud auth not connected on startup")
        ), "startup task never reported the unauthenticated state"

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
        assert _wait_for(
            lambda: _logged(mock_logger.warning, "Plaud auth warmup failed")
        ), "startup task never reported the Plaud warmup failure"

    mock_init_db.assert_called_once()
    mock_plaud_instance.token_status_with_recovery.assert_called_once_with(
        attempt_recovery=True
    )
    mock_logger.warning.assert_called_once()
    assert "Plaud auth warmup failed" in mock_logger.warning.call_args[0][0]
