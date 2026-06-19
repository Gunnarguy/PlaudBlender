import logging
from unittest.mock import patch

import pytest

from src.utils.logger import get_logger, setup_logging


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before and after each test."""
    root_logger = logging.getLogger()
    # Save original handlers and level
    handlers = root_logger.handlers[:]
    default_level = root_logger.level

    # Remove handlers
    for handler in handlers:
        root_logger.removeHandler(handler)

    # Some pytest plugins (like log capture) might add handlers during the test
    # but we want a clean slate for `test_get_logger_calls_setup_if_no_handlers`
    # Let's mock out root_logger.handlers inside that specific test instead if needed.

    yield

    # Restore logging state
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(default_level)


def test_get_logger_returns_logger_with_correct_name():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


@patch("src.utils.logger.setup_logging")
def test_get_logger_calls_setup_if_no_handlers(mock_setup):
    with patch("logging.getLogger") as mock_get_logger:
        # Mock the root logger to have no handlers
        mock_root_logger = logging.getLogger("dummy")
        mock_root_logger.handlers = []

        def side_effect(name=None):
            if name is None:
                return mock_root_logger
            return logging.Logger(name)

        mock_get_logger.side_effect = side_effect

        get_logger("test_module")

        mock_setup.assert_called_once()


@patch("src.utils.logger.setup_logging")
def test_get_logger_does_not_call_setup_if_handlers_exist(mock_setup):
    with patch("logging.getLogger") as mock_get_logger:
        # Mock the root logger to have handlers
        mock_root_logger = logging.getLogger("dummy")
        mock_root_logger.handlers = [logging.NullHandler()]

        def side_effect(name=None):
            if name is None:
                return mock_root_logger
            return logging.Logger(name)

        mock_get_logger.side_effect = side_effect

        get_logger("test_module")

        mock_setup.assert_not_called()


@patch("logging.basicConfig")
def test_setup_logging_sets_correct_level(mock_basicConfig):
    setup_logging("DEBUG")
    mock_basicConfig.assert_called_once_with(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


@patch("logging.basicConfig")
def test_setup_logging_falls_back_to_info_for_invalid_level(mock_basicConfig):
    setup_logging("INVALID_LEVEL")
    mock_basicConfig.assert_called_once_with(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
