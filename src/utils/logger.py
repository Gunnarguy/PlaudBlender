"""Centralized logger configuration.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

This avoids sprinkling basicConfig calls throughout the codebase.
"""
import logging
import os

DEFAULT_LEVEL = os.getenv("PB_LOG_LEVEL", "INFO").upper()


def setup_logging(level: str = DEFAULT_LEVEL) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
