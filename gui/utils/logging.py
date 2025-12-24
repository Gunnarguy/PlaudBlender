"""Logging helpers for the GUI.

Avoids configuring global logging in tests.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("plaudblender.gui")


def log(message: str, level: int = logging.INFO) -> None:
    logger.log(level, message)
