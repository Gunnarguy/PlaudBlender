"""Logs view stub."""

from __future__ import annotations

from dataclasses import dataclass

from gui.views.base import BaseView


@dataclass
class LogsView(BaseView):
    name: str = "logs"
