"""Settings view stub."""

from __future__ import annotations

from dataclasses import dataclass

from gui.views.base import BaseView


@dataclass
class SettingsView(BaseView):
    name: str = "settings"
