"""Main GUI application object.

This repo has multiple front-ends (e.g., Streamlit for Chronos). The unit tests
expect a legacy-style `PlaudBlenderApp` with a couple of methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gui.state import AppState


@dataclass
class PlaudBlenderApp:
    """Minimal app shell used for import/smoke tests."""

    state: AppState = field(default_factory=AppState)

    def run(self) -> None:
        """Run the app.

        In the real app this would start the UI event loop. For tests, it's a no-op.
        """

        return None

    def switch_view(self, view_name: str) -> None:
        """Switch the active view."""

        self.state.current_view = view_name
