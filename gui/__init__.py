"""GUI package for Chronos Streamlit components.

This package contains reusable Streamlit UI components for:
- Device integration panel
- Workflow management panel
- Webhook monitoring panel

The main app is chronos_app.py at the project root.
"""

from dataclasses import dataclass, field


@dataclass
class AppState:
    """Minimal app state for test compatibility."""

    current_view: str = "home"


@dataclass
class PlaudBlenderApp:
    """Minimal app shell for test compatibility."""

    state: AppState = field(default_factory=AppState)

    def run(self) -> None:
        """No-op for tests."""
        return None

    def switch_view(self, view_name: str) -> None:
        """Switch view for tests."""
        self.state.current_view = view_name
