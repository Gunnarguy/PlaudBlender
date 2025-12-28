"""GUI components package.

Provides reusable Streamlit components for the Chronos UI.
"""

# Re-export panel components for easy access
from .plaud_admin_panel import render_plaud_admin_panel
from .device_panel import render_device_panel
from .workflow_panel import render_workflow_panel
from .webhook_panel import render_webhook_panel
from .stat_card import StatCard
from .status_bar import StatusBar

__all__ = [
    "render_plaud_admin_panel",
    "render_device_panel",
    "render_workflow_panel",
    "render_webhook_panel",
    "StatCard",
    "StatusBar",
]
