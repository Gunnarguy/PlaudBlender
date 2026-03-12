"""Callbacks package."""

from app_v2.callbacks.navigation import register_navigation_callbacks
from app_v2.callbacks.day_view import register_day_view_callbacks
from app_v2.callbacks.search import register_search_callbacks
from app_v2.callbacks.graph import register_graph_callbacks
from app_v2.callbacks.recording_detail import register_recording_detail_callbacks


def register_all_callbacks(app):
    """Register all callbacks with the app."""
    register_navigation_callbacks(app)
    register_day_view_callbacks(app)
    register_search_callbacks(app)
    register_graph_callbacks(app)
    register_recording_detail_callbacks(app)
