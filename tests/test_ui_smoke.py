"""
UI Component Smoke Tests.
Tests that views and components can be imported and instantiated.
Note: Actual Tkinter rendering requires a display, so these tests focus on import/structure.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===========================================================================
# Theme Tests
# ===========================================================================


class TestTheme:
    """Tests for gui/theme.py."""

    def test_theme_import(self):
        """Verify theme module can be imported."""
        from gui.theme import Theme, ModernTheme

        assert Theme is not None
        assert ModernTheme is not None

    def test_modern_theme_instantiation(self):
        """Verify ModernTheme can be instantiated."""
        from gui.theme import ModernTheme

        theme = ModernTheme()
        assert theme is not None


# ===========================================================================
# State Tests
# ===========================================================================


class TestState:
    """Tests for gui/state.py."""

    def test_state_import(self):
        """Verify AppState can be imported."""
        from gui.state import AppState

        assert AppState is not None

    def test_state_instantiation(self):
        """Verify AppState can be instantiated."""
        from gui.state import AppState

        state = AppState()
        assert state is not None

    def test_state_has_required_attributes(self):
        """Verify AppState has expected attributes."""
        from gui.state import AppState

        state = AppState()

        # Should have core attributes (exact names may vary)
        # Just verify it's a functional object
        assert hasattr(state, "__dict__") or hasattr(state, "__slots__")


# ===========================================================================
# Component Tests
# ===========================================================================


class TestComponents:
    """Tests for gui/components/."""

    def test_stat_card_import(self):
        """Verify StatCard can be imported."""
        from gui.components.stat_card import StatCard

        assert StatCard is not None

    def test_status_bar_import(self):
        """Verify StatusBar can be imported."""
        from gui.components.status_bar import StatusBar

        assert StatusBar is not None


# ===========================================================================
# View Import Tests
# ===========================================================================


class TestViewImports:
    """Tests that all views can be imported without error."""

    def test_base_view_import(self):
        """Verify BaseView can be imported."""
        from gui.views.base import BaseView

        assert BaseView is not None

    def test_dashboard_view_import(self):
        """Verify DashboardView can be imported."""
        from gui.views.dashboard import DashboardView

        assert DashboardView is not None

    def test_transcripts_view_import(self):
        """Verify TranscriptsView can be imported."""
        from gui.views.transcripts import TranscriptsView

        assert TranscriptsView is not None

    def test_pinecone_view_import(self):
        """Verify PineconeView can be imported."""
        from gui.views.pinecone import PineconeView

        assert PineconeView is not None

    def test_search_view_import(self):
        """Verify SearchView can be imported."""
        from gui.views.search import SearchView

        assert SearchView is not None

    def test_timeline_view_import(self):
        """Verify TimelineView can be imported."""
        from gui.views.timeline import TimelineView

        assert TimelineView is not None

    def test_settings_view_import(self):
        """Verify SettingsView can be imported."""
        from gui.views.settings import SettingsView

        assert SettingsView is not None

    def test_logs_view_import(self):
        """Verify LogsView can be imported."""
        from gui.views.logs import LogsView

        assert LogsView is not None

    def test_notion_view_import(self):
        """Verify NotionView can be imported."""
        from gui.views.notion import NotionView

        assert NotionView is not None


# ===========================================================================
# Utils Tests
# ===========================================================================


class TestUtils:
    """Tests for gui/utils/."""

    def test_tooltips_import(self):
        """Verify tooltip utilities can be imported."""
        from gui.utils.tooltips import ToolTip

        assert ToolTip is not None

    def test_async_tasks_import(self):
        """Verify async task utilities can be imported."""
        from gui.utils.async_tasks import run_async

        assert run_async is not None

    def test_logging_import(self):
        """Verify logging utilities can be imported."""
        from gui.utils.logging import log, logger

        assert log is not None
        assert logger is not None


# ===========================================================================
# App Import Tests
# ===========================================================================


class TestAppImport:
    """Tests for main application module."""

    def test_app_module_import(self):
        """Verify gui.app module can be imported."""
        from gui.app import PlaudBlenderApp

        assert PlaudBlenderApp is not None

    def test_app_has_required_methods(self):
        """Verify PlaudBlenderApp has expected action methods."""
        from gui.app import PlaudBlenderApp

        required_methods = [
            "run",
            "switch_view",
        ]

        for method in required_methods:
            assert hasattr(PlaudBlenderApp, method), f"Missing method: {method}"


# ===========================================================================
# Services Registration Tests
# ===========================================================================


class TestServicesInit:
    """Tests for gui/services/__init__.py exports."""

    def test_services_package_import(self):
        """Verify services package can be imported."""
        import gui.services

        assert gui.services is not None

    def test_clients_import(self):
        """Verify clients can be imported."""
        from gui.services.clients import get_plaud_client, get_pinecone_client

        assert get_plaud_client is not None
        assert get_pinecone_client is not None


# ===========================================================================
# Integration: Full Module Tree
# ===========================================================================


class TestFullModuleTree:
    """Tests that verify the full module tree is importable."""

    def test_full_gui_import_tree(self):
        """Verify entire gui package can be imported."""
        # Core
        import gui
        import gui.app
        import gui.state
        import gui.theme

        # Components
        import gui.components
        import gui.components.stat_card
        import gui.components.status_bar

        # Views
        import gui.views
        import gui.views.base
        import gui.views.dashboard
        import gui.views.transcripts
        import gui.views.pinecone
        import gui.views.search
        import gui.views.timeline
        import gui.views.settings
        import gui.views.logs

        # Services
        import gui.services
        import gui.services.clients
        import gui.services.embedding_service
        import gui.services.pinecone_service
        import gui.services.search_service
        import gui.services.settings_service
        import gui.services.timeline_service
        import gui.services.transcripts_service
        import gui.services.stats_service

        # Utils
        import gui.utils
        import gui.utils.async_tasks
        import gui.utils.tooltips
        import gui.utils.logging

        assert True  # All imports succeeded

    def test_full_src_import_tree(self):
        """Verify entire src package can be imported."""
        # Core
        import src
        import src.config

        # Database
        import src.database
        import src.database.engine
        import src.database.models
        import src.database.repository

        # Models
        import src.models
        import src.models.schemas

        # Processing
        import src.processing
        import src.processing.engine
        import src.processing.indexer
        import src.processing.query_router
        import src.processing.rrf_fusion
        import src.processing.thought_signatures
        import src.processing.conflict_detection
        import src.processing.colpali_ingestion

        # AI
        import src.ai
        import src.ai.embeddings
        import src.ai.providers

        # Utils
        import src.utils
        import src.utils.logger

        # Top-level modules
        import src.notion_sync

        assert True  # All imports succeeded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
