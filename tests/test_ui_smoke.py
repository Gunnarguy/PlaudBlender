"""
UI Component Smoke Tests.
Tests that core modules can be imported and instantiated.
"""

import pytest
import sys
import os

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===========================================================================
# GUI Package Tests
# ===========================================================================


class TestGUIPackage:
    """Tests for gui package (now minimal)."""

    def test_gui_import(self):
        """Verify gui package can be imported."""
        import gui

        assert gui is not None

    def test_app_state_import(self):
        """Verify AppState can be imported from gui."""
        from gui import AppState

        assert AppState is not None

    def test_app_state_instantiation(self):
        """Verify AppState can be instantiated."""
        from gui import AppState

        state = AppState()
        assert state is not None
        assert hasattr(state, "current_view")

    def test_plaudblender_app_import(self):
        """Verify PlaudBlenderApp can be imported."""
        from gui import PlaudBlenderApp

        assert PlaudBlenderApp is not None

    def test_plaudblender_app_methods(self):
        """Verify PlaudBlenderApp has required methods."""
        from gui import PlaudBlenderApp

        app = PlaudBlenderApp()
        assert hasattr(app, "run")
        assert hasattr(app, "switch_view")


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

    def test_device_panel_import(self):
        """Verify device_panel can be imported."""
        from gui.components.device_panel import render_device_panel

        assert render_device_panel is not None

    def test_workflow_panel_import(self):
        """Verify workflow_panel can be imported."""
        from gui.components.workflow_panel import render_workflow_panel

        assert render_workflow_panel is not None

    def test_webhook_panel_import(self):
        """Verify webhook_panel can be imported."""
        from gui.components.webhook_panel import render_webhook_panel

        assert render_webhook_panel is not None


# ===========================================================================
# Src Module Tests
# ===========================================================================


class TestSrcModules:
    """Tests for src package core modules."""

    def test_config_import(self):
        """Verify config can be imported."""
        import src.config

        assert src.config is not None
        from src.config import get_settings

        assert get_settings is not None

    def test_database_import(self):
        """Verify database package can be imported."""
        import src.database
        import src.database.engine
        import src.database.models
        import src.database.repository

        assert src.database is not None

    def test_models_import(self):
        """Verify models package can be imported."""
        import src.models
        import src.models.schemas
        import src.models.chronos_schemas

        assert src.models is not None

    def test_chronos_import(self):
        """Verify chronos package can be imported."""
        import src.chronos

        assert src.chronos is not None

    def test_chronos_modules_import(self):
        """Verify chronos submodules can be imported."""
        from src.chronos.qdrant_client import ChronosQdrantClient
        from src.chronos.embedding_service import ChronosEmbeddingService
        from src.chronos.transcript_processor import TranscriptProcessor
        from src.chronos.ingest_service import ChronosIngestService
        from src.chronos.graph_service import ChronosGraphExtractor

        assert ChronosQdrantClient is not None
        assert ChronosEmbeddingService is not None

    def test_processing_import(self):
        """Verify processing package can be imported."""
        import src.processing
        import src.processing.engine
        import src.processing.indexer

        assert src.processing is not None

    def test_plaud_client_import(self):
        """Verify Plaud client can be imported."""
        from src.plaud_client import PlaudClient

        assert PlaudClient is not None

    def test_plaud_oauth_import(self):
        """Verify Plaud OAuth can be imported."""
        from src.plaud_oauth import PlaudOAuthClient

        assert PlaudOAuthClient is not None

    def test_utils_import(self):
        """Verify utils can be imported."""
        import src.utils
        import src.utils.logger

        assert src.utils is not None


# ===========================================================================
# Integration: Full Module Tree
# ===========================================================================


class TestFullModuleTree:
    """Tests that verify the full module tree is importable."""

    def test_core_imports(self):
        """Verify core packages can be imported."""
        # GUI (minimal)
        import gui
        import gui.components
        import gui.components.stat_card
        import gui.components.status_bar

        # Src core
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
        import src.models.chronos_schemas

        # Processing (legacy, minimal)
        import src.processing
        import src.processing.engine
        import src.processing.indexer

        # Chronos (main pipeline)
        import src.chronos
        import src.chronos.qdrant_client
        import src.chronos.embedding_service
        import src.chronos.transcript_processor
        import src.chronos.ingest_service

        # Utils
        import src.utils
        import src.utils.logger

        assert True  # All imports succeeded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
