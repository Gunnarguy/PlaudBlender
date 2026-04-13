"""
UI Component Smoke Tests.
Tests that core modules can be imported and instantiated.
Covers Dash v2 app (app_v2/) and core src packages.
"""

import pytest
import sys
import os
import threading
from types import SimpleNamespace

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===========================================================================
# Dash v2 App Tests
# ===========================================================================


class TestDashApp:
    """Tests for app_v2/ Dash application."""

    def test_layout_import(self):
        """Verify layout module can be imported."""
        from app_v2 import layout

        assert layout is not None
        assert hasattr(layout, "create_layout")

    def test_components_import(self):
        """Verify all component modules can be imported."""
        from app_v2.components import sidebar
        from app_v2.components import day_view
        from app_v2.components import search
        from app_v2.components import graph
        from app_v2.components import stats
        from app_v2.components import topics
        from app_v2.components import recording_detail

        assert sidebar is not None
        assert day_view is not None
        assert search is not None
        assert graph is not None
        assert stats is not None
        assert topics is not None
        assert recording_detail is not None

    def test_callbacks_import(self):
        """Verify all callback modules can be imported."""
        from app_v2.callbacks import navigation
        from app_v2.callbacks import search as search_cb
        from app_v2.callbacks import day_view as dv_cb
        from app_v2.callbacks import graph as graph_cb

        assert navigation is not None
        assert search_cb is not None
        assert dv_cb is not None
        assert graph_cb is not None

    def test_data_service_import(self):
        """Verify data service can be imported."""
        from app_v2.services.data_service import ChronosDataService

        assert ChronosDataService is not None

    def test_data_service_retries_backend_init(self, monkeypatch):
        """Verify the singleton data service can recover if Qdrant was down at startup."""
        from app_v2.services.data_service import ChronosDataService

        service = ChronosDataService.__new__(ChronosDataService)
        service._qdrant = None
        service._embedder = None
        service._service_init_lock = threading.Lock()

        calls = {"count": 0}

        def fake_init_services():
            calls["count"] += 1
            service.__dict__["_qdrant"] = object()
            service.__dict__["_embedder"] = object()

        monkeypatch.setattr(service, "_init_services", fake_init_services)

        service._ensure_backend_services(require_qdrant=True, require_embedder=True)

        assert calls["count"] == 1
        assert service._qdrant is not None
        assert service._embedder is not None

    def test_embedded_auto_sync_skips_when_systemd_unit_enabled(self, monkeypatch):
        """Dash should not start embedded auto-sync when systemd already owns it."""
        import app_v2.main as app_main

        monkeypatch.delenv("CHRONOS_EMBEDDED_AUTO_SYNC", raising=False)
        monkeypatch.setattr(app_main.platform, "system", lambda: "Linux")
        monkeypatch.setattr(app_main.shutil, "which", lambda _name: "/bin/systemctl")

        def fake_run(args, **kwargs):
            if args[:2] == ["systemctl", "is-enabled"]:
                return SimpleNamespace(stdout="enabled\n", stderr="", returncode=0)
            if args[:2] == ["systemctl", "is-active"]:
                return SimpleNamespace(stdout="active\n", stderr="", returncode=0)
            raise AssertionError(args)

        monkeypatch.setattr(app_main.subprocess, "run", fake_run)

        should_start, reason = app_main._should_start_embedded_auto_sync()

        assert should_start is False
        assert "systemd manages chronos-auto-sync.service" in reason

    def test_embedded_auto_sync_respects_env_override(self, monkeypatch):
        """CHRONOS_EMBEDDED_AUTO_SYNC should allow explicit opt-in override."""
        import app_v2.main as app_main

        monkeypatch.setenv("CHRONOS_EMBEDDED_AUTO_SYNC", "1")
        monkeypatch.setattr(app_main.platform, "system", lambda: "Linux")
        monkeypatch.setattr(app_main.shutil, "which", lambda _name: "/bin/systemctl")

        should_start, reason = app_main._should_start_embedded_auto_sync()

        assert should_start is True
        assert reason == "forced by CHRONOS_EMBEDDED_AUTO_SYNC"


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
        # Dash v2 UI
        from app_v2 import layout
        from app_v2.components import sidebar, day_view, search, graph, stats, topics
        from app_v2.services.data_service import ChronosDataService

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
