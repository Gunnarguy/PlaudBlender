"""
UI Component Smoke Tests.
Tests that core modules can be imported and instantiated.
Covers Dash v2 app (app_v2/) and core src packages.
"""

import pytest
import sys
import os
import threading
from datetime import datetime
from types import SimpleNamespace

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _collect_text(node):
    """Recursively collect text from Dash component trees."""
    texts = []
    if node is None:
        return texts
    if isinstance(node, (str, int, float)):
        return [str(node)]

    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            texts.extend(_collect_text(child))
    elif children is not None:
        texts.extend(_collect_text(children))

    return texts


def _walk_nodes(node):
    """Yield a Dash component tree depth-first."""
    if node is None:
        return
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            yield from _walk_nodes(child)
    elif children is not None and not isinstance(children, (str, int, float)):
        yield from _walk_nodes(children)


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

    def test_sidebar_has_system_navigation(self):
        """Sidebar should expose the dedicated System page."""
        from app_v2.components.sidebar import create_sidebar

        sidebar = create_sidebar()
        found_system = False

        def walk(node):
            nonlocal found_system
            node_id = getattr(node, "id", None)
            if node_id == {"type": "nav-item", "view": "system"}:
                found_system = True
                return
            children = getattr(node, "children", None)
            if isinstance(children, list):
                for child in children:
                    if child is not None:
                        walk(child)
            elif children is not None and not isinstance(children, str):
                walk(children)

        walk(sidebar)
        assert found_system

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

    def test_timeline_view_hides_empty_day_cards(self):
        """Timeline cards should only render days that actually have recordings."""
        from app_v2.components.day_view import create_day_view
        from app_v2.services.data_service import DaySummary, RecordingSummary

        empty_day = DaySummary(
            date="2026-04-18",
            date_display="Saturday, Apr 18",
            total_duration_seconds=0,
            recording_count=0,
            event_count=0,
        )
        recording = RecordingSummary(
            recording_id="rec-001",
            start_time=datetime(2026, 4, 20, 10, 0, 0),
            end_time=datetime(2026, 4, 20, 10, 15, 0),
            duration_seconds=900,
            event_count=3,
            categories={"meeting": 3},
            keywords=["meeting"],
        )
        populated_day = DaySummary(
            date="2026-04-20",
            date_display="Monday, Apr 20",
            total_duration_seconds=900,
            recording_count=1,
            event_count=3,
            recordings=[recording],
            categories={"meeting": 3},
            top_keywords=["meeting"],
        )

        view = create_day_view([empty_day, populated_day])
        children = view.children or []
        if not isinstance(children, list):
            children = [children]
        days_list = children[-1]

        assert len(days_list.children) == 1

    def test_timeline_heatmap_marks_zero_event_recording_days(self):
        """Heatmap cells should show recorded days even when no moments were extracted."""
        from app_v2.components.day_view import create_heat_map_strip
        from app_v2.services.data_service import DaySummary, RecordingSummary

        today = datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
        recording = RecordingSummary(
            recording_id="rec-failed-heat",
            start_time=today,
            end_time=today.replace(hour=9, minute=0),
            duration_seconds=5400,
            event_count=0,
            categories={},
            keywords=[],
            processing_status="failed",
        )
        day = DaySummary(
            date=today.strftime("%Y-%m-%d"),
            date_display=today.strftime("%A, %b %d"),
            total_duration_seconds=5400,
            recording_count=1,
            event_count=0,
            recordings=[recording],
        )

        heatmap = create_heat_map_strip([day], num_calendar_days=7)
        matching_cells = [
            node
            for node in _walk_nodes(heatmap)
            if getattr(node, "id", None)
            == {"type": "heatmap-cell", "date": today.strftime("%Y-%m-%d")}
        ]

        assert matching_cells
        assert "heatmap-recording-only" in getattr(matching_cells[0], "className", "")
        assert "sync failed" in getattr(matching_cells[0], "title", "").lower()

    def test_failed_zero_event_recording_card_surfaces_retry_state(self):
        """Zero-event failed recordings should render as failed sync placeholders, not blanks."""
        from app_v2.components.day_view import create_recording_card
        from app_v2.services.data_service import RecordingSummary

        recording = RecordingSummary(
            recording_id="rec-failed-card",
            start_time=datetime(2026, 4, 20, 10, 0, 0),
            end_time=datetime(2026, 4, 20, 10, 20, 0),
            duration_seconds=1200,
            event_count=0,
            categories={},
            keywords=[],
            processing_status="failed",
        )

        card = create_recording_card(recording, "2026-04-20")
        text = " ".join(_collect_text(card))

        assert "Sync failed" in text
        assert "retry" in text.lower()

    def test_day_card_omits_unknown_flow_for_zero_event_failed_day(self):
        """Failed days without extracted moments should show status, not a fake unknown flow."""
        from app_v2.components.day_view import create_day_card
        from app_v2.services.data_service import DaySummary, RecordingSummary

        recording = RecordingSummary(
            recording_id="rec-failed-day",
            start_time=datetime(2026, 4, 20, 5, 16, 0),
            end_time=datetime(2026, 4, 20, 9, 54, 0),
            duration_seconds=16680,
            event_count=0,
            categories={},
            keywords=[],
            processing_status="failed",
        )
        day = DaySummary(
            date="2026-04-20",
            date_display="Monday, Apr 20",
            total_duration_seconds=16680,
            recording_count=1,
            event_count=0,
            recordings=[recording],
        )

        card = create_day_card(day, expanded=True)
        text = " ".join(_collect_text(card)).lower()

        assert "sync failed" in text
        assert "unknown" not in text

    def test_recording_card_renders_system_status_strip(self):
        """Recording cards should surface sync, Plaud, and Notion bridge state."""
        from app_v2.components.day_view import create_recording_card
        from app_v2.services.data_service import RecordingSummary

        recording = RecordingSummary(
            recording_id="rec-system-001",
            start_time=datetime(2026, 4, 20, 10, 0, 0),
            end_time=datetime(2026, 4, 20, 10, 20, 0),
            duration_seconds=1200,
            event_count=4,
            categories={"meeting": 4},
            keywords=["client", "follow-up"],
            source="plaud_cloud",
            has_plaud_ai=True,
            processing_status="completed",
            plaud_workflow_status="SUCCESS",
            notion_state="linked",
            notion_page_title="Client sync",
            notion_match_count=1,
        )

        card = create_recording_card(recording, "2026-04-20")
        text = " ".join(_collect_text(card))

        assert "Sync ready" in text
        assert "Plaud AI ready" in text
        assert "Notion linked" in text

    def test_recording_detail_renders_system_strip_and_notion_link(self):
        """Detail header should reuse the shared system strip and expose the Notion link."""
        from app_v2.components.recording_detail import create_recording_detail
        from app_v2.services.data_service import RecordingDetail, RecordingSummary

        summary = RecordingSummary(
            recording_id="rec-system-002",
            start_time=datetime(2026, 4, 21, 14, 0, 0),
            end_time=datetime(2026, 4, 21, 14, 30, 0),
            duration_seconds=1800,
            event_count=0,
            categories={},
            keywords=[],
            source="plaud_cloud",
            has_plaud_ai=True,
            processing_status="completed",
            plaud_workflow_status="SUCCESS",
            notion_state="linked",
            notion_page_title="Project note",
            notion_page_url="https://www.notion.so/project-note",
            notion_match_count=1,
        )

        detail = RecordingDetail(summary=summary, events=[])
        view = create_recording_detail(detail, "2026-04-21")
        text = " ".join(_collect_text(view))
        links = [
            node
            for node in _walk_nodes(view)
            if getattr(node, "className", "") == "detail-system-link"
        ]

        assert "Sync ready" in text
        assert "Plaud AI ready" in text
        assert "Notion linked" in text
        assert links
        assert getattr(links[0], "href", "") == "https://www.notion.so/project-note"

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

    def test_notion_api_authorize_url_uses_callback_host(self, monkeypatch):
        """Dash should start Notion OAuth against the API host from NOTION_REDIRECT_URI."""
        import app_v2.main as app_main

        monkeypatch.setattr(
            app_main,
            "NOTION_REDIRECT_URI",
            "https://your-ngrok-domain.ngrok-free.dev/api/v1/auth/notion/callback",
        )

        url = app_main._notion_api_authorize_url("https://ui.example/notion")

        assert url == (
            "https://your-ngrok-domain.ngrok-free.dev/api/v1/auth/notion/web-authorize"
            "?return_to=https%3A%2F%2Fui.example%2Fnotion"
        )

    def test_create_system_view_renders_runtime_details(self, monkeypatch):
        """System view should render host/runtime diagnostics without touching real services."""
        from app_v2.callbacks import navigation

        monkeypatch.setattr(
            navigation,
            "_get_local_runtime_status",
            lambda: {
                "manager_label": "systemd",
                "manager_detail": "Dedicated systemd services own the pipeline",
                "systemd_managed_auto_sync": True,
                "auto_sync_ok": True,
                "auto_sync_label": "Active",
                "auto_sync_detail": "chronos-auto-sync.service: active (enabled)",
                "watchdog_ok": True,
                "watchdog_label": "Active",
                "watchdog_detail": "chronos-watchdog.timer: active (enabled)",
                "plaud_ok": True,
                "plaud_label": "Linked",
                "plaud_detail": "Token valid for ~50 min",
                "ports": [
                    {"label": "UI", "port": 8050, "ok": True},
                    {"label": "API", "port": 8000, "ok": True},
                ],
                "access": {
                    "preferred_kind": "tailscale",
                    "preferred_label": "Tailscale",
                    "preferred_ui_url": "http://100.100.100.100:8050",
                    "preferred_api_url": "http://100.100.100.100:8000/api/v1/health",
                    "entries": [
                        {
                            "label": "Tailscale UI",
                            "url": "http://100.100.100.100:8050",
                            "kind": "tailscale",
                        }
                    ],
                },
            },
        )
        monkeypatch.setattr(
            navigation,
            "_check_services",
            lambda _settings: {
                "plaud": (True, "Token valid"),
                "gemini": (True, "API key valid"),
                "openai": (True, "API key valid"),
                "sqlite": (True, "1 recordings, 2 events"),
                "qdrant": (True, "2 points"),
                "webhook_listener": (True, "Live on :8090"),
                "webhook_config": (True, "Configured"),
            },
        )
        monkeypatch.setattr(
            navigation,
            "_systemd_unit_state",
            lambda _unit: ("active", "enabled"),
        )
        monkeypatch.setattr(
            navigation,
            "_read_log_tail",
            lambda _name, max_lines=12: ["line 1", "line 2"],
        )

        class FakeStats:
            total_events = 12
            total_topics = 4

        class FakeService:
            def get_recording_db_stats(self):
                return {"pending": 1, "processing": 2, "completed": 3, "failed": 0}

            def get_stats(self):
                return FakeStats()

        view = navigation.create_system_view(FakeService())
        rendered = str(view)

        assert "System" in rendered
        assert "Dedicated systemd services own the pipeline" in rendered
        assert "verify-pi.sh" in rendered
        assert "Tailscale" in rendered

    def test_layout_includes_runtime_strip_and_shell_actions(self):
        """The app shell should expose shared runtime status and quick actions."""
        from app_v2.layout import create_layout

        layout = create_layout()
        found_ids = set()

        def walk(node):
            node_id = getattr(node, "id", None)
            if node_id is not None:
                found_ids.add(str(node_id))
            children = getattr(node, "children", None)
            if isinstance(children, list):
                for child in children:
                    if child is not None:
                        walk(child)
            elif children is not None and not isinstance(children, str):
                walk(children)

        walk(layout)

        assert "app-runtime-status" in found_ids
        assert "app-runtime-access-link" in found_ids
        assert "app-workspace-pulse" in found_ids
        assert "shell-open-notion" in found_ids
        assert "shell-open-sync" in found_ids
        assert "shell-open-system" in found_ids

    def test_access_targets_prefer_tailscale_over_lan(self):
        """Runtime access summary should prefer Tailscale when it exists."""
        from app_v2.callbacks import navigation

        summary = navigation._build_access_targets(
            ["10.0.0.1", "100.100.100.100", "172.17.0.1"]
        )

        assert summary["preferred_kind"] == "tailscale"
        assert summary["preferred_label"] == "Tailscale"
        assert summary["preferred_ui_url"] == "http://100.100.100.100:8050"
        assert summary["entries"][0]["label"] == "Tailscale UI"

    def test_workspace_pulse_flags_incomplete_notion_bridge(self):
        """Shell pulse should call out Notion when it is partially configured but not live."""
        from app_v2.callbacks import navigation

        pulse = navigation._build_workspace_pulse(
            {
                "plaud_ok": True,
                "access": {"preferred_label": "Tailscale"},
                "notion": {
                    "connected": False,
                    "has_credentials": True,
                    "has_oauth": False,
                    "label": "Connect",
                    "detail": "Credentials are present, but the Notion bridge is not live yet",
                },
            },
            "timeline",
            0,
        )

        rendered = str(pulse)

        assert "Notion bridge needs one more step" in rendered
        assert "Tailscale" in rendered

    def test_workspace_pulse_highlights_active_background_sync(self):
        """Shell pulse should summarize active background sync work."""
        from app_v2.callbacks import navigation

        pulse = navigation._build_workspace_pulse(
            {
                "plaud_ok": True,
                "access": {"preferred_label": "Tailscale"},
                "activity": {
                    "active": True,
                    "summary": "Pipeline Process 2/5 · Notion 3/25",
                    "detail": "Recording 2/5: Gemini AI… | Notion import is processing 05-31 Field Log",
                },
                "notion": {
                    "connected": True,
                    "has_credentials": True,
                    "label": "Connected",
                    "detail": "Knowledge Inbox",
                },
            },
            "sync",
            0,
        )

        rendered = str(pulse)

        assert "Background sync is active" in rendered
        assert "Pipeline Process 2/5 · Notion 3/25" in rendered

    def test_summarize_background_activity_combines_pipeline_and_notion(
        self, monkeypatch
    ):
        """Runtime shell should combine pipeline and Notion batch state into one activity summary."""
        from app_v2.callbacks import navigation

        monkeypatch.setattr(
            "src.chronos.pipeline_progress.read_progress",
            lambda: {
                "status": "running",
                "current_phase": "process",
                "phases": [
                    {
                        "name": "process",
                        "completed_items": 2,
                        "total_items": 5,
                        "current_step": "Recording 2/5: Gemini AI…",
                        "current_item": "notion:page-1",
                    }
                ],
            },
        )
        monkeypatch.setattr(
            "src.chronos.notion_bridge.get_import_progress",
            lambda: {
                "status": "running",
                "completed": 3,
                "total": 25,
                "current_title": "05-31 Field Log",
            },
        )

        activity = navigation._summarize_background_activity()

        assert activity["active"] is True
        assert activity["summary"] == "Pipeline Process 2/5 · Notion 3/25"
        assert "Recording 2/5: Gemini AI…" in activity["detail"]
        assert "05-31 Field Log" in activity["detail"]

    def test_notion_runtime_status_prefers_live_bridge(self, monkeypatch):
        """Shell Notion status should surface the live database title when connected."""
        from app_v2.callbacks import navigation

        navigation._notion_runtime_status_cache = None
        navigation._notion_runtime_status_ts = 0.0

        class FakeOAuthClient:
            @property
            def token_status(self):
                return {
                    "is_authenticated": True,
                    "has_credentials": True,
                    "workspace_name": "Gunnar Workspace",
                }

        class FakeConnection:
            connected = True
            database_found = True
            database_title = "Knowledge Inbox"
            error = ""

        class FakeService:
            def check_connection(self, quick=False):
                assert quick is True
                return FakeConnection()

        monkeypatch.setattr(
            "src.notion_oauth.NotionOAuthClient",
            FakeOAuthClient,
        )
        monkeypatch.setattr(
            "src.notion_service.get_notion_service",
            lambda: FakeService(),
        )

        status = navigation._get_notion_runtime_status()

        assert status["connected"] is True
        assert status["label"] == "Connected"
        assert "Knowledge Inbox" in status["detail"]

    def test_create_sync_view_hides_archived_failures(self, monkeypatch):
        """Sync view should only surface actionable failures, not archived dead-ends."""
        from app_v2.callbacks import navigation

        monkeypatch.setattr(
            navigation,
            "_get_local_runtime_status",
            lambda: {
                "manager_label": "systemd",
                "manager_detail": "Dedicated systemd services own the pipeline",
                "systemd_managed_auto_sync": True,
                "auto_sync_ok": True,
                "auto_sync_label": "Active",
                "auto_sync_detail": "chronos-auto-sync.service: active (enabled)",
                "watchdog_ok": True,
                "watchdog_label": "Active",
                "watchdog_detail": "chronos-watchdog.timer: active (enabled)",
                "plaud_ok": True,
                "plaud_label": "Linked",
                "plaud_detail": "Token valid for ~50 min",
                "ports": [],
            },
        )

        class FakeStats:
            total_events = 12
            total_days = 4
            total_duration_hours = 2.5
            plaud_cloud_stats = None

        class FakeService:
            def get_stats(self):
                return FakeStats()

            def get_recording_db_stats(self):
                return {
                    "pending": 1,
                    "processing": 0,
                    "completed": 3,
                    "failed": 0,
                    "archived_failed": 6,
                    "total": 10,
                }

            def get_plaud_workflow_stats(self, days_back=30):
                return {
                    "recent_recordings": 10,
                    "workflow_success": 0,
                    "workflow_pending": 0,
                    "workflow_failed": 0,
                    "with_ai_summary": 0,
                }

            def get_sync_failure_summary(self, limit=5):
                return {
                    "actionable_count": 0,
                    "archived_count": 6,
                    "actionable": [],
                    "archived": [
                        {
                            "recording_id": "rec-archived-001",
                            "reason": "Plaud has no transcript available for this recording",
                            "error": "No transcript available in Plaud source_list",
                        }
                    ],
                }

            def get_upload_candidates(self):
                return []

        view = navigation.create_sync_view(FakeService())
        rendered = str(view)

        assert "Retryable Issues" not in rendered
        assert "Archived" not in rendered
        assert "rec-archived-001" not in rendered
        assert ">10<" in rendered or "10" in rendered


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
