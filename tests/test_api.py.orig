"""
Comprehensive tests for the Chronos FastAPI backend.

Uses FastAPI's TestClient with mocked services to avoid hitting real DB/API.
Tests every route for success, 404, 401, and invalid input paths.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fake dataclasses matching data_service shapes ───────────


@dataclass
class FakeEvent:
    id: str = "evt-001"
    recording_id: str = "rec-001"
    start_ts: datetime = None
    end_ts: datetime = None
    day_of_week: str = "Monday"
    hour_of_day: int = 10
    clean_text: str = "Test event text"
    category: str = "meeting"
    category_confidence: float = 0.95
    sentiment: float = 0.5
    keywords: List[str] = field(default_factory=lambda: ["test"])
    speaker: str = "self_talk"
    duration_seconds: float = 120.0

    def __post_init__(self):
        if self.start_ts is None:
            self.start_ts = datetime(2026, 1, 15, 10, 0, 0)
        if self.end_ts is None:
            self.end_ts = datetime(2026, 1, 15, 10, 2, 0)


@dataclass
class FakeRecordingSummary:
    recording_id: str = "rec-001"
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 600.0
    event_count: int = 5
    categories: Dict[str, int] = field(
        default_factory=lambda: {"meeting": 3, "work": 2}
    )
    keywords: List[str] = field(default_factory=lambda: ["test", "demo"])
    avg_sentiment: float = 0.5
    source: str = "plaud_cloud"
    has_plaud_ai: bool = False
    preview_text: str = "Test recording preview"
    event_previews: List[str] = field(default_factory=list)
    sentiment_arc: List[float] = field(default_factory=list)
    time_is_estimated: bool = False
    time_estimate_reason: str = ""
    title: str = "Test Recording"
    plaud_ai_summary: str = None
    cloud_status: str = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime(2026, 1, 15, 10, 0, 0)
        if self.end_time is None:
            self.end_time = datetime(2026, 1, 15, 10, 10, 0)

    @property
    def duration_formatted(self):
        return "10:00"

    @property
    def time_range_formatted(self):
        return "10:00 AM - 10:10 AM"

    @property
    def top_category(self):
        return max(self.categories, key=lambda k: self.categories[k])


@dataclass
class FakeDaySummary:
    date: str = "2026-01-15"
    date_display: str = "Wednesday, Jan 15"
    total_duration_seconds: float = 3600.0
    recording_count: int = 3
    event_count: int = 15
    recordings: list = field(default_factory=list)
    categories: Dict[str, int] = field(
        default_factory=lambda: {"meeting": 10, "work": 5}
    )
    top_keywords: List[str] = field(default_factory=lambda: ["test", "demo"])
    ai_summary: str = "A productive day."

    @property
    def top_category(self):
        return "meeting"


@dataclass
class FakeRecordingDetail:
    summary: FakeRecordingSummary = None
    events: list = field(default_factory=list)

    def __post_init__(self):
        if self.summary is None:
            self.summary = FakeRecordingSummary()
        if not self.events:
            self.events = [FakeEvent()]

    @property
    def category_percentages(self):
        total = sum(self.summary.categories.values())
        if total == 0:
            return {}
        return {k: (v / total) * 100 for k, v in self.summary.categories.items()}

    @property
    def transcript(self):
        return "Full transcript text"

    @property
    def ai_summary(self):
        return "AI summary of recording"

    @property
    def extracted_data(self):
        return {"key": "value"}

    @property
    def workflow_status(self):
        return None

    @property
    def plaud_transcript(self):
        return None


@dataclass
class FakeSearchResult:
    event: FakeEvent = None
    score: float = 0.85
    context_before: str = None
    context_after: str = None

    def __post_init__(self):
        if self.event is None:
            self.event = FakeEvent()


@dataclass
class FakeTopicTimeline:
    topic: str = "test"
    total_occurrences: int = 5
    recording_count: int = 2
    day_count: int = 1
    occurrences: list = field(default_factory=list)


@dataclass
class FakeTopicOccurrence:
    event_id: str = "evt-001"
    recording_id: str = "rec-001"
    timestamp: datetime = None
    text_snippet: str = "Discussed the test topic"
    category: str = "meeting"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime(2026, 1, 15, 10, 0, 0)


@dataclass
class FakeGraphData:
    nodes: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"id": "n1", "label": "Project X", "weight": 5},
        ]
    )
    edges: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"source": "n1", "target": "n2", "weight": 2},
        ]
    )


# ── Mock service factory ────────────────────────────────────


def _make_mock_service():
    """Create a fully mocked ChronosDataService."""
    svc = MagicMock()

    # Timeline
    svc.get_days.return_value = [FakeDaySummary()]
    svc.get_days_filled.return_value = [
        FakeDaySummary(recordings=[FakeRecordingSummary()])
    ]
    svc.get_day_detail.return_value = FakeDaySummary(
        recordings=[FakeRecordingSummary()]
    )

    # Recordings
    svc.get_recording_detail.return_value = FakeRecordingDetail()
    svc.get_events_for_recording.return_value = [FakeEvent()]
    svc.get_transcript.return_value = "Full transcript text"
    svc.get_ai_summary.return_value = "AI summary"
    svc.get_extracted_data.return_value = {"key": "value"}
    svc.get_plaud_workflow_transcript.return_value = "Plaud transcript"
    svc.save_category_override.return_value = None
    svc.get_category_overrides.return_value = {"evt-001": "meeting"}

    # Search
    svc.search.return_value = [FakeSearchResult()]

    # Topics
    svc.get_all_topics.return_value = [("test", 5), ("demo", 3)]
    svc.get_topic_timeline.return_value = FakeTopicTimeline(
        occurrences=[FakeTopicOccurrence()]
    )

    # Graph
    svc.get_graph_data.return_value = FakeGraphData()

    # Stats
    svc.get_stats.return_value = MagicMock(
        total_recordings=10,
        total_events=100,
        total_days=5,
        total_duration_hours=8.5,
        categories={"meeting": 50, "work": 30, "personal": 20},
        sentiment_avg=0.6,
        top_keywords=[("test", 5), ("demo", 3)],
        categories_by_hour={10: {"meeting": 5}, 14: {"work": 3}},
        sentiment_distribution={"positive": 50, "neutral": 30, "negative": 20},
        recent_days=None,
    )
    svc.get_recording_db_stats.return_value = {
        "total": 10,
        "processed": 8,
        "pending": 1,
        "failed": 1,
    }
    svc.get_plaud_workflow_stats.return_value = {
        "total": 5,
        "completed": 3,
        "pending": 2,
    }

    # Sync
    svc.submit_plaud_workflows.return_value = {"submitted": ["rec-001"]}
    svc.refresh_plaud_workflow_statuses.return_value = {"refreshed": 2}
    svc.submit_single_recording_workflow.return_value = "Workflow submitted"
    svc.get_workflow_status_for_recording.return_value = {"status": "pending"}
    svc.reset_stuck_recordings.return_value = 3
    svc.refresh_cache.return_value = None
    svc.get_upload_candidates.return_value = [
        {"recording_id": "rec-001", "filename": "test.wav"}
    ]

    return svc


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture()
def mock_svc():
    return _make_mock_service()


@pytest.fixture()
def client(mock_svc):
    """Create TestClient with mocked dependencies and no auth."""
    # Clear the lru_cache before importing the app
    from api.dependencies import get_service

    get_service.cache_clear()

    # Patch environment so auth is disabled (dev mode)
    with patch.dict(os.environ, {"CHRONOS_API_KEY": ""}, clear=False):
        # Override the get_service dependency
        from api.main import app
        from api.dependencies import get_service as gs

        app.dependency_overrides[gs] = lambda: mock_svc
        yield TestClient(app, raise_server_exceptions=False)
        app.dependency_overrides.clear()


@pytest.fixture()
def authed_client(mock_svc):
    """Create TestClient with auth enabled and provide Bearer token."""
    from api.dependencies import get_service

    get_service.cache_clear()

    with patch.dict(os.environ, {"CHRONOS_API_KEY": "test-secret-key"}, clear=False):
        from api.main import app
        from api.dependencies import get_service as gs

        app.dependency_overrides[gs] = lambda: mock_svc
        yield TestClient(app, raise_server_exceptions=False)
        app.dependency_overrides.clear()


AUTH_HEADER = {"Authorization": "Bearer test-secret-key"}
BAD_AUTH = {"Authorization": "Bearer wrong-key"}


# ═══════════════════════════════════════════════════════════
# HEALTH (no auth required)
# ═══════════════════════════════════════════════════════════


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"

    def test_health_no_auth_needed(self, authed_client):
        """Health endpoint should work without auth header."""
        r = authed_client.get("/api/v1/health")
        assert r.status_code == 200

    def test_status_attempts_plaud_recovery(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            instance = MockPlaud.return_value
            instance.token_status = {
                "is_authenticated": True,
                "has_access_token": True,
                "has_refresh_token": True,
                "token_valid": True,
                "expires_at": "2026-12-31T00:00:00",
                "expires_in_minutes": 60,
                "needs_refresh": False,
            }

            r = client.get("/api/v1/status")
            assert r.status_code == 200
            data = r.json()
            assert data["plaud"]["is_authenticated"] is True
            assert data["plaud"]["recovery_attempted"] is True
            instance.ensure_valid_token.assert_called_once()

    def test_status_reports_plaud_recovery_failure(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            instance = MockPlaud.return_value
            instance.token_status = {
                "is_authenticated": False,
                "has_access_token": True,
                "has_refresh_token": True,
                "token_valid": False,
                "expires_at": "2026-12-31T00:00:00",
                "expires_in_minutes": -5,
                "needs_refresh": True,
            }
            instance.ensure_valid_token.side_effect = Exception("refresh failed")

            r = client.get("/api/v1/status")
            assert r.status_code == 200
            data = r.json()
            assert data["plaud"]["is_authenticated"] is False
            assert data["plaud"]["recovery_attempted"] is True
            assert data["plaud"]["recovery_error"] == "refresh failed"


# ═══════════════════════════════════════════════════════════
# AUTH (test auth enforcement)
# ═══════════════════════════════════════════════════════════


class TestAuthEnforcement:
    def test_unauthenticated_blocked(self, authed_client):
        """When CHRONOS_API_KEY is set, requests without auth header get 401."""
        r = authed_client.get("/api/v1/timeline/days")
        assert r.status_code == 401

    def test_bad_token_blocked(self, authed_client):
        """Wrong API key returns 401."""
        r = authed_client.get("/api/v1/timeline/days", headers=BAD_AUTH)
        assert r.status_code == 401

    def test_valid_token_allowed(self, authed_client):
        """Correct API key lets the request through."""
        r = authed_client.get("/api/v1/timeline/days", headers=AUTH_HEADER)
        assert r.status_code == 200

    def test_dev_mode_no_auth_required(self, client):
        """When CHRONOS_API_KEY is empty, no auth needed."""
        r = client.get("/api/v1/timeline/days")
        assert r.status_code == 200

    def test_require_auth_missing_key_public_mode(self, client):
        """When CHRONOS_REQUIRE_AUTH=1 and CHRONOS_API_KEY is empty, request is rejected."""
        with patch.dict(os.environ, {"CHRONOS_REQUIRE_AUTH": "1", "CHRONOS_API_KEY": ""}, clear=False):
            r = client.get("/api/v1/timeline/days")
            assert r.status_code == 401
            assert "unauthorized" in r.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════
# TIMELINE
# ═══════════════════════════════════════════════════════════


class TestTimeline:
    def test_list_days(self, client):
        r = client.get("/api/v1/timeline/days")
        assert r.status_code == 200
        data = r.json()
        assert "days" in data
        assert data["total"] >= 1
        day = data["days"][0]
        assert day["date"] == "2026-01-15"
        assert day["recording_count"] == 3

    def test_list_days_filled(self, client):
        r = client.get("/api/v1/timeline/days-filled")
        assert r.status_code == 200
        data = r.json()
        assert len(data["days"]) >= 1
        day = data["days"][0]
        assert day["recordings"] is not None
        assert len(day["recordings"]) >= 1

    def test_day_detail(self, client):
        r = client.get("/api/v1/timeline/days/2026-01-15")
        assert r.status_code == 200
        data = r.json()
        assert data["date"] == "2026-01-15"

    def test_day_detail_404(self, client, mock_svc):
        mock_svc.get_day_detail.return_value = None
        r = client.get("/api/v1/timeline/days/1999-01-01")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════
# RECORDINGS
# ═══════════════════════════════════════════════════════════


class TestRecordings:
    def test_recording_detail(self, client):
        r = client.get("/api/v1/recordings/rec-001")
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["recording_id"] == "rec-001"
        assert len(data["events"]) >= 1

    def test_recording_detail_404(self, client, mock_svc):
        mock_svc.get_recording_detail.return_value = None
        r = client.get("/api/v1/recordings/not-found")
        assert r.status_code == 404

    def test_recording_events(self, client):
        r = client.get("/api/v1/recordings/rec-001/events")
        assert r.status_code == 200
        events = r.json()
        assert isinstance(events, list)
        assert len(events) >= 1
        assert events[0]["clean_text"] == "Test event text"

    def test_transcript(self, client):
        r = client.get("/api/v1/recordings/rec-001/transcript")
        assert r.status_code == 200
        assert r.json()["transcript"] == "Full transcript text"

    def test_ai_summary(self, client):
        r = client.get("/api/v1/recordings/rec-001/ai-summary")
        assert r.status_code == 200
        assert r.json()["ai_summary"] == "AI summary"

    def test_extracted_data(self, client):
        r = client.get("/api/v1/recordings/rec-001/extracted-data")
        assert r.status_code == 200
        assert r.json()["extracted_data"] == {"key": "value"}

    def test_plaud_transcript(self, client):
        r = client.get("/api/v1/recordings/rec-001/plaud-transcript")
        assert r.status_code == 200
        assert r.json()["plaud_transcript"] == "Plaud transcript"

    def test_category_overrides_get(self, client):
        r = client.get("/api/v1/recordings/rec-001/category-overrides")
        assert r.status_code == 200
        assert "overrides" in r.json()

    def test_category_override_put(self, client, mock_svc):
        r = client.put(
            "/api/v1/recordings/rec-001/events/evt-001/category",
            json={"category": "personal"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True
        mock_svc.save_category_override.assert_called_once_with("evt-001", "personal")

    def test_category_override_missing_body(self, client):
        r = client.put("/api/v1/recordings/rec-001/events/evt-001/category")
        assert r.status_code == 422  # Validation error


# ═══════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════


class TestSearch:
    def test_search(self, client):
        r = client.post("/api/v1/search", json={"query": "test meeting"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert data["total"] >= 1
        result = data["results"][0]
        assert result["score"] > 0
        assert result["event"]["clean_text"] == "Test event text"

    def test_search_with_filters(self, client):
        r = client.post(
            "/api/v1/search",
            json={
                "query": "meeting",
                "limit": 5,
                "categories": ["meeting"],
                "start_date": "2026-01-01",
                "end_date": "2026-12-31",
            },
        )
        assert r.status_code == 200

    def test_search_empty_query(self, client):
        """Empty query should fail validation (query is required)."""
        r = client.post("/api/v1/search", json={})
        assert r.status_code == 422

    def test_search_limit_bounds(self, client):
        """Limit > 200 should fail validation."""
        r = client.post("/api/v1/search", json={"query": "test", "limit": 201})
        assert r.status_code == 422

    def test_ask_ai(self, client, mock_svc):
        """Test AI Q&A endpoint with mocked OpenAI."""
        with patch("api.routes.search.ChronosAskService") as MockAI:
            ai_instance = MockAI.return_value
            ai_instance.available = True
            ai_instance.ask.return_value = {
                "answer": "Based on your recordings...",
                "model": "gpt-5.4",
                "response_id": "resp_123",
                "config": {"provider": "openai"},
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
            r = client.post(
                "/api/v1/search/ask", json={"question": "What happened today?"}
            )
            assert r.status_code == 200
            data = r.json()
            assert data["answer"] == "Based on your recordings..."
            assert data["model"] == "gpt-5.4"

    def test_ask_ai_unavailable(self, client):
        """Should return 503 when no AI provider is configured."""
        with patch("api.routes.search.ChronosAskService") as MockAI:
            ai_instance = MockAI.return_value
            ai_instance.available = False
            r = client.post("/api/v1/search/ask", json={"question": "What happened?"})
            assert r.status_code == 503

    def test_ask_ai_no_results(self, client, mock_svc):
        mock_svc.search.return_value = []

        with patch("api.routes.search.ChronosAskService") as MockAI:
            ai_instance = MockAI.return_value
            ai_instance.available = True

            r = client.post("/api/v1/search/ask", json={"question": "Missing stuff?"})

        assert r.status_code == 200
        data = r.json()
        assert "couldn't find any relevant events" in data["answer"]


# ═══════════════════════════════════════════════════════════
# TOPICS
# ═══════════════════════════════════════════════════════════


class TestTopics:
    def test_list_topics(self, client):
        r = client.get("/api/v1/topics")
        assert r.status_code == 200
        topics = r.json()
        assert len(topics) == 2
        assert topics[0]["name"] == "test"
        assert topics[0]["count"] == 5

    def test_topic_timeline(self, client):
        r = client.get("/api/v1/topics/test")
        assert r.status_code == 200
        data = r.json()
        assert data["topic"] == "test"
        assert data["total_occurrences"] == 5
        assert len(data["occurrences"]) >= 1

    def test_topic_not_found(self, client, mock_svc):
        mock_svc.get_topic_timeline.return_value = None
        r = client.get("/api/v1/topics/nonexistent")
        assert r.status_code == 200
        data = r.json()
        assert data["total_occurrences"] == 0
        assert data["occurrences"] == []


# ═══════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════


class TestGraph:
    def test_graph_data(self, client):
        r = client.get("/api/v1/graph")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 1


# ═══════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════


class TestStats:
    def test_stats(self, client):
        r = client.get("/api/v1/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_recordings"] == 10
        assert data["total_events"] == 100
        assert isinstance(data["categories"], dict)
        # top_keywords should be list of dicts
        assert isinstance(data["top_keywords"], list)
        if data["top_keywords"]:
            assert "keyword" in data["top_keywords"][0]
        # categories_by_hour keys should be strings
        if data["categories_by_hour"]:
            assert all(isinstance(k, str) for k in data["categories_by_hour"].keys())

    def test_db_stats(self, client):
        r = client.get("/api/v1/stats/db")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 10

    def test_workflow_stats(self, client):
        r = client.get("/api/v1/stats/workflows")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 5


# ═══════════════════════════════════════════════════════════
# SYNC
# ═══════════════════════════════════════════════════════════


class TestSync:
    def test_pipeline_status(self, client):
        with patch("src.chronos.pipeline_progress.read_progress", return_value=None):
            r = client.get("/api/v1/sync/status")
            assert r.status_code == 200
            assert r.json()["status"] == "idle"

    def test_pipeline_status_backfill_metadata(self, client):
        progress = {
            "status": "failed",
            "current_phase": "backfill",
            "sync_mode": "backfill",
            "partial_success": True,
            "warning": "Repeated page signature detected on page 2; stopping backfill.",
            "warnings": ["Partial Plaud backfill preserved earlier pages."],
            "phases": [
                {
                    "name": "backfill",
                    "status": "failed",
                    "total_items": 4,
                    "completed_items": 3,
                    "current_step": "page 4",
                    "current_item": "429",
                    "elapsed_seconds": 42,
                    "summary": "Preserved prior pages before stopping.",
                    "error": "Rate limit while backfilling Plaud history.",
                    "warnings": ["Stopping to avoid an infinite Plaud page loop."],
                }
            ],
        }
        with patch("src.chronos.pipeline_progress.read_progress", return_value=progress):
            r = client.get("/api/v1/sync/status")
            assert r.status_code == 200
            data = r.json()
            assert data["current_phase"] == "backfill"
            assert data["sync_mode"] == "backfill"
            assert data["partial_success"] is True
            assert data["warnings"] == ["Partial Plaud backfill preserved earlier pages."]
            assert data["phases"][0]["warnings"] == ["Stopping to avoid an infinite Plaud page loop."]

    def test_pipeline_status_running(self, client):
        with patch(
            "src.chronos.pipeline_progress.read_progress",
            return_value={"status": "running", "stage": "ingest", "progress": 0.5},
        ):
            r = client.get("/api/v1/sync/status")
            assert r.status_code == 200
            assert r.json()["status"] == "running"

    def test_run_pipeline(self, client):
        with (
            patch("api.routes.sync._pipeline_process_active", return_value=False),
            patch("subprocess.Popen") as mock_popen,
        ):
            r = client.post("/api/v1/sync/run", json={"stage": "full"})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "started"
            assert data["message"] == "Smart sync started for the last 7 days"
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args.args[0]
            assert cmd[2] == "--full"
            assert cmd[-2:] == ["--days-back", "7"]

    @pytest.mark.parametrize("stage", ["backfill", "all_history", "full_history"])
    def test_run_pipeline_backfill_aliases(self, client, stage):
        with (
            patch("api.routes.sync._pipeline_process_active", return_value=False),
            patch("subprocess.Popen") as mock_popen,
        ):
            r = client.post("/api/v1/sync/run", json={"stage": stage})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "started"
            assert data["message"] == "Full-history Plaud backfill started"
            mock_popen.assert_called_once()
            cmd = mock_popen.call_args.args[0]
            assert cmd[2] == "--backfill"
            assert cmd[-2:] == ["--days-back", "7"]

    def test_run_pipeline_when_already_running(self, client):
        running = {"status": "running", "started_at": time.time()}
        with (
            patch("api.routes.sync.read_progress", return_value=running),
            patch("api.routes.sync._pipeline_process_active", return_value=True),
            patch("subprocess.Popen") as mock_popen,
        ):
            r = client.post("/api/v1/sync/run", json={"stage": "full"})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "already_running"
            mock_popen.assert_not_called()

    def test_run_pipeline_stale_progress_without_active_process(self, client):
        stale = {"status": "running", "started_at": time.time()}
        with (
            patch("api.routes.sync.read_progress", return_value=stale),
            patch("api.routes.sync._pipeline_process_active", return_value=False),
            patch("api.routes.sync.progress.finish_run") as mock_finish,
            patch("subprocess.Popen") as mock_popen,
        ):
            r = client.post("/api/v1/sync/run", json={"stage": "full"})
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "started"
            mock_finish.assert_called_once()
            mock_popen.assert_called_once()

    def test_run_pipeline_invalid_stage(self, client):
        r = client.post("/api/v1/sync/run", json={"stage": "invalid"})
        assert r.status_code == 400

    def test_submit_workflows(self, client, mock_svc):
        r = client.post(
            "/api/v1/sync/workflows/submit",
            json={"days_back": 7, "limit": 3},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True
        mock_svc.submit_plaud_workflows.assert_called_once()


class TestSettings:
    def test_get_settings_exposes_autosync_controls(self, authed_client):
        fake_settings = SimpleNamespace(
            chronos_processing_provider="gemini",
            chronos_cleaning_model="gemini-2.5-flash",
            chronos_analyst_model="gemini-2.5-flash",
            chronos_embedding_model="gemini-embedding-2",
            openai_model="gpt-5.5",
            chronos_thinking_level="high",
            openai_temperature=0.7,
            chronos_embedding_dim=768,
            plaud_default_language="en",
            plaud_enable_diarization=True,
            log_level="INFO",
            notion_weekday_start_time="07:30",
            notion_weekend_start_time="12:00",
            qdrant_url="http://localhost:6333",
            qdrant_collection_name="chronos_events",
            chronos_openai_enabled=False,
            chronos_local_llm_enabled=True,
            chronos_local_llm_provider="ollama",
            chronos_local_llm_base_url="http://127.0.0.1:11434",
            chronos_local_llm_model="qwen2.5:0.5b",
            chronos_local_llm_max_context=4096,
            chronos_local_llm_allowed_tasks="json_repair,ask",
            chronos_poll_interval=600,
            chronos_enable_notion_import=True,
            chronos_notion_import_batch_size=6,
            chronos_self_heal_limit=5,
            chronos_embed_batch_size=1,
            chronos_autosync_process_limit=3,
            chronos_autosync_index_limit=4,
            chronos_autosync_graph_limit=3,
            chronos_autosync_max_load_avg=2.0,
            chronos_autosync_min_available_mb=500,
            chronos_autosync_max_swap_used_mb=256,
            chronos_autosync_defer_seconds=60,
            gemini_api_key="test-gemini",
            openai_api_key_configured=False,
            qdrant_api_key=None,
            notion_token=None,
        )

        with patch("api.routes.settings.get_settings", return_value=fake_settings), patch(
            "api.routes.settings.NotionOAuthClient"
        ) as MockNotion:
            MockNotion.return_value.access_token = "notion-oauth-token"
            response = authed_client.get("/api/v1/settings", headers=AUTH_HEADER)

        assert response.status_code == 200
        data = response.json()
        assert data["chronos_poll_interval"] == 600
        assert data["chronos_enable_notion_import"] is True
        assert data["chronos_notion_import_batch_size"] == 6
        assert data["chronos_self_heal_limit"] == 5
        assert data["chronos_embed_batch_size"] == 1
        assert data["chronos_autosync_process_limit"] == 3
        assert data["chronos_autosync_index_limit"] == 4
        assert data["chronos_autosync_graph_limit"] == 3
        assert data["chronos_autosync_max_load_avg"] == 2.0
        assert data["chronos_autosync_min_available_mb"] == 500
        assert data["chronos_autosync_max_swap_used_mb"] == 256
        assert data["chronos_autosync_defer_seconds"] == 60
        assert data["flags"]["has_notion_oauth"] is True

    def test_update_settings_persists_autosync_controls(self, authed_client):
        payload = {
            "chronos_poll_interval": 600,
            "chronos_enable_notion_import": True,
            "chronos_notion_import_batch_size": 6,
            "chronos_self_heal_limit": 5,
            "chronos_embed_batch_size": 1,
            "chronos_autosync_process_limit": 3,
            "chronos_autosync_index_limit": 4,
            "chronos_autosync_graph_limit": 3,
            "chronos_autosync_max_load_avg": 2.0,
            "chronos_autosync_min_available_mb": 500,
            "chronos_autosync_max_swap_used_mb": 256,
            "chronos_autosync_defer_seconds": 60,
        }

        with patch("api.routes.settings._write_env_updates", return_value=12) as mock_write:
            response = authed_client.put(
                "/api/v1/settings", headers=AUTH_HEADER, json=payload
            )

        assert response.status_code == 200
        updates = mock_write.call_args.args[0]
        assert updates["CHRONOS_POLL_INTERVAL"] == "600"
        assert updates["CHRONOS_ENABLE_NOTION_IMPORT"] == "1"
        assert updates["CHRONOS_NOTION_IMPORT_BATCH_SIZE"] == "6"
        assert updates["CHRONOS_SELF_HEAL_LIMIT"] == "5"
        assert updates["CHRONOS_EMBED_BATCH_SIZE"] == "1"
        assert updates["CHRONOS_AUTOSYNC_PROCESS_LIMIT"] == "3"
        assert updates["CHRONOS_AUTOSYNC_INDEX_LIMIT"] == "4"
        assert updates["CHRONOS_AUTOSYNC_GRAPH_LIMIT"] == "3"
        assert updates["CHRONOS_AUTOSYNC_MAX_LOAD_AVG"] == "2.0"
        assert updates["CHRONOS_AUTOSYNC_MIN_AVAILABLE_MB"] == "500"
        assert updates["CHRONOS_AUTOSYNC_MAX_SWAP_USED_MB"] == "256"
        assert updates["CHRONOS_AUTOSYNC_DEFER_SECONDS"] == "60"

    def test_refresh_workflows(self, client, mock_svc):
        r = client.post(
            "/api/v1/sync/workflows/refresh",
            json={"days_back": 30, "limit": 10},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_single_workflow(self, client, mock_svc):
        r = client.post(
            "/api/v1/sync/workflows/rec-001",
            json={"template_id": "summary", "model": "gemini"},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_workflow_status(self, client):
        r = client.get("/api/v1/sync/workflows/rec-001/status")
        assert r.status_code == 200
        data = r.json()
        assert data["recording_id"] == "rec-001"

    def test_db_stats(self, client):
        r = client.get("/api/v1/sync/db-stats")
        assert r.status_code == 200

    def test_sync_failures_hides_archived_by_default(self, client, mock_svc):
        mock_svc.get_sync_failure_summary.return_value = {
            "actionable_count": 0,
            "archived_count": 0,
            "actionable": [],
            "archived": [],
        }

        r = client.get("/api/v1/sync/failures")

        assert r.status_code == 200
        mock_svc.get_sync_failure_summary.assert_called_with(include_archived=False)

    def test_sync_failures_can_include_archived(self, client, mock_svc):
        mock_svc.get_sync_failure_summary.return_value = {
            "actionable_count": 0,
            "archived_count": 2,
            "actionable": [],
            "archived": [{"recording_id": "rec-1", "error": "dead-end"}],
        }

        r = client.get("/api/v1/sync/failures?include_archived=true")

        assert r.status_code == 200
        mock_svc.get_sync_failure_summary.assert_called_with(include_archived=True)

    def test_reset_stuck(self, client, mock_svc):
        r = client.post("/api/v1/sync/reset-stuck")
        assert r.status_code == 200
        assert "Reset 3" in r.json()["message"]

    def test_refresh_cache(self, client, mock_svc):
        r = client.post("/api/v1/sync/refresh-cache")
        assert r.status_code == 200
        mock_svc.refresh_cache.assert_called_once()

    def test_upload_candidates(self, client):
        r = client.get("/api/v1/sync/upload-candidates")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)


# ═══════════════════════════════════════════════════════════
# X-RAY
# ═══════════════════════════════════════════════════════════


class TestXRay:
    def test_get_events(self, client):
        with patch(
            "app_v2.services.xray.get_recent_events",
            return_value=[
                {
                    "seq": 1,
                    "ts": 1700000000.0,
                    "source": "ingest",
                    "op": "fetch",
                    "message": "Fetching recordings",
                    "level": "info",
                }
            ],
        ):
            r = client.get("/api/v1/xray/events")
            assert r.status_code == 200
            data = r.json()
            assert len(data["events"]) == 1
            assert data["latest_seq"] == 1

    def test_get_events_empty(self, client):
        with patch("app_v2.services.xray.get_recent_events", return_value=[]):
            r = client.get("/api/v1/xray/events?since_seq=0&limit=50")
            assert r.status_code == 200
            data = r.json()
            assert data["events"] == []
            assert data["latest_seq"] == 0

    def test_throughput(self, client):
        with patch(
            "app_v2.services.xray.get_throughput",
            return_value=[{"ts": 1700000000, "count": 5}],
        ):
            r = client.get("/api/v1/xray/throughput?buckets=30")
            assert r.status_code == 200

    def test_clear(self, client):
        with patch("app_v2.services.xray.clear_events") as mock_clear:
            r = client.post("/api/v1/xray/clear")
            assert r.status_code == 200
            assert r.json()["success"] is True


# ═══════════════════════════════════════════════════════════
# COSTS
# ═══════════════════════════════════════════════════════════


class TestCosts:
    def test_session_costs(self, client):
        with patch(
            "src.chronos.cost_tracker.get_session_cost",
            return_value={
                "total_cost_usd": 0.05,
                "total_calls": 10,
                "total_input_tokens": 5000,
                "total_output_tokens": 2000,
                "by_model": {},
                "by_type": {},
                "session_minutes": 15.0,
            },
        ):
            r = client.get("/api/v1/costs/session")
            assert r.status_code == 200
            data = r.json()
            assert data["total_cost_usd"] == 0.05
            assert data["total_calls"] == 10

    def test_cost_history(self, client):
        with patch(
            "src.chronos.cost_tracker.get_cost_summary",
            return_value={
                "days": 30,
                "total_cost_usd": 1.25,
                "total_calls": 100,
                "by_model": {},
                "by_day": [],
            },
        ):
            r = client.get("/api/v1/costs/history?days=30")
            assert r.status_code == 200
            data = r.json()
            assert data["total_cost_usd"] == 1.25

    def test_pricing(self, client):
        with patch(
            "src.chronos.cost_tracker.get_model_pricing_table",
            return_value=[{"model": "gemini-3-flash-preview", "input_per_1k": 0.0}],
        ):
            r = client.get("/api/v1/costs/pricing")
            assert r.status_code == 200
            assert "models" in r.json()


# ═══════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════


class TestAuthEndpoints:
    def test_plaud_status(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.token_status = {
                "is_authenticated": True,
                "has_access_token": True,
                "expires_at": "2026-12-31T00:00:00",
            }
            r = client.get("/api/v1/auth/plaud/status")
            assert r.status_code == 200
            data = r.json()
            assert data["is_authenticated"] is True

    def test_plaud_authorize(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.get_authorization_url.return_value = (
                "https://example.com/auth",
                "state123",
            )
            r = client.get("/api/v1/auth/plaud/authorize")
            assert r.status_code == 200
            data = r.json()
            assert "auth_url" in data
            assert data["state"] == "state123"

    def test_plaud_authorize_mobile_uses_api_callback(self, client, monkeypatch):
        monkeypatch.delenv("PLAUD_API_REDIRECT_URI", raising=False)
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.get_authorization_url.return_value = (
                "https://example.com/auth",
                "state123",
            )
            r = client.get("/api/v1/auth/plaud/authorize?mobile=true")
            assert r.status_code == 200
            MockPlaud.assert_called_once_with(
                redirect_uri="http://testserver/api/v1/auth/plaud/callback"
            )

    def test_plaud_token_exchange(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.token_status = {
                "is_authenticated": True,
                "has_access_token": True,
            }
            r = client.post(
                "/api/v1/auth/plaud/token",
                json={"code": "authcode123", "state": "state123"},
            )
            assert r.status_code == 200

    def test_plaud_token_exchange_failure(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.exchange_code_for_token.side_effect = Exception(
                "Invalid code"
            )
            r = client.post(
                "/api/v1/auth/plaud/token",
                json={"code": "bad", "state": "state"},
            )
            assert r.status_code == 400

    def test_plaud_callback_mobile_redirects_to_app(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.get_authorization_url.return_value = (
                "https://example.com/auth",
                "state123",
            )
            r = client.get("/api/v1/auth/plaud/authorize?mobile=true")
            assert r.status_code == 200

            r = client.get(
                "/api/v1/auth/plaud/callback?code=authcode123&state=state123",
                follow_redirects=False,
            )
            assert r.status_code in (302, 307)
            assert r.headers["location"] == "plaudblender://plaud-callback?success=true"

    def test_plaud_callback_web_redirects_back_to_browser(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.get_authorization_url.return_value = (
                "https://example.com/auth",
                "state123",
            )
            r = client.get(
                "/api/v1/auth/plaud/authorize?return_to=http://localhost:8050/settings"
            )
            assert r.status_code == 200

            r = client.get(
                "/api/v1/auth/plaud/callback?code=authcode123&state=state123",
                follow_redirects=False,
            )
            assert r.status_code in (302, 307)
            assert (
                r.headers["location"]
                == "http://localhost:8050/settings?plaud_connected=1"
            )

    def test_plaud_refresh(self, client):
        with patch("src.plaud_oauth.PlaudOAuthClient") as MockPlaud:
            MockPlaud.return_value.token_status = {
                "is_authenticated": True,
                "has_access_token": True,
            }
            r = client.post("/api/v1/auth/plaud/refresh")
            assert r.status_code == 200

    def test_notion_status(self, client):
        with patch("src.notion_oauth.NotionOAuthClient") as MockNotion:
            MockNotion.return_value.token_status = {
                "is_authenticated": True,
                "workspace_name": "My Workspace",
                "workspace_id": "ws-123",
            }
            MockNotion.return_value.access_token = "abc"
            r = client.get("/api/v1/auth/notion/status")
            assert r.status_code == 200
            data = r.json()
            assert data["is_authenticated"] is True

    def test_notion_authorize(self, client):
        with patch("src.notion_oauth.NotionOAuthClient") as MockNotion:
            MockNotion.return_value.get_authorization_url.return_value = (
                "https://notion.so/auth",
                "nonce",
            )
            r = client.get("/api/v1/auth/notion/authorize")
            assert r.status_code == 200

    def test_notion_web_authorize_uses_api_callback(self, client, monkeypatch):
        monkeypatch.delenv("NOTION_REDIRECT_URI", raising=False)
        with patch("src.notion_oauth.NotionOAuthClient") as MockNotion:
            MockNotion.return_value.get_authorization_url.return_value = (
                "https://notion.so/auth",
                "nonce",
            )
            r = client.get("/api/v1/auth/notion/web-authorize", follow_redirects=False)
            assert r.status_code in (302, 307)
            MockNotion.assert_called_once_with(
                redirect_uri="http://testserver/api/v1/auth/notion/callback"
            )

    def test_notion_web_authorize_tracks_return_to(self, client):
        from api.routes import auth as auth_routes

        with patch("src.notion_oauth.NotionOAuthClient") as MockNotion:
            MockNotion.return_value.get_authorization_url.return_value = (
                "https://notion.so/auth",
                "nonce-state",
            )
            r = client.get(
                "/api/v1/auth/notion/web-authorize",
                params={"return_to": "https://dash.example/system"},
                follow_redirects=False,
            )
            assert r.status_code in (302, 307)
            assert auth_routes._notion_oauth_pending["nonce-state"] == {
                "source": "web",
                "return_to": "https://dash.example/system",
            }

    def test_notion_callback_redirects_back_to_originating_web_host(self, client):
        from api.routes import auth as auth_routes

        auth_routes._notion_oauth_pending["nonce-state"] = {
            "source": "web",
            "return_to": "https://dash.example/system?tab=notion",
        }
        with patch("src.notion_oauth.NotionOAuthClient") as MockNotion:
            r = client.get(
                "/api/v1/auth/notion/callback",
                params={"code": "code123", "state": "nonce-state"},
                follow_redirects=False,
            )
            assert r.status_code in (302, 307)
            assert (
                r.headers["location"]
                == "https://dash.example/system?tab=notion&notion_connected=1"
            )

    def test_notion_token_exchange(self, client):
        with patch("src.notion_oauth.NotionOAuthClient") as MockNotion:
            MockNotion.return_value.token_status = {
                "is_authenticated": True,
            }
            MockNotion.return_value.access_token = "token"
            r = client.post("/api/v1/auth/notion/token", json={"code": "code123"})
            assert r.status_code == 200


# ═══════════════════════════════════════════════════════════
# NOTION
# ═══════════════════════════════════════════════════════════


class TestNotion:
    def test_notion_status(self, client):
        with patch("api.routes.notion._get_notion_service") as mock_ns:
            status = MagicMock()
            status.connected = True
            status.total_pages = 42
            status.database_title = "Recordings"
            status.error = None
            mock_ns.return_value.check_connection.return_value = status
            r = client.get("/api/v1/notion/status")
            assert r.status_code == 200
            data = r.json()
            assert data["is_connected"] is True
            assert data["page_count"] == 42

    def test_list_databases(self, client):
        with patch("api.routes.notion._get_notion_service") as mock_ns:
            mock_ns.return_value.list_databases.return_value = [
                {"id": "db1", "title": "My DB"}
            ]
            r = client.get("/api/v1/notion/databases")
            assert r.status_code == 200

    def test_select_database(self, client):
        with patch("api.routes.notion._get_notion_service") as mock_ns:
            r = client.post(
                "/api/v1/notion/databases/select", json={"db_id": "db-uuid"}
            )
            assert r.status_code == 200
            assert r.json()["success"] is True

    def test_list_recordings(self, client):
        with patch("api.routes.notion._get_notion_service") as mock_ns:
            page = MagicMock()
            page.page_id = "page-001"
            page.title = "Meeting Notes"
            page.created_time = "2026-01-15"
            page.last_edited_time = None
            page.url = None
            page.transcript = "Some transcript"
            page.summary = None
            page.date = "2026-01-15"
            page.duration = "10:00"
            page.tags = ["meeting"]
            page.category = "meeting"
            page.matched_recording_id = None
            mock_ns.return_value.fetch_recordings.return_value = [page]
            r = client.get("/api/v1/notion/recordings")
            assert r.status_code == 200
            data = r.json()
            assert len(data["recordings"]) == 1
            assert data["recordings"][0]["page_id"] == "page-001"

    def test_import(self, client):
        with patch(
            "api.routes.notion._build_notion_import_preview",
            return_value={"pending_import": 0},
        ):
            with patch("src.chronos.notion_bridge.import_all_unmatched") as mock_import:
                with patch("src.database.SessionLocal"):
                    mock_import.return_value = (5, 2, [])
                    r = client.post(
                        "/api/v1/notion/import", json={"process": True, "index": True}
                    )
                    assert r.status_code == 200
                    assert "Imported 5" in r.json()["message"]

    def test_import_progress(self, client):
        with patch(
            "src.chronos.notion_bridge.get_import_progress",
            return_value={"status": "running", "progress": 0.5},
        ):
            r = client.get("/api/v1/notion/import/progress")
            assert r.status_code == 200

    def test_coverage(self, client):
        with patch("src.chronos.notion_bridge.get_coverage_calendar") as mock_cov:
            with patch("src.database.SessionLocal"):
                mock_cov.return_value = {
                    "2026-01-15": {"notion": True, "chronos": True}
                }
                r = client.get("/api/v1/notion/coverage")
                assert r.status_code == 200


# ═══════════════════════════════════════════════════════════
# ROUTE COUNT VALIDATION
# ═══════════════════════════════════════════════════════════


class TestRouteCount:
    def test_all_routes_registered(self, client):
        """Verify all expected API routes are registered."""
        from api.main import app

        routes = []
        # The test app instance might not have all routes initialized in this test context
        # if the app was modified or instantiated differently. We use client.app to get the fully initialized app.
        for r in client.app.routes:
            if hasattr(r, "path") and hasattr(r, "methods"):
                routes.append(r.path)
            # handle _IncludedRouter in newer FastAPI versions
            if hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
                for sr in r.original_router.routes:
                    if hasattr(sr, "path") and hasattr(sr, "methods"):
                        routes.append(sr.path)
        # Spot-check key routes
        assert "/api/v1/health" in routes
        assert "/api/v1/timeline/days" in routes
        assert "/api/v1/recordings/{recording_id}" in routes
        assert "/api/v1/search" in routes
        assert "/api/v1/topics" in routes
        assert "/api/v1/graph" in routes
        assert "/api/v1/stats" in routes
        assert "/api/v1/sync/status" in routes
        assert "/api/v1/xray/events" in routes
        assert "/api/v1/costs/session" in routes
        assert "/api/v1/auth/plaud/status" in routes
        assert "/api/v1/notion/status" in routes
        assert "/api/v1/admin/runtime" in routes
        # Should have at least 40 routes
        assert len(routes) >= 40

    def test_admin_runtime_shape(self, client):
        r = client.get("/api/v1/admin/runtime")
        assert r.status_code == 200
        data = r.json()
        assert "captured_at" in data
        assert "runtime_health" in data
        assert "services" in data
        assert "ports" in data
        assert "plaud_auth" in data
