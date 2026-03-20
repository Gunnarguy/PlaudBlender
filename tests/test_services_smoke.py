"""
Service Smoke Tests.
Tests for core Chronos services and database operations.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===========================================================================
# Chronos Services Tests
# ===========================================================================


class TestChronosQdrantClient:
    """Tests for src/chronos/qdrant_client.py."""

    def test_qdrant_client_import(self):
        """Verify ChronosQdrantClient can be imported."""
        from src.chronos.qdrant_client import ChronosQdrantClient

        assert ChronosQdrantClient is not None

    def test_qdrant_client_has_methods(self):
        """Verify ChronosQdrantClient has expected methods."""
        from src.chronos.qdrant_client import ChronosQdrantClient

        assert hasattr(ChronosQdrantClient, "create_collection")
        assert hasattr(ChronosQdrantClient, "upsert_event")
        assert hasattr(ChronosQdrantClient, "upsert_events_batch")
        assert hasattr(ChronosQdrantClient, "search")


class TestChronosEmbeddingService:
    """Tests for src/chronos/embedding_service.py."""

    def test_embedding_service_import(self):
        """Verify ChronosEmbeddingService can be imported."""
        from src.chronos.embedding_service import ChronosEmbeddingService

        assert ChronosEmbeddingService is not None

    def test_embedding_service_has_methods(self):
        """Verify ChronosEmbeddingService has expected methods."""
        from src.chronos.embedding_service import ChronosEmbeddingService

        assert hasattr(ChronosEmbeddingService, "embed_text")
        assert hasattr(ChronosEmbeddingService, "embed_batch")


class TestChronosIngestService:
    """Tests for src/chronos/ingest_service.py."""

    def test_ingest_service_import(self):
        """Verify ChronosIngestService can be imported."""
        from src.chronos.ingest_service import ChronosIngestService

        assert ChronosIngestService is not None


class TestChronosTranscriptProcessor:
    """Tests for src/chronos/transcript_processor.py."""

    def test_transcript_processor_import(self):
        """Verify TranscriptProcessor can be imported."""
        from src.chronos.transcript_processor import TranscriptProcessor

        assert TranscriptProcessor is not None


class TestChronosGraphService:
    """Tests for src/chronos/graph_service.py."""

    def test_graph_service_import(self):
        """Verify ChronosGraphExtractor can be imported."""
        from src.chronos.graph_service import ChronosGraphExtractor

        assert ChronosGraphExtractor is not None


# ===========================================================================
# Processing Engine Tests
# ===========================================================================


class TestProcessingEngine:
    """Tests for src/processing/engine.py."""

    def test_engine_import(self):
        """Verify processing engine can be imported."""
        from src.processing.engine import process_pending_recordings, ChunkingConfig

        assert process_pending_recordings is not None
        assert ChunkingConfig is not None

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig has sensible defaults."""
        from src.processing.engine import ChunkingConfig

        cfg = ChunkingConfig()
        assert cfg.max_words > 0
        assert cfg.overlap_words >= 0


class TestProcessingIndexer:
    """Tests for src/processing/indexer.py."""

    def test_indexer_import(self):
        """Verify indexer can be imported."""
        from src.processing.indexer import index_pending_segments

        assert index_pending_segments is not None


# ===========================================================================
# Plaud Client Tests
# ===========================================================================


class TestPlaudClient:
    """Tests for src/plaud_client.py."""

    def test_plaud_client_import(self):
        """Verify PlaudClient can be imported."""
        from src.plaud_client import PlaudClient

        assert PlaudClient is not None

    def test_plaud_client_has_methods(self):
        """Verify PlaudClient has expected methods."""
        from src.plaud_client import PlaudClient

        assert hasattr(PlaudClient, "list_recordings")
        assert hasattr(PlaudClient, "get_recording")
        assert hasattr(PlaudClient, "get_transcript")
        assert hasattr(PlaudClient, "get_user")


class TestPlaudOAuth:
    """Tests for src/plaud_oauth.py."""

    def test_plaud_oauth_import(self):
        """Verify PlaudOAuthClient can be imported."""
        from src.plaud_oauth import PlaudOAuthClient

        assert PlaudOAuthClient is not None

    def test_exchange_includes_state_in_body(self):
        """Token exchange must include 'state' in POST body — Plaud requires it."""
        from unittest.mock import patch, MagicMock
        from src.plaud_oauth import PlaudOAuthClient

        with patch.dict(
            "os.environ",
            {
                "PLAUD_CLIENT_ID": "test_client_id",
                "PLAUD_CLIENT_SECRET": "test_client_secret",
            },
        ):
            client = PlaudOAuthClient(
                redirect_uri="https://localhost:8050/auth/plaud/callback"
            )

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.reason = "OK"
        mock_resp.text = (
            '{"access_token":"tok","refresh_token":"ref","expires_in":3600}'
        )
        mock_resp.json.return_value = {
            "access_token": "tok",
            "refresh_token": "ref",
            "expires_in": 3600,
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("src.plaud_oauth.requests.post", return_value=mock_resp) as mock_post,
            patch.object(client, "_save_tokens"),
        ):
            client.exchange_code_for_token("test_code", state="test_state_abc")

        # Verify state was included in the POST body
        call_kwargs = mock_post.call_args
        posted_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
        assert posted_data["state"] == "test_state_abc"
        assert posted_data["code"] == "test_code"
        assert "redirect_uri" in posted_data

    def test_exchange_without_state_omits_it(self):
        """When state is None, it should not appear in POST body."""
        from unittest.mock import patch, MagicMock
        from src.plaud_oauth import PlaudOAuthClient

        with patch.dict(
            "os.environ",
            {
                "PLAUD_CLIENT_ID": "test_client_id",
                "PLAUD_CLIENT_SECRET": "test_client_secret",
            },
        ):
            client = PlaudOAuthClient(
                redirect_uri="https://localhost:8050/auth/plaud/callback"
            )

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.reason = "OK"
        mock_resp.text = (
            '{"access_token":"tok","refresh_token":"ref","expires_in":3600}'
        )
        mock_resp.json.return_value = {
            "access_token": "tok",
            "refresh_token": "ref",
            "expires_in": 3600,
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("src.plaud_oauth.requests.post", return_value=mock_resp) as mock_post,
            patch.object(client, "_save_tokens"),
        ):
            client.exchange_code_for_token("test_code")

        posted_data = mock_post.call_args.kwargs.get("data") or mock_post.call_args[
            1
        ].get("data")
        assert "state" not in posted_data

    def test_exchange_uses_basic_auth(self):
        """Token exchange must use Basic auth with base64(client_id:client_secret)."""
        import base64
        from unittest.mock import patch, MagicMock
        from src.plaud_oauth import PlaudOAuthClient

        with patch.dict(
            "os.environ",
            {
                "PLAUD_CLIENT_ID": "test_client_id",
                "PLAUD_CLIENT_SECRET": "test_client_secret",
            },
        ):
            client = PlaudOAuthClient(
                redirect_uri="https://localhost:8050/auth/plaud/callback"
            )

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.reason = "OK"
        mock_resp.text = (
            '{"access_token":"tok","refresh_token":"ref","expires_in":3600}'
        )
        mock_resp.json.return_value = {
            "access_token": "tok",
            "refresh_token": "ref",
            "expires_in": 3600,
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("src.plaud_oauth.requests.post", return_value=mock_resp) as mock_post,
            patch.object(client, "_save_tokens"),
        ):
            client.exchange_code_for_token("test_code", state="s")

        headers = mock_post.call_args.kwargs.get("headers") or mock_post.call_args[
            1
        ].get("headers")
        expected_b64 = base64.b64encode(b"test_client_id:test_client_secret").decode()
        assert headers["Authorization"] == f"Basic {expected_b64}"

    def test_get_authorization_url_includes_state(self):
        """Auth URL must include state parameter for CSRF + server-side validation."""
        from unittest.mock import patch
        from src.plaud_oauth import PlaudOAuthClient

        with patch.dict(
            "os.environ",
            {
                "PLAUD_CLIENT_ID": "test_client_id",
                "PLAUD_CLIENT_SECRET": "test_client_secret",
            },
        ):
            client = PlaudOAuthClient(
                redirect_uri="https://localhost:8050/auth/plaud/callback"
            )

        url, state = client.get_authorization_url()
        assert "state=" in url
        assert state in url
        assert len(state) > 16  # secrets.token_urlsafe(32) is ~43 chars


class TestPlaudWorkflow:
    """Tests for src/plaud_workflow.py."""

    def test_plaud_workflow_import(self):
        """Verify PlaudWorkflowClient can be imported."""
        from src.plaud_workflow import PlaudWorkflowClient

        assert PlaudWorkflowClient is not None


class TestPlaudDevice:
    """Tests for src/plaud_device.py."""

    def test_plaud_device_import(self):
        """Verify PlaudDeviceManager can be imported."""
        from src.plaud_device import PlaudDeviceManager

        assert PlaudDeviceManager is not None


class TestPlaudWebhook:
    """Tests for src/plaud_webhook.py."""

    def test_plaud_webhook_import(self):
        """Verify webhook handler can be imported."""
        from src.plaud_webhook import PlaudWebhookHandler

        assert PlaudWebhookHandler is not None


# ===========================================================================
# Database Tests
# ===========================================================================


class TestDatabaseModels:
    """Tests for src/database/models.py."""

    def test_models_import(self):
        """Verify database models can be imported."""
        from src.database.models import Base, Recording, Segment

        assert Base is not None
        assert Recording is not None
        assert Segment is not None


class TestDatabaseRepository:
    """Tests for src/database/repository.py."""

    def test_repository_import(self):
        """Verify repository functions can be imported."""
        from src.database.repository import upsert_recording, add_segments

        assert upsert_recording is not None
        assert add_segments is not None


class TestChronosRepository:
    """Tests for src/database/chronos_repository.py."""

    def test_chronos_repository_import(self):
        """Verify chronos repository can be imported."""
        from src.database.chronos_repository import (
            upsert_chronos_recording,
            get_chronos_recording,
        )

        assert upsert_chronos_recording is not None
        assert get_chronos_recording is not None


# ===========================================================================
# Integration Test: Full Pipeline Smoke
# ===========================================================================


class TestPipelineSmoke:
    """End-to-end smoke tests for the processing pipeline."""

    def test_full_import_chain(self):
        """Verify all major components can be imported together."""
        # Database
        from src.database.models import Base, Recording, Segment
        from src.database.repository import upsert_recording, add_segments

        # Chronos core
        from src.chronos.qdrant_client import ChronosQdrantClient
        from src.chronos.embedding_service import ChronosEmbeddingService
        from src.chronos.transcript_processor import TranscriptProcessor
        from src.chronos.ingest_service import ChronosIngestService
        from src.chronos.graph_service import ChronosGraphExtractor

        # Processing (legacy)
        from src.processing.engine import process_pending_recordings, ChunkingConfig
        from src.processing.indexer import index_pending_segments

        # Plaud
        from src.plaud_client import PlaudClient
        from src.plaud_oauth import PlaudOAuthClient
        from src.plaud_workflow import PlaudWorkflowClient
        from src.plaud_device import PlaudDeviceManager

        # Models
        from src.models.schemas import RecordingSchema
        from src.models.chronos_schemas import ChronosEvent, TemporalFilter

        # All imports succeeded
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
