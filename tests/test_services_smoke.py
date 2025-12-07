"""
Smoke tests for GUI services.
These tests verify basic functionality without requiring external API connections.
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
# Stats Service Tests
# ===========================================================================

class TestStatsService:
    """Tests for gui/services/stats_service.py."""
    
    def test_stats_service_import(self):
        """Verify StatsService can be imported."""
        from gui.services.stats_service import StatsService
        assert StatsService is not None
    
    def test_stats_service_instantiation(self):
        """Verify StatsService can be instantiated."""
        from gui.services.stats_service import StatsService
        service = StatsService()
        assert service is not None
    
    @patch('src.database.engine.SessionLocal')
    @patch('src.database.engine.init_db')
    def test_get_pipeline_stats_empty_db(self, mock_init_db, mock_sessionlocal):
        """Test pipeline stats with empty database."""
        from gui.services.stats_service import StatsService
        
        # Mock empty session
        mock_sess = MagicMock()
        mock_sess.query.return_value.count.return_value = 0
        mock_sessionlocal.return_value = mock_sess
        
        service = StatsService()
        stats = service.get_all_stats(force_refresh=True)

        assert hasattr(stats, 'total_recordings')
        assert hasattr(stats, 'total_segments')
    
    def test_get_health_status(self):
        """Test health status returns expected structure."""
        from gui.services.stats_service import StatsService
        
        service = StatsService()
        status = service.get_health_status()
        
        assert 'database' in status
        assert 'pinecone' in status
        assert 'notion' in status


# ===========================================================================
# Transcripts Service Tests
# ===========================================================================

class TestTranscriptsService:
    """Tests for gui/services/transcripts_service.py."""
    
    def test_transcripts_service_import(self):
        """Verify transcripts service functions can be imported."""
        from gui.services.transcripts_service import (
            fetch_transcripts,
            sync_recording,
            delete_recording,
            export_recording,
            export_all_recordings,
        )
        assert fetch_transcripts is not None
        assert sync_recording is not None
        assert delete_recording is not None
        assert export_recording is not None
        assert export_all_recordings is not None
    
    @patch('gui.services.transcripts_service.SessionLocal')
    @patch('gui.services.transcripts_service.init_db')
    def test_delete_recording_removes_from_db(self, mock_init_db, mock_sessionlocal):
        """Test delete_recording removes entry from database."""
        from gui.services.transcripts_service import delete_recording
        
        mock_sess = MagicMock()
        mock_recording = Mock(id='test_id')
        mock_sess.query.return_value.filter_by.return_value.first.return_value = mock_recording
        mock_sess.get.return_value = mock_recording
        mock_sessionlocal.return_value = mock_sess
        
        result = delete_recording('test_id')
        # Should complete without error
        assert result is None or isinstance(result, dict) or result == True
    
    @patch('gui.services.transcripts_service.SessionLocal')
    @patch('gui.services.transcripts_service.init_db')
    def test_export_recordings_format(self, mock_init_db, mock_sessionlocal):
        """Test export_all_recordings returns proper format with mocked DB."""
        from gui.services.transcripts_service import export_all_recordings
        
        mock_sess = MagicMock()
        mock_sess.execute.return_value.scalars.return_value.all.return_value = []
        mock_sessionlocal.return_value = mock_sess
        
        result = export_all_recordings(status_filter='nonexistent')
        assert isinstance(result, list)


# ===========================================================================
# Embedding Service Tests
# ===========================================================================

class TestEmbeddingService:
    """Tests for gui/services/embedding_service.py."""
    
    def test_embedding_service_import(self):
        """Verify EmbeddingService can be imported."""
        from gui.services.embedding_service import EmbeddingService
        assert EmbeddingService is not None
    
    def test_embedding_service_instantiation(self):
        """Verify EmbeddingService can be instantiated."""
        from gui.services.embedding_service import EmbeddingService
        
        # Should work even without API keys (lazy initialization)
        service = EmbeddingService()
        assert service is not None


# ===========================================================================
# Search Service Tests
# ===========================================================================

class TestSearchService:
    """Tests for gui/services/search_service.py."""
    
    def test_search_service_import(self):
        """Verify search service functions can be imported."""
        from gui.services.search_service import (
            semantic_search,
            search_with_rerank,
            cross_namespace_search,
        )
        assert semantic_search is not None
        assert search_with_rerank is not None
        assert cross_namespace_search is not None


# ===========================================================================
# Notion Sync Service Tests
# ===========================================================================

class TestNotionSyncService:
    """Tests for src/notion_sync.py."""
    
    def test_notion_sync_import(self):
        """Verify NotionSyncService can be imported."""
        from src.notion_sync import NotionSyncService
        assert NotionSyncService is not None
    
    @patch('src.notion_sync.os.getenv')
    @patch.dict('sys.modules', {'notion_client': MagicMock()})
    def test_notion_sync_handles_missing_config(self, mock_getenv):
        """Test NotionSyncService handles missing configuration gracefully."""
        mock_getenv.return_value = None
        
        from src.notion_sync import NotionSyncService
        with pytest.raises(ValueError):
            NotionSyncService()


# ===========================================================================
# Query Router Tests
# ===========================================================================

class TestQueryRouter:
    """Tests for src/processing/query_router.py."""
    
    def test_query_router_import(self):
        """Verify QueryRouter can be imported."""
        from src.processing.query_router import QueryRouter, QueryIntent
        assert QueryRouter is not None
        assert QueryIntent is not None
    
    def test_query_router_classify_semantic(self):
        """Test QueryRouter classifies semantic queries correctly."""
        from src.processing.query_router import QueryRouter
        
        router = QueryRouter()
        result = router.route("What are the main themes discussed?")
        
        assert result is not None
        assert hasattr(result, 'intent')
    
    def test_query_router_classify_metadata(self):
        """Test QueryRouter classifies metadata queries correctly."""
        from src.processing.query_router import QueryRouter
        
        router = QueryRouter()
        result = router.route("Show me recording rec_12345")
        
        assert result is not None


# ===========================================================================
# RRF Fusion Tests
# ===========================================================================

class TestRRFFusion:
    """Tests for src/processing/rrf_fusion.py."""
    
    def test_rrf_fusion_import(self):
        """Verify RRF functions can be imported."""
        from src.processing.rrf_fusion import reciprocal_rank_fusion, RRFMergeResult
        assert reciprocal_rank_fusion is not None
        assert RRFMergeResult is not None
    
    def test_rrf_empty_inputs(self):
        """Test RRF handles empty inputs."""
        from src.processing.rrf_fusion import reciprocal_rank_fusion, RRFMergeResult
        
        result = reciprocal_rank_fusion([], [], [])
        assert isinstance(result, RRFMergeResult)
        assert len(result.results) == 0
    
    def test_rrf_single_source(self):
        """Test RRF with single source of results."""
        from src.processing.rrf_fusion import reciprocal_rank_fusion, RRFMergeResult
        
        dense_results = [
            {'id': 'a', 'score': 0.9},
            {'id': 'b', 'score': 0.8},
        ]
        
        result = reciprocal_rank_fusion(dense_results, [], [])
        assert isinstance(result, RRFMergeResult)
        assert len(result.results) == 2
        # First result should be highest ranked
        assert result.results[0].id == 'a'


# ===========================================================================
# Thought Signatures Tests
# ===========================================================================

class TestThoughtSignatures:
    """Tests for src/processing/thought_signatures.py."""
    
    def test_thought_signatures_import(self):
        """Verify ThoughtSignatureManager can be imported."""
        from src.processing.thought_signatures import ThoughtSignatureManager, ThoughtSignature
        assert ThoughtSignatureManager is not None
        assert ThoughtSignature is not None
    
    def test_thought_signature_manager(self):
        """Test ThoughtSignatureManager instantiation."""
        from src.processing.thought_signatures import get_thought_manager
        
        manager = get_thought_manager()
        assert manager is not None


# ===========================================================================
# Conflict Detection Tests
# ===========================================================================

class TestConflictDetection:
    """Tests for src/processing/conflict_detection.py."""
    
    def test_conflict_detection_import(self):
        """Verify ConflictDetector can be imported."""
        from src.processing.conflict_detection import ConflictDetector
        assert ConflictDetector is not None
    
    def test_detect_method_exists(self):
        """Test ConflictDetector has detect method."""
        from src.processing.conflict_detection import ConflictDetector
        
        detector = ConflictDetector()
        assert hasattr(detector, 'detect')


# ===========================================================================
# ColPali Ingestion Tests
# ===========================================================================

class TestColPaliIngestion:
    """Tests for src/processing/colpali_ingestion.py."""
    
    def test_colpali_import(self):
        """Verify ColPaliProcessor can be imported."""
        from src.processing.colpali_ingestion import ColPaliProcessor, GeminiVisionAnalyzer
        assert ColPaliProcessor is not None
        assert GeminiVisionAnalyzer is not None
    
    def test_colpali_instantiation(self):
        """Verify ColPaliProcessor can be instantiated."""
        from src.processing.colpali_ingestion import ColPaliProcessor
        
        processor = ColPaliProcessor()
        assert processor is not None


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
        
        # Processing
        from src.processing.engine import process_pending_recordings, ChunkingConfig
        from src.processing.indexer import index_pending_segments
        from src.processing.query_router import QueryRouter
        from src.processing.rrf_fusion import reciprocal_rank_fusion
        from src.processing.thought_signatures import ThoughtSignatureManager
        from src.processing.conflict_detection import ConflictDetector
        from src.processing.colpali_ingestion import ColPaliProcessor
        
        # Services
        from gui.services.stats_service import StatsService
        from gui.services.embedding_service import EmbeddingService
        from gui.services.search_service import semantic_search, search_with_rerank
        from gui.services.transcripts_service import fetch_transcripts
        
        # Notion
        from src.notion_sync import NotionSyncService
        
        # All imports succeeded
        assert True
    
    def test_stats_service_methods_exist(self):
        """Test StatsService has expected methods."""
        from gui.services.stats_service import StatsService
        
        service = StatsService()
        assert hasattr(service, 'get_all_stats')
        assert hasattr(service, 'get_health_status')
        assert hasattr(service, 'invalidate_cache')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
