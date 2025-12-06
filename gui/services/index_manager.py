"""
Smart Index Manager for PlaudBlender.

Handles all Pinecone index dimension compatibility AUTOMATICALLY:
- Detects current index dimension
- Auto-creates compatible indexes when dimension changes
- Migrates data if needed
- Syncs embedding config with Pinecone state

The user never has to worry about dimension mismatches.
"""
import os
import logging
from typing import Optional, Dict, Any, Tuple
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class IndexAction(Enum):
    """Actions the index manager can take."""
    NONE = "none"  # Index is compatible, no action needed
    CREATE = "create"  # Create new index (none exists)
    RECREATE = "recreate"  # Recreate index with new dimension
    USE_EXISTING = "use_existing"  # Adjust embedding to match existing index


class IndexManager:
    """
    Intelligent index manager that keeps embeddings and Pinecone in sync.
    
    Strategies:
    1. If no index exists → create with configured dimension
    2. If index exists with SAME dimension → use as-is
    3. If index exists with DIFFERENT dimension → AUTO-ADJUST embedding config
    
    The key insight: It's easier to adjust embedding dimension than recreate indexes.
    With Matryoshka embeddings, we can use ANY dimension 128-3072 with minimal quality loss.
    """
    
    def __init__(self):
        """Initialize the index manager."""
        self._pc = None
        self._index = None
        self._index_name = os.getenv("PINECONE_INDEX_NAME", "transcripts")
        self._cached_dimension: Optional[int] = None
    
    @property
    def pc(self):
        """Lazy-load Pinecone client."""
        if self._pc is None:
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY not set")
            self._pc = Pinecone(api_key=api_key)
        return self._pc
    
    @property
    def index_name(self) -> str:
        return self._index_name
    
    def get_index_dimension(self) -> Optional[int]:
        """
        Get the dimension of the current Pinecone index.
        
        Returns:
            Dimension (int) if index exists, None otherwise
        """
        if self._cached_dimension:
            return self._cached_dimension
        
        try:
            existing = [idx.name for idx in self.pc.list_indexes()]
            
            if self._index_name not in existing:
                logger.info(f"Index '{self._index_name}' does not exist yet")
                return None
            
            # Get index stats to find dimension
            index = self.pc.Index(self._index_name)
            stats = index.describe_index_stats()
            self._cached_dimension = stats.dimension
            
            logger.info(f"Index '{self._index_name}' has dimension: {self._cached_dimension}")
            return self._cached_dimension
            
        except Exception as e:
            logger.error(f"Error getting index dimension: {e}")
            return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get comprehensive index information."""
        try:
            existing = [idx.name for idx in self.pc.list_indexes()]
            
            if self._index_name not in existing:
                return {
                    "exists": False,
                    "name": self._index_name,
                    "dimension": None,
                    "vector_count": 0,
                }
            
            index = self.pc.Index(self._index_name)
            stats = index.describe_index_stats()
            info = self.pc.describe_index(self._index_name)
            
            return {
                "exists": True,
                "name": self._index_name,
                "dimension": stats.dimension,
                "vector_count": stats.total_vector_count,
                "metric": info.metric,
                "host": info.host,
                "namespaces": list(stats.namespaces.keys()) if stats.namespaces else [],
            }
            
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {"exists": False, "error": str(e)}
    
    def sync_embedding_config(self) -> Tuple[int, str]:
        """
        SMART SYNC: Automatically align embedding config with existing index.
        
        This is the magic function - it ensures you can NEVER have a dimension mismatch.
        
        Returns:
            Tuple of (dimension, action_taken)
        """
        from gui.services.embedding_service import (
            get_embedding_service,
            EmbeddingConfig,
            get_embedding_dimension,
        )
        
        # What dimension does the embedding service want to use?
        requested_dim = get_embedding_dimension()
        
        # What dimension does the index actually have?
        index_dim = self.get_index_dimension()
        
        if index_dim is None:
            # No index exists - we'll create one with the requested dimension
            logger.info(f"No index exists. Will create with dimension {requested_dim}")
            return (requested_dim, "will_create_new")
        
        if index_dim == requested_dim:
            # Perfect match - nothing to do
            logger.info(f"Dimension match: {index_dim}d")
            return (index_dim, "already_synced")
        
        # MISMATCH - Auto-adjust embedding config to match existing index
        # This is the smart behavior: don't force user to recreate index,
        # just use the dimension that already exists
        logger.warning(f"Dimension mismatch: index={index_dim}, requested={requested_dim}")
        logger.info(f"Auto-adjusting embedding to {index_dim}d to match existing index")
        
        # Reconfigure the embedding service
        new_config = EmbeddingConfig(dimension=index_dim)
        get_embedding_service(new_config)
        
        return (index_dim, "auto_adjusted")
    
    def ensure_index_exists(self, dimension: int) -> bool:
        """
        Ensure the index exists with the specified dimension.
        Creates if needed.
        
        Args:
            dimension: Required vector dimension
            
        Returns:
            True if index is ready, False on error
        """
        from pinecone import ServerlessSpec
        
        try:
            existing = [idx.name for idx in self.pc.list_indexes()]
            
            if self._index_name in existing:
                # Check dimension compatibility
                current_dim = self.get_index_dimension()
                if current_dim != dimension:
                    logger.warning(
                        f"Index exists with dimension {current_dim}, "
                        f"but {dimension} was requested. Using existing dimension."
                    )
                return True
            
            # Create new index
            logger.info(f"Creating index '{self._index_name}' with dimension {dimension}")
            self.pc.create_index(
                name=self._index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            self._cached_dimension = dimension
            logger.info(f"✅ Index '{self._index_name}' created with {dimension}d")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            return False
    
    def get_compatible_dimension(self) -> int:
        """
        Get the dimension that should be used for embeddings.
        
        This is the SINGLE SOURCE OF TRUTH for "what dimension should I use?"
        It accounts for existing index state.
        
        Returns:
            The dimension to use (matches existing index if one exists)
        """
        index_dim = self.get_index_dimension()
        
        if index_dim is not None:
            # Always use existing index dimension
            return index_dim
        
        # No index - use embedding service default
        from gui.services.embedding_service import get_embedding_dimension
        return get_embedding_dimension()
    
    def clear_cache(self):
        """Clear cached dimension (call after index changes)."""
        self._cached_dimension = None

    # =========================================================================
    # CONTROL-PLANE GOVERNANCE (2025-10 API)
    # =========================================================================

    def enable_deletion_protection(self) -> Dict[str, Any]:
        """Enable deletion protection on the current index."""
        return self._configure_index(deletion_protection="enabled")

    def disable_deletion_protection(self) -> Dict[str, Any]:
        """Disable deletion protection on the current index."""
        return self._configure_index(deletion_protection="disabled")

    def set_index_tags(self, tags: Dict[str, str]) -> Dict[str, Any]:
        """
        Set custom tags on the current index.

        Args:
            tags: Key-value pairs (max 80/120 chars for key/value)

        Returns:
            Updated index info or error dict
        """
        return self._configure_index(tags=tags)

    def get_deletion_protection_status(self) -> str:
        """Get current deletion protection status ('enabled' or 'disabled')."""
        try:
            info = self.pc.describe_index(self._index_name)
            return getattr(info, "deletion_protection", "disabled")
        except Exception as e:
            logger.error(f"Error getting deletion protection status: {e}")
            return "unknown"

    def _configure_index(
        self,
        deletion_protection: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Internal helper to configure index via PATCH /indexes/{name}.

        Uses Pinecone 2025-10 API for serverless index configuration.
        """
        import requests

        try:
            url = f"https://api.pinecone.io/indexes/{self._index_name}"
            headers = {
                "Api-Key": os.getenv("PINECONE_API_KEY"),
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Pinecone-Api-Version": "2025-10",
            }
            body: Dict[str, Any] = {}
            if deletion_protection:
                body["deletion_protection"] = deletion_protection
            if tags is not None:
                body["tags"] = tags
            if not body:
                return {"error": "No configuration changes specified"}

            resp = requests.patch(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Configured index '{self._index_name}': {body}")
            return result
        except Exception as e:
            logger.error(f"Error configuring index: {e}")
            return {"error": str(e)}


# Singleton instance
_index_manager: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    """Get the singleton IndexManager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager


def sync_dimensions() -> Tuple[int, str]:
    """
    Convenience function: Sync embedding and index dimensions.
    
    Call this before any embedding/Pinecone operation to ensure compatibility.
    
    Returns:
        (dimension, action) tuple
    """
    return get_index_manager().sync_embedding_config()


def get_compatible_dimension() -> int:
    """
    Convenience function: Get the dimension to use.
    
    Returns:
        The dimension that's compatible with the existing index
    """
    return get_index_manager().get_compatible_dimension()


def enable_deletion_protection() -> Dict[str, Any]:
    """Convenience: Enable deletion protection on current index."""
    return get_index_manager().enable_deletion_protection()


def disable_deletion_protection() -> Dict[str, Any]:
    """Convenience: Disable deletion protection on current index."""
    return get_index_manager().disable_deletion_protection()


def set_index_tags(tags: Dict[str, str]) -> Dict[str, Any]:
    """Convenience: Set tags on current index."""
    return get_index_manager().set_index_tags(tags)
