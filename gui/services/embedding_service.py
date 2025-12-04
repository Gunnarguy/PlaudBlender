"""
Centralized Embedding Service for PlaudBlender.

Uses Google's LATEST embedding model: gemini-embedding-001
- Supports flexible dimensions: 128 to 3072
- Task-type optimization for better results
- Matryoshka embeddings (truncate without quality loss)

This is the SINGLE SOURCE OF TRUTH for all embedding operations.
"""
import os
import logging
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import numpy as np

import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDING CONFIGURATION - Fully Customizable
# ============================================================================

class EmbeddingModel(str, Enum):
    """Available embedding models."""
    GEMINI_EMBEDDING_001 = "gemini-embedding-001"  # Latest & best (June 2025)
    TEXT_EMBEDDING_004 = "models/text-embedding-004"  # Legacy (deprecating Oct 2025)


class TaskType(str, Enum):
    """
    Task types for optimized embeddings.
    
    Choose based on your use case for better results:
    - RETRIEVAL_DOCUMENT: For documents being indexed
    - RETRIEVAL_QUERY: For search queries
    - SEMANTIC_SIMILARITY: For comparing text similarity
    - CLASSIFICATION: For categorization tasks
    - CLUSTERING: For grouping similar items
    - QUESTION_ANSWERING: For Q&A systems
    - FACT_VERIFICATION: For fact-checking
    - CODE_RETRIEVAL_QUERY: For code search
    """
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"


class EmbeddingDimension(int, Enum):
    """
    Recommended embedding dimensions.
    
    Higher = more accurate but more storage/compute.
    Lower = faster and cheaper, minimal quality loss.
    
    MTEB Benchmark Scores:
    - 3072: baseline (best)
    - 2048: 68.16
    - 1536: 68.17 (sweet spot!)
    - 768:  67.99
    - 512:  67.55
    - 256:  66.19
    - 128:  63.31
    """
    DIM_3072 = 3072  # Maximum quality (default)
    DIM_2048 = 2048  # Very high quality
    DIM_1536 = 1536  # Recommended sweet spot
    DIM_768 = 768    # Good balance
    DIM_512 = 512    # Compact
    DIM_256 = 256    # Very compact
    DIM_128 = 128    # Minimum (fastest)


# Default configuration
DEFAULT_MODEL = EmbeddingModel.GEMINI_EMBEDDING_001
DEFAULT_DIMENSION = 768  # Good balance for Pinecone (integer, not enum)
DEFAULT_TASK_TYPE_DOCUMENT = TaskType.RETRIEVAL_DOCUMENT
DEFAULT_TASK_TYPE_QUERY = TaskType.RETRIEVAL_QUERY

# Limits
MAX_INPUT_TOKENS = 2048  # gemini-embedding-001 limit
MAX_BATCH_SIZE = 100
MAX_INPUT_CHARS = 8000  # Safe character limit


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingConfig:
    """
    Configuration for embedding generation.
    
    Customize everything:
    - model: Which embedding model to use
    - dimension: Output vector size (128-3072)
    - task_type: Optimization for specific use case
    - normalize: Whether to L2-normalize (required for dims < 3072)
    """
    
    def __init__(
        self,
        model: EmbeddingModel = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        task_type_document: TaskType = DEFAULT_TASK_TYPE_DOCUMENT,
        task_type_query: TaskType = DEFAULT_TASK_TYPE_QUERY,
        normalize: bool = True,  # Required for dims < 3072
    ):
        self.model = model
        self.dimension = dimension
        self.task_type_document = task_type_document
        self.task_type_query = task_type_query
        self.normalize = normalize
        
        # Validate dimension
        if not 128 <= dimension <= 3072:
            raise ValueError(f"Dimension must be 128-3072, got {dimension}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.value,
            "dimension": self.dimension,
            "task_type_document": self.task_type_document.value,
            "task_type_query": self.task_type_query.value,
            "normalize": self.normalize,
        }


class EmbeddingService:
    """
    Fully customizable embedding service using Google's latest models.
    
    Features:
    - Latest gemini-embedding-001 model (June 2025)
    - Flexible dimensions (128-3072)
    - Task-type optimization
    - Automatic normalization for smaller dimensions
    - Batch embedding support
    
    Usage:
        # Default config
        service = EmbeddingService()
        
        # Custom config
        config = EmbeddingConfig(
            dimension=1536,
            task_type_query=TaskType.SEMANTIC_SIMILARITY
        )
        service = EmbeddingService(config)
        
        # Generate embeddings
        doc_vec = service.embed_document("My document text")
        query_vec = service.embed_query("search query")
        batch_vecs = service.embed_batch(["text1", "text2"])
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding service.
        
        Args:
            config: Optional EmbeddingConfig. Uses defaults if not provided.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EmbeddingError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
        
        self.config = config or EmbeddingConfig()
        
        logger.info(f"✅ EmbeddingService initialized")
        logger.info(f"   Model: {self.config.model.value}")
        logger.info(f"   Dimension: {self.config.dimension}")
        logger.info(f"   Task types: doc={self.config.task_type_document.value}, query={self.config.task_type_query.value}")
        logger.info(f"   Normalize: {self.config.normalize}")
    
    @property
    def dimension(self) -> int:
        """Get the configured embedding dimension."""
        return self.config.dimension
    
    @property
    def model(self) -> str:
        """Get the configured model name."""
        return self.config.model.value
    
    # ========================================================================
    # CORE EMBEDDING METHODS
    # ========================================================================
    
    def embed_text(
        self,
        text: str,
        task_type: Optional[TaskType] = None,
        dimension: Optional[int] = None,
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (truncated if too long)
            task_type: Override default task type
            dimension: Override default dimension
            
        Returns:
            Embedding vector (normalized if dim < 3072)
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        # Truncate if needed
        truncated = text[:MAX_INPUT_CHARS]
        if len(text) > MAX_INPUT_CHARS:
            logger.warning(f"Text truncated: {len(text)} → {MAX_INPUT_CHARS} chars")
        
        dim = dimension or self.config.dimension
        task = task_type or self.config.task_type_document
        
        try:
            # Use new API format
            result = genai.embed_content(
                model=self.config.model.value,
                content=truncated,
                task_type=task.value,
                output_dimensionality=dim,
            )
            
            embedding = result["embedding"]
            
            # Normalize for dimensions < 3072 (required for accuracy)
            if self.config.normalize and dim < 3072:
                embedding = self._normalize(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    def embed_batch(
        self,
        texts: List[str],
        task_type: Optional[TaskType] = None,
        dimension: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            task_type: Override default task type
            dimension: Override default dimension
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        dim = dimension or self.config.dimension
        task = task_type or self.config.task_type_document
        
        # Truncate texts
        truncated = [t[:MAX_INPUT_CHARS] for t in texts]
        
        all_embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(truncated), MAX_BATCH_SIZE):
                batch = truncated[i:i + MAX_BATCH_SIZE]
                
                result = genai.embed_content(
                    model=self.config.model.value,
                    content=batch,
                    task_type=task.value,
                    output_dimensionality=dim,
                )
                
                # Handle response format
                embeddings = result["embedding"]
                if not isinstance(embeddings[0], list):
                    embeddings = [embeddings]
                
                # Normalize if needed
                if self.config.normalize and dim < 3072:
                    embeddings = [self._normalize(e) for e in embeddings]
                
                all_embeddings.extend(embeddings)
                logger.debug(f"Embedded batch {i//MAX_BATCH_SIZE + 1}: {len(batch)} texts")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}")
    
    # ========================================================================
    # SEMANTIC METHODS (task-type optimized)
    # ========================================================================
    
    def embed_query(self, query: str, dimension: Optional[int] = None) -> List[float]:
        """
        Embed a SEARCH QUERY (optimized for retrieval).
        
        Uses RETRIEVAL_QUERY task type for best search results.
        """
        return self.embed_text(
            query,
            task_type=self.config.task_type_query,
            dimension=dimension,
        )
    
    def embed_document(self, document: str, dimension: Optional[int] = None) -> List[float]:
        """
        Embed a DOCUMENT for storage/indexing.
        
        Uses RETRIEVAL_DOCUMENT task type for best retrieval.
        """
        return self.embed_text(
            document,
            task_type=self.config.task_type_document,
            dimension=dimension,
        )
    
    def embed_transcript(self, transcript: str, dimension: Optional[int] = None) -> List[float]:
        """Embed a Plaud transcript (alias for embed_document)."""
        return self.embed_document(transcript, dimension)
    
    def embed_for_similarity(self, text: str, dimension: Optional[int] = None) -> List[float]:
        """
        Embed text for SIMILARITY comparison.
        
        Uses SEMANTIC_SIMILARITY task type.
        """
        return self.embed_text(
            text,
            task_type=TaskType.SEMANTIC_SIMILARITY,
            dimension=dimension,
        )
    
    def embed_for_clustering(self, text: str, dimension: Optional[int] = None) -> List[float]:
        """
        Embed text for CLUSTERING.
        
        Uses CLUSTERING task type.
        """
        return self.embed_text(
            text,
            task_type=TaskType.CLUSTERING,
            dimension=dimension,
        )
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _normalize(self, embedding: List[float]) -> List[float]:
        """L2 normalize embedding vector."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    
    def validate_vector(self, vector: List[float]) -> bool:
        """Check if vector has correct dimension."""
        return (
            isinstance(vector, (list, tuple)) and
            len(vector) == self.config.dimension and
            all(isinstance(x, (int, float)) for x in vector)
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.to_dict()
    
    def reconfigure(self, config: EmbeddingConfig):
        """Update configuration at runtime."""
        self.config = config
        logger.info(f"Reconfigured: dim={config.dimension}, model={config.model.value}")


# ============================================================================
# SINGLETON & FACTORY
# ============================================================================

_embedding_service: Optional[EmbeddingService] = None
_current_config: Optional[EmbeddingConfig] = None


def get_embedding_service(config: Optional[EmbeddingConfig] = None) -> EmbeddingService:
    """
    Get or create the EmbeddingService singleton.
    
    Args:
        config: Optional config. If different from current, recreates service.
    """
    global _embedding_service, _current_config
    
    if config and config.to_dict() != (_current_config.to_dict() if _current_config else None):
        _embedding_service = EmbeddingService(config)
        _current_config = config
    elif _embedding_service is None:
        _embedding_service = EmbeddingService(config)
        _current_config = config or EmbeddingConfig()
    
    return _embedding_service


def create_embedding_service(
    dimension: int = DEFAULT_DIMENSION,
    model: EmbeddingModel = DEFAULT_MODEL,
    task_type_document: TaskType = DEFAULT_TASK_TYPE_DOCUMENT,
    task_type_query: TaskType = DEFAULT_TASK_TYPE_QUERY,
    normalize: bool = True,
) -> EmbeddingService:
    """
    Factory function to create a custom EmbeddingService.
    
    Example:
        service = create_embedding_service(
            dimension=1536,
            task_type_query=TaskType.SEMANTIC_SIMILARITY
        )
    """
    config = EmbeddingConfig(
        model=model,
        dimension=dimension,
        task_type_document=task_type_document,
        task_type_query=task_type_query,
        normalize=normalize,
    )
    return EmbeddingService(config)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def embed_text(text: str, dimension: Optional[int] = None) -> List[float]:
    """Embed text using default service."""
    return get_embedding_service().embed_text(text, dimension=dimension)


def embed_query(query: str, dimension: Optional[int] = None) -> List[float]:
    """Embed search query using default service."""
    return get_embedding_service().embed_query(query, dimension=dimension)


def embed_document(document: str, dimension: Optional[int] = None) -> List[float]:
    """Embed document using default service."""
    return get_embedding_service().embed_document(document, dimension=dimension)


def embed_batch(texts: List[str], dimension: Optional[int] = None) -> List[List[float]]:
    """Embed batch using default service."""
    return get_embedding_service().embed_batch(texts, dimension=dimension)


def get_embedding_dimension() -> int:
    """Get current embedding dimension."""
    return get_embedding_service().dimension


def get_embedding_model() -> str:
    """Get current embedding model."""
    return get_embedding_service().model


# Legacy compatibility
EMBEDDING_MODEL = DEFAULT_MODEL.value
EMBEDDING_DIMENSION = DEFAULT_DIMENSION
