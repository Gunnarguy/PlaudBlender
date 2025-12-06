"""
Sparse Embeddings Module for Hybrid Search

Uses Pinecone's hosted sparse embedding model (pinecone-sparse-english-v0)
for lexical/keyword-based search to complement dense semantic embeddings.

Hybrid Search Architecture:
- Dense vectors: Capture semantic meaning (synonyms, paraphrases)
- Sparse vectors: Capture exact keyword matches (domain terms, proper nouns)
- Combined: Best of both worlds, reranked for final relevance

Reference: https://docs.pinecone.io/guides/search/hybrid-search
"""
import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Pinecone sparse model (as of 2025-10)
DEFAULT_SPARSE_MODEL = "pinecone-sparse-english-v0"


@dataclass
class SparseVector:
    """Sparse vector representation with indices and values."""
    indices: List[int]
    values: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Pinecone-compatible dict format."""
        return {
            "indices": self.indices,
            "values": self.values,
        }
    
    def scale(self, factor: float) -> 'SparseVector':
        """Scale values by a factor (for alpha weighting)."""
        return SparseVector(
            indices=self.indices,
            values=[v * factor for v in self.values],
        )


class SparseEmbeddingError(Exception):
    """Raised when sparse embedding generation fails."""
    pass


class PineconeSparseEmbedder:
    """
    Pinecone hosted sparse embeddings using pinecone-sparse-english-v0.
    
    This model generates BM25-style sparse vectors optimized for keyword matching.
    Unlike dense embeddings (768-3072 dimensions), sparse vectors have very high
    dimensionality but mostly zero values, with non-zero values for matching tokens.
    
    Usage:
        embedder = PineconeSparseEmbedder()
        sparse_vec = embedder.embed_text("action items from meeting")
        # Returns SparseVector with indices=[...], values=[...]
    """
    
    def __init__(self, model: str = DEFAULT_SPARSE_MODEL):
        """
        Initialize Pinecone sparse embedder.
        
        Args:
            model: Sparse embedding model name (default: pinecone-sparse-english-v0)
        """
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise SparseEmbeddingError("pinecone package not installed") from e
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise SparseEmbeddingError("PINECONE_API_KEY not found in environment")
        
        self._pc = Pinecone(api_key=api_key)
        self._model = model
        logger.info(f"✅ PineconeSparseEmbedder initialized with model: {model}")
    
    def embed_text(self, text: str, input_type: str = "passage") -> SparseVector:
        """
        Generate sparse embedding for a single text.
        
        Args:
            text: Text to embed
            input_type: 'passage' for documents, 'query' for search queries
        
        Returns:
            SparseVector with indices and values
        """
        if not text or not text.strip():
            raise SparseEmbeddingError("Cannot embed empty text")
        
        try:
            result = self._pc.inference.embed(
                model=self._model,
                inputs=[text],
                parameters={
                    "input_type": input_type,
                    "truncate": "END",
                },
            )
            
            # Extract sparse embedding from response
            embedding = result[0] if isinstance(result, list) else result.data[0]
            
            return SparseVector(
                indices=embedding.get('sparse_indices', []),
                values=embedding.get('sparse_values', []),
            )
            
        except Exception as e:
            logger.error(f"Sparse embedding failed: {e}")
            raise SparseEmbeddingError(f"Failed to generate sparse embedding: {e}") from e
    
    def embed_texts(self, texts: List[str], input_type: str = "passage") -> List[SparseVector]:
        """
        Generate sparse embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            input_type: 'passage' for documents, 'query' for search queries
        
        Returns:
            List of SparseVector objects
        """
        if not texts:
            return []
        
        try:
            result = self._pc.inference.embed(
                model=self._model,
                inputs=texts,
                parameters={
                    "input_type": input_type,
                    "truncate": "END",
                },
            )
            
            sparse_vectors = []
            embeddings = result if isinstance(result, list) else result.data
            
            for emb in embeddings:
                sparse_vectors.append(SparseVector(
                    indices=emb.get('sparse_indices', []),
                    values=emb.get('sparse_values', []),
                ))
            
            return sparse_vectors
            
        except Exception as e:
            logger.error(f"Batch sparse embedding failed: {e}")
            raise SparseEmbeddingError(f"Failed to generate sparse embeddings: {e}") from e
    
    def embed_query(self, query: str) -> SparseVector:
        """Convenience method: embed a search query."""
        return self.embed_text(query, input_type="query")
    
    def embed_document(self, document: str) -> SparseVector:
        """Convenience method: embed a document for indexing."""
        return self.embed_text(document, input_type="passage")
    
    @property
    def model_name(self) -> str:
        return self._model


def hybrid_score_norm(
    dense: List[float],
    sparse: SparseVector,
    alpha: float = 0.5,
) -> Tuple[List[float], Dict]:
    """
    Apply alpha weighting to combine dense and sparse vectors.
    
    Uses convex combination: alpha * dense + (1 - alpha) * sparse
    
    Args:
        dense: Dense vector (list of floats)
        sparse: Sparse vector (SparseVector object)
        alpha: Weight for dense vs sparse (0.0 = sparse only, 1.0 = dense only)
               Default 0.5 = equal weight
    
    Returns:
        Tuple of (weighted_dense, weighted_sparse_dict)
    
    Example:
        dense_vec = embedder.embed_query("find budget discussions")
        sparse_vec = sparse_embedder.embed_query("find budget discussions")
        
        # 70% semantic, 30% keyword
        hdense, hsparse = hybrid_score_norm(dense_vec, sparse_vec, alpha=0.7)
        
        results = index.query(
            vector=hdense,
            sparse_vector=hsparse,
            top_k=10,
        )
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    weighted_dense = [v * alpha for v in dense]
    weighted_sparse = {
        "indices": sparse.indices,
        "values": [v * (1 - alpha) for v in sparse.values],
    }
    
    return weighted_dense, weighted_sparse


# Singleton instance (lazy-loaded)
_sparse_embedder: Optional[PineconeSparseEmbedder] = None


def get_sparse_embedder() -> PineconeSparseEmbedder:
    """Get or create singleton sparse embedder instance."""
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = PineconeSparseEmbedder()
    return _sparse_embedder
