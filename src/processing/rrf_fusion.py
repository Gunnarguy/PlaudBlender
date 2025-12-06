"""
Reciprocal Rank Fusion (RRF) for PlaudBlender

Implements the mathematical algorithm for fusing ranked lists from multiple
retrieval systems (dense vectors, sparse vectors, metadata filters).

RRF is superior to simple score averaging because:
1. Works even when scores are not comparable (different scales)
2. Prioritizes documents that appear near the top of ANY list
3. Simple, parameter-light, and well-studied

Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
- k is a constant (typically 60) to prevent division issues
- rank_i(d) is the rank of document d in list i (1-indexed)

Reference: 
- Cormack, Clarke, Buettcher (2009): "Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods"
- gemini-deep-research2.txt, gemini-final-prompt.txt
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# Default RRF constant (k=60 is standard from the original paper)
DEFAULT_K = 60


@dataclass
class FusedResult:
    """A single result after RRF fusion."""
    id: str
    rrf_score: float
    metadata: Dict = field(default_factory=dict)
    text: Optional[str] = None
    
    # Per-source tracking for transparency
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    metadata_rank: Optional[int] = None
    
    @property
    def sources(self) -> List[str]:
        """List sources this result appeared in."""
        sources = []
        if self.dense_rank is not None:
            sources.append("dense")
        if self.sparse_rank is not None:
            sources.append("sparse")
        if self.metadata_rank is not None:
            sources.append("metadata")
        return sources
    
    @property
    def source_count(self) -> int:
        """Number of sources this result appeared in."""
        return len(self.sources)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "rrf_score": round(self.rrf_score, 4),
            "sources": self.sources,
            "source_count": self.source_count,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "sparse_rank": self.sparse_rank,
            "sparse_score": self.sparse_score,
            "metadata_rank": self.metadata_rank,
            "metadata": self.metadata,
        }


@dataclass
class RRFMergeResult:
    """Complete result of RRF fusion with full transparency."""
    results: List[FusedResult]
    total_candidates: int
    dense_count: int
    sparse_count: int
    metadata_count: int
    multi_source_count: int  # Results found in 2+ sources
    k_constant: int
    
    def to_dict(self) -> Dict:
        return {
            "total_results": len(self.results),
            "total_candidates": self.total_candidates,
            "dense_count": self.dense_count,
            "sparse_count": self.sparse_count,
            "metadata_count": self.metadata_count,
            "multi_source_count": self.multi_source_count,
            "k_constant": self.k_constant,
        }


def reciprocal_rank_fusion(
    dense_results: Optional[List[Any]] = None,
    sparse_results: Optional[List[Any]] = None,
    metadata_results: Optional[List[Any]] = None,
    k: int = DEFAULT_K,
    limit: int = 10,
    weights: Optional[Dict[str, float]] = None,
) -> RRFMergeResult:
    """
    Perform Reciprocal Rank Fusion across multiple result lists.
    
    Args:
        dense_results: Results from dense vector search (semantic)
        sparse_results: Results from sparse vector search (keyword)
        metadata_results: Results from metadata/SQL filter
        k: RRF constant (default 60, higher = more weight to lower ranks)
        limit: Max results to return
        weights: Optional per-source weights (e.g., {"dense": 1.0, "sparse": 0.5})
        
    Returns:
        RRFMergeResult with fused, ranked results and transparency metrics
    """
    dense_results = dense_results or []
    sparse_results = sparse_results or []
    metadata_results = metadata_results or []
    
    # Default weights
    if weights is None:
        weights = {"dense": 1.0, "sparse": 1.0, "metadata": 1.0}
    
    # Track RRF scores and source info
    rrf_scores: Dict[str, float] = defaultdict(float)
    result_info: Dict[str, Dict] = {}
    
    def extract_id(item: Any) -> str:
        """Extract ID from various result formats."""
        if isinstance(item, dict):
            return item.get("id", str(item))
        if hasattr(item, "id"):
            return item.id
        return str(item)
    
    def extract_metadata(item: Any) -> Dict:
        """Extract metadata from various result formats."""
        if isinstance(item, dict):
            return item.get("metadata", {})
        if hasattr(item, "metadata"):
            return item.metadata or {}
        return {}
    
    def extract_score(item: Any) -> Optional[float]:
        """Extract score from various result formats."""
        if isinstance(item, dict):
            return item.get("score")
        if hasattr(item, "score"):
            return item.score
        return None
    
    # Process dense results
    for rank, item in enumerate(dense_results, start=1):
        doc_id = extract_id(item)
        rrf_scores[doc_id] += weights.get("dense", 1.0) / (k + rank)
        
        if doc_id not in result_info:
            result_info[doc_id] = {
                "metadata": extract_metadata(item),
                "text": item.get("text") if isinstance(item, dict) else None,
            }
        
        result_info[doc_id]["dense_rank"] = rank
        result_info[doc_id]["dense_score"] = extract_score(item)
    
    # Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        doc_id = extract_id(item)
        rrf_scores[doc_id] += weights.get("sparse", 1.0) / (k + rank)
        
        if doc_id not in result_info:
            result_info[doc_id] = {
                "metadata": extract_metadata(item),
                "text": item.get("text") if isinstance(item, dict) else None,
            }
        
        result_info[doc_id]["sparse_rank"] = rank
        result_info[doc_id]["sparse_score"] = extract_score(item)
    
    # Process metadata filter results
    for rank, item in enumerate(metadata_results, start=1):
        doc_id = extract_id(item)
        rrf_scores[doc_id] += weights.get("metadata", 1.0) / (k + rank)
        
        if doc_id not in result_info:
            result_info[doc_id] = {
                "metadata": extract_metadata(item),
                "text": item.get("text") if isinstance(item, dict) else None,
            }
        
        result_info[doc_id]["metadata_rank"] = rank
    
    # Build fused results
    fused = []
    for doc_id, rrf_score in rrf_scores.items():
        info = result_info.get(doc_id, {})
        fused.append(FusedResult(
            id=doc_id,
            rrf_score=rrf_score,
            metadata=info.get("metadata", {}),
            text=info.get("text"),
            dense_rank=info.get("dense_rank"),
            dense_score=info.get("dense_score"),
            sparse_rank=info.get("sparse_rank"),
            sparse_score=info.get("sparse_score"),
            metadata_rank=info.get("metadata_rank"),
        ))
    
    # Sort by RRF score (descending)
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    
    # Count multi-source results
    multi_source = sum(1 for r in fused if r.source_count > 1)
    
    logger.info(
        f"🔀 RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"+ {len(metadata_results)} metadata → {len(fused)} unique "
        f"({multi_source} in multiple sources)"
    )
    
    return RRFMergeResult(
        results=fused[:limit],
        total_candidates=len(fused),
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        metadata_count=len(metadata_results),
        multi_source_count=multi_source,
        k_constant=k,
    )


def weighted_rrf(
    result_lists: List[Tuple[str, List[Any], float]],
    k: int = DEFAULT_K,
    limit: int = 10,
) -> RRFMergeResult:
    """
    Generalized RRF for arbitrary number of ranked lists.
    
    Args:
        result_lists: List of (source_name, results, weight) tuples
        k: RRF constant
        limit: Max results
        
    Returns:
        RRFMergeResult
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    result_metadata: Dict[str, Dict] = {}
    source_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
    source_counts = {}
    
    for source_name, results, weight in result_lists:
        source_counts[source_name] = len(results)
        
        for rank, item in enumerate(results, start=1):
            doc_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", str(item))
            rrf_scores[doc_id] += weight / (k + rank)
            source_ranks[doc_id][source_name] = rank
            
            if doc_id not in result_metadata:
                metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
                result_metadata[doc_id] = metadata or {}
    
    # Build fused results (simplified for generalized version)
    fused = []
    for doc_id, score in rrf_scores.items():
        result = FusedResult(
            id=doc_id,
            rrf_score=score,
            metadata=result_metadata.get(doc_id, {}),
        )
        # Map generic source ranks to standard fields if present
        ranks = source_ranks[doc_id]
        if "dense" in ranks:
            result.dense_rank = ranks["dense"]
        if "sparse" in ranks:
            result.sparse_rank = ranks["sparse"]
        if "metadata" in ranks:
            result.metadata_rank = ranks["metadata"]
        
        fused.append(result)
    
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    multi_source = sum(1 for r in fused if r.source_count > 1)
    
    return RRFMergeResult(
        results=fused[:limit],
        total_candidates=len(fused),
        dense_count=source_counts.get("dense", 0),
        sparse_count=source_counts.get("sparse", 0),
        metadata_count=source_counts.get("metadata", 0),
        multi_source_count=multi_source,
        k_constant=k,
    )


def compute_single_rrf_score(ranks: List[int], k: int = DEFAULT_K) -> float:
    """
    Compute RRF score for a single document given its ranks in multiple lists.
    
    Args:
        ranks: List of ranks (1-indexed) from different retrieval systems
        k: RRF constant
        
    Returns:
        Combined RRF score
    """
    return sum(1 / (k + rank) for rank in ranks)


if __name__ == "__main__":
    # Test RRF fusion
    print("\n" + "="*70)
    print("Reciprocal Rank Fusion Test")
    print("="*70)
    
    # Simulated results
    dense = [
        {"id": "doc_a", "score": 0.95, "metadata": {"title": "Doc A"}},
        {"id": "doc_b", "score": 0.88, "metadata": {"title": "Doc B"}},
        {"id": "doc_c", "score": 0.82, "metadata": {"title": "Doc C"}},
        {"id": "doc_d", "score": 0.75, "metadata": {"title": "Doc D"}},
    ]
    
    sparse = [
        {"id": "doc_b", "score": 12.5, "metadata": {"title": "Doc B"}},  # Top in sparse!
        {"id": "doc_e", "score": 10.2, "metadata": {"title": "Doc E"}},
        {"id": "doc_a", "score": 8.1, "metadata": {"title": "Doc A"}},
        {"id": "doc_f", "score": 6.5, "metadata": {"title": "Doc F"}},
    ]
    
    metadata = [
        {"id": "doc_c", "metadata": {"title": "Doc C"}},  # Top in metadata filter
        {"id": "doc_a", "metadata": {"title": "Doc A"}},
    ]
    
    result = reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        metadata_results=metadata,
        limit=5,
    )
    
    print(f"\nFusion stats: {result.to_dict()}")
    print("\nTop results:")
    for i, r in enumerate(result.results, 1):
        print(f"  {i}. {r.id}: RRF={r.rrf_score:.4f} | sources={r.sources}")
        print(f"     dense_rank={r.dense_rank}, sparse_rank={r.sparse_rank}, metadata_rank={r.metadata_rank}")
