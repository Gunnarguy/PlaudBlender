"""
Hybrid Search Service for PlaudBlender

Implements the recommended Pinecone hybrid search pattern:
1. Query dense index (semantic search)
2. Query sparse index (lexical/keyword search)  
3. Merge and deduplicate results
4. Rerank with neural model for final ordering

This achieves ~99% retrieval accuracy by combining:
- Dense vectors: Catch semantic relationships (synonyms, paraphrases)
- Sparse vectors: Catch exact keyword matches (proper nouns, domain terms)
- Neural reranking: Final relevance scoring

Reference: https://docs.pinecone.io/guides/search/hybrid-search
"""
import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
HYBRID_ENABLED = os.getenv("PINECONE_HYBRID_ENABLED", "false").lower() == "true"
HYBRID_ALPHA = float(os.getenv("PINECONE_HYBRID_ALPHA", "0.5"))  # 0=sparse, 1=dense
SPARSE_INDEX_SUFFIX = "_sparse"  # e.g., "transcripts" -> "transcripts_sparse"
DEFAULT_RERANK_MODEL = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")


@dataclass
class HybridSearchResult:
    """Container for hybrid search results with full transparency."""
    query: str
    matches: List[Any]
    dense_matches: int = 0
    sparse_matches: int = 0
    merged_count: int = 0
    reranked: bool = False
    alpha: float = 0.5
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "matches": len(self.matches),
            "dense_matches": self.dense_matches,
            "sparse_matches": self.sparse_matches,
            "merged_count": self.merged_count,
            "reranked": self.reranked,
            "alpha": self.alpha,
            "latency_ms": self.latency_ms,
        }


def _get_sparse_index_name(dense_index_name: str) -> str:
    """
    Derive sparse index name from dense index name.
    
    Convention: If dense index is "transcripts", sparse index is "transcripts_sparse"
    """
    return f"{dense_index_name}{SPARSE_INDEX_SUFFIX}"


def _merge_and_dedupe(
    dense_results: List[Any],
    sparse_results: List[Any],
    id_field: str = "id",
) -> List[Dict]:
    """
    Merge results from dense and sparse searches, deduplicating by ID.
    
    Returns a list of dicts with:
    - id: Vector ID
    - dense_score: Score from dense search (or None)
    - sparse_score: Score from sparse search (or None)
    - metadata: Combined metadata
    - source: 'dense', 'sparse', or 'both'
    """
    merged = {}
    
    # Add dense results
    for match in dense_results:
        vec_id = getattr(match, id_field, str(match))
        if hasattr(match, 'id'):
            vec_id = match.id
        
        merged[vec_id] = {
            "id": vec_id,
            "dense_score": getattr(match, 'score', 0.0),
            "sparse_score": None,
            "metadata": getattr(match, 'metadata', {}) or {},
            "namespace": getattr(match, 'namespace', ''),
            "source": "dense",
        }
    
    # Add/merge sparse results
    for match in sparse_results:
        vec_id = getattr(match, id_field, str(match))
        if hasattr(match, 'id'):
            vec_id = match.id
        
        if vec_id in merged:
            # Already have this from dense, mark as both
            merged[vec_id]["sparse_score"] = getattr(match, 'score', 0.0)
            merged[vec_id]["source"] = "both"
        else:
            merged[vec_id] = {
                "id": vec_id,
                "dense_score": None,
                "sparse_score": getattr(match, 'score', 0.0),
                "metadata": getattr(match, 'metadata', {}) or {},
                "namespace": getattr(match, 'namespace', ''),
                "source": "sparse",
            }
    
    # Sort by combined score (prioritize items found in both)
    def combined_score(item):
        d = item.get("dense_score") or 0
        s = item.get("sparse_score") or 0
        both_bonus = 0.1 if item.get("source") == "both" else 0
        return d + s + both_bonus
    
    return sorted(merged.values(), key=combined_score, reverse=True)


def search_hybrid(
    query: str,
    limit: int = 10,
    alpha: float = HYBRID_ALPHA,
    namespaces: Optional[List[str]] = None,
    filter_dict: Optional[Dict] = None,
    rerank: bool = True,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> HybridSearchResult:
    """
    Perform hybrid search combining dense (semantic) and sparse (keyword) retrieval.
    
    Architecture:
    1. Generate dense embedding for query
    2. Generate sparse embedding for query  
    3. Query dense index (full_text, summaries namespaces)
    4. Query sparse index (same namespaces with _sparse suffix)
    5. Merge and deduplicate results by vector ID
    6. Optionally rerank with neural model
    
    Args:
        query: Natural language search query
        limit: Max results to return
        alpha: Weight between dense (1.0) and sparse (0.0). Default 0.5 = equal.
        namespaces: Namespaces to search (default: full_text, summaries)
        filter_dict: Optional Pinecone metadata filters
        rerank: Whether to apply neural reranking
        rerank_model: Reranker model name
    
    Returns:
        HybridSearchResult with matches and transparency metrics
    """
    import time
    start_time = time.time()
    
    from gui.services.embedding_service import get_embedding_service
    from gui.services.clients import get_pinecone_client
    
    # Default namespaces
    if namespaces is None:
        namespaces = ["full_text", "summaries"]
    
    logger.info(f"🔀 Hybrid search: '{query}' (alpha={alpha}, limit={limit})")
    
    # Get dense embedding
    embedding_service = get_embedding_service()
    dense_vector = embedding_service.embed_query(query)
    
    # Get sparse embedding
    try:
        from src.ai.sparse_embeddings import get_sparse_embedder
        sparse_embedder = get_sparse_embedder()
        sparse_vector = sparse_embedder.embed_query(query)
        has_sparse = True
        logger.info(f"   ✅ Sparse vector: {len(sparse_vector.indices)} non-zero dimensions")
    except Exception as e:
        logger.warning(f"   ⚠️ Sparse embedding unavailable: {e}")
        has_sparse = False
        sparse_vector = None
    
    pinecone_client = get_pinecone_client()
    
    # Query dense index
    dense_results = pinecone_client.query_namespaces(
        query_embedding=dense_vector,
        namespaces=namespaces,
        top_k=limit * 2,  # Get more candidates for merging
        filter_dict=filter_dict,
        include_metadata=True,
    )
    dense_matches = dense_results.matches if dense_results else []
    logger.info(f"   📊 Dense results: {len(dense_matches)}")
    
    # Query sparse index (if available)
    sparse_matches = []
    if has_sparse and sparse_vector:
        try:
            sparse_index_name = _get_sparse_index_name(pinecone_client.index_name)
            # Check if sparse index exists
            existing = pinecone_client.list_indexes()
            if sparse_index_name in existing:
                # Switch to sparse index, query, switch back
                original_index = pinecone_client.index_name
                pinecone_client.switch_index(sparse_index_name)
                
                # Query sparse index using sparse vector
                for ns in namespaces:
                    try:
                        results = pinecone_client.index.query(
                            vector=sparse_vector.indices,  # Sparse query
                            sparse_vector=sparse_vector.to_dict(),
                            namespace=ns,
                            top_k=limit * 2,
                            include_metadata=True,
                        )
                        for match in results.matches:
                            match.namespace = ns
                            sparse_matches.append(match)
                    except Exception as e:
                        logger.warning(f"Sparse query failed for {ns}: {e}")
                
                # Switch back to dense index
                pinecone_client.switch_index(original_index)
                logger.info(f"   📊 Sparse results: {len(sparse_matches)}")
            else:
                logger.info(f"   ℹ️ Sparse index '{sparse_index_name}' not found, dense-only mode")
        except Exception as e:
            logger.warning(f"   ⚠️ Sparse search failed: {e}")
    
    # Merge and deduplicate
    merged = _merge_and_dedupe(dense_matches, sparse_matches)
    logger.info(f"   🔀 Merged: {len(merged)} unique results")
    
    # Convert merged dicts back to match-like objects for formatting
    @dataclass
    class HybridMatch:
        id: str
        score: float
        metadata: Dict
        namespace: str
        dense_score: Optional[float] = None
        sparse_score: Optional[float] = None
        source: str = "dense"
    
    hybrid_matches = []
    for item in merged[:limit * 2]:  # Keep extra for reranking
        # Compute combined score based on alpha
        dense_s = item.get("dense_score") or 0
        sparse_s = item.get("sparse_score") or 0
        combined = alpha * dense_s + (1 - alpha) * sparse_s
        
        hybrid_matches.append(HybridMatch(
            id=item["id"],
            score=combined,
            metadata=item["metadata"],
            namespace=item["namespace"],
            dense_score=item.get("dense_score"),
            sparse_score=item.get("sparse_score"),
            source=item["source"],
        ))
    
    # Apply reranking if enabled
    reranked = False
    if rerank and hybrid_matches:
        try:
            docs = []
            match_map = {}
            for i, match in enumerate(hybrid_matches):
                meta = match.metadata or {}
                text = meta.get('synthesis') or meta.get('text') or meta.get('summary') or meta.get('title') or ''
                if text:
                    doc_id = f"doc_{i}"
                    docs.append({"id": doc_id, "text": text[:2000]})
                    match_map[doc_id] = match
            
            if docs:
                rerank_result = pinecone_client.rerank(
                    query=query,
                    documents=docs,
                    model=rerank_model,
                    top_n=limit,
                    return_documents=False,
                )
                
                if "error" not in rerank_result:
                    reranked_matches = []
                    for item in rerank_result.get("data", []):
                        doc_id = item.get("document", {}).get("id") or f"doc_{item.get('index', 0)}"
                        if doc_id in match_map:
                            match = match_map[doc_id]
                            # Stash original scores and update with rerank score
                            match.metadata['_dense_score'] = match.dense_score
                            match.metadata['_sparse_score'] = match.sparse_score
                            match.metadata['_hybrid_score'] = match.score
                            match.score = item.get("score", match.score)
                            reranked_matches.append(match)
                    
                    hybrid_matches = reranked_matches
                    reranked = True
                    logger.info(f"   🏆 Reranked: {len(hybrid_matches)} results")
        except Exception as e:
            logger.warning(f"   ⚠️ Reranking failed: {e}")
    
    # Trim to final limit
    final_matches = hybrid_matches[:limit]
    
    latency_ms = (time.time() - start_time) * 1000
    
    return HybridSearchResult(
        query=query,
        matches=final_matches,
        dense_matches=len(dense_matches),
        sparse_matches=len(sparse_matches),
        merged_count=len(merged),
        reranked=reranked,
        alpha=alpha,
        latency_ms=latency_ms,
    )


def format_hybrid_results(
    result: HybridSearchResult,
    include_context: bool = True,
) -> str:
    """
    Format hybrid search results for display with full transparency.
    
    Shows:
    - Dense score (semantic similarity)
    - Sparse score (keyword match)
    - Hybrid score (alpha-weighted combination)
    - Rerank score (if reranked)
    """
    if not result.matches:
        return f"No matches found for '{result.query}'."
    
    lines = [
        f"{'═' * 65}",
        f"🔀 HYBRID SEARCH RESULTS for: '{result.query}'",
        f"   Found {len(result.matches)} matches (dense={result.dense_matches}, sparse={result.sparse_matches})",
        f"   Alpha: {result.alpha:.2f} (dense) / {1-result.alpha:.2f} (sparse)",
        f"   Latency: {result.latency_ms:.0f}ms | Reranked: {'✅' if result.reranked else '❌'}",
        f"{'═' * 65}",
        "",
    ]
    
    for idx, match in enumerate(result.matches, 1):
        meta = match.metadata or {}
        
        # Build comprehensive score display
        score_parts = []
        if result.reranked:
            score_parts.append(f"rerank: {match.score*100:.1f}%")
        
        if hasattr(match, 'dense_score') and match.dense_score is not None:
            score_parts.append(f"dense: {match.dense_score*100:.1f}%")
        elif meta.get('_dense_score') is not None:
            score_parts.append(f"dense: {meta['_dense_score']*100:.1f}%")
        
        if hasattr(match, 'sparse_score') and match.sparse_score is not None:
            score_parts.append(f"sparse: {match.sparse_score*100:.1f}%")
        elif meta.get('_sparse_score') is not None:
            score_parts.append(f"sparse: {meta['_sparse_score']*100:.1f}%")
        
        if meta.get('_hybrid_score') is not None:
            score_parts.append(f"hybrid: {meta['_hybrid_score']*100:.1f}%")
        
        score_str = " │ ".join(score_parts) if score_parts else f"score: {match.score*100:.1f}%"
        
        # Source indicator
        source = getattr(match, 'source', 'unknown')
        source_icon = "🔀" if source == "both" else ("📊" if source == "dense" else "🔤")
        
        title = meta.get('title', 'Untitled')
        ns_icon = "📄" if match.namespace == "full_text" else "📝"
        
        lines.append(f"#{idx} │ {title}")
        lines.append(f"   {source_icon} Source: {source} │ {ns_icon} {match.namespace}")
        lines.append(f"   📊 {score_str}")
        lines.append(f"   🏷️  Themes: {meta.get('themes', '—')}")
        
        # Date
        date_val = meta.get('date') or meta.get('start_at') or meta.get('created') or '—'
        if isinstance(date_val, str) and len(date_val) > 10:
            date_val = date_val[:10]
        lines.append(f"   📅 Date: {date_val}")
        
        # Context snippet
        if include_context:
            text = (
                meta.get('synthesis') or 
                meta.get('text') or 
                meta.get('summary') or 
                meta.get('content') or 
                ''
            )
            if text:
                snippet = (text[:400] + '…') if len(text) > 400 else text
                lines.append("")
                lines.append(f"   💬 {snippet}")
        
        lines.append("")
        lines.append(f"{'─' * 65}")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# PUBLIC API
# ============================================================================

def search_hybrid_formatted(
    query: str,
    limit: int = 5,
    alpha: float = HYBRID_ALPHA,
    namespaces: Optional[List[str]] = None,
    rerank: bool = True,
) -> str:
    """
    Perform hybrid search and return formatted results string.
    
    This is the main entry point for the Search UI.
    """
    result = search_hybrid(
        query=query,
        limit=limit,
        alpha=alpha,
        namespaces=namespaces,
        rerank=rerank,
    )
    return format_hybrid_results(result)


def is_hybrid_enabled() -> bool:
    """Check if hybrid search is enabled via environment."""
    return HYBRID_ENABLED


def get_hybrid_alpha() -> float:
    """Get current alpha setting (dense vs sparse weight)."""
    return HYBRID_ALPHA
