"""
Hybrid Search Service for PlaudBlender

Implements a dense + sparse hybrid search pattern:
1. Query dense index (semantic search)
2. Query sparse index (lexical/keyword search)
3. Merge and deduplicate results
4. Rerank with neural model for final ordering

This achieves strong retrieval accuracy by combining:
- Dense vectors: Catch semantic relationships (synonyms, paraphrases)
- Sparse vectors: Catch exact keyword matches (proper nouns, domain terms)
- Neural reranking: Final relevance scoring

Reference: https://qdrant.tech/articles/hybrid-search/ (concept applies across providers)
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION (provider-neutral with Pinecone fallbacks for legacy .env)
# ============================================================================
HYBRID_ENABLED = (
    os.getenv(
        "VECTOR_HYBRID_ENABLED", os.getenv("PINECONE_HYBRID_ENABLED", "false")
    ).lower()
    == "true"
)
HYBRID_ALPHA = float(
    os.getenv("VECTOR_HYBRID_ALPHA", os.getenv("PINECONE_HYBRID_ALPHA", "0.5"))
)  # 0=sparse, 1=dense
SPARSE_INDEX_SUFFIX = "_sparse"  # e.g., "transcripts" -> "transcripts_sparse"
DEFAULT_RERANK_MODEL = os.getenv(
    "VECTOR_RERANK_MODEL",
    os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3"),
)


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
        if hasattr(match, "id"):
            vec_id = match.id

        merged[vec_id] = {
            "id": vec_id,
            "dense_score": getattr(match, "score", 0.0),
            "sparse_score": None,
            "metadata": getattr(match, "metadata", {}) or {},
            "namespace": getattr(match, "namespace", ""),
            "source": "dense",
        }

    # Add/merge sparse results
    for match in sparse_results:
        vec_id = getattr(match, id_field, str(match))
        if hasattr(match, "id"):
            vec_id = match.id

        if vec_id in merged:
            # Already have this from dense, mark as both
            merged[vec_id]["sparse_score"] = getattr(match, "score", 0.0)
            merged[vec_id]["source"] = "both"
        else:
            merged[vec_id] = {
                "id": vec_id,
                "dense_score": None,
                "sparse_score": getattr(match, "score", 0.0),
                "metadata": getattr(match, "metadata", {}) or {},
                "namespace": getattr(match, "namespace", ""),
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
    """Perform hybrid search combining dense (semantic) and optional sparse (keyword) retrieval.

    Notes on backends:
    - Dense search works for all providers through the vector-client shim.
    - The sparse-index strategy implemented here matches Pinecone's API surface.
      For Qdrant, we currently skip sparse querying (conceptually possible, but not
      implemented in our compatibility layer).
    """

    import time

    from gui.services.embedding_service import get_embedding_service
    from gui.services.clients import get_vector_db_client

    start_time = time.time()

    try:
        from src.vector_store import is_pinecone

        pinecone_mode = bool(is_pinecone())
    except Exception:
        pinecone_mode = False

    if namespaces is None:
        namespaces = ["full_text", "summaries"]

    embedding_service = get_embedding_service()
    dense_vector = embedding_service.embed_query(query)

    # Sparse embedding (optional)
    sparse_vector = None
    has_sparse = False
    if pinecone_mode:
        try:
            from src.ai.sparse_embeddings import get_sparse_embedder

            sparse_embedder = get_sparse_embedder()
            sparse_vector = sparse_embedder.embed_query(query)
            has_sparse = True
            logger.info(
                f"   ✅ Sparse vector: {len(getattr(sparse_vector, 'indices', []) or [])} non-zero dimensions"
            )
        except Exception as e:
            logger.warning(f"   ⚠️ Sparse embedding unavailable: {e}")

    # The vector client returned here is a provider-specific shim with a superset
    # of methods (list_indexes/switch_index/rerank/...). The shared protocol type
    # is intentionally minimal, so we cast for static checkers.
    from typing import cast

    vector_client = cast(Any, get_vector_db_client())

    # Dense query (provider-agnostic)
    dense_results = vector_client.query_namespaces(
        query_embedding=dense_vector,
        namespaces=namespaces,
        top_k=limit * 2,
        filter_dict=filter_dict,
        include_metadata=True,
    )
    dense_matches = dense_results.matches if dense_results else []

    # Sparse query (Pinecone only, best-effort)
    sparse_matches: List[Any] = []
    if pinecone_mode and has_sparse and sparse_vector is not None:
        try:
            sparse_index_name = _get_sparse_index_name(vector_client.index_name)
            existing = vector_client.list_indexes()

            if sparse_index_name not in existing:
                logger.info(
                    f"   ℹ️ Sparse index '{sparse_index_name}' not found; skipping sparse query"
                )
            else:
                original_index = vector_client.index_name
                vector_client.switch_index(sparse_index_name)
                for ns in namespaces:
                    try:
                        results = vector_client.index.query(
                            vector=getattr(sparse_vector, "indices", []),
                            sparse_vector=sparse_vector.to_dict(),
                            namespace=ns,
                            top_k=limit * 2,
                            include_metadata=True,
                        )
                        for match in getattr(results, "matches", []) or []:
                            match.namespace = ns
                            sparse_matches.append(match)
                    except Exception as e:
                        logger.warning(f"Sparse query failed for {ns}: {e}")
                vector_client.switch_index(original_index)
        except Exception as e:
            logger.warning(f"   ⚠️ Sparse search unavailable: {e}")

    merged = _merge_and_dedupe(dense_matches, sparse_matches)

    # Apply alpha weighting into a final score and sort.
    for item in merged:
        d = item.get("dense_score") or 0.0
        s = item.get("sparse_score") or 0.0
        item["score"] = alpha * d + (1.0 - alpha) * s
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Optional rerank (Pinecone inference API compatible surface)
    reranked = False
    if rerank and pinecone_mode and merged:
        try:
            docs = []
            for i, item in enumerate(merged[: max(limit * 3, 10)]):
                meta = item.get("metadata") or {}
                text = (
                    meta.get("synthesis")
                    or meta.get("text")
                    or meta.get("summary")
                    or meta.get("title")
                    or ""
                )
                if text:
                    docs.append({"id": str(i), "text": text[:2000]})

            if docs and hasattr(vector_client, "rerank"):
                rr = vector_client.rerank(
                    query=query,
                    documents=docs,
                    model=rerank_model,
                    top_n=min(limit, len(docs)),
                    return_documents=False,
                )
                if isinstance(rr, dict) and "data" in rr:
                    order = [str(d.get("index")) for d in rr.get("data", [])]
                    by_id = {str(i): m for i, m in enumerate(merged)}
                    merged = [by_id[i] for i in order if i in by_id]
                    reranked = True
        except Exception as e:
            logger.warning(f"   ⚠️ Reranking failed: {e}")

    final_matches = merged[:limit]

    latency_ms = (time.time() - start_time) * 1000.0
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


class HybridSearchService:
    """Thin OO wrapper used by the GUI app layer."""

    def hybrid_search(
        self,
        query: str,
        alpha: float = HYBRID_ALPHA,
        limit: int = 10,
        namespaces: Optional[List[str]] = None,
        rerank: bool = True,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        filter_dict: Optional[Dict] = None,
    ) -> str:
        result = search_hybrid(
            query=query,
            limit=limit,
            alpha=alpha,
            namespaces=namespaces,
            filter_dict=filter_dict,
            rerank=rerank,
            rerank_model=rerank_model,
        )
        return format_hybrid_results(result)


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
        # search_hybrid currently returns dict-shaped matches (provider-neutral).
        # Keep this formatter resilient in case callers pass SDK match objects.
        if isinstance(match, dict):
            meta = match.get("metadata") or {}
            namespace = match.get("namespace") or ""
            score = float(match.get("score") or 0.0)
            dense_score = match.get("dense_score")
            sparse_score = match.get("sparse_score")
            source = match.get("source") or "unknown"
        else:
            meta = getattr(match, "metadata", {}) or {}
            namespace = getattr(match, "namespace", "")
            score = float(getattr(match, "score", 0.0) or 0.0)
            dense_score = getattr(match, "dense_score", None)
            sparse_score = getattr(match, "sparse_score", None)
            source = getattr(match, "source", "unknown")

        # Build comprehensive score display
        score_parts = []
        # We only mark reranked at the result-level. If a per-match rerank score
        # exists in metadata, show it.
        if result.reranked and meta.get("_rerank_score") is not None:
            score_parts.append(f"rerank: {float(meta['_rerank_score'])*100:.1f}%")

        if dense_score is not None:
            score_parts.append(f"dense: {float(dense_score)*100:.1f}%")
        elif meta.get("_dense_score") is not None:
            score_parts.append(f"dense: {meta['_dense_score']*100:.1f}%")

        if sparse_score is not None:
            score_parts.append(f"sparse: {float(sparse_score)*100:.1f}%")
        elif meta.get("_sparse_score") is not None:
            score_parts.append(f"sparse: {meta['_sparse_score']*100:.1f}%")

        if meta.get("_hybrid_score") is not None:
            score_parts.append(f"hybrid: {meta['_hybrid_score']*100:.1f}%")

        score_str = (
            " │ ".join(score_parts) if score_parts else f"score: {score*100:.1f}%"
        )

        # Source indicator
        source_icon = (
            "🔀" if source == "both" else ("📊" if source == "dense" else "🔤")
        )

        title = meta.get("title", "Untitled")
        ns_icon = "📄" if namespace == "full_text" else "📝"

        lines.append(f"#{idx} │ {title}")
        lines.append(f"   {source_icon} Source: {source} │ {ns_icon} {namespace}")
        lines.append(f"   📊 {score_str}")
        lines.append(f"   🏷️  Themes: {meta.get('themes', '—')}")

        # Date
        date_val = (
            meta.get("date") or meta.get("start_at") or meta.get("created") or "—"
        )
        if isinstance(date_val, str) and len(date_val) > 10:
            date_val = date_val[:10]
        lines.append(f"   📅 Date: {date_val}")

        # Context snippet
        if include_context:
            text = (
                meta.get("synthesis")
                or meta.get("text")
                or meta.get("summary")
                or meta.get("content")
                or ""
            )
            if text:
                snippet = (text[:400] + "…") if len(text) > 400 else text
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


# ============================================================================
# ROUTED SMART SEARCH (with Query Router + RRF Fusion)
# ============================================================================
# Reference: gemini-deep-research2.txt, gemini-final-prompt.txt
#
# This integrates:
# 1. Query Router: Pre-classifies intent to choose optimal strategy
# 2. RRF Fusion: Proper mathematical fusion of ranked lists
# 3. GraphRAG fallback: For aggregation/global queries


@dataclass
class SmartSearchResult:
    """Result from router-aware smart search."""

    query: str
    matches: List[Any]
    routing_decision: Optional[Dict] = None
    rrf_stats: Optional[Dict] = None
    graphrag_answer: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "match_count": len(self.matches),
            "routing": self.routing_decision,
            "rrf_stats": self.rrf_stats,
            "has_graphrag": self.graphrag_answer is not None,
            "latency_ms": self.latency_ms,
        }


def smart_search(
    query: str,
    limit: int = 10,
    use_router: bool = True,
    use_graphrag: bool = True,
) -> SmartSearchResult:
    """
    Intelligent search using Query Router + RRF Fusion.

    This is the RECOMMENDED entry point for production search:
    1. Router classifies query intent (metadata/keyword/semantic/aggregation)
    2. Optimal alpha and filters are auto-selected
    3. Results are fused using proper RRF algorithm
    4. GraphRAG is used for aggregation queries

    Args:
        query: User's natural language query
        limit: Max results to return
        use_router: Enable query routing (recommended)
        use_graphrag: Enable GraphRAG for aggregation queries

    Returns:
        SmartSearchResult with matches and full transparency
    """
    import time

    start_time = time.time()

    routing_decision = None
    rrf_stats = None
    graphrag_answer = None

    # Step 1: Route the query
    if use_router:
        try:
            from src.processing.query_router import route_query, QueryIntent

            decision = route_query(query)
            routing_decision = decision.to_dict()

            # Handle aggregation queries with GraphRAG
            if use_graphrag and decision.intent == QueryIntent.AGGREGATION:
                try:
                    from src.processing.graph_rag import answer_global_query

                    global_result = answer_global_query(query)
                    graphrag_answer = global_result.get("answer")
                    logger.info(f"🧠 GraphRAG answered aggregation query")
                except Exception as e:
                    logger.warning(f"GraphRAG failed: {e}")

            # Use router's recommended alpha and filters
            alpha = decision.alpha
            auto_filters = (
                decision.filters.to_pinecone_filter() if decision.filters else None
            )

        except ImportError:
            logger.warning("Query router not available, using defaults")
            alpha = HYBRID_ALPHA
            auto_filters = None
    else:
        alpha = HYBRID_ALPHA
        auto_filters = None

    # Step 2: Execute hybrid search
    from gui.services.embedding_service import get_embedding_service
    from gui.services.clients import get_vector_db_client

    embedding_service = get_embedding_service()
    dense_vector = embedding_service.embed_query(query)

    vector_client = get_vector_db_client()
    namespaces = ["full_text", "summaries"]

    # Dense results
    dense_results = vector_client.query_namespaces(
        query_embedding=dense_vector,
        namespaces=namespaces,
        top_k=limit * 2,
        filter_dict=auto_filters,
        include_metadata=True,
    )
    dense_matches = dense_results.matches if dense_results else []

    # Sparse results (if available)
    sparse_matches = []
    try:
        from src.ai.sparse_embeddings import get_sparse_embedder

        sparse_embedder = get_sparse_embedder()
        sparse_vector = sparse_embedder.embed_query(query)
        # Would query sparse index here if available
    except Exception:
        pass

    # Step 3: Apply RRF Fusion
    try:
        from src.processing.rrf_fusion import reciprocal_rank_fusion

        rrf_result = reciprocal_rank_fusion(
            dense_results=dense_matches,
            sparse_results=sparse_matches,
            metadata_results=[],  # Could add metadata-filtered results here
            limit=limit,
            weights={
                "dense": alpha,
                "sparse": 1 - alpha,
                "metadata": 1.0,
            },
        )

        # Convert RRF results back to match format
        matches = []
        for fused in rrf_result.results:

            @dataclass
            class SmartMatch:
                id: str
                score: float
                metadata: Dict
                namespace: str = "full_text"
                rrf_score: float = 0.0
                sources: List[str] = field(default_factory=list)

            matches.append(
                SmartMatch(
                    id=fused.id,
                    score=fused.rrf_score,
                    metadata=fused.metadata,
                    rrf_score=fused.rrf_score,
                    sources=fused.sources,
                )
            )

        rrf_stats = rrf_result.to_dict()

    except ImportError:
        logger.warning("RRF fusion not available, using simple merge")
        matches = dense_matches[:limit]

    latency_ms = (time.time() - start_time) * 1000

    return SmartSearchResult(
        query=query,
        matches=matches,
        routing_decision=routing_decision,
        rrf_stats=rrf_stats,
        graphrag_answer=graphrag_answer,
        latency_ms=latency_ms,
    )


def format_smart_results(result: SmartSearchResult) -> str:
    """Format smart search results with full transparency."""
    lines = [
        f"{'═' * 70}",
        f"🧠 SMART SEARCH RESULTS for: '{result.query}'",
    ]

    # Show routing decision
    if result.routing_decision:
        rd = result.routing_decision
        lines.append(
            f"   🧭 Router: {rd.get('intent', '?')} (conf: {rd.get('confidence', 0):.2f})"
        )
        lines.append(
            f"   ⚖️  Alpha: {rd.get('alpha', 0.5):.2f} (dense) / {1-rd.get('alpha', 0.5):.2f} (sparse)"
        )

    # Show RRF stats
    if result.rrf_stats:
        rrf = result.rrf_stats
        lines.append(
            f"   🔀 RRF: {rrf.get('dense_count', 0)} dense + {rrf.get('sparse_count', 0)} sparse → {rrf.get('total_results', 0)} results"
        )
        lines.append(f"   🎯 Multi-source matches: {rrf.get('multi_source_count', 0)}")

    lines.append(f"   ⏱️  Latency: {result.latency_ms:.0f}ms")
    lines.append(f"{'═' * 70}")

    # Show GraphRAG answer if available
    if result.graphrag_answer:
        lines.append("")
        lines.append(f"🧠 GLOBAL SYNTHESIS (GraphRAG):")
        lines.append(f"   {result.graphrag_answer}")
        lines.append("")
        lines.append(f"{'─' * 70}")

    # Show individual results
    lines.append("")
    for idx, match in enumerate(result.matches, 1):
        meta = match.metadata if hasattr(match, "metadata") else {}
        score = (
            match.rrf_score
            if hasattr(match, "rrf_score")
            else getattr(match, "score", 0)
        )
        sources = match.sources if hasattr(match, "sources") else []

        title = meta.get("title", "Untitled")
        source_str = ", ".join(sources) if sources else "dense"

        lines.append(f"#{idx} │ {title}")
        lines.append(f"   📊 RRF Score: {score:.4f} │ Sources: {source_str}")
        lines.append(f"   🏷️  Themes: {meta.get('themes', '—')}")

        text = meta.get("synthesis") or meta.get("text") or meta.get("summary") or ""
        if text:
            snippet = (text[:300] + "…") if len(text) > 300 else text
            lines.append(f"   💬 {snippet}")

        lines.append("")

    return "\n".join(lines)
