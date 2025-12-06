"""
Search Service for PlaudBlender.

Provides ultra-granular, clearly-named search actions:
- search_single_namespace(): Search one namespace
- search_all_namespaces(): Cross-namespace parallel search
- search_full_text(): Search only the full_text namespace
- search_summaries(): Search only the summaries namespace
- search_with_rerank(): Search + Pinecone rerank for better relevance

All embedding is delegated to the centralized EmbeddingService.
Dimensions are automatically synced with Pinecone via IndexManager.
"""
import os
from typing import List, Optional, Dict, Any

from gui.services.embedding_service import get_embedding_service, EmbeddingError
from gui.services.clients import get_pinecone_client
from gui.utils.logging import log


# ============================================================================
# CONFIGURATION
# ============================================================================
RERANK_ENABLED = os.getenv("PINECONE_RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")


# ============================================================================
# AUTO-SYNC HELPER
# ============================================================================

def _ensure_dimension_sync():
    """
    Automatically sync embedding dimension with Pinecone index.
    Called before any search operation to prevent dimension mismatches.
    """
    try:
        from gui.services.index_manager import sync_dimensions
        dim, action = sync_dimensions()
        if action == "auto_adjusted":
            log('INFO', f"🔄 Auto-adjusted embedding to {dim}d to match index")
    except Exception as e:
        log('WARNING', f"Could not sync dimensions: {e}")


# ============================================================================
# NAMESPACE CONSTANTS
# ============================================================================
NAMESPACE_FULL_TEXT = "full_text"
NAMESPACE_SUMMARIES = "summaries"
ALL_NAMESPACES = [NAMESPACE_FULL_TEXT, NAMESPACE_SUMMARIES]


# ============================================================================
# GRANULAR SEARCH ACTIONS
# ============================================================================

def search_full_text(
    query: str,
    limit: int = 5,
    include_context: bool = True,
    filter_dict: Optional[Dict] = None,
) -> str:
    """
    🔍 SEARCH FULL TEXT NAMESPACE
    
    Searches the 'full_text' namespace containing complete transcript chunks.
    Use this when you want to find specific passages or detailed content.
    
    Args:
        query: Natural language search query
        limit: Max results to return (default: 5)
        include_context: Include text snippets in results
        filter_dict: Optional Pinecone metadata filters
    
    Returns:
        Formatted search results as string
    """
    log('INFO', f"🔍 search_full_text: '{query}' (limit={limit})")
    return _execute_search(
        query=query,
        namespace=NAMESPACE_FULL_TEXT,
        limit=limit,
        include_context=include_context,
        filter_dict=filter_dict,
        namespace_label="📄 Full Text"
    )


def search_summaries(
    query: str,
    limit: int = 5,
    include_context: bool = True,
    filter_dict: Optional[Dict] = None,
) -> str:
    """
    📝 SEARCH SUMMARIES NAMESPACE
    
    Searches the 'summaries' namespace containing AI-generated syntheses.
    Use this when you want high-level topic matching or thematic search.
    
    Args:
        query: Natural language search query
        limit: Max results to return (default: 5)
        include_context: Include summary text in results
        filter_dict: Optional Pinecone metadata filters
    
    Returns:
        Formatted search results as string
    """
    log('INFO', f"📝 search_summaries: '{query}' (limit={limit})")
    return _execute_search(
        query=query,
        namespace=NAMESPACE_SUMMARIES,
        limit=limit,
        include_context=include_context,
        filter_dict=filter_dict,
        namespace_label="📝 Summaries"
    )


def search_all_namespaces(
    query: str,
    limit: int = 5,
    include_context: bool = True,
    filter_dict: Optional[Dict] = None,
) -> str:
    """
    🌐 CROSS-NAMESPACE SEARCH (PARALLEL)
    
    Searches BOTH 'full_text' AND 'summaries' namespaces in parallel.
    Results are merged and ranked by relevance score.
    
    This is the most comprehensive search - finds both specific passages
    AND high-level thematic matches.
    
    Args:
        query: Natural language search query
        limit: Max results to return after merging (default: 5)
        include_context: Include text/summary in results
        filter_dict: Optional Pinecone metadata filters
    
    Returns:
        Formatted search results with namespace indicators
    """
    log('INFO', f"🌐 search_all_namespaces: '{query}' (limit={limit})")
    
    if not query.strip():
        return "❌ Error: Please provide a search query."
    
    # AUTO-SYNC: Ensure embedding dimension matches Pinecone index
    _ensure_dimension_sync()
    
    # Get embedding from centralized service
    try:
        embedding_service = get_embedding_service()
        vector = embedding_service.embed_query(query)
        log('INFO', f"   ✅ Generated {len(vector)}-dim embedding")
    except EmbeddingError as e:
        return f"❌ Embedding Error: {e}"
    
    # Get Pinecone client
    pinecone_client = get_pinecone_client()
    
    # Execute cross-namespace query
    try:
        results = pinecone_client.query_namespaces(
            query_embedding=vector,
            namespaces=ALL_NAMESPACES,
            top_k=limit,
            filter_dict=filter_dict,
            include_metadata=True,
        )
        
        return _format_results(
            query=query,
            matches=results.matches,
            include_context=include_context,
            show_namespace=True,
        )
        
    except Exception as e:
        log('ERROR', f"Cross-namespace search failed: {e}")
        return f"❌ Search Error: {e}"


def search_with_rerank(
    query: str,
    limit: int = 5,
    include_context: bool = True,
    filter_dict: Optional[Dict] = None,
    namespaces: Optional[List[str]] = None,
    rerank_model: str = RERANK_MODEL,
) -> str:
    """
    🏆 SEARCH + RERANK (highest quality)

    Performs cross-namespace search then reranks results via Pinecone inference
    for the best relevance ordering.

    Args:
        query: Natural language search query
        limit: Max results to return after reranking
        include_context: Include text snippets in results
        filter_dict: Optional Pinecone metadata filters
        namespaces: Namespaces to search (default: all)
        rerank_model: Pinecone reranker model (default: bge-reranker-v2-m3)

    Returns:
        Formatted search results, reranked by query relevance
    """
    log('INFO', f"🏆 search_with_rerank: '{query}' (limit={limit}, model={rerank_model})")

    if not query.strip():
        return "❌ Error: Please provide a search query."

    _ensure_dimension_sync()

    try:
        embedding_service = get_embedding_service()
        vector = embedding_service.embed_query(query)
        log('INFO', f"   ✅ Generated {len(vector)}-dim embedding")
    except EmbeddingError as e:
        return f"❌ Embedding Error: {e}"

    pinecone_client = get_pinecone_client()
    target_namespaces = namespaces or ALL_NAMESPACES

    # Fetch more candidates than needed for reranking
    candidate_limit = max(limit * 3, 15)

    try:
        results = pinecone_client.query_namespaces(
            query_embedding=vector,
            namespaces=target_namespaces,
            top_k=candidate_limit,
            filter_dict=filter_dict,
            include_metadata=True,
        )

        if not results.matches:
            return f"No matches found for '{query}'."

        # Build documents for reranking
        docs = []
        match_map = {}
        for i, match in enumerate(results.matches):
            meta = match.metadata or {}
            text = meta.get('synthesis') or meta.get('text') or meta.get('summary') or meta.get('title') or ''
            if text:
                doc_id = f"doc_{i}"
                docs.append({"id": doc_id, "text": text[:2000]})  # Truncate for reranker
                match_map[doc_id] = match

        if not docs:
            log('WARNING', "No text found in matches for reranking; returning unranked results")
            return _format_results(query, results.matches[:limit], include_context, show_namespace=True)

        # Call rerank
        rerank_result = pinecone_client.rerank(
            query=query,
            documents=docs,
            model=rerank_model,
            top_n=limit,
            return_documents=False,
        )

        if "error" in rerank_result:
            log('WARNING', f"Rerank failed: {rerank_result['error']}; returning unranked results")
            return _format_results(query, results.matches[:limit], include_context, show_namespace=True)

        # Rebuild matches in reranked order, stashing original retrieval score
        reranked_matches = []
        for item in rerank_result.get("data", []):
            doc_id = item.get("document", {}).get("id") or f"doc_{item.get('index', 0)}"
            if doc_id in match_map:
                match = match_map[doc_id]
                # Stash original vector similarity score for transparency
                if match.metadata is None:
                    match.metadata = {}
                match.metadata['_retrieval_score'] = match.score
                # Update to rerank score
                match.score = item.get("score", match.score)
                reranked_matches.append(match)

        log('INFO', f"   🏆 Reranked {len(reranked_matches)} results")
        return _format_results(query, reranked_matches, include_context, show_namespace=True, reranked=True)

    except Exception as e:
        log('ERROR', f"Search with rerank failed: {e}")
        return f"❌ Search Error: {e}"


def search_single_namespace(
    query: str,
    namespace: str = "",
    limit: int = 5,
    include_context: bool = True,
    filter_dict: Optional[Dict] = None,
) -> str:
    """
    🎯 SEARCH SINGLE NAMESPACE
    
    Searches a specific namespace (or default if empty).
    Use search_full_text() or search_summaries() for clearer intent.
    
    Args:
        query: Natural language search query
        namespace: Namespace to search (empty = default)
        limit: Max results to return
        include_context: Include text snippets
        filter_dict: Optional metadata filters
    
    Returns:
        Formatted search results
    """
    log('INFO', f"🎯 search_single_namespace: '{query}' in '{namespace or 'default'}' (limit={limit})")
    return _execute_search(
        query=query,
        namespace=namespace,
        limit=limit,
        include_context=include_context,
        filter_dict=filter_dict,
    )


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def semantic_search(
    query: str, 
    limit: int = 5, 
    include_context: bool = True,
    namespace: Optional[str] = None,
) -> str:
    """
    Legacy function - redirects to search_single_namespace.
    
    Kept for backward compatibility with existing code.
    """
    return search_single_namespace(
        query=query,
        namespace=namespace or "",
        limit=limit,
        include_context=include_context,
    )


def cross_namespace_search(
    query: str,
    limit: int = 5,
    include_context: bool = True,
    namespaces: Optional[List[str]] = None,
) -> str:
    """
    Legacy function - redirects to search_all_namespaces.
    
    Kept for backward compatibility with existing code.
    """
    return search_all_namespaces(
        query=query,
        limit=limit,
        include_context=include_context,
    )


# Keep old function name for imports
def get_embedding(query: str) -> Optional[List[float]]:
    """
    Legacy function - use embedding_service.embed_query() instead.
    
    Kept for backward compatibility.
    """
    try:
        return get_embedding_service().embed_query(query)
    except EmbeddingError:
        return None


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _execute_search(
    query: str,
    namespace: str,
    limit: int,
    include_context: bool,
    filter_dict: Optional[Dict] = None,
    namespace_label: Optional[str] = None,
) -> str:
    """Internal: Execute a single-namespace search."""
    
    if not query.strip():
        return "❌ Error: Please provide a search query."
    
    # AUTO-SYNC: Ensure embedding dimension matches Pinecone index
    _ensure_dimension_sync()
    
    # Get embedding from centralized service
    try:
        embedding_service = get_embedding_service()
        vector = embedding_service.embed_query(query)
        log('INFO', f"   ✅ Generated {len(vector)}-dim embedding")
    except EmbeddingError as e:
        return f"❌ Embedding Error: {e}"
    
    # Get Pinecone client and query
    pinecone_client = get_pinecone_client()
    
    try:
        matches = pinecone_client.query_similar(
            query_embedding=vector,
            top_k=limit,
            filter_dict=filter_dict,
            namespace=namespace,
        )
        
        return _format_results(
            query=query,
            matches=matches,
            include_context=include_context,
            namespace_label=namespace_label,
        )
        
    except Exception as e:
        log('ERROR', f"Search failed: {e}")
        return f"❌ Search Error: {e}"


def _format_results(
    query: str,
    matches: list,
    include_context: bool,
    show_namespace: bool = False,
    namespace_label: Optional[str] = None,
    reranked: bool = False,
) -> str:
    """
    Internal: Format search results for display.
    
    Shows retrieval_score (vector similarity) and rerank_score (semantic relevance)
    when reranking is enabled, giving users transparency into ranking quality.
    """
    
    if not matches:
        return f"No matches found for '{query}'."
    
    # Header with search mode indicator
    mode_indicator = "🏆 RERANKED " if reranked else "🔍 "
    lines = [
        f"{'═' * 60}",
        f"{mode_indicator}SEARCH RESULTS for: '{query}'",
        f"   Found {len(matches)} matches",
        f"{'═' * 60}",
        "",
    ]

    for idx, match in enumerate(matches, 1):
        meta = match.metadata or {}
        
        # ──────────── SCORE DISPLAY (granular) ────────────
        # Primary score: always show as percentage
        score_pct = match.score * 100
        
        # Check if this is reranked (score came from reranker vs vector similarity)
        # Rerank scores are typically 0-1 scale from the cross-encoder
        retrieval_score = meta.get('_retrieval_score')  # Stashed original if reranked
        
        # Build score string with transparency
        if reranked and retrieval_score is not None:
            # Show both scores for full transparency
            score_str = f"rerank: {score_pct:.1f}% │ retrieval: {retrieval_score*100:.1f}%"
        else:
            score_str = f"score: {score_pct:.1f}%"
        
        title = meta.get('title', 'Untitled')
        header = f"#{idx} │ {title}"
        
        # Add namespace indicator
        if show_namespace and hasattr(match, 'namespace'):
            ns_icon = "📄" if match.namespace == NAMESPACE_FULL_TEXT else "📝"
            header += f" │ {ns_icon} {match.namespace}"
        elif namespace_label:
            header += f" │ {namespace_label}"
        
        lines.append(header)
        lines.append(f"   📊 {score_str}")
        lines.append(f"   🏷️  Themes: {meta.get('themes', '—')}")
        
        # Handle date field variations
        date_val = meta.get('date') or meta.get('start_at') or meta.get('created') or '—'
        if isinstance(date_val, str) and len(date_val) > 10:
            date_val = date_val[:10]
        lines.append(f"   📅 Date: {date_val}")
        
        # Include context snippet
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
                lines.append(f"   💬 Snippet: {snippet}")
        
        lines.append("")
        lines.append(f"{'─' * 60}")
        lines.append("")

    log('INFO', f"   📊 Returned {len(matches)} formatted results (reranked={reranked})")
    return "\n".join(lines)
