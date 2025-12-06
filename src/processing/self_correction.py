"""
Self-Correction Loop for RAG (Retrieval-Augmented Generation)

Implements confidence-aware retrieval with automatic re-querying:
1. Initial retrieval with confidence scoring
2. Low-confidence detection (score < threshold)
3. Query expansion and reformulation
4. Hierarchical retrieval (child → parent escalation)
5. Multi-strategy fallback (dense → hybrid → full-text)

This achieves higher accuracy by:
- Detecting when retrieval quality is insufficient
- Automatically trying alternative strategies
- Expanding queries to capture semantic variations
- Escalating to broader context when needed
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies in order of preference."""
    DENSE = "dense"           # Pure semantic search
    HYBRID = "hybrid"         # Dense + sparse combined
    SPARSE = "sparse"         # Pure keyword/lexical
    PARENT_CONTEXT = "parent" # Escalate to parent chunks
    FULL_TEXT = "full_text"   # Search full transcript namespace
    EXPANDED = "expanded"     # Query expansion with synonyms


@dataclass
class RetrievalResult:
    """Result from a retrieval attempt with confidence metrics."""
    matches: List[Dict]
    strategy: RetrievalStrategy
    confidence: float          # 0.0 - 1.0 overall confidence
    top_score: float          # Highest individual match score
    score_variance: float     # Variance in scores (low = confident)
    query_used: str           # Actual query executed
    latency_ms: float         # Time taken
    
    @property
    def is_confident(self) -> bool:
        """Check if result meets confidence threshold."""
        return self.confidence >= 0.7 and self.top_score >= 0.5
    
    @property
    def needs_correction(self) -> bool:
        """Check if self-correction should be attempted."""
        return (
            self.confidence < 0.5 or 
            self.top_score < 0.4 or
            len(self.matches) == 0
        )


@dataclass
class CorrectionAttempt:
    """Record of a self-correction attempt."""
    strategy: RetrievalStrategy
    query: str
    result: Optional[RetrievalResult]
    success: bool
    reason: str


@dataclass 
class SelfCorrectionResult:
    """Final result after self-correction loop."""
    final_result: RetrievalResult
    attempts: List[CorrectionAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0
    corrections_applied: int = 0
    
    @property
    def was_corrected(self) -> bool:
        """Check if correction was needed and applied."""
        return self.corrections_applied > 0


class SelfCorrectionLoop:
    """
    Implements self-correction for RAG retrieval.
    
    When initial retrieval has low confidence, automatically:
    1. Reformulate the query
    2. Try alternative retrieval strategies
    3. Expand context with parent chunks
    4. Combine results from multiple strategies
    """
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.6"))
    SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))
    MAX_CORRECTION_ATTEMPTS = int(os.getenv("RAG_MAX_CORRECTIONS", "3"))
    
    def __init__(
        self,
        dense_search_fn: Optional[Callable] = None,
        hybrid_search_fn: Optional[Callable] = None,
        sparse_search_fn: Optional[Callable] = None,
        parent_context_fn: Optional[Callable] = None,
        query_expand_fn: Optional[Callable] = None,
    ):
        """
        Initialize with search function callbacks.
        
        Args:
            dense_search_fn: Function for dense/semantic search
            hybrid_search_fn: Function for hybrid (dense+sparse) search
            sparse_search_fn: Function for sparse/keyword search
            parent_context_fn: Function to fetch parent chunk context
            query_expand_fn: Function to expand query with synonyms
        """
        self.dense_search = dense_search_fn
        self.hybrid_search = hybrid_search_fn
        self.sparse_search = sparse_search_fn
        self.parent_context = parent_context_fn
        self.query_expand = query_expand_fn
        
        # Strategy order (try in sequence until confident)
        self.strategy_order = [
            RetrievalStrategy.DENSE,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.EXPANDED,
            RetrievalStrategy.PARENT_CONTEXT,
            RetrievalStrategy.SPARSE,
        ]
        
    def search_with_correction(
        self,
        query: str,
        limit: int = 10,
        initial_strategy: RetrievalStrategy = RetrievalStrategy.DENSE,
    ) -> SelfCorrectionResult:
        """
        Execute search with automatic self-correction.
        
        Args:
            query: User's search query
            limit: Maximum results to return
            initial_strategy: Starting retrieval strategy
            
        Returns:
            SelfCorrectionResult with final matches and correction history
        """
        import time
        start_time = time.perf_counter()
        
        attempts: List[CorrectionAttempt] = []
        corrections_applied = 0
        
        # Try initial strategy
        logger.info(f"🔍 Initial search: '{query}' via {initial_strategy.value}")
        result = self._execute_strategy(initial_strategy, query, limit)
        
        attempts.append(CorrectionAttempt(
            strategy=initial_strategy,
            query=query,
            result=result,
            success=result is not None and result.is_confident,
            reason="initial"
        ))
        
        # Check if correction needed
        if result is None or result.needs_correction:
            logger.info(f"⚠️ Low confidence ({result.confidence if result else 0:.2f}), starting self-correction")
            
            # Try alternative strategies
            for strategy in self.strategy_order:
                if strategy == initial_strategy:
                    continue
                    
                if corrections_applied >= self.MAX_CORRECTION_ATTEMPTS:
                    logger.info(f"⏹️ Max correction attempts ({self.MAX_CORRECTION_ATTEMPTS}) reached")
                    break
                
                # Expand query for certain strategies
                search_query = query
                if strategy == RetrievalStrategy.EXPANDED and self.query_expand:
                    search_query = self.query_expand(query)
                    logger.info(f"📝 Expanded query: '{search_query}'")
                
                logger.info(f"🔄 Correction attempt {corrections_applied + 1}: {strategy.value}")
                alt_result = self._execute_strategy(strategy, search_query, limit)
                corrections_applied += 1
                
                attempts.append(CorrectionAttempt(
                    strategy=strategy,
                    query=search_query,
                    result=alt_result,
                    success=alt_result is not None and alt_result.is_confident,
                    reason=f"correction_{corrections_applied}"
                ))
                
                # Check if this result is better
                if alt_result and alt_result.is_confident:
                    logger.info(f"✅ Correction successful via {strategy.value} (conf: {alt_result.confidence:.2f})")
                    result = alt_result
                    break
                elif alt_result and (result is None or alt_result.confidence > result.confidence):
                    logger.info(f"📈 Improved confidence: {result.confidence if result else 0:.2f} → {alt_result.confidence:.2f}")
                    result = alt_result
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # If still no result, create empty one
        if result is None:
            result = RetrievalResult(
                matches=[],
                strategy=initial_strategy,
                confidence=0.0,
                top_score=0.0,
                score_variance=0.0,
                query_used=query,
                latency_ms=total_latency
            )
        
        return SelfCorrectionResult(
            final_result=result,
            attempts=attempts,
            total_latency_ms=total_latency,
            corrections_applied=corrections_applied
        )
    
    def _execute_strategy(
        self, 
        strategy: RetrievalStrategy, 
        query: str, 
        limit: int
    ) -> Optional[RetrievalResult]:
        """Execute a single retrieval strategy."""
        import time
        start = time.perf_counter()
        
        try:
            matches = []
            
            if strategy == RetrievalStrategy.DENSE and self.dense_search:
                matches = self.dense_search(query, limit)
            elif strategy == RetrievalStrategy.HYBRID and self.hybrid_search:
                matches = self.hybrid_search(query, limit)
            elif strategy == RetrievalStrategy.SPARSE and self.sparse_search:
                matches = self.sparse_search(query, limit)
            elif strategy == RetrievalStrategy.PARENT_CONTEXT and self.parent_context:
                matches = self.parent_context(query, limit)
            elif strategy == RetrievalStrategy.EXPANDED and self.dense_search:
                # Use dense search with expanded query
                matches = self.dense_search(query, limit)
            else:
                logger.warning(f"Strategy {strategy.value} not available")
                return None
            
            latency = (time.perf_counter() - start) * 1000
            
            # Calculate confidence metrics
            if matches:
                scores = [m.get('score', 0) for m in matches]
                top_score = max(scores)
                avg_score = sum(scores) / len(scores)
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                
                # Confidence based on top score, average, and result count
                confidence = (
                    0.4 * top_score +
                    0.3 * avg_score +
                    0.3 * min(1.0, len(matches) / limit)
                )
            else:
                top_score = 0.0
                variance = 0.0
                confidence = 0.0
            
            return RetrievalResult(
                matches=matches,
                strategy=strategy,
                confidence=confidence,
                top_score=top_score,
                score_variance=variance,
                query_used=query,
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Strategy {strategy.value} failed: {e}")
            return None
    
    def get_correction_summary(self, result: SelfCorrectionResult) -> str:
        """Generate human-readable summary of correction process."""
        lines = [
            f"{'═' * 50}",
            f"🔄 SELF-CORRECTION SUMMARY",
            f"{'═' * 50}",
            f"",
            f"Query: {result.final_result.query_used}",
            f"Final Strategy: {result.final_result.strategy.value}",
            f"Confidence: {result.final_result.confidence:.2f}",
            f"Top Score: {result.final_result.top_score:.2f}",
            f"Results: {len(result.final_result.matches)}",
            f"Total Latency: {result.total_latency_ms:.0f}ms",
            f"Corrections Applied: {result.corrections_applied}",
            f"",
        ]
        
        if result.attempts:
            lines.append("Attempts:")
            for i, attempt in enumerate(result.attempts, 1):
                status = "✅" if attempt.success else "❌"
                conf = attempt.result.confidence if attempt.result else 0
                lines.append(f"  {i}. {status} {attempt.strategy.value}: conf={conf:.2f} ({attempt.reason})")
        
        lines.append(f"{'═' * 50}")
        
        return "\n".join(lines)


class QueryExpander:
    """
    Expand queries with synonyms and related terms.
    
    Uses a simple rule-based approach; can be enhanced with LLM.
    """
    
    # Common domain-specific expansions
    EXPANSIONS = {
        'meeting': ['discussion', 'call', 'conversation', 'talk'],
        'project': ['initiative', 'plan', 'work', 'task'],
        'team': ['group', 'squad', 'department', 'colleagues'],
        'deadline': ['due date', 'timeline', 'schedule', 'target date'],
        'issue': ['problem', 'bug', 'concern', 'challenge'],
        'feature': ['capability', 'functionality', 'enhancement'],
        'customer': ['client', 'user', 'consumer'],
        'revenue': ['sales', 'income', 'earnings'],
        'cost': ['expense', 'spending', 'budget'],
    }
    
    def expand(self, query: str) -> str:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with OR clauses
        """
        words = query.lower().split()
        expanded_parts = []
        
        for word in words:
            if word in self.EXPANSIONS:
                # Add word and its synonyms
                synonyms = self.EXPANSIONS[word]
                expanded_parts.append(f"({word} OR {' OR '.join(synonyms[:2])})")
            else:
                expanded_parts.append(word)
        
        return ' '.join(expanded_parts)
    
    def expand_with_llm(self, query: str, llm_fn: Callable) -> str:
        """
        Expand query using LLM for smarter reformulation.
        
        Args:
            query: Original query
            llm_fn: Function to call LLM with prompt
            
        Returns:
            LLM-expanded query
        """
        prompt = f"""Reformulate this search query to find more relevant results.
Add synonyms and related terms, but keep it concise.

Original: {query}

Return only the expanded query, nothing else."""
        
        try:
            expanded = llm_fn(prompt)
            return expanded.strip()
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return self.expand(query)  # Fall back to rule-based
