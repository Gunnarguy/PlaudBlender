"""
Query Router / Intent Classifier for PlaudBlender

Pre-classifies user queries BEFORE vector search to route to optimal strategy:
- METADATA_LOOKUP: Exact ID/date/keyword lookup (SQL/filter-first)
- SEMANTIC_EXPLORATION: Broad conceptual search (dense-heavy)
- KEYWORD_MATCH: Specific terms, acronyms, proper nouns (sparse-heavy)
- HYBRID_BALANCED: Mixed intent (balanced alpha)
- AGGREGATION: Questions requiring cross-document synthesis (GraphRAG)

The Router Pattern is critical for 99.9% accuracy because:
1. Dense vectors FAIL on exact matches (part numbers, IDs, acronyms)
2. Sparse vectors FAIL on conceptual similarity
3. GraphRAG is REQUIRED for multi-hop reasoning queries
4. SQL filters GUARANTEE scope accuracy for metadata lookups

Reference: gemini-deep-research2.txt, gemini-final-prompt.txt
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Classification of user query intent for optimal routing."""
    METADATA_LOOKUP = "metadata_lookup"      # Exact ID/date/recording lookup
    SEMANTIC_EXPLORATION = "semantic"        # Broad conceptual exploration
    KEYWORD_MATCH = "keyword"                # Specific terms, acronyms
    HYBRID_BALANCED = "hybrid"               # Mixed intent
    AGGREGATION = "aggregation"              # Cross-document synthesis
    ENTITY_LOOKUP = "entity"                 # GraphRAG entity search


@dataclass
class ExtractedFilters:
    """Metadata filters auto-extracted from natural language query."""
    recording_id: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    
    def to_pinecone_filter(self) -> Optional[Dict]:
        """Convert to Pinecone filter dict."""
        filters = {}
        
        if self.recording_id:
            filters["recording_id"] = {"$eq": self.recording_id}
        
        if self.date_start and self.date_end:
            filters["timestamp"] = {
                "$gte": self.date_start,
                "$lte": self.date_end,
            }
        elif self.date_start:
            filters["timestamp"] = {"$gte": self.date_start}
        elif self.date_end:
            filters["timestamp"] = {"$lte": self.date_end}
        
        if self.themes:
            if len(self.themes) == 1:
                filters["theme"] = {"$eq": self.themes[0]}
            else:
                filters["theme"] = {"$in": self.themes}
        
        return filters if filters else None
    
    @property
    def has_filters(self) -> bool:
        """Check if any filters were extracted."""
        return bool(
            self.recording_id or 
            self.date_start or 
            self.date_end or 
            self.keywords or
            self.themes
        )


@dataclass
class RoutingDecision:
    """Complete routing decision with transparency."""
    intent: QueryIntent
    alpha: float                          # Hybrid search alpha (0=sparse, 1=dense)
    use_rerank: bool                      # Whether to apply neural reranking
    use_graphrag: bool                    # Whether to use GraphRAG
    filters: Optional[ExtractedFilters]   # Auto-extracted filters
    confidence: float                     # Router confidence in decision
    reasoning: str                        # Human-readable explanation
    cleaned_query: str                    # Query with filters removed
    
    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "alpha": self.alpha,
            "use_rerank": self.use_rerank,
            "use_graphrag": self.use_graphrag,
            "has_filters": self.filters.has_filters if self.filters else False,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class QueryRouter:
    """
    Intelligent query router using pattern matching and optional LLM classification.
    
    Routes queries to optimal search strategy:
    - Pattern-based rules for fast classification (< 5ms)
    - Optional LLM fallback for ambiguous queries
    
    Example Classifications:
    - "recording abc123" → METADATA_LOOKUP (alpha=0.0, filter by ID)
    - "what themes emerged" → AGGREGATION (use GraphRAG)
    - "mentions of Project X" → KEYWORD_MATCH (alpha=0.2)
    - "how does the team feel about" → SEMANTIC_EXPLORATION (alpha=0.9)
    - "action items from last week" → HYBRID_BALANCED (alpha=0.5, date filter)
    """
    
    # Pattern matchers for intent classification
    METADATA_PATTERNS = [
        r'\b(recording|id|rec)[:\s]*([\w-]+)',           # recording:abc123
        r'\b(from|on|date)[:\s]*(\d{4}-\d{2}-\d{2})',    # from:2025-01-15
        r'\b(before|after|since)[:\s]*(\d{4}-\d{2}-\d{2})',
    ]
    
    AGGREGATION_PATTERNS = [
        r'\b(what|which)\s+(themes?|topics?|patterns?|trends?)',
        r'\b(summarize|overview|aggregate|across all)',
        r'\b(how many|count|total|most common)',
        r'\b(compare|contrast|relationship between)',
        r'\b(all|every)\s+(mention|time|instance)',
    ]
    
    ENTITY_PATTERNS = [
        r'\b(who|person|people|team|mentioned)',
        r'\b(which project|what project|project\s+\w+)',
        r'\b(organization|company|org)',
        r'\b(connected to|related to|involved with)',
    ]
    
    KEYWORD_INDICATORS = [
        r'"[^"]+"',                    # Quoted exact phrases
        r'\b[A-Z]{2,}\b',             # Acronyms (API, CEO, KPI)
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+', # CamelCase (ProjectAlpha)
        r'\b\d+[A-Za-z]+|\b[A-Za-z]+\d+', # AlphaNum IDs (V2, Phase3)
    ]
    
    SEMANTIC_INDICATORS = [
        r'\b(feel|think|believe|seems?|appears?)',
        r'\b(meaning|concept|idea|implication)',
        r'\b(similar|like|related|about)',
        r'\b(understand|explain|what is)',
    ]
    
    def __init__(self, use_llm_fallback: bool = False):
        """
        Initialize router.
        
        Args:
            use_llm_fallback: Use Gemini for ambiguous queries (slower, more accurate)
        """
        self.use_llm = use_llm_fallback
    
    def route(self, query: str) -> RoutingDecision:
        """
        Route query to optimal search strategy.
        
        Args:
            query: User's natural language query
            
        Returns:
            RoutingDecision with intent, alpha, filters, and reasoning
        """
        query_lower = query.lower().strip()
        
        # Extract metadata filters first
        filters = self._extract_filters(query)
        cleaned_query = self._clean_query(query, filters)
        
        # Pattern-based classification
        intent, confidence, reasoning = self._classify_intent(query_lower, filters)
        
        # Determine alpha based on intent
        alpha = self._compute_alpha(intent, query_lower)
        
        # Determine if reranking helps
        use_rerank = intent not in [QueryIntent.METADATA_LOOKUP]
        
        # Determine if GraphRAG is needed
        use_graphrag = intent in [QueryIntent.AGGREGATION, QueryIntent.ENTITY_LOOKUP]
        
        decision = RoutingDecision(
            intent=intent,
            alpha=alpha,
            use_rerank=use_rerank,
            use_graphrag=use_graphrag,
            filters=filters,
            confidence=confidence,
            reasoning=reasoning,
            cleaned_query=cleaned_query,
        )
        
        logger.info(f"🧭 Router: '{query[:50]}...' → {intent.value} (α={alpha:.2f})")
        return decision
    
    def _extract_filters(self, query: str) -> ExtractedFilters:
        """Extract metadata filters from natural language."""
        filters = ExtractedFilters()
        
        # Recording ID
        rec_match = re.search(r'\b(?:recording|rec|id)[:\s]*([\w-]+)', query, re.I)
        if rec_match:
            filters.recording_id = rec_match.group(1)
        
        # Date extraction
        date_patterns = [
            (r'(?:from|since|after)[:\s]*(\d{4}-\d{2}-\d{2})', 'start'),
            (r'(?:to|until|before)[:\s]*(\d{4}-\d{2}-\d{2})', 'end'),
            (r'(?:on|date)[:\s]*(\d{4}-\d{2}-\d{2})', 'exact'),
        ]
        
        for pattern, date_type in date_patterns:
            match = re.search(pattern, query, re.I)
            if match:
                date_val = match.group(1)
                if date_type in ['start', 'exact']:
                    filters.date_start = date_val
                if date_type in ['end', 'exact']:
                    filters.date_end = date_val
        
        # Time-relative expressions
        time_relative = {
            r'last\s+week': ('7_days_ago', None),
            r'last\s+month': ('30_days_ago', None),
            r'yesterday': ('yesterday', 'yesterday'),
            r'today': ('today', 'today'),
        }
        
        for pattern, (start_marker, end_marker) in time_relative.items():
            if re.search(pattern, query, re.I):
                # In production, convert to actual dates
                filters.date_start = start_marker
                if end_marker:
                    filters.date_end = end_marker
        
        # Quoted keywords
        quoted = re.findall(r'"([^"]+)"', query)
        filters.keywords.extend(quoted)
        
        return filters
    
    def _clean_query(self, query: str, filters: ExtractedFilters) -> str:
        """Remove extracted filter syntax from query."""
        cleaned = query
        
        # Remove filter syntax
        patterns_to_remove = [
            r'\b(?:recording|rec|id)[:\s]*[\w-]+',
            r'\b(?:from|to|since|until|before|after|on|date)[:\s]*\d{4}-\d{2}-\d{2}',
            r'\blast\s+(?:week|month|year)',
            r'\b(?:yesterday|today)',
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.I)
        
        # Clean up whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip() or query
    
    def _classify_intent(
        self, 
        query: str, 
        filters: ExtractedFilters
    ) -> Tuple[QueryIntent, float, str]:
        """Classify query intent using pattern matching."""
        
        # Check for metadata lookup (highest priority)
        if filters.recording_id:
            return (
                QueryIntent.METADATA_LOOKUP, 
                0.95, 
                f"Recording ID '{filters.recording_id}' specified"
            )
        
        if filters.date_start or filters.date_end:
            if not any(re.search(p, query) for p in self.SEMANTIC_INDICATORS):
                return (
                    QueryIntent.METADATA_LOOKUP,
                    0.85,
                    "Date filter specified with no semantic indicators"
                )
        
        # Check aggregation patterns
        for pattern in self.AGGREGATION_PATTERNS:
            if re.search(pattern, query, re.I):
                return (
                    QueryIntent.AGGREGATION,
                    0.90,
                    f"Aggregation pattern matched: requires cross-document synthesis"
                )
        
        # Check entity patterns
        for pattern in self.ENTITY_PATTERNS:
            if re.search(pattern, query, re.I):
                return (
                    QueryIntent.ENTITY_LOOKUP,
                    0.85,
                    "Entity/relationship query: GraphRAG recommended"
                )
        
        # Check keyword indicators
        keyword_matches = sum(
            1 for p in self.KEYWORD_INDICATORS if re.search(p, query)
        )
        if keyword_matches >= 2:
            return (
                QueryIntent.KEYWORD_MATCH,
                0.80,
                f"{keyword_matches} keyword indicators (acronyms, quoted phrases)"
            )
        
        # Check semantic indicators
        semantic_matches = sum(
            1 for p in self.SEMANTIC_INDICATORS if re.search(p, query, re.I)
        )
        if semantic_matches >= 1:
            return (
                QueryIntent.SEMANTIC_EXPLORATION,
                0.75,
                "Semantic exploration query: conceptual matching needed"
            )
        
        # Default to hybrid
        return (
            QueryIntent.HYBRID_BALANCED,
            0.60,
            "Mixed/ambiguous intent: balanced hybrid search"
        )
    
    def _compute_alpha(self, intent: QueryIntent, query: str) -> float:
        """Compute optimal alpha (sparse/dense weight) for intent."""
        
        alpha_map = {
            QueryIntent.METADATA_LOOKUP: 0.0,       # Pure sparse/filter
            QueryIntent.KEYWORD_MATCH: 0.2,         # Mostly sparse
            QueryIntent.HYBRID_BALANCED: 0.5,       # Equal balance
            QueryIntent.AGGREGATION: 0.6,           # Slightly dense
            QueryIntent.ENTITY_LOOKUP: 0.5,         # Balanced for entities
            QueryIntent.SEMANTIC_EXPLORATION: 0.9,  # Mostly dense
        }
        
        base_alpha = alpha_map.get(intent, 0.5)
        
        # Adjust based on query characteristics
        if filters_keywords := re.findall(r'"[^"]+"', query):
            # Has quoted phrases: reduce alpha (more keyword)
            base_alpha = max(0.1, base_alpha - 0.2)
        
        if re.search(r'\b[A-Z]{2,}\b', query):
            # Has acronyms: reduce alpha
            base_alpha = max(0.1, base_alpha - 0.1)
        
        return round(base_alpha, 2)


class LLMQueryRouter(QueryRouter):
    """
    Extended router using Gemini for ambiguous query classification.
    
    Falls back to LLM when pattern-based confidence is low.
    """
    
    CLASSIFICATION_PROMPT = """Classify the following search query into one of these categories:

Categories:
- METADATA_LOOKUP: Looking for specific recording by ID, date, or exact filter
- KEYWORD_MATCH: Looking for specific terms, acronyms, proper nouns, exact phrases
- SEMANTIC_EXPLORATION: Broad conceptual exploration, feelings, implications
- AGGREGATION: Requires synthesizing across multiple documents (themes, trends, comparisons)
- ENTITY_LOOKUP: Looking for people, projects, organizations, relationships

Query: "{query}"

Respond with ONLY the category name and confidence (0.0-1.0), like:
SEMANTIC_EXPLORATION 0.85

Category and confidence:"""

    def __init__(self):
        super().__init__(use_llm_fallback=True)
    
    def route(self, query: str) -> RoutingDecision:
        """Route with LLM fallback for ambiguous queries."""
        # First try pattern-based
        decision = super().route(query)
        
        # If low confidence, use LLM
        if decision.confidence < 0.7 and self.use_llm:
            llm_intent, llm_confidence = self._classify_with_llm(query)
            if llm_confidence > decision.confidence:
                decision.intent = llm_intent
                decision.confidence = llm_confidence
                decision.reasoning = f"LLM classification (confidence: {llm_confidence:.2f})"
                decision.alpha = self._compute_alpha(llm_intent, query.lower())
        
        return decision
    
    def _classify_with_llm(self, query: str) -> Tuple[QueryIntent, float]:
        """Use Gemini to classify ambiguous query."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return (QueryIntent.HYBRID_BALANCED, 0.5)
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            response = model.generate_content(
                self.CLASSIFICATION_PROMPT.format(query=query),
                generation_config=genai.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=20,
                )
            )
            
            text = response.text.strip().upper()
            
            # Parse response
            intent_map = {
                "METADATA_LOOKUP": QueryIntent.METADATA_LOOKUP,
                "KEYWORD_MATCH": QueryIntent.KEYWORD_MATCH,
                "SEMANTIC_EXPLORATION": QueryIntent.SEMANTIC_EXPLORATION,
                "AGGREGATION": QueryIntent.AGGREGATION,
                "ENTITY_LOOKUP": QueryIntent.ENTITY_LOOKUP,
            }
            
            for intent_name, intent_enum in intent_map.items():
                if intent_name in text:
                    # Extract confidence if present
                    conf_match = re.search(r'(\d\.\d+)', text)
                    confidence = float(conf_match.group(1)) if conf_match else 0.75
                    return (intent_enum, confidence)
            
            return (QueryIntent.HYBRID_BALANCED, 0.5)
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return (QueryIntent.HYBRID_BALANCED, 0.5)


def get_query_router(use_llm: bool = False) -> QueryRouter:
    """Factory function for query router."""
    if use_llm:
        return LLMQueryRouter()
    return QueryRouter()


# ============================================================================
# Convenience Functions
# ============================================================================

def route_query(query: str) -> RoutingDecision:
    """Quick-access routing function."""
    router = get_query_router()
    return router.route(query)


def should_use_graphrag(query: str) -> bool:
    """Check if query needs GraphRAG."""
    decision = route_query(query)
    return decision.use_graphrag


def get_optimal_alpha(query: str) -> float:
    """Get optimal hybrid alpha for query."""
    decision = route_query(query)
    return decision.alpha


if __name__ == "__main__":
    # Test the router
    test_queries = [
        "recording:abc123",
        "what themes emerged from last week's meetings",
        "mentions of Project Alpha",
        '"API integration" error handling',
        "how does the team feel about the new process",
        "action items from 2025-01-15",
        "who is working with Bob on the budget project",
        "CEO KPI Q4 targets",
        "summarize all discussions about pricing",
    ]
    
    router = get_query_router()
    
    print("\n" + "="*70)
    print("Query Router Test")
    print("="*70)
    
    for query in test_queries:
        decision = router.route(query)
        print(f"\n📝 Query: {query}")
        print(f"   Intent: {decision.intent.value}")
        print(f"   Alpha: {decision.alpha}")
        print(f"   Rerank: {decision.use_rerank} | GraphRAG: {decision.use_graphrag}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Reasoning: {decision.reasoning}")
        if decision.filters and decision.filters.has_filters:
            print(f"   Filters: {decision.filters.to_pinecone_filter()}")
