"""
Conflict Detection and Source-Aware RAG Responses

Research Context (Dec 2025):
- RAG systems must handle contradicting data sources
- Instead of hallucinating a compromise, explicitly flag conflicts
- "The database says X, but the document says Y"
- Automated evaluation should inject contradictory data to test

Why this matters:
- SQL DB says "Inventory: 0" 
- Vector store has doc saying "New shipment arrived"
- Bad RAG: hallucinates "inventory is being restocked"
- Good RAG: "Database shows 0 inventory, but a shipping manifest suggests stock may be available. Please verify."

Implementation:
- Detect semantic contradictions between sources
- Flag confidence levels per source
- Generate source-aware responses with explicit citations
- Test suite with intentionally conflicting data
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

class ConflictType(Enum):
    """Types of data conflicts."""
    NUMERICAL = "numerical"        # Different numbers (price, quantity, date)
    CATEGORICAL = "categorical"    # Different categories (status, state)
    FACTUAL = "factual"           # Contradicting facts
    TEMPORAL = "temporal"         # Different time references
    NEGATION = "negation"         # One affirms, other denies
    AMBIGUOUS = "ambiguous"       # Unclear relationship


class SourceType(Enum):
    """Types of data sources."""
    DATABASE = "database"          # SQL/structured data
    VECTOR_STORE = "vector_store"  # Pinecone/vector search
    API = "api"                   # External API
    DOCUMENT = "document"         # Uploaded document
    CACHE = "cache"               # Cached computation
    USER = "user"                 # User-provided


@dataclass
class SourcedFact:
    """A fact with source attribution."""
    fact: str                      # The factual claim
    value: Any                     # Extracted value if applicable
    source_type: SourceType
    source_id: str                 # ID of source (doc ID, table name, etc.)
    source_name: str               # Human-readable source name
    confidence: float = 1.0        # 0-1 confidence in extraction
    timestamp: Optional[str] = None  # When the fact was recorded
    context: str = ""              # Surrounding context
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact,
            "value": self.value,
            "source": f"{self.source_type.value}:{self.source_name}",
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


@dataclass
class ConflictReport:
    """Report of detected conflicts between sources."""
    conflict_type: ConflictType
    description: str
    facts: List[SourcedFact]
    severity: str = "medium"       # low, medium, high, critical
    resolution_hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.conflict_type.value,
            "description": self.description,
            "severity": self.severity,
            "sources": [f.to_dict() for f in self.facts],
            "resolution_hint": self.resolution_hint
        }


@dataclass
class SourceAwareResponse:
    """RAG response with full source attribution and conflict detection."""
    answer: str
    facts_used: List[SourcedFact]
    conflicts: List[ConflictReport]
    confidence: float
    has_conflicts: bool = False
    conflict_warning: Optional[str] = None
    
    # Metadata
    sources_consulted: List[str] = field(default_factory=list)
    retrieval_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "has_conflicts": self.has_conflicts,
            "conflict_warning": self.conflict_warning,
            "facts": [f.to_dict() for f in self.facts_used],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "sources": self.sources_consulted
        }


# ═══════════════════════════════════════════════════════════════════════════
# Conflict Detector
# ═══════════════════════════════════════════════════════════════════════════

class ConflictDetector:
    """
    Detect conflicts between retrieved facts from different sources.
    
    Uses:
    - LLM to identify semantic contradictions
    - Rules for numerical/temporal conflicts
    - Heuristics for common conflict patterns
    
    Usage:
        detector = ConflictDetector()
        
        facts = [
            SourcedFact("Inventory is 0", 0, SourceType.DATABASE, ...),
            SourcedFact("New shipment arrived today", None, SourceType.DOCUMENT, ...)
        ]
        
        conflicts = detector.detect(facts)
    """
    
    # Common conflict patterns (regex-like matching)
    NEGATION_PAIRS = [
        ("available", "unavailable"),
        ("active", "inactive"),
        ("enabled", "disabled"),
        ("open", "closed"),
        ("complete", "incomplete"),
        ("approved", "rejected"),
        ("valid", "invalid"),
        ("success", "failure"),
    ]
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize detector.
        
        Args:
            use_llm: Whether to use LLM for semantic conflict detection
        """
        self.use_llm = use_llm
        
        if use_llm:
            self._init_llm()
        
        logger.info(f"✅ ConflictDetector initialized (LLM: {use_llm})")
    
    def _init_llm(self):
        """Initialize LLM for semantic analysis."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, LLM conflict detection disabled")
            self.use_llm = False
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        except ImportError:
            logger.warning("google-generativeai not installed")
            self.use_llm = False
    
    def detect(self, facts: List[SourcedFact]) -> List[ConflictReport]:
        """
        Detect conflicts among a list of facts.
        
        Args:
            facts: List of sourced facts to analyze
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Skip if only one fact
        if len(facts) < 2:
            return conflicts
        
        # Rule-based detection
        conflicts.extend(self._detect_numerical_conflicts(facts))
        conflicts.extend(self._detect_negation_conflicts(facts))
        
        # LLM-based semantic detection
        if self.use_llm and len(facts) >= 2:
            semantic_conflicts = self._detect_semantic_conflicts(facts)
            conflicts.extend(semantic_conflicts)
        
        # Deduplicate conflicts
        conflicts = self._deduplicate_conflicts(conflicts)
        
        return conflicts
    
    def _detect_numerical_conflicts(self, facts: List[SourcedFact]) -> List[ConflictReport]:
        """Detect conflicts in numerical values."""
        conflicts = []
        
        # Group facts by what they might be measuring
        # (Simple heuristic: facts with numerical values)
        numerical_facts = [f for f in facts if isinstance(f.value, (int, float))]
        
        if len(numerical_facts) < 2:
            return conflicts
        
        # Compare pairs
        for i, fact1 in enumerate(numerical_facts):
            for fact2 in numerical_facts[i+1:]:
                # Check if they seem to be about the same thing
                if self._facts_about_same_topic(fact1, fact2):
                    if fact1.value != fact2.value:
                        # Calculate discrepancy
                        if fact1.value and fact2.value:
                            diff_pct = abs(fact1.value - fact2.value) / max(abs(fact1.value), abs(fact2.value)) * 100
                        else:
                            diff_pct = 100
                        
                        severity = "low" if diff_pct < 10 else "medium" if diff_pct < 50 else "high"
                        
                        conflicts.append(ConflictReport(
                            conflict_type=ConflictType.NUMERICAL,
                            description=f"Numerical discrepancy: {fact1.value} vs {fact2.value} ({diff_pct:.1f}% difference)",
                            facts=[fact1, fact2],
                            severity=severity,
                            resolution_hint="Verify which source has the most recent/authoritative data"
                        ))
        
        return conflicts
    
    def _detect_negation_conflicts(self, facts: List[SourcedFact]) -> List[ConflictReport]:
        """Detect negation/opposite conflicts."""
        conflicts = []
        
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                # Check for negation patterns
                text1 = fact1.fact.lower()
                text2 = fact2.fact.lower()
                
                for pos, neg in self.NEGATION_PAIRS:
                    if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                        conflicts.append(ConflictReport(
                            conflict_type=ConflictType.NEGATION,
                            description=f"Contradicting states: one says '{pos}', other says '{neg}'",
                            facts=[fact1, fact2],
                            severity="high",
                            resolution_hint="These sources directly contradict each other"
                        ))
                        break
        
        return conflicts
    
    def _detect_semantic_conflicts(self, facts: List[SourcedFact]) -> List[ConflictReport]:
        """Use LLM to detect semantic contradictions."""
        conflicts = []
        
        if not self.use_llm or not hasattr(self, 'model'):
            return conflicts
        
        # Build prompt with all facts
        facts_text = "\n".join([
            f"[Source: {f.source_name}] {f.fact}"
            for f in facts
        ])
        
        prompt = f"""Analyze these facts from different sources for contradictions:

{facts_text}

For each pair of contradicting facts, respond with JSON:
{{
    "conflicts": [
        {{
            "fact1_index": 0,
            "fact2_index": 1,
            "type": "factual|temporal|categorical",
            "description": "brief description of contradiction",
            "severity": "low|medium|high"
        }}
    ]
}}

If no contradictions, return {{"conflicts": []}}.
Only report genuine contradictions, not complementary information."""

        try:
            import google.generativeai as genai
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            import json
            result = json.loads(response.text)
            
            for c in result.get("conflicts", []):
                idx1 = c.get("fact1_index", 0)
                idx2 = c.get("fact2_index", 1)
                
                if idx1 < len(facts) and idx2 < len(facts):
                    conflict_type = ConflictType.FACTUAL
                    if c.get("type") == "temporal":
                        conflict_type = ConflictType.TEMPORAL
                    elif c.get("type") == "categorical":
                        conflict_type = ConflictType.CATEGORICAL
                    
                    conflicts.append(ConflictReport(
                        conflict_type=conflict_type,
                        description=c.get("description", "Semantic contradiction detected"),
                        facts=[facts[idx1], facts[idx2]],
                        severity=c.get("severity", "medium")
                    ))
        
        except Exception as e:
            logger.warning(f"LLM conflict detection failed: {e}")
        
        return conflicts
    
    def _facts_about_same_topic(self, fact1: SourcedFact, fact2: SourcedFact) -> bool:
        """Heuristic check if two facts are about the same topic."""
        # Simple word overlap check
        words1 = set(fact1.fact.lower().split())
        words2 = set(fact2.fact.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further", "then", "once"}
        
        words1 -= stop_words
        words2 -= stop_words
        
        # Check overlap
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        
        return overlap >= 2 or (min_len > 0 and overlap / min_len > 0.3)
    
    def _deduplicate_conflicts(self, conflicts: List[ConflictReport]) -> List[ConflictReport]:
        """Remove duplicate conflicts."""
        seen = set()
        unique = []
        
        for c in conflicts:
            # Create key from fact contents
            key = tuple(sorted([f.fact for f in c.facts]))
            
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique


# ═══════════════════════════════════════════════════════════════════════════
# Source-Aware Response Generator
# ═══════════════════════════════════════════════════════════════════════════

class SourceAwareGenerator:
    """
    Generate RAG responses with explicit source attribution and conflict handling.
    
    Instead of:
        "The inventory is low" (hallucinated average)
    
    Generates:
        "According to the database (updated 2h ago), inventory is 0 units.
         However, a shipping manifest from today indicates a new shipment arrived.
         ⚠️ These sources may conflict - please verify current inventory."
    
    Usage:
        generator = SourceAwareGenerator()
        
        response = generator.generate(
            query="What's our current inventory?",
            facts=[...],
            conflicts=[...]
        )
    """
    
    RESPONSE_PROMPT = """You are a helpful assistant that provides accurate, source-attributed answers.

IMPORTANT RULES:
1. Always cite your sources inline: "According to [Source Name]..."
2. If sources conflict, DO NOT average or guess. Instead, present BOTH viewpoints.
3. When conflicts exist, add a clear warning about the discrepancy.
4. Rate your confidence based on source agreement.

Query: {query}

Facts from sources:
{facts}

{conflict_section}

Generate a response that:
1. Answers the query using the facts
2. Cites sources for each claim
3. If conflicts exist, explicitly mentions both sides
4. Includes a confidence assessment

Format:
{{
    "answer": "Your source-attributed answer here",
    "confidence": 0.0-1.0,
    "conflict_warning": "Warning message if conflicts exist, or null"
}}"""

    def __init__(self):
        """Initialize generator."""
        self.conflict_detector = ConflictDetector()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
            self._genai = genai
        else:
            self.model = None
        
        logger.info("✅ SourceAwareGenerator initialized")
    
    def generate(
        self,
        query: str,
        facts: List[SourcedFact],
        detect_conflicts: bool = True
    ) -> SourceAwareResponse:
        """
        Generate source-aware response.
        
        Args:
            query: User query
            facts: Retrieved facts with source attribution
            detect_conflicts: Whether to run conflict detection
            
        Returns:
            SourceAwareResponse with citations and conflict warnings
        """
        # Detect conflicts
        conflicts = []
        if detect_conflicts and len(facts) >= 2:
            conflicts = self.conflict_detector.detect(facts)
        
        has_conflicts = len(conflicts) > 0
        
        # Generate response
        if self.model:
            response = self._generate_with_llm(query, facts, conflicts)
        else:
            response = self._generate_simple(query, facts, conflicts)
        
        response.facts_used = facts
        response.conflicts = conflicts
        response.has_conflicts = has_conflicts
        response.sources_consulted = list(set(f.source_name for f in facts))
        
        return response
    
    def _generate_with_llm(
        self,
        query: str,
        facts: List[SourcedFact],
        conflicts: List[ConflictReport]
    ) -> SourceAwareResponse:
        """Generate response using LLM."""
        # Format facts
        facts_text = "\n".join([
            f"- [{f.source_name}] {f.fact}" + 
            (f" (confidence: {f.confidence:.0%})" if f.confidence < 1.0 else "")
            for f in facts
        ])
        
        # Format conflicts
        if conflicts:
            conflict_section = "DETECTED CONFLICTS:\n" + "\n".join([
                f"⚠️ {c.description}\n   Sources: {', '.join(f.source_name for f in c.facts)}"
                for c in conflicts
            ])
        else:
            conflict_section = "No conflicts detected between sources."
        
        prompt = self.RESPONSE_PROMPT.format(
            query=query,
            facts=facts_text,
            conflict_section=conflict_section
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self._genai.GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            import json
            result = json.loads(response.text)
            
            return SourceAwareResponse(
                answer=result.get("answer", "Unable to generate response"),
                facts_used=[],  # Will be filled by caller
                conflicts=[],   # Will be filled by caller
                confidence=result.get("confidence", 0.5),
                conflict_warning=result.get("conflict_warning")
            )
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_simple(query, facts, conflicts)
    
    def _generate_simple(
        self,
        query: str,
        facts: List[SourcedFact],
        conflicts: List[ConflictReport]
    ) -> SourceAwareResponse:
        """Generate simple response without LLM."""
        # Build answer from facts
        answer_parts = []
        
        for fact in facts:
            answer_parts.append(f"According to {fact.source_name}: {fact.fact}")
        
        answer = "\n".join(answer_parts)
        
        # Add conflict warning
        conflict_warning = None
        if conflicts:
            conflict_warning = f"⚠️ {len(conflicts)} conflict(s) detected between sources. Please verify."
            answer += f"\n\n{conflict_warning}"
        
        # Calculate confidence
        if conflicts:
            confidence = 0.5
        elif len(facts) > 0:
            confidence = sum(f.confidence for f in facts) / len(facts)
        else:
            confidence = 0.0
        
        return SourceAwareResponse(
            answer=answer,
            facts_used=[],
            conflicts=[],
            confidence=confidence,
            conflict_warning=conflict_warning
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test Suite with Conflicting Data
# ═══════════════════════════════════════════════════════════════════════════

class ConflictTestSuite:
    """
    Test suite that injects conflicting data to verify conflict handling.
    
    Used to validate that RAG system properly:
    1. Detects contradictions
    2. Doesn't hallucinate compromises
    3. Flags conflicts to users
    
    Usage:
        suite = ConflictTestSuite()
        results = suite.run_all_tests(generator)
    """
    
    # Test cases with intentionally conflicting data
    TEST_CASES = [
        {
            "name": "numerical_inventory",
            "query": "What's the current inventory level?",
            "facts": [
                {"fact": "Current inventory is 0 units", "value": 0, "source": "database", "source_name": "Inventory DB"},
                {"fact": "Received shipment of 100 units today", "value": 100, "source": "document", "source_name": "Shipping Manifest"},
            ],
            "expected_conflict": True,
            "expected_type": ConflictType.NUMERICAL,
        },
        {
            "name": "status_contradiction",
            "query": "Is the project active?",
            "facts": [
                {"fact": "Project status: Active", "value": "active", "source": "database", "source_name": "Project DB"},
                {"fact": "Project was marked inactive last week", "value": "inactive", "source": "document", "source_name": "Meeting Notes"},
            ],
            "expected_conflict": True,
            "expected_type": ConflictType.NEGATION,
        },
        {
            "name": "temporal_conflict",
            "query": "When was the last update?",
            "facts": [
                {"fact": "Last updated: 2025-12-01", "value": "2025-12-01", "source": "database", "source_name": "System Log"},
                {"fact": "No updates since November", "value": "2025-11", "source": "document", "source_name": "Status Report"},
            ],
            "expected_conflict": True,
            "expected_type": ConflictType.TEMPORAL,
        },
        {
            "name": "no_conflict",
            "query": "What's the project name?",
            "facts": [
                {"fact": "Project name: Alpha", "value": "Alpha", "source": "database", "source_name": "Project DB"},
                {"fact": "Working on Project Alpha", "value": "Alpha", "source": "document", "source_name": "Meeting Notes"},
            ],
            "expected_conflict": False,
            "expected_type": None,
        },
    ]
    
    def __init__(self):
        """Initialize test suite."""
        self.detector = ConflictDetector()
    
    def run_all_tests(self, generator: Optional[SourceAwareGenerator] = None) -> Dict[str, Any]:
        """
        Run all test cases.
        
        Args:
            generator: Optional generator to test full response generation
            
        Returns:
            Test results summary
        """
        results = {
            "total": len(self.TEST_CASES),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test in self.TEST_CASES:
            result = self._run_test(test, generator)
            results["details"].append(result)
            
            if result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        logger.info(f"Conflict tests: {results['passed']}/{results['total']} passed")
        return results
    
    def _run_test(self, test: Dict, generator: Optional[SourceAwareGenerator]) -> Dict[str, Any]:
        """Run a single test case."""
        # Create sourced facts
        facts = [
            SourcedFact(
                fact=f["fact"],
                value=f.get("value"),
                source_type=SourceType(f["source"]),
                source_id=f["source_name"],
                source_name=f["source_name"]
            )
            for f in test["facts"]
        ]
        
        # Detect conflicts
        conflicts = self.detector.detect(facts)
        
        # Check expectations
        conflict_detected = len(conflicts) > 0
        passed = conflict_detected == test["expected_conflict"]
        
        # Check conflict type if expected
        if test["expected_conflict"] and conflicts:
            type_match = any(c.conflict_type == test["expected_type"] for c in conflicts)
            passed = passed and type_match
        
        result = {
            "name": test["name"],
            "passed": passed,
            "expected_conflict": test["expected_conflict"],
            "conflict_detected": conflict_detected,
            "conflicts": [c.to_dict() for c in conflicts]
        }
        
        # Test full generation if generator provided
        if generator:
            response = generator.generate(test["query"], facts)
            result["response"] = response.to_dict()
            
            # Verify conflict warning is present when expected
            if test["expected_conflict"]:
                has_warning = response.conflict_warning is not None or response.has_conflicts
                result["warning_present"] = has_warning
                if not has_warning:
                    result["passed"] = False
        
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def detect_conflicts(facts: List[SourcedFact]) -> List[ConflictReport]:
    """
    Convenience function to detect conflicts in facts.
    
    Args:
        facts: List of sourced facts
        
    Returns:
        List of detected conflicts
    """
    detector = ConflictDetector()
    return detector.detect(facts)


def generate_source_aware_response(
    query: str,
    facts: List[SourcedFact]
) -> SourceAwareResponse:
    """
    Convenience function to generate source-aware response.
    
    Args:
        query: User query
        facts: Retrieved facts
        
    Returns:
        Source-aware response with conflict handling
    """
    generator = SourceAwareGenerator()
    return generator.generate(query, facts)


def run_conflict_tests() -> Dict[str, Any]:
    """Run the conflict detection test suite."""
    suite = ConflictTestSuite()
    generator = SourceAwareGenerator()
    return suite.run_all_tests(generator)
