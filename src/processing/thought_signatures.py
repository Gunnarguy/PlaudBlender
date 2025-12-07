"""
Thought Signatures and Context Caching for Agentic Workflows

Research Context (Dec 2025 - Gemini 3):
- "Reasoning drift" occurs in multi-step agent workflows
- When agents call tools, they lose internal reasoning state
- Thought Signatures: Encrypted/compressed reasoning state snapshots
- Context Caching: Reuse expensive reasoning across requests

Why this matters:
- Agent calls Tool A, gets result, calls Tool B
- By Tool B, the original "plan" is degraded
- Thought Signatures let agents "remember" their reasoning

Implementation:
- Serialize reasoning state before tool calls
- Cache expensive computations (embeddings, analyses)
- Restore context efficiently after interruptions
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Cache directory
CACHE_DIR = Path(os.getenv("PLAUDBLENDER_CACHE_DIR", "data/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default TTLs
DEFAULT_THOUGHT_TTL = 3600  # 1 hour for thought signatures
DEFAULT_EMBEDDING_TTL = 86400 * 7  # 7 days for embeddings
DEFAULT_ANALYSIS_TTL = 86400  # 1 day for LLM analyses


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ThoughtSignature:
    """
    Compressed representation of agent reasoning state.
    
    Captures:
    - Original query/task
    - Current plan/steps
    - Intermediate results
    - Decision points and rationale
    """
    signature_id: str
    created_at: str
    
    # Original context
    original_query: str
    task_description: str
    
    # Reasoning state
    current_step: int
    total_steps: int
    plan: List[str]
    completed_steps: List[Dict[str, Any]]
    
    # Intermediate results
    tool_results: Dict[str, Any] = field(default_factory=dict)
    accumulated_context: str = ""
    
    # Decision tracking
    decisions: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    model_used: str = ""
    confidence: float = 1.0
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ThoughtSignature':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)
    
    def add_tool_result(self, tool_name: str, result: Any):
        """Record a tool call result."""
        self.tool_results[tool_name] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def add_decision(self, decision: str, rationale: str):
        """Record a decision point."""
        self.decisions.append({
            "decision": decision,
            "rationale": rationale,
            "step": self.current_step,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def advance_step(self, result: Dict[str, Any]):
        """Move to next step, recording the completed one."""
        self.completed_steps.append({
            "step_num": self.current_step,
            "description": self.plan[self.current_step] if self.current_step < len(self.plan) else "unknown",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.current_step += 1


@dataclass
class CacheEntry:
    """Generic cache entry with TTL."""
    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > (self.created_at + self.ttl_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at


# ═══════════════════════════════════════════════════════════════════════════
# Thought Signature Manager
# ═══════════════════════════════════════════════════════════════════════════

class ThoughtSignatureManager:
    """
    Manage thought signatures for agentic workflows.
    
    Enables:
    - Creating snapshots of reasoning state
    - Restoring state after tool calls
    - Tracking multi-step execution progress
    
    Usage:
        manager = ThoughtSignatureManager()
        
        # Create signature at start of agent task
        sig = manager.create_signature(
            query="Analyze Q3 report and compare to Q2",
            plan=["Fetch Q3 report", "Analyze Q3", "Fetch Q2 report", "Compare"]
        )
        
        # Before tool call, save state
        manager.save(sig)
        
        # After tool call, restore
        sig = manager.load(sig.signature_id)
        sig.add_tool_result("fetch_q3", {"revenue": 1000000})
        sig.advance_step({"status": "Q3 fetched"})
        
        # Continue with restored context
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        """Initialize manager."""
        self.cache_dir = cache_dir / "thought_signatures"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ ThoughtSignatureManager initialized at {self.cache_dir}")
    
    def create_signature(
        self,
        query: str,
        plan: List[str],
        task_description: str = "",
        model: str = ""
    ) -> ThoughtSignature:
        """
        Create a new thought signature for an agent task.
        
        Args:
            query: Original user query
            plan: List of planned steps
            task_description: Human-readable task description
            model: Model being used
            
        Returns:
            New ThoughtSignature
        """
        # Generate unique ID from query + timestamp
        sig_id = hashlib.sha256(
            f"{query}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        signature = ThoughtSignature(
            signature_id=sig_id,
            created_at=datetime.utcnow().isoformat(),
            original_query=query,
            task_description=task_description or query,
            current_step=0,
            total_steps=len(plan),
            plan=plan,
            completed_steps=[],
            model_used=model
        )
        
        logger.info(f"Created thought signature: {sig_id}")
        logger.info(f"  Query: {query[:50]}...")
        logger.info(f"  Steps: {len(plan)}")
        
        return signature
    
    def save(self, signature: ThoughtSignature) -> str:
        """
        Save thought signature to disk.
        
        Args:
            signature: ThoughtSignature to save
            
        Returns:
            Path to saved file
        """
        filepath = self.cache_dir / f"{signature.signature_id}.json"
        
        with open(filepath, 'w') as f:
            f.write(signature.to_json())
        
        logger.debug(f"Saved thought signature: {signature.signature_id}")
        return str(filepath)
    
    def load(self, signature_id: str) -> Optional[ThoughtSignature]:
        """
        Load thought signature from disk.
        
        Args:
            signature_id: ID of signature to load
            
        Returns:
            ThoughtSignature or None if not found
        """
        filepath = self.cache_dir / f"{signature_id}.json"
        
        if not filepath.exists():
            logger.warning(f"Thought signature not found: {signature_id}")
            return None
        
        with open(filepath, 'r') as f:
            signature = ThoughtSignature.from_json(f.read())
        
        logger.debug(f"Loaded thought signature: {signature_id}")
        return signature
    
    def delete(self, signature_id: str) -> bool:
        """Delete a thought signature."""
        filepath = self.cache_dir / f"{signature_id}.json"
        
        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Deleted thought signature: {signature_id}")
            return True
        return False
    
    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """
        Remove expired thought signatures.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of signatures deleted
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        deleted = 0
        
        for filepath in self.cache_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                created = datetime.fromisoformat(data['created_at'])
                if created < cutoff:
                    filepath.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Error checking signature {filepath}: {e}")
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired thought signatures")
        
        return deleted


# ═══════════════════════════════════════════════════════════════════════════
# Context Cache
# ═══════════════════════════════════════════════════════════════════════════

class ContextCache:
    """
    Cache expensive computations for reuse.
    
    Caches:
    - Embeddings (avoid re-embedding same text)
    - LLM analyses (avoid re-analyzing same content)
    - Tool results (avoid redundant API calls)
    
    Usage:
        cache = ContextCache()
        
        # Cache embedding
        embedding = cache.get_or_compute(
            key="embed:my_text_hash",
            compute_fn=lambda: embedder.embed("my text"),
            ttl=86400 * 7  # 7 days
        )
        
        # Decorator usage
        @cache.cached(ttl=3600)
        def expensive_analysis(text: str) -> dict:
            return llm.analyze(text)
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        """Initialize cache."""
        self.cache_dir = cache_dir / "context_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for hot entries
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._max_memory_entries = 1000
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        logger.info(f"✅ ContextCache initialized at {self.cache_dir}")
    
    def _make_key(self, key: str) -> str:
        """Create filesystem-safe cache key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        safe_key = self._make_key(key)
        
        # Check memory cache first
        if safe_key in self._memory_cache:
            entry = self._memory_cache[safe_key]
            if not entry.is_expired:
                entry.hit_count += 1
                self._hits += 1
                return entry.value
            else:
                del self._memory_cache[safe_key]
        
        # Check disk cache
        filepath = self.cache_dir / f"{safe_key}.json"
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry(
                    key=key,
                    value=data['value'],
                    created_at=data['created_at'],
                    ttl_seconds=data['ttl_seconds'],
                    hit_count=data.get('hit_count', 0) + 1
                )
                
                if not entry.is_expired:
                    # Promote to memory cache
                    self._memory_cache[safe_key] = entry
                    self._hits += 1
                    return entry.value
                else:
                    # Clean up expired entry
                    filepath.unlink()
            except Exception as e:
                logger.warning(f"Error reading cache entry {key}: {e}")
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = DEFAULT_EMBEDDING_TTL):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        safe_key = self._make_key(key)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl
        )
        
        # Store in memory
        self._memory_cache[safe_key] = entry
        
        # Evict old entries if needed
        if len(self._memory_cache) > self._max_memory_entries:
            self._evict_memory_cache()
        
        # Store on disk
        filepath = self.cache_dir / f"{safe_key}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'key': key,
                    'value': value,
                    'created_at': entry.created_at,
                    'ttl_seconds': ttl,
                    'hit_count': 0
                }, f)
        except Exception as e:
            logger.warning(f"Error writing cache entry {key}: {e}")
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: int = DEFAULT_EMBEDDING_TTL
    ) -> Any:
        """
        Get from cache or compute and cache.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time-to-live in seconds
            
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        
        if value is not None:
            return value
        
        # Compute and cache
        value = compute_fn()
        self.set(key, value, ttl)
        
        return value
    
    def cached(self, ttl: int = DEFAULT_EMBEDDING_TTL, key_fn: Optional[Callable] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time-to-live in seconds
            key_fn: Optional function to generate cache key from args
            
        Usage:
            @cache.cached(ttl=3600)
            def expensive_fn(x, y):
                return compute(x, y)
        """
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    key = f"{fn.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                
                return self.get_or_compute(key, lambda: fn(*args, **kwargs), ttl)
            
            return wrapper
        return decorator
    
    def _evict_memory_cache(self):
        """Evict least recently used entries from memory cache."""
        # Sort by hit count (ascending) and age (descending)
        entries = list(self._memory_cache.items())
        entries.sort(key=lambda x: (x[1].hit_count, -x[1].age_seconds))
        
        # Remove bottom 20%
        to_remove = len(entries) // 5
        for key, _ in entries[:to_remove]:
            del self._memory_cache[key]
        
        logger.debug(f"Evicted {to_remove} entries from memory cache")
    
    def invalidate(self, key: str):
        """Remove entry from cache."""
        safe_key = self._make_key(key)
        
        if safe_key in self._memory_cache:
            del self._memory_cache[safe_key]
        
        filepath = self.cache_dir / f"{safe_key}.json"
        if filepath.exists():
            filepath.unlink()
    
    def clear(self):
        """Clear entire cache."""
        self._memory_cache.clear()
        
        for filepath in self.cache_dir.glob("*.json"):
            filepath.unlink()
        
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(list(self.cache_dir.glob("*.json")))
        }


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Cache (Specialized)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingCache:
    """
    Specialized cache for embeddings.
    
    Avoids re-embedding the same text, which is:
    - Expensive (API calls)
    - Slow (network latency)
    - Wasteful (same text = same embedding)
    
    Usage:
        cache = EmbeddingCache()
        
        # Get or compute embedding
        embedding = cache.get_embedding(
            text="my text to embed",
            embedder=my_embedder
        )
    """
    
    def __init__(self, cache: Optional[ContextCache] = None):
        """Initialize embedding cache."""
        self.cache = cache or ContextCache()
    
    def _text_key(self, text: str, model: str = "default") -> str:
        """Generate cache key for text."""
        # Use hash of text + model for key
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding:{model}:{text_hash}"
    
    def get_embedding(
        self,
        text: str,
        embedder: Callable[[str], List[float]],
        model: str = "default",
        ttl: int = DEFAULT_EMBEDDING_TTL
    ) -> List[float]:
        """
        Get embedding from cache or compute.
        
        Args:
            text: Text to embed
            embedder: Function to compute embedding
            model: Model identifier (for key uniqueness)
            ttl: Cache TTL
            
        Returns:
            Embedding vector
        """
        key = self._text_key(text, model)
        
        return self.cache.get_or_compute(
            key=key,
            compute_fn=lambda: embedder(text),
            ttl=ttl
        )
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        embedder: Callable[[str], List[float]],
        batch_embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
        model: str = "default"
    ) -> List[List[float]]:
        """
        Get embeddings for multiple texts, using cache where possible.
        
        Args:
            texts: List of texts
            embedder: Single-text embedder
            batch_embedder: Optional batch embedder for uncached texts
            model: Model identifier
            
        Returns:
            List of embeddings
        """
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._text_key(text, model)
            cached = self.cache.get(key)
            
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Compute uncached embeddings
        if uncached_texts:
            if batch_embedder and len(uncached_texts) > 1:
                # Use batch embedder if available
                new_embeddings = batch_embedder(uncached_texts)
            else:
                # Fall back to single embedder
                new_embeddings = [embedder(t) for t in uncached_texts]
            
            # Cache and store results
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                embedding = new_embeddings[i]
                key = self._text_key(text, model)
                self.cache.set(key, embedding)
                results[idx] = embedding
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Agentic Context Manager
# ═══════════════════════════════════════════════════════════════════════════

class AgenticContextManager:
    """
    High-level manager for agentic workflows.
    
    Combines thought signatures and caching to enable
    reliable multi-step agent execution.
    
    Usage:
        manager = AgenticContextManager()
        
        # Start a new agent task
        with manager.task("Analyze and compare reports") as ctx:
            ctx.plan(["Fetch report A", "Analyze A", "Fetch report B", "Compare"])
            
            # Execute steps with state persistence
            result_a = ctx.execute_step(fetch_report, "report_a")
            ctx.record_decision("Using Q3 data", "Most recent available")
            
            analysis_a = ctx.execute_step(analyze, result_a)
            
            result_b = ctx.execute_step(fetch_report, "report_b")
            
            comparison = ctx.execute_step(compare, analysis_a, result_b)
            
        # Context automatically saved/cleaned up
    """
    
    def __init__(self):
        """Initialize manager."""
        self.thought_manager = ThoughtSignatureManager()
        self.cache = ContextCache()
        self.embedding_cache = EmbeddingCache(self.cache)
        
        logger.info("✅ AgenticContextManager initialized")
    
    def task(self, description: str):
        """Create a context manager for a task."""
        return AgenticTaskContext(
            description=description,
            thought_manager=self.thought_manager,
            cache=self.cache
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats."""
        return {
            "cache": self.cache.stats()
        }


class AgenticTaskContext:
    """Context manager for a single agent task."""
    
    def __init__(
        self,
        description: str,
        thought_manager: ThoughtSignatureManager,
        cache: ContextCache
    ):
        self.description = description
        self.thought_manager = thought_manager
        self.cache = cache
        self.signature: Optional[ThoughtSignature] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up signature on completion
        if self.signature:
            self.thought_manager.delete(self.signature.signature_id)
        return False
    
    def plan(self, steps: List[str]):
        """Set the plan for this task."""
        self.signature = self.thought_manager.create_signature(
            query=self.description,
            plan=steps,
            task_description=self.description
        )
        self.thought_manager.save(self.signature)
    
    def execute_step(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute a step with state persistence.
        
        Saves thought signature before execution,
        records result after.
        """
        if not self.signature:
            raise ValueError("Must call plan() before execute_step()")
        
        # Save state before execution
        self.thought_manager.save(self.signature)
        
        # Execute
        result = fn(*args, **kwargs)
        
        # Record result and advance
        self.signature.advance_step({"result_type": type(result).__name__})
        self.thought_manager.save(self.signature)
        
        return result
    
    def record_decision(self, decision: str, rationale: str):
        """Record a decision point."""
        if self.signature:
            self.signature.add_decision(decision, rationale)
            self.thought_manager.save(self.signature)
    
    def add_context(self, context: str):
        """Add accumulated context."""
        if self.signature:
            self.signature.accumulated_context += f"\n{context}"
            self.thought_manager.save(self.signature)


# ═══════════════════════════════════════════════════════════════════════════
# Global Instances
# ═══════════════════════════════════════════════════════════════════════════

# Singleton instances for convenience
_context_cache: Optional[ContextCache] = None
_embedding_cache: Optional[EmbeddingCache] = None
_thought_manager: Optional[ThoughtSignatureManager] = None


def get_context_cache() -> ContextCache:
    """Get global context cache instance."""
    global _context_cache
    if _context_cache is None:
        _context_cache = ContextCache()
    return _context_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(get_context_cache())
    return _embedding_cache


def get_thought_manager() -> ThoughtSignatureManager:
    """Get global thought signature manager."""
    global _thought_manager
    if _thought_manager is None:
        _thought_manager = ThoughtSignatureManager()
    return _thought_manager
