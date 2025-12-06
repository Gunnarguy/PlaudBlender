"""
Hierarchical Chunking Module for PlaudBlender

Implements parent/child chunking strategy for improved RAG accuracy:
- Parent chunks: ~2000 tokens, provide full context to LLM
- Child chunks: ~512 tokens, indexed for precise retrieval
- At query time: Retrieve child chunks, return parent chunks to LLM

This solves the "lost in the middle" problem where important context
gets cut off in fixed-size chunks.

Reference: gemini-deep-research-RAG.txt Section on Hierarchical Chunking
"""
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import tiktoken

logger = logging.getLogger(__name__)

# Chunk size configuration (in tokens)
PARENT_CHUNK_SIZE = 2000    # Large chunks for LLM context
CHILD_CHUNK_SIZE = 512      # Small chunks for precise retrieval
CHILD_OVERLAP = 64          # Overlap between child chunks
PARENT_OVERLAP = 200        # Overlap between parent chunks


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    parent_id: Optional[str] = None
    chunk_type: str = "child"  # "parent" or "child"
    chunk_index: int = 0
    total_chunks: int = 1
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            **self.metadata,
        }


class HierarchicalChunker:
    """
    Creates hierarchical parent/child chunks for improved RAG retrieval.
    
    Strategy:
    1. Split text into parent chunks (~2000 tokens each)
    2. For each parent, create child chunks (~512 tokens each)
    3. Index ONLY child chunks in vector store
    4. Store parent_id in child metadata
    5. At retrieval time, fetch parent chunks for matched children
    
    Benefits:
    - Child chunks are small → precise vector matching
    - Parent chunks are large → full context to LLM
    - Eliminates "lost in the middle" problem
    - Better handling of multi-topic documents
    
    Usage:
        chunker = HierarchicalChunker()
        chunks = chunker.chunk_document(
            text="Long transcript...",
            doc_id="recording_123",
            metadata={"title": "Q3 Review", "date": "2024-10-15"}
        )
        
        # chunks contains both parent and child chunks
        # Index only children, but include parent_id for retrieval
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
    """
    
    def __init__(
        self,
        parent_size: int = PARENT_CHUNK_SIZE,
        child_size: int = CHILD_CHUNK_SIZE,
        parent_overlap: int = PARENT_OVERLAP,
        child_overlap: int = CHILD_OVERLAP,
        tokenizer_name: str = "cl100k_base",
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            parent_size: Target size for parent chunks (tokens)
            child_size: Target size for child chunks (tokens)
            parent_overlap: Overlap between parent chunks (tokens)
            child_overlap: Overlap between child chunks (tokens)
            tokenizer_name: Tiktoken encoding name
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        logger.info(f"✅ HierarchicalChunker initialized: parent={parent_size}, child={child_size}")
    
    def _generate_chunk_id(self, doc_id: str, chunk_type: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{doc_id}:{chunk_type}:{index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _split_into_tokens(self, text: str) -> List[int]:
        """Tokenize text."""
        return self.tokenizer.encode(text)
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        return self.tokenizer.decode(tokens)
    
    def _create_chunks_from_tokens(
        self,
        tokens: List[int],
        text: str,
        chunk_size: int,
        overlap: int,
        doc_id: str,
        chunk_type: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Create chunks from token sequence.
        
        Uses sliding window with overlap.
        """
        chunks = []
        num_tokens = len(tokens)
        
        if num_tokens <= chunk_size:
            # Single chunk, no splitting needed
            chunk_id = self._generate_chunk_id(doc_id, chunk_type, 0)
            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                parent_id=parent_id,
                chunk_type=chunk_type,
                chunk_index=0,
                total_chunks=1,
                token_count=num_tokens,
                char_start=0,
                char_end=len(text),
                metadata=metadata or {},
            ))
            return chunks
        
        # Sliding window chunking
        step = chunk_size - overlap
        start = 0
        index = 0
        
        while start < num_tokens:
            end = min(start + chunk_size, num_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokens_to_text(chunk_tokens)
            
            chunk_id = self._generate_chunk_id(doc_id, chunk_type, index)
            
            # Calculate approximate char positions
            # This is approximate since tokenization isn't 1:1 with chars
            char_ratio = len(text) / num_tokens if num_tokens > 0 else 1
            char_start = int(start * char_ratio)
            char_end = int(end * char_ratio)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                parent_id=parent_id,
                chunk_type=chunk_type,
                chunk_index=index,
                total_chunks=-1,  # Will update after
                token_count=len(chunk_tokens),
                char_start=char_start,
                char_end=min(char_end, len(text)),
                metadata=metadata or {},
            ))
            
            index += 1
            start += step
            
            # Don't create tiny trailing chunks
            if num_tokens - start < overlap:
                break
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict] = None,
        return_parents: bool = True,
    ) -> List[Chunk]:
        """
        Create hierarchical chunks from a document.
        
        Args:
            text: Full document text
            doc_id: Unique document identifier
            metadata: Additional metadata to attach to all chunks
            return_parents: Whether to include parent chunks in output
        
        Returns:
            List of Chunk objects (children, and optionally parents)
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for doc_id={doc_id}")
            return []
        
        tokens = self._split_into_tokens(text)
        total_tokens = len(tokens)
        
        logger.info(f"📄 Chunking document {doc_id}: {total_tokens} tokens")
        
        all_chunks = []
        
        # Step 1: Create parent chunks
        parent_chunks = self._create_chunks_from_tokens(
            tokens=tokens,
            text=text,
            chunk_size=self.parent_size,
            overlap=self.parent_overlap,
            doc_id=doc_id,
            chunk_type="parent",
            parent_id=None,
            metadata=metadata,
        )
        
        logger.info(f"   📦 Created {len(parent_chunks)} parent chunks")
        
        if return_parents:
            all_chunks.extend(parent_chunks)
        
        # Step 2: For each parent, create child chunks
        total_children = 0
        for parent in parent_chunks:
            parent_tokens = self._split_into_tokens(parent.text)
            
            child_chunks = self._create_chunks_from_tokens(
                tokens=parent_tokens,
                text=parent.text,
                chunk_size=self.child_size,
                overlap=self.child_overlap,
                doc_id=f"{doc_id}_p{parent.chunk_index}",
                chunk_type="child",
                parent_id=parent.chunk_id,
                metadata={
                    **(metadata or {}),
                    "parent_chunk_index": parent.chunk_index,
                },
            )
            
            total_children += len(child_chunks)
            all_chunks.extend(child_chunks)
        
        logger.info(f"   📦 Created {total_children} child chunks")
        logger.info(f"   ✅ Total chunks: {len(all_chunks)}")
        
        return all_chunks
    
    def get_children_only(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filter to only child chunks (for indexing)."""
        return [c for c in chunks if c.chunk_type == "child"]
    
    def get_parents_only(self, chunks: List[Chunk]) -> List[Chunk]:
        """Filter to only parent chunks (for context retrieval)."""
        return [c for c in chunks if c.chunk_type == "parent"]
    
    def get_parent_for_child(
        self,
        child_chunk: Chunk,
        all_chunks: List[Chunk],
    ) -> Optional[Chunk]:
        """
        Find the parent chunk for a given child chunk.
        
        Args:
            child_chunk: The child chunk to find parent for
            all_chunks: All chunks from the same document
        
        Returns:
            Parent Chunk or None if not found
        """
        if not child_chunk.parent_id:
            return None
        
        for chunk in all_chunks:
            if chunk.chunk_id == child_chunk.parent_id:
                return chunk
        
        return None


# Store parent chunks in memory for fast retrieval
# In production, this would be a database or cache
_parent_chunk_store: Dict[str, Chunk] = {}


def store_parent_chunk(chunk: Chunk) -> None:
    """Store a parent chunk for later retrieval."""
    if chunk.chunk_type == "parent":
        _parent_chunk_store[chunk.chunk_id] = chunk


def get_parent_chunk(parent_id: str) -> Optional[Chunk]:
    """Retrieve a parent chunk by ID."""
    return _parent_chunk_store.get(parent_id)


def store_all_parents(chunks: List[Chunk]) -> int:
    """Store all parent chunks from a list. Returns count stored."""
    count = 0
    for chunk in chunks:
        if chunk.chunk_type == "parent":
            store_parent_chunk(chunk)
            count += 1
    return count


def expand_to_parents(child_ids: List[str], chunks: List[Chunk]) -> List[Chunk]:
    """
    Given a list of child chunk IDs, return their parent chunks.
    
    This is used at retrieval time to provide full context to the LLM.
    
    Args:
        child_ids: List of retrieved child chunk IDs
        chunks: Full list of chunks (or use global store)
    
    Returns:
        List of unique parent chunks
    """
    parent_ids = set()
    child_map = {c.chunk_id: c for c in chunks if c.chunk_type == "child"}
    
    for child_id in child_ids:
        child = child_map.get(child_id)
        if child and child.parent_id:
            parent_ids.add(child.parent_id)
    
    # Retrieve parents
    parents = []
    parent_map = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}
    
    for parent_id in parent_ids:
        parent = parent_map.get(parent_id) or get_parent_chunk(parent_id)
        if parent:
            parents.append(parent)
    
    return parents


# Singleton instance
_chunker: Optional[HierarchicalChunker] = None


def get_hierarchical_chunker() -> HierarchicalChunker:
    """Get or create singleton chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = HierarchicalChunker()
    return _chunker
