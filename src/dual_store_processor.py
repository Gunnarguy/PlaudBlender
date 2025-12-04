"""
Enhanced Transcript Processor with Dual Namespace Architecture
Optimized for PlaudAI → Notion → Zapier → Pinecone pipeline

Features:
- Dual namespace storage (full_text + summaries)
- gRPC transport for improved performance
- AI-powered theme extraction and synthesis
- Configurable embedding dimensions via EmbeddingService
"""
import os
import sys
import tiktoken
from datetime import datetime
from typing import List, Dict, Optional
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging

# Use gRPC client for better performance on data operations
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    USING_GRPC = True
except ImportError:
    from pinecone import Pinecone
    USING_GRPC = False

from pinecone import ServerlessSpec

# Import embedding service for dimension configuration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from gui.services.embedding_service import get_embedding_dimension, get_embedding_model
except ImportError:
    # Fallback if embedding service not available
    def get_embedding_dimension():
        return 768
    def get_embedding_model():
        return "gemini-embedding-001"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class DualStoreTranscriptProcessor:
    """
    Process transcripts with dual namespace storage:
    - 'full_text': Complete transcripts (chunked for long ones)
    - 'summaries': AI-generated syntheses
    """
    
    # Constants
    MAX_CHUNK_TOKENS = 8000  # Pinecone limit consideration
    OVERLAP_TOKENS = 200     # Overlap between chunks for context
    
    def __init__(self):
        """Initialize with dual namespace support"""
        logger.info("🚀 Initializing Dual Store Transcript Processor...")
        
        # Initialize Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY must be set")
        
        self.llm = Gemini(
            model="models/gemini-2.0-flash-exp",
            api_key=gemini_key,
            temperature=0.7
        )
        Settings.llm = self.llm
        logger.info("✅ Gemini LLM initialized")
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize Pinecone with optimized settings
        pc_key = os.getenv("PINECONE_API_KEY")
        if not pc_key:
            raise ValueError("PINECONE_API_KEY must be set")
        
        self.pc = Pinecone(api_key=pc_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "transcripts")
        
        transport = "gRPC" if USING_GRPC else "HTTP"
        logger.info(f"✅ Pinecone initialized [{transport}]")
        
        # Create or connect to index
        self._ensure_index_exists()
        
        # Get index for both namespaces
        self.pinecone_index = self.pc.Index(self.index_name)
        
        # Create vector stores for both namespaces
        self.full_text_store = PineconeVectorStore(
            pinecone_index=self.pinecone_index,
            namespace="full_text"
        )
        self.summary_store = PineconeVectorStore(
            pinecone_index=self.pinecone_index,
            namespace="summaries"
        )
        
        # Create indexes for both
        self.full_text_index = VectorStoreIndex.from_vector_store(self.full_text_store)
        self.summary_index = VectorStoreIndex.from_vector_store(self.summary_store)
        
        logger.info("✅ Dual namespace vector stores ready")
        logger.info(f"   📦 Namespace 'full_text': For complete transcripts")
        logger.info(f"   📦 Namespace 'summaries': For AI syntheses")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist with correct embedding dimension"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        # Get dimension from centralized EmbeddingService config
        embedding_dim = get_embedding_dimension()
        embedding_model = get_embedding_model()
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            logger.info(f"   Dimension: {embedding_dim} (from {embedding_model})")
            self.pc.create_index(
                name=self.index_name,
                dimension=embedding_dim,  # From EmbeddingService config
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"✅ Index '{self.index_name}' created")
        else:
            logger.info(f"✅ Connected to existing index: {self.index_name}")
    
    def _chunk_text(self, text: str, page_id: str, title: str, metadata: Dict) -> List[Document]:
        """
        Chunk long transcripts for efficient storage and retrieval
        
        Args:
            text: Full transcript text
            page_id: Notion page ID
            title: Transcript title
            metadata: Additional metadata
            
        Returns:
            List of Document objects (chunks)
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.MAX_CHUNK_TOKENS:
            # No chunking needed
            return [Document(
                text=text,
                metadata={
                    **metadata,
                    "page_id": page_id,
                    "title": title,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "doc_type": "full_transcript"
                }
            )]
        
        # Chunk with overlap
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(tokens):
            end = start + self.MAX_CHUNK_TOKENS
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(Document(
                text=chunk_text,
                metadata={
                    **metadata,
                    "page_id": page_id,
                    "title": title,
                    "chunk_index": chunk_index,
                    "total_chunks": -1,  # Will update later
                    "doc_type": "full_transcript_chunk"
                }
            ))
            
            chunk_index += 1
            start = end - self.OVERLAP_TOKENS  # Overlap for context
        
        # Update total_chunks in all chunks
        total = len(chunks)
        for doc in chunks:
            doc.metadata["total_chunks"] = total
        
        logger.info(f"📄 Chunked transcript into {total} parts")
        return chunks
    
    def extract_themes(self, text: str) -> List[str]:
        """Extract key themes using LLM"""
        try:
            prompt = f"""Extract 3-5 key themes from this transcript.
Return ONLY a JSON array: ["theme1", "theme2", ...]

Transcript (first 3000 chars):
{text[:3000]}

Themes:"""
            
            response = self.llm.complete(prompt).text.strip()
            
            # Clean response
            if response.startswith("```"):
                response = response.split("```")[1].strip()
                if response.startswith("json"):
                    response = response[4:].strip()
            
            import json
            themes = json.loads(response)
            return [str(t).lower().strip() for t in themes[:5]]
            
        except Exception as e:
            logger.warning(f"Theme extraction failed: {e}")
            return ["general"]
    
    def generate_synthesis(self, transcript_text: str, title: str, themes: List[str]) -> str:
        """Generate comprehensive AI synthesis"""
        prompt = f"""Analyze this transcript and provide a detailed synthesis.

Title: {title}
Themes: {', '.join(themes)}

Transcript:
{transcript_text}

Provide:
1. **Overview** (2-3 sentences capturing the essence)
2. **Key Insights** (3-5 bullet points)
3. **Main Themes** (Brief explanation of themes)
4. **Connections** (How this relates to broader topics)
5. **Action Items** (If any)

Keep it comprehensive but concise (max 500 words)."""
        
        return self.llm.complete(prompt).text
    
    def process_transcript(
        self, 
        transcript_text: str, 
        page_id: str, 
        title: str = "Untitled",
        created_at: Optional[str] = None
    ) -> Dict:
        """
        Process transcript with dual storage strategy
        
        Args:
            transcript_text: Full transcript text
            page_id: Notion page ID
            title: Transcript title
            created_at: Creation timestamp
            
        Returns:
            Dict with processing results
        """
        logger.info(f"📝 Processing: {title[:50]}")
        
        try:
            # Extract themes
            themes = self.extract_themes(transcript_text)
            logger.info(f"🎯 Themes: {themes}")
            
            # Calculate metadata
            word_count = len(transcript_text.split())
            token_count = len(self.tokenizer.encode(transcript_text))
            
            base_metadata = {
                "title": title,
                "themes": ", ".join(themes),
                "created": created_at or str(datetime.now()),
                "word_count": word_count,
                "token_count": token_count
            }
            
            # === STEP 1: Store full transcript in 'full_text' namespace ===
            chunks = self._chunk_text(transcript_text, page_id, title, base_metadata)
            for chunk in chunks:
                self.full_text_index.insert(chunk)
            
            logger.info(f"✅ Stored {len(chunks)} chunk(s) in 'full_text' namespace")
            
            # === STEP 2: Generate and store synthesis in 'summaries' namespace ===
            synthesis = self.generate_synthesis(transcript_text, title, themes)
            logger.info(f"✅ Generated synthesis ({len(synthesis)} chars)")
            
            summary_doc = Document(
                text=synthesis,
                metadata={
                    **base_metadata,
                    "page_id": page_id,
                    "doc_type": "summary",
                    "full_text_chunks": len(chunks)
                }
            )
            
            self.summary_index.insert(summary_doc)
            logger.info(f"✅ Stored synthesis in 'summaries' namespace")
            
            # === STEP 3: Find related transcripts (query summaries for high-level similarity) ===
            query_engine = self.summary_index.as_query_engine(
                similarity_top_k=5,
                response_mode="no_text"
            )
            
            query_text = f"Themes: {', '.join(themes)}. {synthesis[:300]}"
            related = query_engine.query(query_text)
            
            connections = []
            for node in related.source_nodes:
                related_id = node.node.metadata.get("page_id")
                if related_id and related_id != page_id and related_id not in connections:
                    connections.append(related_id)
            
            logger.info(f"🔗 Found {len(connections)} related transcripts")
            
            return {
                "success": True,
                "page_id": page_id,
                "title": title,
                "themes": themes,
                "synthesis": synthesis,
                "full_text": transcript_text,  # Include for visualization
                "word_count": word_count,
                "token_count": token_count,
                "chunks_stored": len(chunks),
                "connections": connections[:10],
                "created": base_metadata["created"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {page_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "page_id": page_id
            }
    
    def query_full_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query the full transcript namespace for detailed search
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching chunks with metadata
        """
        query_engine = self.full_text_index.as_query_engine(
            similarity_top_k=top_k
        )
        
        response = query_engine.query(query)
        
        results = []
        for node in response.source_nodes:
            results.append({
                "page_id": node.node.metadata.get("page_id"),
                "title": node.node.metadata.get("title"),
                "text": node.node.text,
                "chunk_index": node.node.metadata.get("chunk_index"),
                "score": node.score
            })
        
        return results
    
    def query_summaries(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Query the summaries namespace for high-level similarity
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching summaries with metadata
        """
        query_engine = self.summary_index.as_query_engine(
            similarity_top_k=top_k
        )
        
        response = query_engine.query(query)
        
        results = []
        for node in response.source_nodes:
            results.append({
                "page_id": node.node.metadata.get("page_id"),
                "title": node.node.metadata.get("title"),
                "synthesis": node.node.text,
                "themes": node.node.metadata.get("themes"),
                "score": node.score
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics for both namespaces"""
        try:
            stats = self.pinecone_index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "namespaces": stats.get("namespaces", {}),
                "dimension": stats.get("dimension", 1536)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
