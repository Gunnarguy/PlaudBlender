"""
LLM Processor using LlamaIndex and Google Gemini
"""
import os
from datetime import datetime
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class TranscriptProcessor:
    """Process transcripts using LlamaIndex, Gemini, and Pinecone"""
    
    def __init__(self):
        """Initialize LLM and vector store"""
        logger.info("Initializing Transcript Processor...")
        
        # Initialize Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY must be set in .env file")
        
        self.llm = Gemini(
            model="models/gemini-2.0-flash-exp",
            api_key=gemini_key,
            temperature=0.7
        )
        Settings.llm = self.llm
        logger.info("Gemini LLM initialized")
        
        # Initialize Pinecone
        pc_key = os.getenv("PINECONE_API_KEY")
        if not pc_key:
            raise ValueError("PINECONE_API_KEY must be set in .env file")
        
        pc = Pinecone(api_key=pc_key)
        
        # Create or connect to index
        index_name = os.getenv("PINECONE_INDEX_NAME", "transcripts")
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        else:
            logger.info(f"Connected to existing Pinecone index: {index_name}")
        
        pinecone_index = pc.Index(index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.pc = pc
        self.index_name = index_name
        
        logger.info("Transcript Processor ready")
    
    def extract_themes(self, text):
        """
        Extract key themes from transcript using LLM
        
        Args:
            text: Transcript text
            
        Returns:
            List of theme strings
        """
        try:
            prompt = f"""Extract 3-5 key themes or topics from this transcript. 
Return ONLY a JSON array of theme strings, nothing else.

Example format: ["productivity", "mental health", "career development"]

Transcript:
{text[:3000]}

Themes (JSON array only):"""
            
            response = self.llm.complete(prompt).text.strip()
            
            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                
                themes = json.loads(response)
                if isinstance(themes, list):
                    return [str(t).lower().strip() for t in themes[:5]]
            except json.JSONDecodeError:
                # Fallback: split by commas
                themes = [t.strip(' "[]\'') for t in response.split(',')]
                return [t.lower() for t in themes if t][:5]
            
            return ["general"]
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return ["general"]
    
    def process_transcript(self, transcript_text, page_id, title="Untitled"):
        """
        Process single transcript: embed, synthesize, find connections
        
        Args:
            transcript_text: Full transcript text
            page_id: Notion page ID
            title: Page title
            
        Returns:
            Dict with synthesis, connections, and themes
        """
        logger.info(f"Processing transcript: {title[:50]}")
        
        try:
            # Extract themes first
            themes = self.extract_themes(transcript_text)
            logger.info(f"Extracted themes: {themes}")
            
            # Create document with metadata
            document = Document(
                text=transcript_text,
                metadata={
                    "page_id": page_id,
                    "title": title,
                    "created": str(datetime.now()),
                    "themes": ", ".join(themes)
                }
            )
            
            # Insert into vector store (auto-generates embeddings)
            self.index.insert(document)
            logger.info(f"Embedded and stored document for {page_id[:8]}...")
            
            # Generate synthesis
            synthesis_prompt = f"""Analyze this transcript and provide a comprehensive synthesis.

Title: {title}

Transcript:
{transcript_text}

Provide:
1. **Key Insights** (3-5 bullet points of the most important takeaways)
2. **Main Themes** (2-3 overarching themes)
3. **Connections** (How this relates to broader concepts or potential other topics)
4. **Action Items** (If any actionable insights emerge)

Keep the synthesis concise but insightful (max 400 words)."""
            
            synthesis = self.llm.complete(synthesis_prompt).text
            logger.info(f"Generated synthesis ({len(synthesis)} chars)")
            
            # Find related transcripts using semantic search
            query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="no_text"
            )
            
            # Query with key themes and summary
            query_text = f"Themes: {', '.join(themes)}. Summary: {transcript_text[:500]}"
            related = query_engine.query(query_text)
            
            # Extract unique page IDs (excluding current page)
            connections = []
            for node in related.source_nodes:
                related_id = node.node.metadata.get("page_id")
                if related_id and related_id != page_id and related_id not in connections:
                    connections.append(related_id)
            
            logger.info(f"Found {len(connections)} related transcripts")
            
            return {
                "synthesis": synthesis,
                "connections": connections[:10],  # Limit to top 10
                "themes": themes,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing transcript {page_id}: {e}")
            return {
                "synthesis": f"Error processing: {str(e)}",
                "connections": [],
                "themes": ["error"],
                "success": False
            }
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index"""
        try:
            idx = self.pc.Index(self.index_name)
            stats = idx.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}
    
    def query_similar_transcripts(self, query_text, top_k=5):
        """
        Query for similar transcripts
        
        Args:
            query_text: Text to search for
            top_k: Number of results
            
        Returns:
            List of similar documents with metadata
        """
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text"
            )
            
            results = query_engine.query(query_text)
            
            similar = []
            for node in results.source_nodes:
                similar.append({
                    "page_id": node.node.metadata.get("page_id", "unknown"),
                    "title": node.node.metadata.get("title", "Untitled"),
                    "themes": node.node.metadata.get("themes", ""),
                    "score": node.score,
                    "text_preview": node.node.text[:200] + "..."
                })
            
            return similar
            
        except Exception as e:
            logger.error(f"Error querying similar transcripts: {e}")
            return []
