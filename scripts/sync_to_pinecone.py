#!/usr/bin/env python3
"""
Dump all Plaud transcripts into Pinecone

Fetches transcripts from Plaud API and stores embeddings in Pinecone.
Uses the centralized EmbeddingService with configurable model/dimension.

Default: gemini-embedding-001 @ 768 dimensions
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.plaud_client import PlaudClient
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Import centralized embedding service with new configurable API
from gui.services.embedding_service import (
    get_embedding_service,
    get_embedding_dimension,
    get_embedding_model,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingDimension,
    TaskType,
)


def get_embedding(text: str) -> list:
    """
    Get embedding using centralized EmbeddingService.
    
    Returns:
        Vector with configured dimension (default 768)
    """
    service = get_embedding_service()
    return service.embed_document(text)


def extract_themes_with_gemini(text: str) -> list:
    """Extract themes using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""Extract 3-5 key themes or topics from this transcript. 
Return ONLY a JSON array of theme strings, nothing else.
Example: ["productivity", "mental health", "career development"]

Transcript (first 2000 chars):
{text[:2000]}

Themes:"""
    
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Clean up response
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        themes = json.loads(result)
        return [str(t).lower().strip() for t in themes[:5]]
    except Exception as e:
        print(f"  ⚠️ Could not extract themes: {e}")
        return ["general"]


def main():
    print("\n" + "="*60)
    print("🚀 PLAUD → PINECONE SYNC")
    print("="*60)
    
    # Initialize clients
    print("\n📡 Initializing clients...")
    
    # AUTO-SYNC: Ensure embedding dimension matches Pinecone
    try:
        from gui.services.index_manager import sync_dimensions, get_compatible_dimension
        dim, action = sync_dimensions()
        print(f"  🔄 Dimension sync: {dim}d ({action})")
    except Exception as e:
        print(f"  ⚠️ Could not sync dimensions: {e}")
    
    # Get embedding config (now synced with index)
    embedding_dim = get_embedding_dimension()
    embedding_model = get_embedding_model()
    
    # Plaud
    plaud = PlaudClient()
    print("  ✅ Plaud API connected")
    
    # Google AI
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("  ❌ GEMINI_API_KEY not set!")
        return
    genai.configure(api_key=gemini_key)
    print(f"  ✅ Google AI configured")
    print(f"     Model: {embedding_model}")
    print(f"     Dimension: {embedding_dim}")
    
    # Pinecone
    pc_key = os.getenv("PINECONE_API_KEY")
    if not pc_key:
        print("  ❌ PINECONE_API_KEY not set!")
        return
    
    pc = Pinecone(api_key=pc_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "transcripts")
    
    # Check/create index with correct dimension (from config)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"  📦 Creating Pinecone index: {index_name}")
        print(f"     Dimension: {embedding_dim} ({embedding_model})")
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,  # From EmbeddingService config
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"  ✅ Pinecone index: {index_name} ({stats.total_vector_count} vectors)")
    
    # Fetch all recordings
    print("\n📥 Fetching recordings from Plaud...")
    files = plaud.list_recordings(limit=100)
    print(f"  Found {len(files)} recordings")
    
    # Process each
    print("\n🔄 Processing transcripts...")
    success = 0
    failed = 0
    
    for i, file in enumerate(files, 1):
        file_id = file.get('id')
        name = file.get('name', 'Untitled')[:50]
        duration = file.get('duration', 0) // 1000
        
        print(f"\n[{i}/{len(files)}] {name}")
        print(f"         ID: {file_id[:12]}... | Duration: {duration}s")
        
        # Get transcript
        try:
            transcript = plaud.get_transcript_text(file_id)
            
            if not transcript or len(transcript.strip()) < 50:
                print(f"         ⏭️  Skipping (transcript too short: {len(transcript)} chars)")
                failed += 1
                continue
            
            print(f"         📝 Transcript: {len(transcript)} chars")
            
            # Extract themes
            themes = extract_themes_with_gemini(transcript)
            print(f"         🏷️  Themes: {', '.join(themes)}")
            
            # Generate embedding
            print(f"         🧮 Generating embedding...")
            embedding = get_embedding(transcript[:8000])  # Limit to 8k chars for embedding
            
            # Upsert to Pinecone
            metadata = {
                "title": file.get('name', 'Untitled'),
                "plaud_id": file_id,
                "duration": duration,
                "start_at": file.get('start_at', ''),
                "themes": ", ".join(themes),
                "text": transcript,  # Store full transcript
                "char_count": len(transcript),
                "synced_at": datetime.now().isoformat()
            }
            
            index.upsert(
                vectors=[{
                    "id": file_id,
                    "values": embedding,
                    "metadata": metadata
                }]
            )
            
            print(f"         ✅ Stored in Pinecone")
            success += 1
            
        except Exception as e:
            print(f"         ❌ Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 SYNC COMPLETE")
    print("="*60)
    print(f"  ✅ Successfully processed: {success}")
    print(f"  ❌ Failed/skipped: {failed}")
    
    # Final stats
    stats = index.describe_index_stats()
    print(f"\n  📦 Pinecone now has: {stats.total_vector_count} vectors")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
