# Qdrant Cloud Migration Guide for PlaudBlender

> **Current stance:** Qdrant is now the default/primary vector store. Any remaining “Pinecone” mentions are for legacy compatibility only. If you see Pinecone keys/env vars, treat them as fallbacks; the live stack runs on Qdrant.

> **Purpose:** Help Gunnar understand what migrating from Pinecone to Qdrant involves, why it might be better for PlaudBlender, and what the actual work looks like.

## ✅ MIGRATION STATUS: PHASE 1 COMPLETE

| Component | Status | Notes |
|-----------|--------|-------|
| Qdrant Docker running | ✅ Done | `docker ps` to verify |
| Qdrant Python client | ✅ Done | `src/qdrant_client.py` |
| Abstraction layer | ✅ Done | `src/vector_store.py` |
| Feature flag | ✅ Done | `VECTOR_DB=qdrant` in `.env` (Pinecone keys kept only as fallback) |
| clients.py updated | ✅ Done | Uses abstraction layer |
| Tests passing | ✅ Done | 57 tests green |
| **Dashboard access** | ✅ Done | http://localhost:6333/dashboard |
| **Pinecone references** | ⚠️ Legacy | Left only for compatibility; safe to remove when desired |

**Next steps:** Re-embed your transcripts into Qdrant (Phase 2)

---

## TL;DR — The Honest Assessment

| Aspect | Verdict |
|--------|---------|
| **Is it worth it?** | Probably yes, especially for your use case (personal knowledge base, local-first, transparency) |
| **How hard?** | Medium — ~15-20 files touched, 2-3 days of focused work |
| **Risk level** | Low — you can run both in parallel during migration |
| **Cost difference** | Qdrant Cloud free tier is more generous; self-hosted is $0 |

---

## Part 1: Why Qdrant Might Be Better for PlaudBlender

### 1.1 Your Use Case Profile

Based on your project, you need:
- ✅ Personal knowledge base (not enterprise scale)
- ✅ Transparency/introspection (you love seeing what's happening under the hood)
- ✅ Local-first option (data sovereignty matters)
- ✅ Cost-effective for a solo developer
- ✅ Rich metadata filtering
- ✅ Full-text + vector hybrid search

### 1.2 Pinecone Pain Points

| Issue | How It Affects You |
|-------|-------------------|
| **Opaque** | You can't see your data, just query it. No browsing vectors in a dashboard. |
| **Serverless lock-in** | Can't run locally; always depends on their cloud |
| **Pricing at scale** | $0.33/1M reads seems cheap until you're debugging queries |
| **No built-in full-text** | You have to bring your own sparse embeddings for hybrid search |
| **Namespace quirks** | Namespaces aren't first-class; can't iterate/list vectors easily |

### 1.3 Qdrant Advantages

| Feature | Why It Matters for You |
|---------|----------------------|
| **Open source** | Run locally on your Mac for free during development |
| **Web dashboard** | Actually SEE your vectors, browse collections, debug visually |
| **Built-in hybrid search** | Native sparse+dense fusion — no BM25 hacks needed |
| **Rich filtering** | Full boolean logic, nested fields, geo, datetime |
| **Snapshots/backups** | First-class backup to S3/local — you control your data |
| **Payload (metadata)** | No 40KB limit like Pinecone; store full text if you want |
| **Self-host option** | Docker container, even runs on a Raspberry Pi |
| **Qdrant Cloud** | Managed option with generous free tier (1GB storage) |

### 1.4 Qdrant Potential Downsides

| Concern | Reality Check |
|---------|--------------|
| **Smaller ecosystem** | True, but Python SDK is mature and well-documented |
| **No gRPC in cloud** | REST is fine for your scale; gRPC for self-hosted |
| **Less "enterprise"** | You don't need SOC2 compliance for your voice memos |
| **Learning curve** | Different concepts, but you'll grok it fast |

---

## Part 2: Conceptual Mapping — Pinecone → Qdrant

| Pinecone Concept | Qdrant Equivalent | Notes |
|------------------|-------------------|-------|
| **Index** | **Collection** | Same idea — container for vectors of same dimension |
| **Namespace** | **Payload filter** OR **separate collection** | Qdrant doesn't have namespaces; use `{"namespace": "full_text"}` in payload, or create `transcripts_full_text` and `transcripts_summaries` collections |
| **Metadata** | **Payload** | Qdrant's term; same thing, more flexible |
| **Vector ID** | **Point ID** | String or integer; Qdrant calls vectors "points" |
| **Serverless Spec** | **Cloud cluster config** | Or just run Docker locally |
| **query()** | **search()** | Almost identical API |
| **upsert()** | **upsert()** | Identical concept |
| **query_namespaces()** | **Multi-collection query** or **filter** | Either filter by payload field or query multiple collections |

---

## Part 3: Your Current Pinecone Footprint

Based on code analysis, here's what needs to change:

### 3.1 Core Files to Modify

| File | Purpose | Effort |
|------|---------|--------|
| `src/pinecone_client.py` (789 lines) | Core vector operations | **Replace entirely** with `src/qdrant_client.py` |
| `gui/services/pinecone_service.py` (276 lines) | GUI wrapper | Rename + adapt to Qdrant client |
| `gui/services/search_service.py` (659 lines) | Search logic | Update client calls, similar patterns |
| `gui/services/index_manager.py` | Index/dimension management | Adapt to collection API |
| `gui/services/clients.py` | Client factory | Add Qdrant client factory |
| `src/models/vector_metadata.py` (112 lines) | Metadata schema | Minor tweaks (Qdrant is less restrictive) |
| `gui/views/pinecone.py` | GUI view | Rename to `vector_db.py`, update methods |
| `.env` | Config | Replace `PINECONE_*` with `QDRANT_*` vars |

### 3.2 Files That Stay the Same

| File | Why |
|------|-----|
| `gui/services/embedding_service.py` | Embeddings are provider-agnostic |
| `src/database/*` | SQLite is your source of truth — unchanged |
| `gui/services/transcripts_service.py` | Fetches from SQLite, not vector DB |
| All GUI views except pinecone.py | They call services, not clients directly |

### 3.3 Patterns to Preserve

Your codebase already has good patterns that translate directly:
- `build_metadata()` helper → works with Qdrant payloads
- `text_hash` for dedup → same approach
- Dimension checking → Qdrant has similar `collection_info()`
- Background threading for I/O → unchanged

---

## Part 4: Migration Strategy

### Option A: Big Bang (Not Recommended)
Replace Pinecone entirely in one go. Risky if something breaks.

### Option B: Parallel Run (Recommended) ✅

1. **Create abstraction layer** — `VectorStoreClient` interface
2. **Implement both backends** — `PineconeVectorStore`, `QdrantVectorStore`
3. **Feature flag** — `VECTOR_DB=pinecone` or `VECTOR_DB=qdrant` in `.env`
4. **Migrate data** — Export from Pinecone, import to Qdrant
5. **Validate** — Run same queries, compare results
6. **Cut over** — Switch flag, remove Pinecone code

### Option C: Fresh Start (If Pinecone Data Is Small)

Since your source of truth is SQLite + Plaud API:
1. **Build Qdrant client** 
2. **Re-embed everything** from SQLite transcripts
3. **Delete Pinecone resources** when satisfied

---

## Part 5: Qdrant Python SDK Primer

### 5.1 Installation

```bash
pip install qdrant-client
```

### 5.2 Connection

```python
from qdrant_client import QdrantClient

# Option 1: Qdrant Cloud
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Option 2: Local Docker
client = QdrantClient(host="localhost", port=6333)

# Option 3: In-memory (for testing)
client = QdrantClient(":memory:")
```

### 5.3 Create Collection (= Pinecone Index)

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="transcripts",
    vectors_config=VectorParams(
        size=768,  # Your embedding dimension
        distance=Distance.COSINE,
    ),
)
```

### 5.4 Upsert Vectors

```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="transcripts",
    points=[
        PointStruct(
            id="rec_abc123_chunk_0",  # String ID works!
            vector=[0.1, 0.2, ...],   # Your embedding
            payload={                  # Your metadata
                "recording_id": "abc123",
                "namespace": "full_text",  # Replaces Pinecone namespace
                "text": "Full transcript text...",
                "text_hash": "sha256...",
                "title": "Meeting Notes",
                "start_at": "2024-12-15T10:30:00Z",
                "themes": ["work", "planning"],
            }
        ),
    ]
)
```

### 5.5 Search with Filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="transcripts",
    query_vector=[0.1, 0.2, ...],
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="namespace",
                match=MatchValue(value="full_text"),
            )
        ]
    ),
)

for hit in results:
    print(f"ID: {hit.id}, Score: {hit.score}")
    print(f"Title: {hit.payload['title']}")
```

### 5.6 Hybrid Search (Sparse + Dense)

This is where Qdrant really shines:

```python
from qdrant_client.models import SparseVector

# Create collection with both dense and sparse vectors
client.create_collection(
    collection_name="transcripts_hybrid",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": {},  # BM25-style sparse embeddings
    },
)

# Query with both
results = client.query_points(
    collection_name="transcripts_hybrid",
    prefetch=[
        {"query": dense_vector, "using": "dense", "limit": 20},
        {"query": sparse_vector, "using": "sparse", "limit": 20},
    ],
    query={"fusion": "rrf"},  # Reciprocal Rank Fusion
    limit=5,
)
```

---

## Part 6: What the New Client Would Look Like

Here's a sketch of `src/qdrant_client.py`:

```python
"""
Qdrant client for vector operations

Drop-in replacement for PineconeClient with same interface.
"""
import os
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
)
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantVectorClient:
    """
    Qdrant client matching PineconeClient interface.
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if url and api_key:
            # Qdrant Cloud
            self.client = QdrantClient(url=url, api_key=api_key)
        elif url:
            # Self-hosted (no auth)
            self.client = QdrantClient(url=url)
        else:
            # Local Docker default
            self.client = QdrantClient(host="localhost", port=6333)
        
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "transcripts")
        logger.info(f"Qdrant client initialized for collection: {self.collection_name}")

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> bool:
        """Create a new collection (= Pinecone index)."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dotproduct": Distance.DOT,
        }
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance_map.get(metric, Distance.COSINE),
                ),
            )
            logger.info(f"Created Qdrant collection '{name}' ({dimension}d, {metric})")
            return True
        except Exception as e:
            if "already exists" in str(e):
                return True
            logger.error(f"Error creating collection: {e}")
            return False

    def list_collections(self) -> List[str]:
        """List all collections (= Pinecone list_indexes)."""
        collections = self.client.get_collections().collections
        return [c.name for c in collections]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection stats (= Pinecone describe_index_stats)."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "total_vectors": info.points_count,
            "dimension": info.config.params.vectors.size,
            "metric": str(info.config.params.vectors.distance),
            "status": info.status,
        }

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        namespace: str = "",
    ) -> List[Any]:
        """Search for similar vectors."""
        # Build Qdrant filter from namespace + filter_dict
        conditions = []
        if namespace:
            conditions.append(
                FieldCondition(key="namespace", match=MatchValue(value=namespace))
            )
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        
        query_filter = Filter(must=conditions) if conditions else None
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        return results

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
    ) -> bool:
        """Upsert vectors with metadata."""
        points = []
        for vec in vectors:
            payload = vec.get("metadata", {}).copy()
            if namespace:
                payload["namespace"] = namespace
            points.append(
                PointStruct(
                    id=vec["id"],
                    vector=vec["values"],
                    payload=payload,
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return True

    def delete_vectors(self, ids: List[str], namespace: str = "") -> bool:
        """Delete vectors by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
        return True

    # ... more methods matching PineconeClient interface
```

---

## Part 7: Environment Variables

### Current (.env for Pinecone)
```bash
PINECONE_API_KEY=pcsk_xxxxx
PINECONE_INDEX_NAME=transcripts
PINECONE_RERANK_ENABLED=false
PINECONE_RERANK_MODEL=bge-reranker-v2-m3
```

### New (.env for Qdrant)
```bash
# Qdrant Cloud
QDRANT_URL=https://your-cluster-xyz.qdrant.io
QDRANT_API_KEY=your-api-key-here
QDRANT_COLLECTION=transcripts

# OR for local Docker
QDRANT_URL=http://localhost:6333
# No API key needed for local

# Feature flag during migration
VECTOR_DB=qdrant  # or "pinecone"
```

---

## Part 8: Cost Comparison

### Pinecone Serverless
| Metric | Price |
|--------|-------|
| Storage | $0.33/GB/month |
| Reads | $8.25/1M queries |
| Writes | $2/1M upserts |
| Free tier | 100K vectors (then pay) |

### Qdrant Cloud
| Metric | Price |
|--------|-------|
| Free tier | 1GB storage, 1M vectors |
| Starter | $25/month for 4GB |
| Self-hosted | **$0** (just compute costs) |

### For Your Scale
With ~1000 Plaud recordings, you're looking at maybe 10K-50K vectors:
- **Pinecone**: Probably free tier, but hits limits fast with debugging
- **Qdrant Cloud**: Comfortably free tier
- **Qdrant Docker**: Zero cost, runs on your Mac

---

## Part 9: Action Items / Roadmap

### Phase 1: Setup & Exploration (Day 1)
- [ ] Sign up for Qdrant Cloud (free tier)
- [ ] OR run Docker locally: `docker run -p 6333:6333 qdrant/qdrant`
- [ ] Install SDK: `pip install qdrant-client`
- [ ] Play with web dashboard at `http://localhost:6333/dashboard`
- [ ] Create test collection, upsert dummy vectors, query them

### Phase 2: Build Abstraction (Day 1-2)
- [ ] Create `src/vector_store/__init__.py` with base interface
- [ ] Move Pinecone code to `src/vector_store/pinecone_store.py`
- [ ] Create `src/vector_store/qdrant_store.py`
- [ ] Add factory function reading `VECTOR_DB` env var
- [ ] Update `gui/services/clients.py` to use factory

### Phase 3: Migrate Services (Day 2)
- [ ] Update `gui/services/pinecone_service.py` → `vector_service.py`
- [ ] Update `gui/services/search_service.py` for new client
- [ ] Update `gui/services/index_manager.py` → `collection_manager.py`
- [ ] Test all search functions work

### Phase 4: Re-embed & Validate (Day 2-3)
- [ ] Use existing `reembed_all_into_index()` logic with Qdrant backend
- [ ] Compare search results between Pinecone and Qdrant
- [ ] Verify GUI displays data correctly

### Phase 5: Cleanup (Day 3)
- [ ] Remove Pinecone-specific code
- [ ] Update documentation
- [ ] Update tests
- [ ] Delete Pinecone index to stop billing

---

## Part 10: Quick Reference — Qdrant CLI

```bash
# Run locally
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

# Web UI
open http://localhost:6333/dashboard

# List collections via REST
curl http://localhost:6333/collections

# Collection info
curl http://localhost:6333/collections/transcripts
```

---

## Summary: Should You Do This?

**Yes**, if you value:
- 🔍 **Transparency** — actually see your vectors in a UI
- 💰 **Cost control** — free self-hosting option
- 🏠 **Data sovereignty** — your vectors on your machine
- 🔧 **Flexibility** — richer filtering, hybrid search built-in
- 🧪 **Developer experience** — easier to debug and iterate

**Timing**: 2-3 focused days. You can run both in parallel, so zero downtime risk.

**Next step**: Run `docker run -p 6333:6333 qdrant/qdrant` and open `http://localhost:6333/dashboard` to see what you've been missing.

---

*Document created: December 16, 2024*
*Last updated: December 16, 2024*
