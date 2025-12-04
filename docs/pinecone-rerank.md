# Rerank Results

> Rerank search results based on their relevance to a query using Pinecone's hosted reranking models.

## Endpoint

```
POST /rerank
```

**Base URL:** `https://api.pinecone.io`

---

## Authorization

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Api-Key` | `string` | ✅ | Your Pinecone API key |

---

## Headers

| Header | Type | Default | Required |
|--------|------|---------|----------|
| `Content-Type` | `application/json` | - | ✅ |
| `X-Pinecone-Api-Version` | `string` | `2025-10` | ✅ |

---

## Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | ✅ | Reranking model name |
| `query` | `string` | ✅ | Query to rerank against |
| `documents` | `array` | ✅ | Documents to rerank |
| `top_n` | `integer` | ❌ | Number of results to return (default: all) |
| `return_documents` | `boolean` | ❌ | Include document text in response (default: `true`) |
| `parameters` | `object` | ❌ | Model-specific parameters |

### Document Format

Documents can be strings or objects with a `text` field:

```json
// String format
["Document 1 text", "Document 2 text"]

// Object format
[
  {"id": "doc1", "text": "Document 1 text", "custom_field": "value"},
  {"id": "doc2", "text": "Document 2 text"}
]
```

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | Model used for reranking |
| `data` | `array` | Reranked results |
| `usage` | `object` | Token usage statistics |

### Result Object

| Field | Type | Description |
|-------|------|-------------|
| `index` | `integer` | Original document index |
| `score` | `number` | Relevance score (0-1) |
| `document` | `object` | Document content (if `return_documents=true`) |

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid parameters |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `429` | `RESOURCE_EXHAUSTED` | Rate limit exceeded |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Basic Rerank

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://api.pinecone.io/rerank" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "model": "pinecone-rerank-v0",
        "query": "What is machine learning?",
        "documents": [
          "The weather today is sunny with a high of 75°F.",
          "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
          "The stock market closed higher today.",
          "Deep learning uses neural networks with multiple layers to process data."
        ],
        "top_n": 2
      }'
```

### Response

```json
{
  "model": "pinecone-rerank-v0",
  "data": [
    {
      "index": 1,
      "score": 0.92,
      "document": {
        "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
      }
    },
    {
      "index": 3,
      "score": 0.78,
      "document": {
        "text": "Deep learning uses neural networks with multiple layers to process data."
      }
    }
  ],
  "usage": {
    "rerank_units": 1
  }
}
```

### Rerank with Document Objects

```bash
curl -X POST "https://api.pinecone.io/rerank" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "model": "pinecone-rerank-v0",
        "query": "python programming",
        "documents": [
          {"id": "doc1", "text": "Python is a versatile programming language.", "source": "wiki"},
          {"id": "doc2", "text": "Java is used for enterprise applications.", "source": "blog"},
          {"id": "doc3", "text": "Python has extensive libraries for data science.", "source": "tutorial"}
        ],
        "top_n": 2
      }'
```

### Response

```json
{
  "model": "pinecone-rerank-v0",
  "data": [
    {
      "index": 0,
      "score": 0.95,
      "document": {
        "id": "doc1",
        "text": "Python is a versatile programming language.",
        "source": "wiki"
      }
    },
    {
      "index": 2,
      "score": 0.88,
      "document": {
        "id": "doc3",
        "text": "Python has extensive libraries for data science.",
        "source": "tutorial"
      }
    }
  ],
  "usage": {
    "rerank_units": 1
  }
}
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Basic rerank
result = pc.inference.rerank(
    model="pinecone-rerank-v0",
    query="What is machine learning?",
    documents=[
        "The weather today is sunny.",
        "Machine learning enables systems to learn from data.",
        "The stock market closed higher.",
        "Deep learning uses neural networks."
    ],
    top_n=2
)

for item in result.data:
    print(f"Index {item.index}: Score {item.score:.3f}")
    print(f"  Text: {item.document['text']}")

# Rerank search results from Pinecone query
index = pc.Index(host="INDEX_HOST")

# First, get initial results
search_results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=20,
    include_metadata=True
)

# Prepare documents for reranking
documents = [
    {"id": m.id, "text": m.metadata.get("content", "")}
    for m in search_results.matches
]

# Rerank
reranked = pc.inference.rerank(
    model="pinecone-rerank-v0",
    query="What is machine learning?",
    documents=documents,
    top_n=5
)

# Get final results
for item in reranked.data:
    original_match = search_results.matches[item.index]
    print(f"ID: {original_match.id}")
    print(f"  Vector score: {original_match.score:.3f}")
    print(f"  Rerank score: {item.score:.3f}")
```

---

## Available Models

| Model | Max Tokens | Max Batch | Provider | Description |
|-------|------------|-----------|----------|-------------|
| `pinecone-rerank-v0` | 512 | 100 | Pinecone | State-of-the-art, handles 1-2 paragraphs |
| `bge-reranker-v2-m3` | 1024 | 100 | BAAI | Multilingual reranking |
| `cohere-rerank-3.5` | 40000 | 200 | Cohere | Long document support |

---

## Two-Stage Retrieval Pattern

Reranking is typically used in a two-stage retrieval pattern:

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Query     │ ──▶ │   Stage 1    │ ──▶ │  Stage 2   │
│             │     │   Vector     │     │  Rerank    │
│             │     │   Search     │     │            │
└─────────────┘     │  (top_k=50)  │     │  (top_n=5) │
                    └──────────────┘     └────────────┘
                           │                    │
                           ▼                    ▼
                    Fast, approximate      Precise, slower
                    High recall            High precision
```

### Benefits

1. **Better precision** - Rerankers understand query-document relevance better
2. **Cost efficient** - Only rerank top candidates from vector search
3. **Hybrid signals** - Combine semantic similarity with relevance

---

## Truncation

| Model | Truncation Behavior |
|-------|---------------------|
| `pinecone-rerank-v0` | Truncates at 512 tokens |
| `bge-reranker-v2-m3` | Truncates at 1024 tokens |
| `cohere-rerank-3.5` | Truncates at 40000 tokens |

Set `parameters.truncate` to control:
- `"END"` - Truncate from end (default)
- `"NONE"` - Error if exceeds limit

---

## Limits

| Parameter | Limit |
|-----------|-------|
| Documents per request | Model-specific (100-200) |
| Document length | Model-specific (512-40000 tokens) |
| Query length | 512 tokens |

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/inference/rerank)
