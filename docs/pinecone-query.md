# Query Vectors

> Search a namespace using a query vector to find the most similar vectors.

## Endpoint

```
POST /query
```

**Base URL:** `https://{index_host}` (your index endpoint)

> **Note:** This is a **Data Plane** endpoint. Send requests to your index host URL, not `api.pinecone.io`.

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
| `vector` | `array` | ✅* | Query vector (must match index dimension) |
| `id` | `string` | ✅* | Query by existing vector ID |
| `top_k` | `integer` | ✅ | Number of results to return (1-10000) |
| `namespace` | `string` | ❌ | Namespace to query (default: `""`) |
| `filter` | `object` | ❌ | Metadata filter |
| `include_values` | `boolean` | ❌ | Return vector values (default: `false`) |
| `include_metadata` | `boolean` | ❌ | Return metadata (default: `false`) |
| `sparse_vector` | `object` | ❌ | Sparse vector for hybrid search |

> *Either `vector` or `id` is required, but not both.

### Sparse Vector Object

| Field | Type | Description |
|-------|------|-------------|
| `indices` | `array<integer>` | Sparse vector indices |
| `values` | `array<number>` | Sparse vector values |

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `matches` | `array` | Matching vectors sorted by similarity |
| `namespace` | `string` | Namespace queried |
| `usage` | `object` | Read units consumed |

### Match Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Vector ID |
| `score` | `number` | Similarity score |
| `values` | `array` | Vector values (if requested) |
| `metadata` | `object` | Metadata (if requested) |
| `sparse_values` | `object` | Sparse values (if applicable) |

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid query parameters |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `404` | `NOT_FOUND` | Namespace not found |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Basic Query

```bash
INDEX_HOST="your-index-abc123.svc.pinecone.io"
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://$INDEX_HOST/query" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "vector": [0.1, 0.2, 0.3, ...],
        "top_k": 10,
        "namespace": "my-namespace",
        "include_metadata": true
      }'
```

### Response

```json
{
  "matches": [
    {
      "id": "vec1",
      "score": 0.95,
      "metadata": {
        "title": "Document 1",
        "category": "tech"
      }
    },
    {
      "id": "vec2",
      "score": 0.87,
      "metadata": {
        "title": "Document 2",
        "category": "science"
      }
    }
  ],
  "namespace": "my-namespace",
  "usage": {
    "read_units": 5
  }
}
```

### Query with Filter

```bash
curl -X POST "https://$INDEX_HOST/query" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "vector": [0.1, 0.2, 0.3, ...],
        "top_k": 10,
        "filter": {
          "category": {"$eq": "tech"},
          "year": {"$gte": 2020}
        },
        "include_metadata": true
      }'
```

### Query by ID

```bash
curl -X POST "https://$INDEX_HOST/query" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "id": "existing-vec-id",
        "top_k": 10,
        "include_metadata": true
      }'
```

### Hybrid Search (Dense + Sparse)

```bash
curl -X POST "https://$INDEX_HOST/query" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "vector": [0.1, 0.2, 0.3, ...],
        "sparse_vector": {
          "indices": [10, 45, 120],
          "values": [0.5, 0.3, 0.2]
        },
        "top_k": 10
      }'
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index(host="INDEX_HOST")

# Basic query
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    namespace="my-namespace",
    include_metadata=True
)

for match in results.matches:
    print(f"{match.id}: {match.score}")
    print(f"  Metadata: {match.metadata}")

# Query with filter
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={
        "category": {"$eq": "tech"},
        "year": {"$gte": 2020}
    },
    include_metadata=True
)

# Query by ID (find similar to existing vector)
results = index.query(
    id="existing-vec-id",
    top_k=10,
    include_metadata=True
)

# Hybrid search with sparse vector
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    sparse_vector={
        "indices": [10, 45, 120],
        "values": [0.5, 0.3, 0.2]
    },
    top_k=10
)
```

---

## Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"field": {"$eq": "value"}}` |
| `$ne` | Not equals | `{"field": {"$ne": "value"}}` |
| `$gt` | Greater than | `{"field": {"$gt": 10}}` |
| `$gte` | Greater than or equal | `{"field": {"$gte": 10}}` |
| `$lt` | Less than | `{"field": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"field": {"$lte": 100}}` |
| `$in` | In array | `{"field": {"$in": ["a", "b"]}}` |
| `$nin` | Not in array | `{"field": {"$nin": ["x", "y"]}}` |
| `$exists` | Field exists | `{"field": {"$exists": true}}` |

### Combining Filters

```json
{
  "$and": [
    {"category": {"$eq": "tech"}},
    {"year": {"$gte": 2020}}
  ]
}
```

```json
{
  "$or": [
    {"category": {"$eq": "tech"}},
    {"category": {"$eq": "science"}}
  ]
}
```

---

## Similarity Scores

| Metric | Score Range | Best Match |
|--------|-------------|------------|
| `cosine` | -1 to 1 | Highest (1) |
| `euclidean` | 0 to ∞ | Lowest (0) |
| `dotproduct` | -∞ to ∞ | Highest |

---

## Limits

| Parameter | Limit |
|-----------|-------|
| `top_k` | 10,000 max |
| Vector dimension | Must match index |
| Filter complexity | Varies by plan |
| Metadata size | 40 KB per vector |

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/data-plane/query)
