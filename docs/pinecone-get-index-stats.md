# Get Index Statistics

> Returns statistics about an index's contents, including vector counts, namespace breakdown, and capacity utilization.

## Endpoint

```
POST /describe_index_stats
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
| `filter` | `object` | ❌ | Metadata filter to count subset of vectors |

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `dimension` | `integer` | Vector dimensions in the index |
| `index_fullness` | `number` | Utilization (0.0 to 1.0, pod-based only) |
| `total_vector_count` | `integer` | Total vectors across all namespaces |
| `namespaces` | `object` | Breakdown by namespace |
| `metric` | `string` | Distance metric used (`cosine`, `euclidean`, `dotproduct`) |
| `vector_type` | `string` | Vector storage type (`dense` or `sparse`) |

### Namespace Object

| Field | Type | Description |
|-------|------|-------------|
| `vector_count` | `integer` | Number of vectors in namespace |

### Dedicated Index Additional Fields

For serverless dedicated indexes:

| Field | Type | Description |
|-------|------|-------------|
| `memory_fullness` | `number` | Memory utilization (0.0 to 1.0) |
| `storage_fullness` | `number` | Storage utilization (0.0 to 1.0) |

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid filter syntax |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Basic Request

```bash
INDEX_HOST="your-index-abc123.svc.aped-4627-b74a.pinecone.io"
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://$INDEX_HOST/describe_index_stats" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{}'
```

### Response (Serverless On-Demand)

```json
{
  "dimension": 768,
  "index_fullness": 0.0,
  "total_vector_count": 23,
  "metric": "cosine",
  "vector_type": "dense",
  "namespaces": {
    "": {
      "vector_count": 15
    },
    "summaries": {
      "vector_count": 5
    },
    "full_text": {
      "vector_count": 3
    }
  }
}
```

### Response (Serverless Dedicated)

```json
{
  "dimension": 1536,
  "index_fullness": 0.0,
  "total_vector_count": 125000,
  "metric": "cosine",
  "vector_type": "dense",
  "memory_fullness": 0.35,
  "storage_fullness": 0.22,
  "namespaces": {
    "documents": {
      "vector_count": 100000
    },
    "summaries": {
      "vector_count": 25000
    }
  }
}
```

### Response (Pod-Based)

```json
{
  "dimension": 1536,
  "index_fullness": 0.45,
  "total_vector_count": 500000,
  "metric": "dotproduct",
  "vector_type": "dense",
  "namespaces": {
    "production": {
      "vector_count": 450000
    },
    "staging": {
      "vector_count": 50000
    }
  }
}
```

### With Metadata Filter

Count only vectors matching specific metadata:

```bash
curl -X POST "https://$INDEX_HOST/describe_index_stats" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "filter": {
          "genre": {"$eq": "action"},
          "year": {"$gte": 2020}
        }
      }'
```

### Filtered Response

```json
{
  "dimension": 1536,
  "index_fullness": 0.15,
  "total_vector_count": 1250,
  "namespaces": {
    "movies": {
      "vector_count": 1250
    }
  }
}
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index("docs-example-index")

# Get full stats
stats = index.describe_index_stats()
print(f"Dimension: {stats.dimension}")
print(f"Total vectors: {stats.total_vector_count}")
print(f"Namespaces: {stats.namespaces}")

# With filter
filtered_stats = index.describe_index_stats(
    filter={
        "genre": {"$eq": "action"}
    }
)
```

### Output

```python
Dimension: 768
Total vectors: 23
Namespaces: {
    '': NamespaceSummary(vector_count=15),
    'summaries': NamespaceSummary(vector_count=5),
    'full_text': NamespaceSummary(vector_count=3)
}
```

---

## Understanding Index Fullness

| Value | Meaning |
|-------|---------|
| `0.0` | Empty or serverless (no limit) |
| `0.5` | 50% capacity used |
| `0.9` | 90% capacity - consider scaling |
| `1.0` | At capacity - scaling required |

> **Note:** `index_fullness` is only meaningful for **pod-based** indexes. Serverless indexes always return `0.0`.

---

## Use Cases

- **Monitoring** - Track vector counts over time
- **Capacity Planning** - Monitor index_fullness for pod-based indexes
- **Namespace Management** - Understand data distribution across namespaces
- **Debugging** - Verify upserts succeeded by checking counts
- **Cost Optimization** - Identify empty or underutilized namespaces

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

### Combining Filters

```bash
curl -X POST "https://$INDEX_HOST/describe_index_stats" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "filter": {
          "$and": [
            {"genre": {"$eq": "action"}},
            {"year": {"$gte": 2020}},
            {"rating": {"$gt": 7.5}}
          ]
        }
      }'
```

---

## gRPC Support

For high-throughput scenarios, use the gRPC interface:

```python
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key="YOUR_API_KEY")
index = pc.Index("docs-example-index")

# gRPC stats call
stats = index.describe_index_stats()
```

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/data-plane/describeindexstats)
