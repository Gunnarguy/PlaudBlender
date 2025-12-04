# Upsert Vectors

> Insert or update vectors in an index namespace.

## Endpoint

```
POST /vectors/upsert
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
| `vectors` | `array` | ✅ | Array of vectors to upsert |
| `namespace` | `string` | ❌ | Target namespace (default: `""`) |

### Vector Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | ✅ | Unique vector ID (max 512 chars) |
| `values` | `array<number>` | ✅ | Dense vector values |
| `metadata` | `object` | ❌ | Key-value metadata (max 40 KB) |
| `sparse_values` | `object` | ❌ | Sparse vector components |

### Sparse Values Object

| Field | Type | Description |
|-------|------|-------------|
| `indices` | `array<integer>` | Sparse vector indices |
| `values` | `array<number>` | Sparse vector values |

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `upserted_count` | `integer` | Number of vectors upserted |

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid vector format or dimension mismatch |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `413` | `PAYLOAD_TOO_LARGE` | Request exceeds size limit |
| `429` | `RESOURCE_EXHAUSTED` | Rate limit exceeded |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Basic Upsert

```bash
INDEX_HOST="your-index-abc123.svc.pinecone.io"
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://$INDEX_HOST/vectors/upsert" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "vectors": [
          {
            "id": "vec1",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
          },
          {
            "id": "vec2",
            "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
          }
        ],
        "namespace": "my-namespace"
      }'
```

### Response

```json
{
  "upserted_count": 2
}
```

### Upsert with Metadata

```bash
curl -X POST "https://$INDEX_HOST/vectors/upsert" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "vectors": [
          {
            "id": "doc-1",
            "values": [0.1, 0.2, 0.3, ...],
            "metadata": {
              "title": "Introduction to ML",
              "category": "tech",
              "year": 2024,
              "tags": ["machine-learning", "ai"]
            }
          }
        ],
        "namespace": "documents"
      }'
```

### Upsert with Sparse Values (Hybrid)

```bash
curl -X POST "https://$INDEX_HOST/vectors/upsert" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "vectors": [
          {
            "id": "hybrid-1",
            "values": [0.1, 0.2, 0.3, ...],
            "sparse_values": {
              "indices": [15, 42, 128, 256],
              "values": [0.5, 0.3, 0.8, 0.2]
            },
            "metadata": {"type": "hybrid"}
          }
        ]
      }'
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index(host="INDEX_HOST")

# Basic upsert
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, 0.3, ...]},
        {"id": "vec2", "values": [0.2, 0.3, 0.4, ...]}
    ],
    namespace="my-namespace"
)

# Upsert with metadata
index.upsert(
    vectors=[
        {
            "id": "doc-1",
            "values": [0.1, 0.2, 0.3, ...],
            "metadata": {
                "title": "Introduction to ML",
                "category": "tech",
                "year": 2024
            }
        }
    ],
    namespace="documents"
)

# Batch upsert (recommended for large datasets)
def chunks(iterable, batch_size=100):
    """Yield successive chunks from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

vectors = [{"id": f"vec-{i}", "values": [...]} for i in range(10000)]

for batch in chunks(vectors, batch_size=100):
    index.upsert(vectors=batch, namespace="large-dataset")

# Using tuples (alternative format)
index.upsert(
    vectors=[
        ("vec1", [0.1, 0.2, ...]),                     # id, values
        ("vec2", [0.2, 0.3, ...], {"key": "value"})    # id, values, metadata
    ]
)
```

---

## gRPC (High Throughput)

```python
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key="YOUR_API_KEY")
index = pc.Index(host="INDEX_HOST")

# gRPC upsert - same API, better performance
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, ...]},
        {"id": "vec2", "values": [0.2, 0.3, ...]}
    ],
    namespace="my-namespace"
)
```

---

## Metadata Guidelines

### Supported Types

| Type | Example |
|------|---------|
| String | `"hello"` |
| Number | `42`, `3.14` |
| Boolean | `true`, `false` |
| Array of strings | `["a", "b", "c"]` |
| Null | `null` |

### Limits

| Constraint | Limit |
|------------|-------|
| Metadata size | 40 KB per vector |
| Key length | 512 bytes |
| Nested depth | Not supported |

### Indexing

By default, all metadata fields are indexed. For large metadata, consider:

```python
# Create index with selective metadata indexing
pc.create_index(
    name="my-index",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    metadata_config={
        "indexed": ["category", "year"]  # Only index these fields
    }
)
```

---

## Limits

| Parameter | Limit |
|-----------|-------|
| Vectors per request | 1000 |
| Request size | 2 MB |
| Vector ID length | 512 characters |
| Metadata per vector | 40 KB |
| Sparse indices | 1000 max |

---

## Best Practices

1. **Batch operations** - Upsert in batches of 100-1000 vectors
2. **Use gRPC** - 10-20% faster for high-volume operations
3. **Parallel upserts** - Use multiple threads for large datasets
4. **Validate dimensions** - Ensure vectors match index dimension
5. **Normalize vectors** - For cosine similarity, normalize to unit length

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/data-plane/upsert)
