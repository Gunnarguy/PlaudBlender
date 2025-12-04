# Create an Index

> Create a Pinecone index. Specify the measure of similarity, the dimension of vectors, cloud provider, and deployment configuration.

## Endpoint

```
POST /indexes
```

**Base URL:** `https://api.pinecone.io`

---

## Authorization

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Api-Key` | `string` | ✅ | Your Pinecone API key |

---

## Headers

| Header | Type | Required |
|--------|------|----------|
| `Content-Type` | `application/json` | ✅ |
| `X-Pinecone-Api-Version` | `2025-10` | ✅ |

---

## Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | ✅ | Index name (1-45 chars, lowercase alphanumeric or `-`) |
| `dimension` | `integer` | ❌ | Vector dimensions (1-20000). Required for dense vectors. |
| `metric` | `string` | ❌ | `"cosine"` (default), `"euclidean"`, or `"dotproduct"` |
| `vector_type` | `string` | ❌ | `"dense"` (default) or `"sparse"` |
| `spec` | `object` | ✅ | Serverless, Pod-based, or BYOC configuration |
| `deletion_protection` | `string` | ❌ | `"disabled"` (default) or `"enabled"` |
| `tags` | `object` | ❌ | Custom key-value tags |

### Spec Options

Choose **one** of:

#### Serverless Spec

```json
{
  "serverless": {
    "cloud": "aws",
    "region": "us-east-1",
    "read_capacity": { "mode": "OnDemand" }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cloud` | `string` | ✅ | `"aws"`, `"gcp"`, or `"azure"` |
| `region` | `string` | ✅ | Cloud region (e.g., `"us-east-1"`) |
| `read_capacity` | `object` | ❌ | `OnDemand` (default) or `Dedicated` |
| `source_collection` | `string` | ❌ | Create from existing collection |
| `schema` | `object` | ❌ | Metadata filtering configuration |

#### Pod-Based Spec

```json
{
  "pod": {
    "environment": "us-east-1-aws",
    "pod_type": "p1.x1",
    "pods": 1,
    "replicas": 1,
    "shards": 1
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `environment` | `string` | ✅ | - | Pod environment |
| `pod_type` | `string` | ✅ | `p1.x1` | Pod size |
| `pods` | `integer` | ❌ | `1` | Total pods |
| `replicas` | `integer` | ❌ | `1` | Replicas for throughput |
| `shards` | `integer` | ❌ | `1` | Shards for storage |
| `metadata_config` | `object` | ❌ | - | Indexed metadata fields |
| `source_collection` | `string` | ❌ | - | Create from collection |

#### BYOC Spec

```json
{
  "byoc": {
    "environment": "aws-us-east-1-b921"
  }
}
```

---

## Response

### `201 Created`

Returns an `IndexModel` object. See [Describe Index](./pinecone-describe-index.md) for full schema.

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid request parameters |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `402` | `PAYMENT_REQUIRED` | Delinquent payment |
| `403` | `FORBIDDEN` | Quota exceeded |
| `404` | `NOT_FOUND` | Invalid cloud/region |
| `409` | `ALREADY_EXISTS` | Index name already exists |
| `422` | `UNPROCESSABLE_ENTITY` | Malformed request body |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Serverless Index (On-Demand)

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://api.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "name": "movie-recommendations",
        "dimension": 1536,
        "metric": "cosine",
        "deletion_protection": "enabled",
        "spec": {
          "serverless": {
            "cloud": "gcp",
            "region": "us-east1",
            "read_capacity": {
              "mode": "OnDemand"
            }
          }
        }
      }'
```

### Serverless Index (Dedicated)

```bash
curl -X POST "https://api.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "name": "dedicated-index",
        "dimension": 1536,
        "metric": "cosine",
        "deletion_protection": "enabled",
        "spec": {
          "serverless": {
            "cloud": "gcp",
            "region": "us-east1",
            "read_capacity": {
              "mode": "Dedicated",
              "dedicated": {
                "node_type": "b1",
                "scaling": "Manual",
                "manual": {
                  "shards": 2,
                  "replicas": 3
                }
              }
            }
          }
        }
      }'
```

### Sparse Index

```bash
curl -X POST "https://api.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "name": "sparse-index",
        "metric": "dotproduct",
        "vector_type": "sparse",
        "deletion_protection": "enabled",
        "spec": {
          "serverless": {
            "cloud": "gcp",
            "region": "us-east1",
            "read_capacity": {
              "mode": "OnDemand"
            }
          }
        }
      }'
```

### Pod-Based Index

```bash
curl -X POST "https://api.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "name": "movie-recommendations",
        "dimension": 1536,
        "metric": "cosine",
        "deletion_protection": "enabled",
        "spec": {
          "pod": {
            "environment": "us-east-1-aws",
            "pod_type": "p1.x1",
            "pods": 1,
            "replicas": 1,
            "shards": 1,
            "metadata_config": {
              "indexed": ["genre", "title", "imdb_rating"]
            }
          }
        }
      }'
```

### Response

```json
{
  "name": "movie-recommendations",
  "dimension": 1536,
  "metric": "cosine",
  "vector_type": "dense",
  "host": "movie-recommendations-abc123.svc.us-east1-gcp.pinecone.io",
  "status": {
    "ready": false,
    "state": "Initializing"
  },
  "spec": {
    "serverless": {
      "cloud": "gcp",
      "region": "us-east1",
      "read_capacity": {
        "mode": "OnDemand",
        "status": {
          "state": "Initializing"
        }
      }
    }
  },
  "deletion_protection": "enabled"
}
```

---

## Python SDK

```python
from pinecone import Pinecone, ServerlessSpec, PodSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# Serverless index
pc.create_index(
    name="movie-recommendations",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
    deletion_protection="enabled"
)

# Pod-based index
pc.create_index(
    name="pod-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east-1-aws",
        pod_type="p1.x1",
        pods=1
    )
)

# Wait for ready
import time
while not pc.describe_index("movie-recommendations").status.ready:
    time.sleep(1)
```

---

## Configuration Details

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `cosine` | Cosine similarity (default) | Text embeddings, normalized vectors |
| `euclidean` | L2 distance | Image features, spatial data |
| `dotproduct` | Dot product | Sparse vectors, pre-normalized data |

> **Note:** Sparse indexes (`vector_type: "sparse"`) require `metric: "dotproduct"`.

### Pod Types

| Type | Description |
|------|-------------|
| `s1.x1` | Storage optimized (base) |
| `s1.x2/x4/x8` | Storage optimized (2x/4x/8x) |
| `p1.x1` | Performance optimized (base) |
| `p1.x2/x4/x8` | Performance optimized (2x/4x/8x) |
| `p2.x1` | High performance (base) |
| `p2.x2/x4/x8` | High performance (2x/4x/8x) |

### Dedicated Node Types

| Type | Description |
|------|-------------|
| `b1` | Base dedicated nodes |
| `t1` | Enhanced processing power and memory |

### Metadata Schema (Serverless)

```json
{
  "schema": {
    "fields": {
      "genre": { "filterable": true },
      "year": { "filterable": true },
      "description": { "filterable": true }
    }
  }
}
```

> **Note:** Only `filterable: true` is currently supported.

---

## Tags

Custom tags for organization (max 20 per index):

- **Key:** 1-80 characters, alphanumeric, `_`, or `-`
- **Value:** 1-120 characters, alphanumeric, `;`, `@`, `_`, `-`, `.`, `+`, or space

```json
{
  "tags": {
    "environment": "production",
    "team": "ml-platform",
    "project": "search-v2"
  }
}
```

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/control-plane/create_index)
