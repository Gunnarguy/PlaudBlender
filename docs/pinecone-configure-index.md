# Configure an Index

> Configure an existing index. For serverless indexes, configure deletion protection, tags, and integrated inference embedding settings. For pod-based indexes, configure pod size, replicas, tags, and deletion protection.

## Endpoint

```
PATCH /indexes/{index_name}
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

## Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `index_name` | `string` | ✅ | Name of the index to configure |

---

## Request Body

All fields are **optional**. Only include fields you want to change.

| Field | Type | Description |
|-------|------|-------------|
| `spec` | `object` | Scaling configuration (varies by index type) |
| `deletion_protection` | `string` | `"enabled"` or `"disabled"` |
| `tags` | `object` | Custom metadata tags |
| `embed` | `object` | Integrated embedding configuration (serverless only) |

### Serverless Spec

```json
{
  "spec": {
    "serverless": {
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
}
```

### Pod-Based Spec

| Field | Type | Description |
|-------|------|-------------|
| `pod.replicas` | `integer` | Number of replicas (min: 1) |
| `pod.pod_type` | `string` | Pod size (e.g., `"p1.x2"`) |

> ⚠️ Pod type **cannot be changed** after creation. Create a collection and new index instead.

---

## Response

### `202 Accepted`

Returns the updated `IndexModel` object. See [Describe Index](./pinecone-describe-index.md) for full schema.

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid configuration |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `402` | `PAYMENT_REQUIRED` | Delinquent payment |
| `403` | `FORBIDDEN` | Quota exceeded |
| `404` | `NOT_FOUND` | Index not found |
| `422` | `UNPROCESSABLE_ENTITY` | Invalid scaling for index type |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Enable Deletion Protection

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X PATCH "https://api.pinecone.io/indexes/docs-example-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "deletion_protection": "enabled"
      }'
```

### Update Tags

```bash
curl -X PATCH "https://api.pinecone.io/indexes/docs-example-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "tags": {
          "tag0": "new-val",
          "tag1": ""
        }
      }'
```

> **Note:** Set a tag value to empty string `""` to delete it.

### Scale Pod-Based Index (Vertical)

```bash
curl -X PATCH "https://api.pinecone.io/indexes/movie-embeddings" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "spec": {
          "pod": {
            "pod_type": "p1.x2"
          }
        }
      }'
```

### Scale Pod-Based Index (Horizontal)

```bash
curl -X PATCH "https://api.pinecone.io/indexes/movie-embeddings" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "spec": {
          "pod": {
            "replicas": 4
          }
        }
      }'
```

### Scale Serverless Dedicated Index

```bash
curl -X PATCH "https://api.pinecone.io/indexes/dedicated-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "spec": {
          "serverless": {
            "read_capacity": {
              "mode": "Dedicated",
              "dedicated": {
                "node_type": "t1",
                "scaling": "Manual",
                "manual": {
                  "shards": 2,
                  "replicas": 4
                }
              }
            }
          }
        }
      }'
```

### Update Integrated Embedding Configuration

```bash
curl -X PATCH "https://api.pinecone.io/indexes/multilingual-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "embed": {
          "field_map": {
            "text": "new-text-field"
          },
          "read_parameters": {
            "input_type": "query",
            "truncate": "NONE"
          }
        }
      }'
```

> **Note:** The `model` cannot be changed after index creation.

### Convert to Integrated Embedding Index

You can add integrated embedding to an existing index:

```bash
curl -X PATCH "https://api.pinecone.io/indexes/existing-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "embed": {
          "model": "multilingual-e5-large",
          "field_map": {
            "text": "content"
          }
        }
      }'
```

> ⚠️ Index dimension and metric must match the model's requirements.

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Enable deletion protection
pc.configure_index(
    "docs-example-index",
    deletion_protection="enabled"
)

# Add/update tags
pc.configure_index(
    "docs-example-index",
    tags={
        "environment": "production",
        "team": "ml-platform"
    }
)

# Scale pod-based index
pc.configure_index(
    "movie-embeddings",
    replicas=4,
    pod_type="p1.x2"
)

# Update embed configuration (serverless with integrated embedding)
pc.configure_index(
    "multilingual-index",
    embed={
        "field_map": {"text": "new-field"},
        "read_parameters": {"input_type": "query"}
    }
)
```

---

## Configuration Details

### Tags

- **Max:** 20 tags per index
- **Key:** 1-80 characters, alphanumeric, `_`, or `-`
- **Value:** 1-120 characters, alphanumeric, `;`, `@`, `_`, `-`, `.`, `+`, or space
- **Delete a tag:** Set value to empty string `""`

### Deletion Protection

| Value | Description |
|-------|-------------|
| `"disabled"` | Index can be deleted (default) |
| `"enabled"` | Index cannot be deleted until disabled |

### Pod Types

| Type | Description |
|------|-------------|
| `s1.x1/x2/x4/x8` | Storage optimized |
| `p1.x1/x2/x4/x8` | Performance optimized |
| `p2.x1/x2/x4/x8` | High performance |

### Dedicated Node Types (Serverless)

| Type | Description |
|------|-------------|
| `b1` | Base dedicated nodes |
| `t1` | Enhanced processing power and memory |

### Scaling Replicas

- **Replicas:** Duplicate compute resources for throughput
- **Min:** 0 (disables index but reduces costs)
- **Effect:** Higher throughput, better availability

### Scaling Shards

- **Shards:** Storage capacity units (250 GB each)
- **Min:** 1
- **Effect:** More storage capacity

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/control-plane/configure_index)
