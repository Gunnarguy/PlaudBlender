# Create an Index with Integrated Embedding

> Create an index with integrated embedding where Pinecone automatically embeds text using a hosted model during upsert and search operations.

## Endpoint

```
POST /indexes/create-for-model
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
| `cloud` | `string` | ✅ | `"aws"`, `"gcp"`, or `"azure"` |
| `region` | `string` | ✅ | Deployment region (e.g., `"us-east-1"`) |
| `embed` | `object` | ✅ | Embedding configuration |
| `deletion_protection` | `string` | ❌ | `"disabled"` (default) or `"enabled"` |
| `tags` | `object` | ❌ | Custom key-value tags |
| `schema` | `object` | ❌ | Metadata indexing configuration |
| `read_capacity` | `object` | ❌ | On-demand (default) or dedicated |

### Embed Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | ✅ | Embedding model name |
| `field_map` | `object` | ✅ | Maps document fields to embedding input |
| `metric` | `string` | ❌ | `"cosine"`, `"euclidean"`, or `"dotproduct"` |
| `dimension` | `integer` | ❌ | Override default model dimension |
| `read_parameters` | `object` | ❌ | Query parameters |
| `write_parameters` | `object` | ❌ | Upsert parameters |

### Available Embedding Models

| Model | Dimension | Vector Type | Description |
|-------|-----------|-------------|-------------|
| `llama-text-embed-v2` | 1024 | Dense | LLaMA-based text embedding |
| `multilingual-e5-large` | 1024 | Dense | Multilingual E5 large |
| `pinecone-sparse-english-v0` | N/A | Sparse | English sparse vectors |

### Embed Parameters

| Field | Type | Description |
|-------|------|-------------|
| `input_type` | `string` | `"passage"` for documents, `"query"` for search |
| `truncate` | `string` | `"END"` (truncate from end) or `"NONE"` (error if too long) |
| `dimension` | `integer` | Override default dimension (some models only) |

---

## Response

### `201 Created`

Returns an `IndexModel` object with the `embed` field populated. See [Describe Index](./pinecone-describe-index.md) for full schema.

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid request parameters |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `404` | `NOT_FOUND` | Invalid cloud/region |
| `409` | `ALREADY_EXISTS` | Index name already exists |
| `422` | `UNPROCESSABLE_ENTITY` | Malformed request body |
| `500` | `UNKNOWN` | Internal server error |

---

## Example

### Request

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://api.pinecone.io/indexes/create-for-model" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "name": "multilingual-e5-large-index",
        "cloud": "gcp",
        "region": "us-east1",
        "deletion_protection": "enabled",
        "embed": {
          "model": "multilingual-e5-large",
          "metric": "cosine",
          "field_map": {
            "text": "your-text-field"
          },
          "write_parameters": {
            "input_type": "passage",
            "truncate": "END"
          },
          "read_parameters": {
            "input_type": "query",
            "truncate": "END"
          }
        }
      }'
```

### Response

```json
{
  "name": "multilingual-e5-large-index",
  "dimension": 1024,
  "metric": "cosine",
  "vector_type": "dense",
  "host": "multilingual-e5-large-index-abc123.svc.us-east1-gcp.pinecone.io",
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
  "deletion_protection": "enabled",
  "embed": {
    "model": "multilingual-e5-large",
    "field_map": {
      "text": "your-text-field"
    },
    "dimension": 1024,
    "metric": "cosine",
    "write_parameters": {
      "input_type": "passage",
      "truncate": "END"
    },
    "read_parameters": {
      "input_type": "query",
      "truncate": "END"
    },
    "vector_type": "dense"
  }
}
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Create index with integrated embedding
pc.create_index_for_model(
    name="multilingual-e5-large-index",
    cloud="gcp",
    region="us-east1",
    embed={
        "model": "multilingual-e5-large",
        "metric": "cosine",
        "field_map": {"text": "your-text-field"},
        "write_parameters": {"input_type": "passage", "truncate": "END"},
        "read_parameters": {"input_type": "query", "truncate": "END"}
    },
    deletion_protection="enabled"
)

# Wait for ready
import time
while not pc.describe_index("multilingual-e5-large-index").status.ready:
    time.sleep(1)
```

---

## Using Integrated Embedding

### Upsert Text (No Vectors Required)

With integrated embedding, you upsert raw text and Pinecone generates vectors automatically:

```python
index = pc.Index("multilingual-e5-large-index")

# Upsert text records (no vectors!)
index.upsert_records(
    namespace="articles",
    records=[
        {
            "_id": "article-1",
            "your-text-field": "Machine learning is transforming industries...",
            "category": "tech"
        },
        {
            "_id": "article-2", 
            "your-text-field": "Climate change impacts global agriculture...",
            "category": "environment"
        }
    ]
)
```

### Search with Text

```python
# Search with text (no query vector needed!)
results = index.search(
    namespace="articles",
    data={"your-text-field": "AI applications in healthcare"},
    top_k=10
)
```

---

## Field Map Configuration

The `field_map` tells Pinecone which field in your documents contains the text to embed:

```json
{
  "field_map": {
    "text": "your-text-field"
  }
}
```

- **Key (`text`):** Always `"text"` - the model's expected input
- **Value (`your-text-field`):** Your document field name containing text

### Example Document Structure

```json
{
  "_id": "doc-123",
  "your-text-field": "This is the text that will be embedded",
  "title": "Document Title",
  "category": "example"
}
```

---

## Input Types

| Type | Use For | Description |
|------|---------|-------------|
| `passage` | Documents/upserts | Optimized for longer content being indexed |
| `query` | Search queries | Optimized for search query text |

> **Best Practice:** Use `"passage"` for `write_parameters` and `"query"` for `read_parameters`.

---

## Important Notes

- ⚠️ **Model cannot be changed** after index creation
- The index `dimension` and `metric` are set by the model (or can be overridden where supported)
- The `embed` configuration can be updated later to change `field_map`, `read_parameters`, or `write_parameters`
- Use the [Configure Index](./pinecone-configure-index.md) endpoint to modify embedding settings

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/control-plane/create_for_model)
