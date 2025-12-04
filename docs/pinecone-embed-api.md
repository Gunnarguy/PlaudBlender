# Pinecone Embed API

> Generate vector embeddings using Pinecone's hosted embedding models. Supports both dense and sparse embeddings.

## Endpoint

```
POST /embed
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
| `model` | `string` | ✅ | The embedding model to use |
| `inputs` | `array` | ✅ | List of inputs to embed |
| `parameters` | `object` | ❌ | Model-specific parameters |

### Input Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | ✅ | The text to embed |

### Parameters Object

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_type` | `string` | - | `"query"` or `"passage"` (required for most models) |
| `truncate` | `string` | `"END"` | `"END"`, `"START"`, or `"NONE"` |
| `dimension` | `integer` | model default | Output dimension (if model supports it) |

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | Model used for embedding |
| `vector_type` | `string` | `"dense"` or `"sparse"` |
| `data` | `array` | Generated embeddings |
| `usage` | `object` | Token usage statistics |

### Dense Embedding Object

```json
{
  "values": [0.0493, -0.0131, -0.0113, ...],
  "vector_type": "dense"
}
```

### Sparse Embedding Object

```json
{
  "sparse_values": [0.1, 0.2, 0.3],
  "sparse_indices": [10, 3, 156],
  "vector_type": "sparse",
  "sparse_tokens": ["quick", "brown", "fox"]
}
```

### Usage Object

| Field | Type | Description |
|-------|------|-------------|
| `total_tokens` | `integer` | Total tokens processed |

---

## Examples

### Dense Embeddings (llama-text-embed-v2)

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X POST "https://api.pinecone.io/embed" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "model": "llama-text-embed-v2",
        "parameters": {
          "input_type": "passage",
          "truncate": "END"
        },
        "inputs": [
          {"text": "Apple is a popular fruit known for its sweetness."},
          {"text": "The tech company Apple is known for innovative products."}
        ]
      }'
```

#### Response

```json
{
  "data": [
    {
      "values": [0.0493, -0.0131, -0.0113, ...],
      "vector_type": "dense"
    },
    {
      "values": [0.0521, -0.0145, -0.0098, ...],
      "vector_type": "dense"
    }
  ],
  "model": "llama-text-embed-v2",
  "usage": {
    "total_tokens": 26
  }
}
```

### Sparse Embeddings (pinecone-sparse-english-v0)

```bash
curl -X POST "https://api.pinecone.io/embed" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "model": "pinecone-sparse-english-v0",
        "parameters": {
          "input_type": "passage",
          "return_tokens": true
        },
        "inputs": [
          {"text": "The quick brown fox jumps over the lazy dog."}
        ]
      }'
```

#### Response

```json
{
  "data": [
    {
      "sparse_values": [0.234, 0.567, 0.123, 0.890, 0.456],
      "sparse_indices": [1024, 2048, 4096, 8192, 16384],
      "vector_type": "sparse",
      "sparse_tokens": ["quick", "brown", "fox", "jumps", "lazy"]
    }
  ],
  "model": "pinecone-sparse-english-v0",
  "usage": {
    "total_tokens": 10
  }
}
```

### Custom Dimensions (Matryoshka Embeddings)

```bash
curl -X POST "https://api.pinecone.io/embed" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -d '{
        "model": "llama-text-embed-v2",
        "parameters": {
          "input_type": "query",
          "dimension": 512
        },
        "inputs": [
          {"text": "What is machine learning?"}
        ]
      }'
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Dense embeddings
dense_result = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[
        {"text": "First document to embed"},
        {"text": "Second document to embed"}
    ],
    parameters={
        "input_type": "passage",
        "truncate": "END"
    }
)

# Access embeddings
for item in dense_result.data:
    print(f"Vector: {item.values[:5]}...")  # First 5 values

# Sparse embeddings
sparse_result = pc.inference.embed(
    model="pinecone-sparse-english-v0",
    inputs=[
        {"text": "Query for sparse retrieval"}
    ],
    parameters={
        "input_type": "query",
        "return_tokens": True
    }
)

# Access sparse embeddings
for item in sparse_result.data:
    print(f"Indices: {item.sparse_indices}")
    print(f"Values: {item.sparse_values}")
    print(f"Tokens: {item.sparse_tokens}")
```

---

## Available Models

### Dense Embedding Models

| Model | Dimensions | Max Tokens | Provider |
|-------|------------|------------|----------|
| `llama-text-embed-v2` | 384, 512, 768, 1024, 2048 | 2048 | NVIDIA |
| `multilingual-e5-large` | 1024 | 507 | Microsoft |

### Sparse Embedding Models

| Model | Max Tokens | Provider |
|-------|------------|----------|
| `pinecone-sparse-english-v0` | 512 | Pinecone |

---

## Input Type Guidelines

| Input Type | Use Case |
|------------|----------|
| `"passage"` | Documents being indexed/stored |
| `"query"` | Search queries at retrieval time |

> **Best Practice:** Use `"passage"` when upserting, `"query"` when searching.

---

## Truncation Options

| Value | Behavior |
|-------|----------|
| `"END"` | Truncate from end (keeps beginning) |
| `"START"` | Truncate from start (keeps end) |
| `"NONE"` | Error if input exceeds max tokens |

---

## Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `INVALID_ARGUMENT` | Invalid model or parameters |
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `429` | `RESOURCE_EXHAUSTED` | Rate limit exceeded |
| `500` | `UNKNOWN` | Internal server error |

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/inference/embed)
