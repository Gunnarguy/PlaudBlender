# List Models

> List all embedding and reranking models hosted by Pinecone.

## Endpoint

```
GET /models
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
| `X-Pinecone-Api-Version` | `string` | `2025-10` | ✅ |

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `models` | `array` | List of available models |

### Model Object

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | Model identifier |
| `short_description` | `string` | Brief description |
| `type` | `string` | `"embed"` or `"rerank"` |
| `vector_type` | `string` | `"dense"` or `"sparse"` (embed only) |
| `default_dimension` | `integer` | Default output dimension (dense only) |
| `modality` | `string` | Input type (`"text"`) |
| `max_sequence_length` | `integer` | Maximum input tokens |
| `max_batch_size` | `integer` | Maximum inputs per request |
| `provider_name` | `string` | Model provider |
| `supported_metrics` | `array` | Compatible distance metrics |
| `supported_dimensions` | `array` | Available output dimensions |
| `supported_parameters` | `array` | Model-specific parameters |

---

## Example

### Request

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl "https://api.pinecone.io/models" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

### Response

```json
{
  "models": [
    {
      "model": "llama-text-embed-v2",
      "short_description": "High performance dense embedding model optimized for multilingual text with support for long documents and dynamic embedding size.",
      "type": "embed",
      "vector_type": "dense",
      "default_dimension": 1024,
      "modality": "text",
      "max_sequence_length": 2048,
      "max_batch_size": 96,
      "provider_name": "NVIDIA",
      "supported_metrics": ["Cosine", "DotProduct"],
      "supported_dimensions": [384, 512, 768, 1024, 2048],
      "supported_parameters": [
        {
          "parameter": "input_type",
          "required": true,
          "type": "one_of",
          "value_type": "string",
          "allowed_values": ["query", "passage"]
        },
        {
          "parameter": "truncate",
          "required": false,
          "default": "END",
          "type": "one_of",
          "value_type": "string",
          "allowed_values": ["END", "NONE", "START"]
        },
        {
          "parameter": "dimension",
          "required": false,
          "default": 1024,
          "type": "one_of",
          "value_type": "integer",
          "allowed_values": [384, 512, 768, 1024, 2048]
        }
      ]
    },
    {
      "model": "multilingual-e5-large",
      "short_description": "High-performance dense embedding model for multilingual text retrieval.",
      "type": "embed",
      "vector_type": "dense",
      "default_dimension": 1024,
      "modality": "text",
      "max_sequence_length": 507,
      "max_batch_size": 96,
      "provider_name": "Microsoft",
      "supported_metrics": ["Cosine", "Euclidean"],
      "supported_dimensions": [1024],
      "supported_parameters": [
        {
          "parameter": "input_type",
          "required": true,
          "type": "one_of",
          "value_type": "string",
          "allowed_values": ["query", "passage"]
        },
        {
          "parameter": "truncate",
          "required": false,
          "default": "END",
          "type": "one_of",
          "value_type": "string",
          "allowed_values": ["END", "NONE"]
        }
      ]
    },
    {
      "model": "pinecone-sparse-english-v0",
      "short_description": "Sparse embedding model for keyword/hybrid search.",
      "type": "embed",
      "vector_type": "sparse",
      "modality": "text",
      "max_sequence_length": 512,
      "max_batch_size": 96,
      "provider_name": "Pinecone",
      "supported_metrics": ["DotProduct"],
      "supported_parameters": [
        {
          "parameter": "input_type",
          "required": true,
          "type": "one_of",
          "value_type": "string",
          "allowed_values": ["query", "passage"]
        },
        {
          "parameter": "return_tokens",
          "required": false,
          "default": false,
          "type": "any",
          "value_type": "boolean"
        }
      ]
    },
    {
      "model": "pinecone-rerank-v0",
      "short_description": "State of the art reranking model.",
      "type": "rerank",
      "modality": "text",
      "max_sequence_length": 512,
      "max_batch_size": 100,
      "provider_name": "Pinecone",
      "supported_parameters": [
        {
          "parameter": "truncate",
          "required": false,
          "default": "END",
          "type": "one_of",
          "value_type": "string",
          "allowed_values": ["END", "NONE"]
        }
      ]
    },
    {
      "model": "bge-reranker-v2-m3",
      "short_description": "High-performance multilingual reranking model.",
      "type": "rerank",
      "modality": "text",
      "max_sequence_length": 1024,
      "max_batch_size": 100,
      "provider_name": "BAAI"
    },
    {
      "model": "cohere-rerank-3.5",
      "short_description": "Cohere's leading reranking model for enterprise search.",
      "type": "rerank",
      "modality": "text",
      "max_sequence_length": 40000,
      "max_batch_size": 200,
      "provider_name": "Cohere"
    }
  ]
}
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# List all models
models = pc.inference.list_models()

# Filter embedding models
embed_models = [m for m in models.models if m.type == "embed"]
for model in embed_models:
    print(f"{model.model}: {model.short_description}")
    print(f"  Dimensions: {model.supported_dimensions}")
    print(f"  Max tokens: {model.max_sequence_length}")

# Filter reranking models
rerank_models = [m for m in models.models if m.type == "rerank"]
for model in rerank_models:
    print(f"{model.model}: {model.short_description}")
    print(f"  Provider: {model.provider_name}")
```

---

## Model Categories

### Embedding Models

| Model | Type | Dimensions | Max Tokens | Provider |
|-------|------|------------|------------|----------|
| `llama-text-embed-v2` | Dense | 384-2048 | 2048 | NVIDIA |
| `multilingual-e5-large` | Dense | 1024 | 507 | Microsoft |
| `pinecone-sparse-english-v0` | Sparse | N/A | 512 | Pinecone |

### Reranking Models

| Model | Max Tokens | Max Batch | Provider |
|-------|------------|-----------|----------|
| `pinecone-rerank-v0` | 512 | 100 | Pinecone |
| `bge-reranker-v2-m3` | 1024 | 100 | BAAI |
| `cohere-rerank-3.5` | 40000 | 200 | Cohere |

---

## Use Cases

- **Model Selection** - Choose the right model for your use case
- **Capacity Planning** - Understand batch sizes and token limits
- **Metric Compatibility** - Ensure index metric matches model requirements
- **Dimension Planning** - Select appropriate dimensions for your index

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/inference/list_models)
