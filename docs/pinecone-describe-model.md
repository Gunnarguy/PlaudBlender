# Describe a Model

> Get detailed information about a specific embedding or reranking model hosted by Pinecone.

## Endpoint

```
GET /models/{model_name}
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

## Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | `string` | ✅ | The name of the model (e.g., `llama-text-embed-v2`) |

---

## Response

### `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | Model identifier |
| `short_description` | `string` | Brief description of the model |
| `type` | `string` | `"embed"` or `"rerank"` |
| `vector_type` | `string` | `"dense"` or `"sparse"` (embed models only) |
| `default_dimension` | `integer` | Default output dimension (dense only) |
| `modality` | `string` | Input type (`"text"`) |
| `max_sequence_length` | `integer` | Maximum input tokens |
| `max_batch_size` | `integer` | Maximum inputs per request |
| `provider_name` | `string` | Model provider |
| `supported_metrics` | `array` | Compatible distance metrics |
| `supported_dimensions` | `array` | Available output dimensions |
| `supported_parameters` | `array` | Model-specific parameters |

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `404` | `NOT_FOUND` | Model not found |
| `500` | `UNKNOWN` | Internal server error |

---

## Examples

### Request

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl "https://api.pinecone.io/models/llama-text-embed-v2" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

### Response: Dense Embedding Model

```json
{
  "model": "llama-text-embed-v2",
  "short_description": "High performance dense embedding model optimized for multilingual text with support for long documents (up to 2048 tokens) and dynamic embedding size (Matryoshka Embeddings).",
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
}
```

### Response: Sparse Embedding Model

```json
{
  "model": "pinecone-sparse-english-v0",
  "short_description": "A sparse embedding model for converting text to sparse vectors for keyword or hybrid semantic/keyword search. Built on the innovations of the DeepImpact architecture.",
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
      "parameter": "truncate",
      "required": false,
      "default": "END",
      "type": "one_of",
      "value_type": "string",
      "allowed_values": ["END", "NONE"]
    },
    {
      "parameter": "return_tokens",
      "required": false,
      "default": false,
      "type": "any",
      "value_type": "boolean"
    }
  ]
}
```

### Response: Reranking Model

```json
{
  "model": "pinecone-rerank-v0",
  "short_description": "A state of the art reranking model that out-performs competitors on widely accepted benchmarks. It can handle chunks up to 512 tokens (1-2 paragraphs).",
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
}
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Get model details
model = pc.inference.describe_model("llama-text-embed-v2")

print(f"Model: {model.model}")
print(f"Type: {model.type}")
print(f"Vector type: {model.vector_type}")
print(f"Default dimension: {model.default_dimension}")
print(f"Max tokens: {model.max_sequence_length}")
print(f"Max batch: {model.max_batch_size}")
print(f"Supported dimensions: {model.supported_dimensions}")
print(f"Supported metrics: {model.supported_metrics}")

# Check supported parameters
for param in model.supported_parameters:
    print(f"  {param.parameter}: {param.value_type}")
    if hasattr(param, 'allowed_values'):
        print(f"    Allowed: {param.allowed_values}")
```

---

## Supported Parameter Types

| Type | Description |
|------|-------------|
| `one_of` | Value must be one of `allowed_values` |
| `numeric_range` | Value must be between `min` and `max` |
| `any` | Any valid value of `value_type` |

---

## Use Cases

- **Model selection** - Compare models before choosing one
- **Validation** - Verify parameter values before embedding
- **Capacity planning** - Check batch sizes and token limits
- **Integration** - Ensure index metric matches model requirements

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/inference/describe_model)
