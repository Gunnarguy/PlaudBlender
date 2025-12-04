# Pinecone API Documentation

> Comprehensive reference documentation for Pinecone vector database APIs.

---

## Control Plane APIs

Manage your Pinecone indexes and infrastructure.

| Document | Description |
|----------|-------------|
| [List Indexes](pinecone-list-indexes.md) | List all indexes in your project |
| [Create Index](pinecone-create-index.md) | Create a new serverless or pod-based index |
| [Create Index (Integrated)](pinecone-create-index-integrated.md) | Create index with built-in embedding |
| [Describe Index](pinecone-describe-index.md) | Get details about a specific index |
| [Configure Index](pinecone-configure-index.md) | Update index settings and scaling |
| [Delete Index](pinecone-delete-index.md) | Delete an index permanently |

---

## Data Plane APIs

Read and write vector data.

| Document | Description |
|----------|-------------|
| [Upsert Vectors](pinecone-upsert.md) | Insert or update vectors |
| [Query Vectors](pinecone-query.md) | Search for similar vectors |
| [Get Index Stats](pinecone-get-index-stats.md) | Get vector counts and namespace breakdown |

---

## Inference APIs

Use Pinecone's hosted embedding and reranking models.

| Document | Description |
|----------|-------------|
| [Embed API](pinecone-embed-api.md) | Generate dense and sparse embeddings |
| [Rerank Results](pinecone-rerank.md) | Rerank search results by relevance |
| [List Models](pinecone-list-models.md) | List available hosted models |
| [Describe Model](pinecone-describe-model.md) | Get model details and parameters |

---

## SDKs

| Document | Description |
|----------|-------------|
| [Python SDK](pinecone-python-sdk.md) | Official Python client library |

---

## Quick Reference

### Base URLs

| API Type | URL |
|----------|-----|
| Control Plane | `https://api.pinecone.io` |
| Data Plane | `https://{index_host}` |

### Required Headers

```
Api-Key: YOUR_API_KEY
X-Pinecone-Api-Version: 2025-10
Content-Type: application/json
```

### Common Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created |
| `202` | Accepted (async operation) |
| `400` | Invalid request |
| `401` | Unauthorized |
| `403` | Forbidden (quota/protection) |
| `404` | Not found |
| `409` | Conflict (already exists) |
| `422` | Unprocessable entity |
| `429` | Rate limited |
| `500` | Server error |

---

## Available Embedding Models

| Model | Type | Dimensions | Max Tokens |
|-------|------|------------|------------|
| `llama-text-embed-v2` | Dense | 384-2048 | 2048 |
| `multilingual-e5-large` | Dense | 1024 | 507 |
| `pinecone-sparse-english-v0` | Sparse | N/A | 512 |

## Available Reranking Models

| Model | Max Tokens | Provider |
|-------|------------|----------|
| `pinecone-rerank-v0` | 512 | Pinecone |
| `bge-reranker-v2-m3` | 1024 | BAAI |
| `cohere-rerank-3.5` | 40000 | Cohere |

---

## Index Types

### Serverless On-Demand

- Pay per operation
- Auto-scales automatically
- Best for variable workloads

### Serverless Dedicated

- Fixed capacity with predictable costs
- Manual scaling control
- Best for steady, high-volume workloads

### Pod-Based

- Legacy deployment model
- Fixed compute resources
- Best for predictable, latency-sensitive workloads

---

## Resources

- [Official Pinecone Docs](https://docs.pinecone.io)
- [API Reference](https://docs.pinecone.io/reference/api/introduction)
- [Python SDK Reference](https://sdk.pinecone.io/python/)
- [LLMs.txt Index](https://docs.pinecone.io/llms.txt)

---

*Last updated: December 2024 - API version 2025-10*
