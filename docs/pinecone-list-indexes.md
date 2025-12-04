# List Indexes

> List all indexes in a project.

## Endpoint

```
GET /indexes
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

```json
{
  "indexes": [IndexModel, ...]
}
```

Returns an array of `IndexModel` objects with the same structure as [Describe Index](./pinecone-describe-index.md).

### Error Responses

| Status | Description |
|--------|-------------|
| `401` | Unauthorized - Invalid API key |
| `500` | Internal server error |

---

## Example

### Request

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X GET "https://api.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

### Response

```json
{
  "indexes": [
    {
      "name": "example-serverless-dedicated-index",
      "vector_type": "dense",
      "metric": "cosine",
      "dimension": 1536,
      "status": {
        "ready": true,
        "state": "Ready"
      },
      "host": "example-serverless-dedicated-index-bhnyigt.svc.aped-4627-b74a.pinecone.io",
      "spec": {
        "serverless": {
          "region": "us-east-1",
          "cloud": "aws",
          "read_capacity": {
            "mode": "Dedicated",
            "dedicated": {
              "node_type": "b1",
              "scaling": "Manual",
              "manual": {
                "shards": 1,
                "replicas": 2
              }
            },
            "status": {
              "state": "Scaling",
              "current_shards": 1,
              "current_replicas": 1
            }
          }
        }
      },
      "deletion_protection": "enabled",
      "tags": {
        "tag0": "value0",
        "tag1": "value1"
      }
    },
    {
      "name": "example-serverless-ondemand-index",
      "vector_type": "dense",
      "metric": "cosine",
      "dimension": 1024,
      "status": {
        "ready": true,
        "state": "Ready"
      },
      "host": "example-serverless-ondemand-index-bhnyigt.svc.aped-4627-b74a.pinecone.io",
      "spec": {
        "serverless": {
          "region": "us-east-1",
          "cloud": "aws",
          "read_capacity": {
            "mode": "OnDemand",
            "status": {
              "state": "Ready",
              "current_shards": null,
              "current_replicas": null
            }
          }
        }
      },
      "deletion_protection": "enabled",
      "tags": {
        "tag1": "value1",
        "tag2": "value2"
      },
      "embed": {
        "model": "llama-text-embed-v2",
        "field_map": {
          "text": "text"
        },
        "dimension": 1024,
        "metric": "cosine",
        "write_parameters": {
          "dimension": 1024,
          "input_type": "passage",
          "truncate": "END"
        },
        "read_parameters": {
          "dimension": 1024,
          "input_type": "query",
          "truncate": "END"
        },
        "vector_type": "dense"
      }
    },
    {
      "name": "example-pod-index",
      "vector_type": "dense",
      "metric": "cosine",
      "dimension": 768,
      "status": {
        "ready": true,
        "state": "Ready"
      },
      "host": "example-pod-index-bhnyigt.svc.us-east-1-aws.pinecone.io",
      "spec": {
        "pod": {
          "replicas": 1,
          "shards": 1,
          "pods": 1,
          "pod_type": "s1.x1",
          "environment": "us-east-1-aws"
        }
      },
      "deletion_protection": "disabled",
      "tags": null
    }
  ]
}
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# List all indexes
indexes = pc.list_indexes()

for index in indexes:
    print(f"{index.name}: {index.dimension}d, {index.metric}, {index.status.state}")
```

---

## Index Types in Response

The response includes all index types in your project:

| Type | Spec Field | Description |
|------|------------|-------------|
| **Serverless On-Demand** | `spec.serverless.read_capacity.mode: "OnDemand"` | Auto-scaling, pay-per-use |
| **Serverless Dedicated** | `spec.serverless.read_capacity.mode: "Dedicated"` | Provisioned hardware |
| **Pod-Based** | `spec.pod` | Legacy pod infrastructure |
| **BYOC** | `spec.byoc` | Bring Your Own Cloud |

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/control-plane/list_indexes)
