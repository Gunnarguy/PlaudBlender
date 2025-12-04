# Describe an Index

> Get a description of an index.

## Endpoint

```
GET /indexes/{index_name}
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
| `index_name` | `string` | ✅ | Name of the index to describe |

---

## Response

### `200 OK` - IndexModel

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | ✅ | Index name (1-45 chars, lowercase alphanumeric or `-`) |
| `vector_type` | `string` | ✅ | `"dense"` or `"sparse"` |
| `metric` | `string` | ✅ | `"cosine"`, `"euclidean"`, or `"dotproduct"` |
| `dimension` | `integer` | ❌ | Vector dimensions (1-20000). Not set for sparse indexes. |
| `host` | `string` | ✅ | Index URL endpoint |
| `private_host` | `string` | ❌ | Private endpoint URL (if configured) |
| `status` | `object` | ✅ | `{ready: boolean, state: string}` |
| `spec` | `object` | ✅ | Serverless, Pod-based, or BYOC configuration |
| `deletion_protection` | `string` | ❌ | `"enabled"` or `"disabled"` |
| `tags` | `object` | ❌ | Custom key-value tags |
| `embed` | `object` | ❌ | Integrated embedding configuration (if enabled) |

### Status States

| State | Description |
|-------|-------------|
| `Initializing` | Index is being created |
| `InitializationFailed` | Creation failed |
| `ScalingUp` | Adding replicas/pods |
| `ScalingDown` | Removing replicas/pods |
| `ScalingUpPodSize` | Increasing pod size |
| `ScalingDownPodSize` | Decreasing pod size |
| `Terminating` | Index being deleted |
| `Ready` | Index is operational |
| `Disabled` | Index is disabled |

### Error Responses

| Status | Description |
|--------|-------------|
| `401` | Unauthorized - Invalid API key |
| `404` | Index not found |
| `500` | Internal server error |

---

## Example

### Request

```bash
PINECONE_API_KEY="YOUR_API_KEY"
INDEX_NAME="YOUR_INDEX_NAME"

curl "https://api.pinecone.io/indexes/$INDEX_NAME" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

### Response: Serverless Index (On-Demand)

```json
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
}
```

### Response: Serverless Index (Dedicated)

```json
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
}
```

### Response: Pod-Based Index

```json
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
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Get index details
index_info = pc.describe_index("example-index")

print(index_info.name)         # "example-index"
print(index_info.dimension)    # 1024
print(index_info.vector_type)  # "dense"
print(index_info.metric)       # "cosine"
print(index_info.host)         # "example-index-abc123..."
print(index_info.status)       # {"ready": True, "state": "Ready"}
```

---

## Spec Object Details

### Serverless Spec

| Field | Type | Description |
|-------|------|-------------|
| `cloud` | `string` | `"aws"`, `"gcp"`, or `"azure"` |
| `region` | `string` | Deployment region |
| `read_capacity` | `object` | On-demand or dedicated configuration |
| `source_collection` | `string` | Source collection name (optional) |
| `schema` | `object` | Metadata indexing configuration (optional) |

### Read Capacity Modes

| Mode | Description |
|------|-------------|
| `OnDemand` | Automatic scaling (default) |
| `Dedicated` | Provisioned hardware with manual scaling |

### Read Capacity Status

| Field | Type | Description |
|-------|------|-------------|
| `state` | `string` | `"Ready"`, `"Scaling"`, `"Migrating"`, or `"Error"` |
| `current_shards` | `integer` | Current number of shards |
| `current_replicas` | `integer` | Current number of replicas |
| `error_message` | `string` | Error details if state is `"Error"` |

### Dedicated Configuration

| Field | Type | Description |
|-------|------|-------------|
| `node_type` | `string` | `"b1"` or `"t1"` (enhanced processing/memory) |
| `scaling` | `string` | `"Manual"` |
| `manual.shards` | `integer` | Storage units (250 GB each) |
| `manual.replicas` | `integer` | Compute replicas for throughput |

### Pod Spec

| Field | Type | Description |
|-------|------|-------------|
| `environment` | `string` | Pod environment (e.g., `"us-east-1-aws"`) |
| `pod_type` | `string` | `"s1.x1"`, `"p1.x1"`, `"p2.x1"`, etc. |
| `pods` | `integer` | Total pods (`shards × replicas`) |
| `replicas` | `integer` | Number of replicas |
| `shards` | `integer` | Number of shards |
| `metadata_config` | `object` | Indexed metadata fields |

---

## Embed Object (Integrated Inference)

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | Embedding model name (e.g., `"llama-text-embed-v2"`) |
| `field_map` | `object` | Maps document fields to embedding input |
| `dimension` | `integer` | Vector dimensions |
| `metric` | `string` | Distance metric |
| `vector_type` | `string` | `"dense"` or `"sparse"` |
| `write_parameters` | `object` | Upsert parameters |
| `read_parameters` | `object` | Query parameters |

### Embed Parameters

| Field | Type | Description |
|-------|------|-------------|
| `input_type` | `string` | `"passage"` for documents, `"query"` for search |
| `truncate` | `string` | `"END"` or `"NONE"` |
| `dimension` | `integer` | Override default model dimension |

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/control-plane/describe_index)
