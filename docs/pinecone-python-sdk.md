# Pinecone Python SDK

> The official Python SDK for Pinecone vector database.

## Installation

```bash
# Basic installation
pip install pinecone

# With gRPC support (recommended for high throughput)
pip install "pinecone[grpc]"

# With asyncio support
pip install "pinecone[asyncio]"
```

---

## Requirements

- Python 3.9 or later (tested up to 3.13)

---

## SDK Version Compatibility

| API Version | SDK Version |
|-------------|-------------|
| `2025-04` | v7.x |
| `2025-01` | v6.x |
| `2024-10` | v5.3.x |
| `2024-07` | v5.0.x-v5.2.x |
| `2024-04` | v4.x |

---

## Initialization

### Standard HTTP Client

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
```

### gRPC Client (Recommended for High Throughput)

```python
from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
```

---

## Index Management

### Create Serverless Index

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
    name="docs-example",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
    deletion_protection="disabled",
    tags={"environment": "development"}
)
```

### Create Pod-Based Index

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
    name="docs-example",
    dimension=768,
    metric="cosine",
    spec=PodSpec(
        environment="us-west-1-gcp",
        pod_type="p1.x1",
        pods=1
    )
)
```

### List Indexes

```python
indexes = pc.list_indexes()
for index in indexes:
    print(f"{index.name}: {index.dimension}d, {index.metric}")
```

### Describe Index

```python
info = pc.describe_index("docs-example")
print(f"Host: {info.host}")
print(f"Status: {info.status}")
```

### Delete Index

```python
pc.delete_index("docs-example")
```

---

## Data Operations

### Connect to Index

```python
# By name (requires additional API call)
index = pc.Index("docs-example")

# By host (recommended)
index = pc.Index(host="docs-example-abc123.svc.pinecone.io")
```

### Upsert Vectors

```python
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, 0.3, ...],  # 768 dimensions
            "metadata": {"title": "Document 1", "category": "tech"}
        },
        {
            "id": "vec2",
            "values": [0.4, 0.5, 0.6, ...],
            "metadata": {"title": "Document 2", "category": "science"}
        }
    ],
    namespace="my-namespace"
)
```

### Query Vectors

```python
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    namespace="my-namespace",
    include_metadata=True,
    include_values=False,
    filter={"category": {"$eq": "tech"}}
)

for match in results.matches:
    print(f"{match.id}: {match.score}")
    print(f"  Metadata: {match.metadata}")
```

### Fetch Vectors by ID

```python
response = index.fetch(
    ids=["vec1", "vec2"],
    namespace="my-namespace"
)

for id, vector in response.vectors.items():
    print(f"{id}: {vector.values[:5]}...")
```

### Update Vector Metadata

```python
index.update(
    id="vec1",
    set_metadata={"category": "updated"},
    namespace="my-namespace"
)
```

### Delete Vectors

```python
# Delete by ID
index.delete(ids=["vec1", "vec2"], namespace="my-namespace")

# Delete by filter
index.delete(
    filter={"category": {"$eq": "old"}},
    namespace="my-namespace"
)

# Delete all in namespace
index.delete(delete_all=True, namespace="my-namespace")
```

### Get Index Statistics

```python
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
print(f"Dimension: {stats.dimension}")
for ns, ns_stats in stats.namespaces.items():
    print(f"  {ns or '(default)'}: {ns_stats.vector_count} vectors")
```

---

## Async Operations

```python
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec

async def main():
    async with PineconeAsyncio(api_key="YOUR_API_KEY") as pc:
        # Create index
        await pc.create_index(
            name="async-example",
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
        # Get index
        async with pc.IndexAsyncio(host="INDEX_HOST") as idx:
            # Upsert
            await idx.upsert(vectors=[...])
            
            # Query
            results = await idx.query(vector=[...], top_k=10)

asyncio.run(main())
```

---

## Query Across Namespaces

```python
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key="YOUR_API_KEY")
index = pc.Index(
    name="docs-example",
    pool_threads=50  # For parallel queries
)

combined_results = index.query_namespaces(
    vector=[0.1, 0.2, ...],
    namespaces=["ns1", "ns2", "ns3"],
    metric="cosine",
    top_k=10,
    include_metadata=True,
    filter={"category": {"$eq": "tech"}}
)

for match in combined_results.matches:
    print(f"{match.namespace}/{match.id}: {match.score}")
```

---

## Upsert from DataFrame

```python
import pandas as pd
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index(host="INDEX_HOST")

# DataFrame with 'id', 'values', and optionally 'metadata' columns
df = pd.DataFrame({
    "id": ["vec1", "vec2", "vec3"],
    "values": [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
    "metadata": [{"a": 1}, {"a": 2}, {"a": 3}]
})

index.upsert_from_dataframe(df, batch_size=100)
```

---

## Proxy Configuration

```python
from pinecone import Pinecone
from urllib3 import make_headers

pc = Pinecone(
    api_key="YOUR_API_KEY",
    proxy_url="https://your-proxy.com",
    proxy_headers=make_headers(proxy_basic_auth="username:password"),
    ssl_ca_certs="path/to/cert-bundle.pem",
    ssl_verify=True  # Default; set False to disable (not recommended)
)
```

---

## Inference API

### Generate Embeddings

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Dense embeddings
result = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[
        {"text": "First document"},
        {"text": "Second document"}
    ],
    parameters={
        "input_type": "passage",
        "truncate": "END"
    }
)

for item in result.data:
    print(f"Vector: {item.values[:5]}...")

# Sparse embeddings
sparse_result = pc.inference.embed(
    model="pinecone-sparse-english-v0",
    inputs=[{"text": "Query text"}],
    parameters={"input_type": "query", "return_tokens": True}
)
```

### Rerank Results

```python
rerank_result = pc.inference.rerank(
    model="pinecone-rerank-v0",
    query="What is machine learning?",
    documents=[
        "Machine learning is a subset of AI...",
        "The weather today is sunny...",
        "ML algorithms learn from data..."
    ],
    top_n=2
)

for item in rerank_result.data:
    print(f"Index {item.index}: {item.score}")
```

---

## gRPC vs HTTP

| Feature | HTTP | gRPC |
|---------|------|------|
| Installation | `pip install pinecone` | `pip install "pinecone[grpc]"` |
| Performance | Good | Better (10-20% faster) |
| Connection pooling | Manual config | Automatic |
| Streaming | No | Yes |
| Recommended for | Low-medium volume | High volume |

### gRPC Example

```python
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key="YOUR_API_KEY")
index = pc.Index(host="INDEX_HOST")

# All operations work the same
index.upsert(vectors=[...])
results = index.query(vector=[...], top_k=10)
```

---

## Error Handling

```python
from pinecone import Pinecone
from pinecone.exceptions import (
    PineconeException,
    NotFoundException,
    UnauthorizedException,
    ServiceException
)

pc = Pinecone(api_key="YOUR_API_KEY")

try:
    info = pc.describe_index("nonexistent")
except NotFoundException:
    print("Index not found")
except UnauthorizedException:
    print("Invalid API key")
except ServiceException as e:
    print(f"Service error: {e}")
except PineconeException as e:
    print(f"Pinecone error: {e}")
```

---

## Resources

- [SDK Documentation](https://sdk.pinecone.io/python/)
- [GitHub Repository](https://github.com/pinecone-io/pinecone-python-client)
- [Upgrade Guide](https://github.com/pinecone-io/pinecone-python-client/blob/main/docs/upgrading.md)
- [Report Issues](https://github.com/pinecone-io/pinecone-python-client/issues)

---

> **Reference:** [Pinecone Python SDK Documentation](https://docs.pinecone.io/reference/python-sdk)
