# Delete an Index

> Delete an existing index permanently.

## Endpoint

```
DELETE /indexes/{index_name}
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
| `index_name` | `string` | ✅ | Name of the index to delete |

---

## Response

### `202 Accepted`

**Empty response** - Index deletion has been initiated.

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `401` | `UNAUTHENTICATED` | Invalid API key |
| `403` | `FORBIDDEN` | Deletion protection is enabled |
| `404` | `NOT_FOUND` | Index not found |
| `412` | `FAILED_PRECONDITION` | Pending collections from this index |
| `500` | `UNKNOWN` | Internal server error |

#### Error Response Format

```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "Resource protected from deletion"
  },
  "status": 403
}
```

---

## Example

### Request

```bash
PINECONE_API_KEY="YOUR_API_KEY"

curl -X DELETE "https://api.pinecone.io/indexes/docs-example-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

### Response

```
HTTP/1.1 202 Accepted
(empty body)
```

---

## Python SDK

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Delete the index
pc.delete_index("docs-example-index")
```

---

## Important Notes

> ⚠️ **This operation is irreversible.** All data in the index will be permanently deleted.

### Deletion Protection

If `deletion_protection` is enabled on the index, you must first disable it:

```bash
# Disable deletion protection first
curl -X PATCH "https://api.pinecone.io/indexes/docs-example-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10" \
  -H "Content-Type: application/json" \
  -d '{"deletion_protection": "disabled"}'

# Then delete the index
curl -X DELETE "https://api.pinecone.io/indexes/docs-example-index" \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "X-Pinecone-Api-Version: 2025-10"
```

### Pending Collections Precondition

If there are pending collections being created from this index, deletion will fail:

```json
{
  "error": {
    "code": "FAILED_PRECONDITION",
    "message": "Unable to delete an index. There are pending collections for this index: ['test-collection']"
  },
  "status": 412
}
```

Wait for collections to complete or delete them first before deleting the index.

### Python with Protection Check

```python
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# Check if protection is enabled
index_info = pc.describe_index("docs-example-index")
if index_info.deletion_protection == "enabled":
    # Disable protection first
    pc.configure_index(
        "docs-example-index",
        deletion_protection="disabled"
    )

# Now delete
pc.delete_index("docs-example-index")
```

### Data Preservation

Before deleting an index, consider creating a backup:

#### For Serverless Indexes

```python
# Create a backup
pc.create_backup(
    backup_name="pre-deletion-backup",
    source_index_name="docs-example-index"
)
```

#### For Pod-Based Indexes

```python
import time

# Create a collection (acts as backup)
pc.create_collection(
    name="backup-collection",
    source="docs-example-index"
)

# Wait for collection to be ready
while pc.describe_collection("backup-collection").status != "Ready":
    time.sleep(1)

# Now safe to delete
pc.delete_index("docs-example-index")
```

---

## Use Cases

- **Cleanup** - Remove unused or test indexes
- **Migration** - Delete old index after migrating data to new one
- **Cost Management** - Reduce costs by removing unnecessary indexes
- **Environment Reset** - Clear development/staging environments

---

> **Reference:** [Pinecone API Documentation](https://docs.pinecone.io/reference/api/2025-10/control-plane/delete_index)
