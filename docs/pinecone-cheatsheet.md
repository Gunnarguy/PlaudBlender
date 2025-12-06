# Pinecone Cheatsheet for PlaudBlender

Curated pointers to Pinecone docs that matter for this project (serverless index, namespaces, search/upsert, and optional Assistant/MCP hooks). Use these links when extending the vector/search flows or adding new integrations.

## Core data plane (vectors)
- Upsert vectors: https://docs.pinecone.io/reference/api/2025-10/data-plane/upsert
- Query/search vectors: https://docs.pinecone.io/reference/api/2025-10/data-plane/query
- Fetch by id: https://docs.pinecone.io/reference/api/2025-10/data-plane/fetch
- Fetch by metadata: https://docs.pinecone.io/reference/api/2025-10/data-plane/fetch_by_metadata
- Update vector metadata/values: https://docs.pinecone.io/reference/api/2025-10/data-plane/update
- Delete vectors: https://docs.pinecone.io/reference/api/2025-10/data-plane/delete
- Describe index stats: https://docs.pinecone.io/reference/api/2025-10/data-plane/describeindexstats
- List namespaces: https://docs.pinecone.io/reference/api/2025-10/data-plane/listnamespaces
- Create namespace: https://docs.pinecone.io/reference/api/2025-10/data-plane/createnamespace
- Delete namespace: https://docs.pinecone.io/reference/api/2025-10/data-plane/deletenamespace

## Control plane (indexes)
- Create index: https://docs.pinecone.io/reference/api/2025-10/control-plane/create_index
- Describe index: https://docs.pinecone.io/reference/api/2025-10/control-plane/describe_index
- List indexes: https://docs.pinecone.io/reference/api/2025-10/control-plane/list_indexes
- Configure index (deletion protection, tags, integrated embedding): https://docs.pinecone.io/reference/api/2025-10/control-plane/configure_index

## Assistant & OpenAI-compatible chat (optional)
- Standard assistant chat (citations/context): https://docs.pinecone.io/reference/api/2025-10/assistant/chat_assistant
- OpenAI-compatible assistant chat: https://docs.pinecone.io/reference/api/2025-10/assistant/chat_completion_assistant
- Retrieve context snippets: https://docs.pinecone.io/reference/api/2025-10/assistant/context_assistant
- Assistant MCP server (connect agents): https://docs.pinecone.io/guides/assistant/mcp-server

## Integrations & inference
- Hosted embedding / generate vectors: https://docs.pinecone.io/reference/api/2025-10/inference/generate-embeddings
- Rerank results: https://docs.pinecone.io/reference/api/2025-10/inference/rerank
- List available models: https://docs.pinecone.io/reference/api/2025-10/inference/list_models

## Ops & cost
- Error handling patterns: https://docs.pinecone.io/guides/production/error-handling
- Monitor performance: https://docs.pinecone.io/guides/production/monitoring
- Manage cost: https://docs.pinecone.io/guides/manage-cost/manage-cost

## Quick notes for PlaudBlender
- We primarily use serverless indexes, namespaces `full_text` and `summaries`.
- Keep metadata stable (`recording_id`, `segment_id`, `source`, `model`) to allow deletes/filters.
- When changing embedding dimensions/providers, ensure index dimension matches (see existing auto-fix logic in Pinecone view).
- Assistant endpoints are optional; current app uses direct vector search. If adding assistants, choose between standard chat (more features/citations) or OpenAI-compatible chat for simplicity.
- MCP: We already ship `scripts/mcp_server.py` (OpenAI Responses). Pinecone also offers an Assistant MCP server if you need agent access to assistants.
