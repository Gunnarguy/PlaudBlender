# Pinecone Integration Playbook (Authoritative for Agents)

This document tells the next agent exactly what to do to keep Pinecone usage correct, modern, and safe. Always read this before touching Pinecone code or search flows.

## Scope & Current Stack
- Python SDK `pinecone[grpc]` ≥ 6.x, serverless indexes (default `aws/us-east-1`).
- Namespaces in use: `full_text` (chunked transcripts), `summaries` (AI syntheses).
- Embeddings via EmbeddingService (Gemini/OpenAI/Pinecone hosted). Provider set by `AI_PROVIDER` env var.
- Core touchpoints: `src/pinecone_client.py`, `gui/services/pinecone_service.py`, `gui/services/search_service.py`, `gui/services/index_manager.py`, `src/dual_store_processor.py`.

## Non-Negotiables (before any change)
1. **No secrets in code.** Everything comes from `.env` via `python-dotenv`.
2. **Dimension alignment.** Respect the index dimension; use `IndexManager`/`EmbeddingService` to sync. Never upsert mismatched vectors.
3. **Metadata contract.** Every vector must carry stable IDs and hashes to allow idempotent upsert/delete.
   - Required: `recording_id`, `source`, `model`, `dimension`, `text_hash` (sha256 of text), `start_at`, `duration_ms` (if available).
   - Optional but encouraged: `segment_id`, `themes/tags`, `provider`, `title`, `version`, `indexed_at`.
   - Schema defined in `src/models/vector_metadata.py`. Use `build_metadata()` helper.
4. **Namespaces explicit.** Do not rely on empty namespace; pass `full_text` or `summaries` deliberately.
5. **Tests.** Run at least `python -m pytest tests/` and any new targeted tests you add.

## Implemented Features (✅ done)

### 1. Extended data-plane coverage (`src/pinecone_client.py`)
- `fetch_by_metadata(filter_dict, namespace, limit)` — fetch vectors by metadata filter
- `list_namespaces_v2(limit, prefix)` — 2025-10 /namespaces GET endpoint
- `create_namespace(name, schema)` — create namespace with optional schema
- `delete_namespace(name)` — delete namespace
- `configure_index(deletion_protection, tags)` — control-plane PATCH
- `rerank(query, documents, model, top_n)` — Pinecone inference reranker
- `generate_embeddings(texts, model, input_type)` — Pinecone hosted embeddings

### 2. Metadata schema enforcement
- `src/models/vector_metadata.py` defines `VectorMetadata` Pydantic model.
- `build_metadata()` helper produces validated, Pinecone-ready dicts.
- `gui/services/pinecone_service.py::reembed_all_into_index` uses schema + `fetch_by_metadata` for dedup.

### 3. Rerank-enabled search
- `gui/services/search_service.py::search_with_rerank()` — cross-namespace search + Pinecone rerank.
- Controlled by env vars `PINECONE_RERANK_ENABLED` (default false) and `PINECONE_RERANK_MODEL`.

### 4. Hosted embeddings option
- `src/ai/embeddings.py::PineconeEmbedder` — uses Pinecone /embed API.
- Set `AI_PROVIDER=pinecone` and optionally `PINECONE_EMBEDDING_MODEL` (default multilingual-e5-large).

### 5. Control-plane governance
- `gui/services/index_manager.py` helpers: `enable_deletion_protection()`, `disable_deletion_protection()`, `set_index_tags(tags)`.
- Status check: `get_deletion_protection_status()`.

## Remaining / Future Action Items
1. **Surface governance toggles in GUI settings view** — add checkboxes/buttons for deletion protection and tags.
2. **Monitoring dashboard** — log read/write units and latency; show in status bar or dedicated view.
3. **Batch upsert safety valve** — pause + toast if error rate spikes.
4. **Assistant integration (optional)** — add `gui/services/assistant_service.py` for Pinecone Assistant chat endpoints if needed.

## Implementation Notes
- Cheatsheet: `docs/pinecone-cheatsheet.md` (pinned API links, 2025-10).
- Prefer `query_namespaces` for cross-namespace search; keep manual fallback.
- Batch size defaults: upsert 50–100; fetch 100.
- Rerank: configurable model; do not hardcode.
- Heavy I/O stays off Tk main thread.

## Testing & Verification
- Unit: run `python -m pytest tests/` (7 tests as of last check).
- Import smoke: `python -c "from src.models.vector_metadata import build_metadata; from gui.services.search_service import search_with_rerank"`
- Manual: from GUI, ensure index stats render, cross-namespace search works, and re-embed skips unchanged items.

## What NOT to do
- Do **not** change index dimensions silently; either auto-adjust embeddings (explicit log) or block with a clear error.
- Do **not** upsert without `recording_id` or `text_hash`.
- Do **not** invent new namespaces without surfacing them in settings/state.

## Quick Pointers for Next Agent
- Primary files: `src/pinecone_client.py`, `gui/services/pinecone_service.py`, `gui/services/search_service.py`, `gui/services/index_manager.py`, `src/dual_store_processor.py`.
- Metadata schema: `src/models/vector_metadata.py`.
- Settings persistence lives in `gui/services/settings_service.py`; use it for new toggles.
- Keep functions concise; add meaningful comments when behavior is non-obvious.
