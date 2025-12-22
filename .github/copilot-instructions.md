# PlaudBlender — AI Agent Instructions

> **This file is auto-loaded into every Copilot conversation.** It tells you how to navigate and work in this project.

## Quick Orientation

| What            | Where                                                                              |
| --------------- | ---------------------------------------------------------------------------------- |
| **Full docs**   | `docs/PROJECT_GUIDE.md` — read this first for architecture, structure, and roadmap |
| **MVP spec**    | `docs/chronos-mvp.md` — complete Chronos system architecture                       |
| **Entry point** | `streamlit run chronos_app.py` — launches the Streamlit UI                         |
| **Pipeline**    | `python scripts/chronos_pipeline.py --full` — ingest → process → index → graph     |
| **Tests**       | `python -m pytest tests/` — 57 tests, run before committing                        |

## What This Project Does

**Chronos** transforms **Plaud voice recordings** into a **searchable knowledge base**:

- Fetches audio from Plaud API (OAuth) and stores locally
- Processes through Gemini 3 for cognitive cleaning (removes "ums", extracts events)
- Indexes to Qdrant with temporal metadata (day-of-week, hour, category)
- Extracts entities and builds knowledge graph (NetworkX)
- Provides Streamlit UI with timeline and semantic search
- Optional: Notion sync, MCP server

## Project Structure (Simplified)

```
chronos_app.py          → Streamlit UI (Master-Detail with timeline)
plaud_setup.py          → Setup wizard + OAuth
scripts/                → CLI tools (chronos_pipeline.py, mcp_server.py, plaud_auth_utils.py)
src/chronos/            → Core Chronos system (engine, qdrant_client, ingest, graph, analytics)
src/                    → Shared logic (plaud_client, database/, models/, ai/graph_rag.py)
tests/                  → Pytest suite
docs/                   → PROJECT_GUIDE.md, chronos-mvp.md
archive/                → Legacy GUI and Pinecone code (don't import from here)
```

## User Philosophy

> _"Gunnar loves data, granularity, and depth—the ability to drill down and see what's happening under the hood."_

- **Expose metrics** (latency, read units, scores) in the UI
- **Add tooltips** explaining what every control does
- **Show provenance** — link vectors back to source recordings
- **Never hide information** that could help debug or understand behavior

## Coding Rules

1. **Environment:** All secrets from `.env` via `python-dotenv`. Never hardcode.
2. **Imports:** Use `from src.X import Y` pattern. All `src/` subdirs have `__init__.py`.
3. **Schemas:** Validate data with Pydantic (`src/models/chronos_schemas.py` for Chronos, `src/models/schemas.py` for legacy)
4. **Qdrant:** Use `src/chronos/qdrant_client.py` — native Qdrant API with temporal indexes
5. **Gemini:** Use `src/chronos/engine.py` for audio processing and `src/chronos/embedding_service.py` for vectors
6. **Tests:** Run `pytest tests/` before any commit. Currently 57 tests.

## Before You Code

1. **Read `docs/chronos-mvp.md`** for Chronos architecture
2. **Read `docs/PROJECT_GUIDE.md`** if you need full context
3. **Check `src/chronos/` modules** — this is the active codebase
4. **Run tests** after any change: `python -m pytest tests/ -q`

## Key Chronos Services (src/chronos/)

| Service             | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `ingest_service`    | Download audio from Plaud, verify checksums, store locally |
| `engine`            | Gemini File API integration for cognitive cleaning         |
| `qdrant_client`     | Native Qdrant client with temporal payload indexes         |
| `embedding_service` | Gemini text-embedding-004 batch embedding                  |
| `analytics`         | Day-of-week pattern analysis, sentiment aggregation        |
| `graph_service`     | Entity extraction and NetworkX graph building              |

## Don't

- Don't import from `archive/` — that's retired code (legacy GUI, Pinecone shims)
- Don't use chunking — Gemini's 1M token context processes full recordings
- Don't scatter `load_dotenv()` — use `src/config.py`
- Don't reference Pinecone — we're 100% Qdrant now
