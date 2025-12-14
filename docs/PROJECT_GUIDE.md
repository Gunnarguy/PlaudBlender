# PlaudBlender — Complete Project Documentation

> **Single source of truth** for architecture, roadmap, implementation status, and next steps.

---

## Table of Contents
1. [What Is PlaudBlender?](#what-is-plaudblender)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Data Flow & Pipeline](#data-flow--pipeline)
5. [Project Structure](#project-structure)
6. [Environment Variables](#environment-variables)
7. [GUI Overview](#gui-overview)
8. [RAG & Search Capabilities](#rag--search-capabilities)
9. [Implementation Status](#implementation-status)
10. [What's Missing / Next Steps](#whats-missing--next-steps)
11. [Developer Notes](#developer-notes)
12. [Archived Reference](#archived-reference)

---

## What Is PlaudBlender?

PlaudBlender transforms **Plaud voice recordings** into a **searchable knowledge base** with:
- A **GUI control plane** (Tkinter) for managing recordings, vectors, search, and visualization
- A **data pipeline** (SQLite → AI processing → Pinecone) for durable storage and fast retrieval
- **Advanced RAG capabilities** (hybrid search, reranking, GraphRAG, self-correction)
- **Optional integrations** (Notion sync, OpenAI Chat, MCP server)

### Core Philosophy
> _"Gunnar loves data, granularity, and depth—the ability to drill down and see what's happening under the hood."_

The GUI exposes metrics (latency, read units, scores), tooltips explaining every control, and provenance back to source recordings.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Authenticate with Plaud
```bash
python plaud_setup.py
```

### 4. Launch the GUI
```bash
python gui.py
```

### Other entry points
| Command                                            | Purpose                               |
| -------------------------------------------------- | ------------------------------------- |
| `python scripts/sync_to_pinecone.py`               | Batch sync Plaud → Pinecone           |
| `python scripts/process_pending.py`                | Process SQL recordings into segments  |
| `python scripts/plaud_auth_utils.py --check-token` | Validate Plaud OAuth token            |
| `python -m scripts.mcp_server`                     | Start MCP server (OpenAI Responses)   |
| `python verify_integration.py`                     | Developer smoke test for all features |
| `python -m pytest tests/`                          | Run test suite                        |

---

## Architecture

### North Star
```
┌─────────────────────────────────────────────────────────────┐
│                     GUI (Control Plane)                      │
│  Dashboard │ Transcripts │ Pinecone │ Search │ Settings     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Services Layer                             │
│  transcripts_service │ pinecone_service │ search_service    │
│  embedding_service │ stats_service │ chat_service           │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌────────────┐   ┌──────────┐
    │ SQLite   │   │  Pinecone  │   │  Notion  │
    │ (truth)  │   │  (serving) │   │ (optional)│
    └──────────┘   └────────────┘   └──────────┘
```

### Key Principles
- **SQLite is the source of truth** (`data/brain.db`) — all recordings and segments persist here
- **Pinecone is the serving index** — fast vector search with metadata pointing back to SQL IDs
- **Pydantic enforces contracts** — validated schemas at ingestion boundaries
- **GUI stays responsive** — all I/O runs in background threads

---

## Data Flow & Pipeline

### Canonical pipeline (SQL-first)
```
1. INGEST:  Plaud API → validate → SQLite (recordings, status=raw)
2. PROCESS: recordings → chunk → SQLite (segments, status=pending)
3. INDEX:   segments → embed → Pinecone (metadata: recording_id, segment_id)
4. SERVE:   GUI reads Pinecone for search, SQL for provenance/details
```

### Namespaces
- `full_text` — chunked transcript segments
- `summaries` — AI-generated syntheses

### Metadata contract (every vector)
Required: `recording_id`, `source`, `model`, `dimension`, `text_hash`
Recommended: `segment_id`, `themes`, `title`, `indexed_at`

---

## Project Structure

```
PlaudBlender/
├── gui.py                      # Entry point → gui/app.py
├── plaud_setup.py              # Setup wizard + OAuth
├── verify_integration.py       # Developer smoke tests
├── generate_mindmap.py         # Mind map from Notion data
├── requirements.txt
├── pyproject.toml              # Project metadata + pytest config
├── .env.example                # Environment template
│
├── gui/                        # GUI package (control plane)
│   ├── app.py                  # Main application + action registry
│   ├── state.py                # Centralized state
│   ├── theme.py                # Styling
│   ├── views/                  # Dashboard, Transcripts, Pinecone, Search, etc.
│   ├── services/               # Business logic (transcripts, pinecone, search, etc.)
│   ├── components/             # Reusable widgets (stat_card, status_bar)
│   └── utils/                  # Threading, tooltips, logging
│
├── src/                        # Core processing & clients
│   ├── config.py               # Single .env loader
│   ├── plaud_oauth.py          # OAuth 2.0 client
│   ├── plaud_client.py         # Plaud API wrapper
│   ├── pinecone_client.py      # Pinecone wrapper
│   ├── notion_sync.py          # Direct Notion integration
│   ├── notion_client.py        # Notion API client
│   ├── visualizer.py           # Mind map generation
│   ├── ai/                     # Embedding providers
│   ├── database/               # SQLAlchemy engine + models
│   ├── models/                 # Pydantic schemas
│   └── processing/             # Chunking, GraphRAG, self-correction, etc.
│
├── scripts/                    # CLI tools
│   ├── sync_to_pinecone.py     # Batch Plaud → Pinecone
│   ├── process_pending.py      # Process SQL recordings
│   ├── plaud_auth_utils.py     # OAuth diagnostics
│   └── mcp_server.py           # MCP server (OpenAI Responses)
│
├── tests/                      # Pytest suite
├── data/                       # Local data (brain.db, caches) — gitignored
├── docs/                       # This guide + references
│   ├── PROJECT_GUIDE.md        # ← You are here (consolidated)
│   ├── audit-checklist.md      # Live task tracker
│   ├── pinecone-cheatsheet.md  # Quick API links
│   ├── pinecone-integration-playbook.md  # Agent instructions
│   └── archive/                # Historical reference docs
│
├── archive/                    # Retired code (gui_legacy.py, etc.)
├── lib/                        # Vendor libraries (vis.js, tom-select)
└── .vscode/                    # Workspace settings
```

---

## Environment Variables

### Required (Plaud)
```
PLAUD_CLIENT_ID=
PLAUD_CLIENT_SECRET=
PLAUD_REDIRECT_URI=http://localhost:8080/callback
```

### Optional (enables features)
```
# AI Processing
GEMINI_API_KEY=             # Gemini for theme extraction
PINECONE_API_KEY=           # Vector database
PINECONE_INDEX_NAME=transcripts

# OpenAI (Chat + MCP)
OPENAI_API_KEY=
OPENAI_DEFAULT_MODEL=gpt-4.1

# Notion (optional sync)
NOTION_TOKEN=
NOTION_DATABASE_ID=

# Embedding provider
AI_PROVIDER=gemini          # gemini | openai | pinecone
```

---

## GUI Overview

### Views
| View                | Purpose                                                 |
| ------------------- | ------------------------------------------------------- |
| **Dashboard**       | Stats overview, quick actions, recent activity          |
| **Transcripts**     | Browse/filter/sync/delete Plaud recordings              |
| **Pinecone**        | Vector workspace (CRUD, search, namespace management)   |
| **Search**          | Semantic search with hybrid/rerank/self-correct toggles |
| **Knowledge Graph** | Interactive vis.js visualization of entities            |
| **Chat**            | OpenAI Responses chat interface                         |
| **Notion**          | Two-way Notion sync controls                            |
| **Settings**        | API configuration, embedding provider selection         |
| **Logs**            | Application log stream                                  |

### Key UI features
- Status bar shows latency, read units, active namespace
- Tooltips on every control
- Async operations with busy indicators
- Export to JSON/CSV/GraphML

---

## RAG & Search Capabilities

### Implemented (✅)
| Feature                               | Location                                  |
| ------------------------------------- | ----------------------------------------- |
| Dense vector search                   | `search_service.py`                       |
| Hybrid search (dense + sparse)        | `hybrid_search_service.py`                |
| Reranking (Pinecone/Cohere)           | `search_with_rerank()`                    |
| Hierarchical chunking (parent/child)  | `src/processing/hierarchical_chunking.py` |
| GraphRAG entity extraction            | `src/processing/graph_rag.py`             |
| Community summarization               | `src/processing/graph_rag.py`             |
| Query routing (intent classification) | `src/processing/query_router.py`          |
| Reciprocal Rank Fusion                | `src/processing/rrf_fusion.py`            |
| Self-correction loop                  | `src/processing/self_correction.py`       |
| LLM-as-Judge evaluation               | `src/processing/rag_evaluation.py`        |
| Thought signatures (agentic state)    | `src/processing/thought_signatures.py`    |
| Conflict detection                    | `src/processing/conflict_detection.py`    |
| ColPali vision ingestion              | `src/processing/colpali_ingestion.py`     |
| Audio embeddings (CLAP)               | `src/processing/audio_processor.py`       |

### Search modes in UI
- 🔍 Standard search
- 🔀 Hybrid search (alpha slider)
- 🏆 Rerank search
- 🔄 Self-correcting search
- 🧠 Smart search (router + RRF + GraphRAG)
- 🎵 Audio similarity search

---

## Implementation Status

### Done ✅
- [x] Modular GUI architecture (views/services/components)
- [x] SQLite database layer with Recording/Segment models
- [x] Full Pinecone 2025-10 API coverage
- [x] Metadata schema enforcement (`VectorMetadata`)
- [x] All RAG features from research docs
- [x] Direct Notion integration (replace Zapier)
- [x] OpenAI Chat tab + MCP server
- [x] Knowledge Graph visualization
- [x] Audio processing pipeline

### In Progress (~)
- [ ] Unified processor (`dual_store_processor` → `src/processing/engine.py`)
- [ ] Single canonical pipeline (SQL-first everywhere)

### Not Started
- [ ] Bulk import/export UX in Pinecone view
- [ ] Settings validation + persistence clarity
- [ ] Transcripts pagination/virtualization
- [ ] RAG health metrics in Dashboard
- [ ] Background worker CLI (`scripts/worker.py --loop`)

---

## What's Missing / Next Steps

### Priority 1: Pipeline consolidation
Pick SQL-first as the canonical path. Keep `sync_to_pinecone.py` as a developer convenience but ensure all GUI actions go through SQL → Pinecone.

### Priority 2: GUI polish (from audit-checklist)
- Pinecone bulk import/export with progress feedback
- Settings save/load with validation errors surfaced
- Transcripts view pagination
- "Last sync" timestamp consistency

### Priority 3: Testing hardening
- Extend `tests/` with mocked tests for processing, services
- CI integration

### Priority 4: Pipeline unification
- Consolidate the two pipeline flows (direct-to-Pinecone vs SQL-first) into one canonical path
- Ensure all ingestion goes through validated Pydantic schemas

---

## Developer Notes

### Threading rules
- All network/IO in background threads via `run_async()`
- Never call Tk methods from non-main thread
- Use `root.after()` for UI updates from callbacks

### Adding a new view
1. Create `gui/views/my_view.py` extending base pattern
2. Register in `gui/app.py::_create_views()`
3. Add sidebar button in `_build_layout()`
4. Wire actions in `self.actions` dict

### Adding a new service
1. Create `gui/services/my_service.py`
2. Import in `gui/services/__init__.py`
3. Use from views/app via import

### Metadata schema
Always use `src/models/vector_metadata.py::build_metadata()` when upserting to Pinecone.

---

## Archived Reference

Historical documentation and retired code is preserved in `archive/` and `docs/archive/`:

### `archive/` (retired code)
- `gui_legacy.py` — Original monolithic GUI (~5900 lines, replaced by modular `gui/`)
- `test_components.py` — Legacy live component tests
- `REFACTORING_PLAN.md` — Previous refactoring roadmap
- `src/dual_store_processor.py` — Original AI processing pipeline (superseded by `scripts/sync_to_pinecone.py`)
- `src/llm_processor.py` — LlamaIndex + Gemini processor (unused)
- `src/notion_mcp_client.py` — MCP-based Notion client (unused)
- `src/processing/hierarchical_chunking.py` — Hierarchical chunker (unused)
- `src/processing/rag_evaluation.py` — RAG evaluation metrics (unused)

### `docs/archive/` (reference docs)
- 14 Pinecone API reference documents
- 3 Gemini RAG research documents
- `architecture-roadmap.md`, `README.md` — Superseded by this guide

These informed the current implementation but are not actively maintained.

---

*Last updated: December 2025*
