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
- A **Chronos system** for temporal-aware semantic search and knowledge graph
- A **GUI control plane** (Streamlit) for managing recordings, vectors, search, and visualization
- A **data pipeline** (SQLite → Gemini processing → Qdrant) for durable storage and fast retrieval
- **Full Plaud API integration** (OAuth, Workflows, AI Summary, Webhooks, Device Management)
- **Advanced RAG capabilities** (hybrid search, reranking, GraphRAG, self-correction)
- **Optional integrations** (Notion sync, MCP server)

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
│                     Chronos UI (Streamlit)                   │
│  Timeline │ Search │ Insights │ Knowledge Graph │ Settings  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Plaud API Integration                      │
│  plaud_client │ plaud_workflow │ plaud_device │ webhook     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌────────────┐   ┌──────────┐
    │ SQLite   │   │  Qdrant    │   │  Notion  │
    │ (truth)  │   │  (vectors) │   │ (optional)│
    └──────────┘   └────────────┘   └──────────┘
```

### Key Principles
- **SQLite is the source of truth** (`data/brain.db`) — all recordings and segments persist here
- **Qdrant is the vector index** — local-first, granular temporal metadata, fast semantic search
- **Pydantic enforces contracts** — validated schemas at ingestion boundaries
- **GUI stays responsive** — all I/O runs in background threads

---

## Data Flow & Pipeline

### Canonical pipeline (SQL-first)
```
1. INGEST:  Plaud API → validate → SQLite (recordings, status=raw)
2. PROCESS: recordings → Gemini cognitive cleaning → SQLite (events, status=pending)
3. INDEX:   events → embed → Qdrant (temporal metadata: day_of_week, hour, category)
4. SERVE:   Chronos UI reads Qdrant for search, SQL for provenance/details
```

### Temporal Metadata (Chronos)
Each event in Qdrant includes:
- `day_of_week` (0-6) — enables "What do I do on Mondays?" queries
- `hour` (0-23) — time-of-day patterns
- `category` — event type (meeting, idea, task, etc.)
- `sentiment` — AI-detected emotional tone
- `entities` — extracted people, places, concepts

### Plaud API Integration
| Module           | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `plaud_client`   | OAuth, recordings, transcripts                      |
| `plaud_workflow` | AI workflow orchestration (transcription + ETL)     |
| `plaud_device`   | Device management (NotePin, Note, NotePro)          |
| `plaud_webhook`  | Async event handling (transcription complete, etc.) |

---

## Project Structure

```
PlaudBlender/
├── chronos_app.py              # Entry point → Streamlit UI
├── plaud_setup.py              # Setup wizard + OAuth
├── verify_integration.py       # Developer smoke tests
├── generate_mindmap.py         # Mind map from Notion data
├── requirements.txt
├── pyproject.toml              # Project metadata + pytest config
├── docker-compose.yml          # Qdrant + services
├── .env.example                # Environment template
│
├── src/                        # Core modules
│   ├── config.py               # Single .env loader
│   ├── plaud_oauth.py          # OAuth 2.0 client
│   ├── plaud_client.py         # Plaud API wrapper (recordings, transcripts)
│   ├── plaud_workflow.py       # Plaud Workflow API (AI pipelines)
│   ├── plaud_device.py         # Plaud Device management
│   ├── plaud_webhook.py        # Webhook handler for async events
│   ├── vector_store.py         # Qdrant abstraction layer
│   ├── notion_sync.py          # Direct Notion integration
│   ├── notion_client.py        # Notion API client
│   ├── visualizer.py           # Mind map generation
│   │
│   ├── chronos/                # Chronos system (temporal search)
│   │   ├── engine.py           # Gemini File API for cognitive cleaning
│   │   ├── qdrant_client.py    # Native Qdrant with temporal indexes
│   │   ├── embedding_service.py # Gemini embeddings
│   │   ├── ingest_service.py   # Audio download and storage
│   │   ├── analytics.py        # Day-of-week patterns, sentiment
│   │   ├── graph_service.py    # Entity extraction, NetworkX
│   │   └── transcript_processor.py # Text processing
│   │
│   ├── ai/                     # AI/embedding providers
│   ├── database/               # SQLAlchemy engine + models
│   ├── models/                 # Pydantic schemas
│   └── processing/             # Chunking, GraphRAG, self-correction
│
├── gui/                        # Legacy GUI package (being replaced by Streamlit)
│
├── scripts/                    # CLI tools
│   ├── chronos_pipeline.py     # Full pipeline: ingest → process → index
│   ├── mcp_server.py           # MCP server for AI agents
│   ├── plaud_auth_utils.py     # OAuth diagnostics
│   └── migrate_rename_vector_id.py # Database migration
│
├── tests/                      # Pytest suite (57+ tests)
├── data/                       # Local data (brain.db, audio, caches)
├── docs/                       # Documentation
│   ├── PROJECT_GUIDE.md        # ← You are here
│   ├── chronos-mvp.md          # Chronos system architecture
│   ├── PlaudDocs/              # Plaud API documentation
│   └── archive/                # Historical reference
│
└── archive/                    # Retired code
```

---

## Environment Variables

### Required (Plaud OAuth)
```
PLAUD_CLIENT_ID=
PLAUD_CLIENT_SECRET=
PLAUD_REDIRECT_URI=http://localhost:8080/callback
```

### Plaud API Settings (optional)
```
PLAUD_WEBHOOK_SECRET=       # For webhook signature verification
PLAUD_WEBHOOK_URL=          # Your webhook endpoint
PLAUD_DEFAULT_LANGUAGE=en   # Default transcription language
PLAUD_ENABLE_DIARIZATION=1  # Enable speaker diarization
PLAUD_WORKFLOW_TIMEOUT=600  # Workflow timeout in seconds
```

### AI Processing
```
GEMINI_API_KEY=             # Gemini for processing + embeddings
CHRONOS_CLEANING_MODEL=gemini-3-flash-preview
CHRONOS_EMBEDDING_MODEL=gemini-embedding-001
CHRONOS_ANALYST_MODEL=gemini-3-pro-preview
```

### Qdrant (Vector Store)
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=             # Optional for Qdrant Cloud
QDRANT_COLLECTION_NAME=chronos_events
```

### Optional Integrations
```
# Notion (two-way sync)
NOTION_TOKEN=
NOTION_DATABASE_ID=
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
