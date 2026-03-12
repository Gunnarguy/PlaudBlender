# PlaudBlender — Complete Project Documentation

> **Single source of truth** for architecture, roadmap, implementation status, and next steps.
>
> _Last updated: March 10, 2026_

---

## Table of Contents

1. [What Is PlaudBlender?](#what-is-plaudblender)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Data Flow & Pipeline](#data-flow--pipeline)
5. [Project Structure](#project-structure)
6. [Environment Variables](#environment-variables)
7. [UI Overview (Dash v2)](#ui-overview-dash-v2)
8. [Implementation Status](#implementation-status)
9. [What's Next — Tier 3](#whats-next--tier-3)
10. [Developer Notes](#developer-notes)

---

## What Is PlaudBlender?

PlaudBlender transforms **Plaud voice recordings** into a **searchable knowledge timeline** with:

- A **Chronos system** for temporal-aware semantic search and knowledge graph
- A **Dash v2 UI** (`app_v2/`) — recording-centric interface with 7 views and 23 callbacks
- A **data pipeline** (Plaud API → SQLite → Gemini → Qdrant) for durable storage and fast retrieval
- **Full Plaud API integration** (OAuth, transcripts, devices, workflows)
- **Knowledge Graph** visualization (entity extraction → NetworkX → Cytoscape)

### Core Philosophy

> _"Gunnar loves data, granularity, and depth—the ability to drill down and see what's happening under the hood."_

- **Expose metrics** (latency, scores) in the UI
- **Show command previews** before running pipelines
- **Progressive disclosure** — simple by default, advanced options collapsed
- **Never hide information** that could help debug or understand behavior

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)
```

### 3. Start Qdrant

```bash
docker-compose up -d
```

### 4. Authenticate with Plaud

```bash
python plaud_setup.py
```

### 5. Run the pipeline

```bash
# Full pipeline: download from Plaud → process through Gemini → index to Qdrant → build graph
python scripts/chronos_pipeline.py --full
```

### 6. Launch the UI

```bash
python scripts/launch_app.py
# → http://localhost:8050
```

### Other entry points

| Command                                        | Purpose                            |
| ---------------------------------------------- | ---------------------------------- |
| `python scripts/chronos_pipeline.py --ingest`  | Download recordings from Plaud API |
| `python scripts/chronos_pipeline.py --process` | Process pending through Gemini     |
| `python scripts/chronos_pipeline.py --index`   | Embed + index to Qdrant            |
| `python scripts/chronos_pipeline.py --graph`   | Build knowledge graph              |
| `python scripts/fix_recordings.py`             | Diagnose stuck/failed recordings   |
| `python scripts/fix_recordings.py --fix`       | Reset stuck recordings to pending  |
| `python scripts/index_unindexed.py`            | Index events missing from Qdrant   |
| `python scripts/chronos_pipeline.py --reindex` | Re-embed all events (model change) |
| `python -m pytest tests/`                      | Run test suite (90 tests)          |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Chronos UI (Dash v2)                       │
│  Days │ Search │ Graph │ Stats │ Sync │ Settings │ Topics   │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌────────────┐   ┌──────────────┐
    │ SQLite   │   │  Qdrant    │   │  Gemini AI   │
    │ (truth)  │   │  (vectors) │   │ (processing) │
    │ brain.db │   │  :6333     │   │  Flash/Pro   │
    └──────────┘   └────────────┘   └──────────────┘
          ▲
          │
    ┌──────────┐
    │ Plaud API│
    │  (OAuth) │
    └──────────┘
```

### Key Principles

- **SQLite is source of truth** (`data/brain.db`) — all recordings and events persist here
- **Qdrant is the vector index** — local-first, 768-dim Gemini Embedding 2 vectors (multimodal), temporal payload indexes
- **Pydantic enforces contracts** — validated schemas at all boundaries
- **100% local** — no cloud dependencies except Gemini API and Plaud API

---

## Data Flow & Pipeline

### Canonical pipeline (SQL-first)

```
1. INGEST:   Plaud API → validate → SQLite (chronos_recordings, status=pending)
2. PROCESS:  recordings → fetch transcript → Gemini cognitive cleaning → SQLite (chronos_events)
3. INDEX:    events → Gemini Embedding 2 (768-dim, multimodal text+audio) → Qdrant (with temporal metadata)
4. GRAPH:    events → entity extraction → NetworkX graph → pickle cache
5. SERVE:    Dash UI reads Qdrant for search, SQLite for provenance/details
```

### Temporal Metadata (Qdrant Payload Indexes)

Each event in Qdrant includes payload indexes for filtered search:

- `day_of_week` (0-6) — enables "What do I do on Mondays?" queries
- `hour_of_day` (0-23) — time-of-day patterns
- `timestamp` — ISO datetime for date range queries
- `category` — event type (work, meeting, personal, health, etc.)
- `sentiment` — AI-detected emotional tone (-1.0 to 1.0)
- `keywords` — extracted key terms
- `speaker` — speaker identification

### Gemini Processing

The engine (`src/chronos/engine.py`) uses a structured prompt with `{{RECORDING_DATE}}` placeholder to extract:

- Timeline events with start/end timestamps
- Category classification
- Sentiment analysis
- Key entities and concepts
- Raw transcript snippets with reasoning

Models used:

| Model                        | Purpose                          | Notes                                                                   |
| ---------------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| `gemini-3-flash-preview`     | Primary processing (cleaning)    | Fast, free on standard tier                                             |
| `gemini-3.1-pro-preview`     | Fallback for complex transcripts | Replaced `gemini-3-pro-preview` (shut down 2026-03-09)                  |
| `gemini-embedding-2-preview` | Multimodal embeddings (768-dim)  | Text + audio + image + video + PDF. MRL dims 128–3072.                  |
| OpenAI `gpt-5.4`             | RAG responses (Responses API)    | 1.05M context, 128K output, reasoning levels. Preferred for ask_chronos |

### Embedding Details

- **Model:** `gemini-embedding-2-preview` — fully multimodal (text, audio, image, video, PDF)
- **Dimensions:** 768 (via Matryoshka Representation Learning truncation from 3072 native)
- **L2 normalization:** Auto-applied when dim < 3072 for unit-length vectors
- **Multimodal fusion:** When audio is available (`ChronosRecording.local_audio_path`), text + audio are embedded together for richer vectors. Falls back to text-only if audio unavailable.
- **Audio limits:** WAV or MP3, ≤ 80 seconds
- **Task types:** `RETRIEVAL_DOCUMENT` (indexing), `RETRIEVAL_QUERY` (search), `QUESTION_ANSWERING` (RAG/ask_chronos)
- **Incompatible with `gemini-embedding-001`** — switching models requires `--reindex`

---

## Project Structure

```
PlaudBlender/
├── app_v2/                     # ← MAIN UI (Dash v2)
│   ├── main.py                 # Entry point: python -m app_v2.main
│   ├── layout.py               # 3-column layout (sidebar | content | detail)
│   ├── assets/style.css        # ~3950 lines of dark-theme CSS
│   ├── components/             # UI components
│   │   ├── sidebar.py          # Navigation sidebar (7 views)
│   │   ├── day_view.py         # Date-grouped event timeline
│   │   ├── search.py           # Semantic search + category/date filters
│   │   ├── graph.py            # Cytoscape knowledge graph
│   │   ├── stats.py            # Analytics dashboard (8 cards + charts)
│   │   ├── topics.py           # Category-grouped view
│   │   └── recording_detail.py # Recording detail panel + transcript viewer
│   ├── callbacks/              # Dash interactivity (23 callbacks)
│   │   ├── navigation.py       # Main nav + sync/settings views (~1550 lines)
│   │   ├── search.py           # Search + filter callbacks
│   │   ├── day_view.py         # Day view interactions
│   │   └── graph.py            # Graph layout + node click
│   └── services/
│       └── data_service.py     # Data access layer (~1250 lines)
│
├── scripts/                    # CLI tools
│   ├── chronos_pipeline.py     # Full pipeline runner (~688 lines)
│   ├── launch_app.py           # App launcher (debug=False)
│   ├── fix_recordings.py       # Diagnose + repair stuck recordings
│   ├── index_unindexed.py      # Batch index events to Qdrant
│   ├── mcp_server.py           # Production MCP server (11 tools, FastMCP)
│   └── plaud_auth_utils.py     # OAuth diagnostics
│
├── src/                        # Core modules
│   ├── config.py               # Single .env loader
│   ├── plaud_oauth.py          # OAuth 2.0 client
│   ├── plaud_client.py         # Plaud API wrapper
│   ├── plaud_device.py         # Device management
│   ├── plaud_workflow.py       # Workflow API
│   ├── chronos/                # Chronos engine
│   │   ├── engine.py           # Gemini processing + prompt template
│   │   ├── transcript_processor.py  # Recording → events pipeline
│   │   ├── qdrant_client.py    # Native Qdrant with temporal indexes
│   │   ├── embedding_service.py     # Gemini Embedding 2 — multimodal (text+audio), L2 norm
│   │   ├── ingest_service.py   # Plaud API → SQLite ingestion
│   │   ├── analytics.py        # Day-of-week patterns, heatmaps
│   │   ├── graph_service.py    # Entity extraction, NetworkX graphs
│   │   ├── graph_rag.py        # Graph-enhanced RAG with community detection
│   │   ├── openai_service.py   # OpenAI Responses API wrapper (GPT-5.4 RAG)
│   │   └── genai_helpers.py    # Gemini client + model selection
│   ├── database/               # SQLAlchemy engine + models
│   │   ├── engine.py           # SessionLocal, init_db
│   │   ├── models.py           # ChronosRecording, ChronosEvent
│   │   └── chronos_repository.py    # CRUD operations
│   └── models/
│       └── chronos_schemas.py  # Pydantic: ChronosEvent, TemporalFilter, etc.
│
├── tests/                      # 90 tests
│   ├── test_database_models.py
│   ├── test_device_integration.py
│   ├── test_processing_engine.py
│   ├── test_processing_indexer.py
│   ├── test_schemas.py
│   ├── test_services_smoke.py
│   └── test_ui_smoke.py
│
├── data/                       # Local data (gitignored)
│   ├── brain.db                # SQLite database
│   ├── audio/                  # Cached audio files
│   ├── cache/graphs/           # NetworkX pickle files
│   └── processed/              # Processing artifacts
│
├── docs/
│   ├── PROJECT_GUIDE.md        # ← You are here
│   ├── chronos-mvp.md          # Chronos system architecture spec
│   └── PlaudDocs/              # Plaud API documentation
│
└── archive/                    # Retired code (do NOT import)
```

---

## Environment Variables

### Required

```bash
# Plaud OAuth
PLAUD_CLIENT_ID=
PLAUD_CLIENT_SECRET=
PLAUD_REDIRECT_URI=http://localhost:8050/auth/plaud/callback

# Gemini AI
GEMINI_API_KEY=
```

### Processing Models (optional, have defaults)

```bash
CHRONOS_CLEANING_MODEL=gemini-3-flash-preview          # Cognitive cleaning
CHRONOS_EMBEDDING_MODEL=gemini-embedding-2-preview      # Multimodal embeddings
CHRONOS_EMBEDDING_DIM=768                               # MRL dim (128–3072)
CHRONOS_ANALYST_MODEL=gemini-3.1-pro-preview            # Deep analysis fallback
```

> **⚠️ Changing `CHRONOS_EMBEDDING_MODEL` or `CHRONOS_EMBEDDING_DIM` requires re-indexing:**
>
> ```bash
> python scripts/chronos_pipeline.py --reindex
> ```
>
> Different embedding models produce incompatible vector spaces.

### Qdrant (optional, defaults shown)

```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=chronos_events
```

### Optional Integrations

```bash
NOTION_TOKEN=               # Notion sync
NOTION_DATABASE_ID=
OPENAI_API_KEY=             # RAG responses via Responses API (preferred over Gemini)
OPENAI_MODEL=gpt-5.4        # gpt-5.4, gpt-5.4-pro, gpt-5-mini, gpt-5-nano, etc.
OPENAI_TEMPERATURE=0.7      # 0.0 = deterministic, 2.0 = creative
```

---

## UI Overview (Dash v2)

The UI runs on **Dash 4.0** at `http://localhost:8050` with a 3-column layout:

### Views (7 total)

| View         | Purpose                                                                          |
| ------------ | -------------------------------------------------------------------------------- |
| **Days**     | Date-grouped event timeline, click recording → detail panel                      |
| **Topics**   | Events grouped by category (work, meeting, personal, etc.)                       |
| **Search**   | Semantic search with category/date filters, results ranked by score              |
| **Graph**    | Interactive Cytoscape knowledge graph, 6 layout algorithms, node click → details |
| **Stats**    | 8 stat cards, sentiment chart, productivity insights, pipeline health            |
| **Sync**     | Pipeline dashboard: status counts, Full Sync button, Reset Stuck button          |
| **Settings** | Real connectivity checks for Plaud, Gemini, Qdrant with latency                  |

### Key Features

- **23 callbacks** registered for full interactivity
- **Auto-refresh** every 60 seconds on Days view
- **Recording detail panel** with transcript viewer (collapsible, word/char count)
- **Search filters** combine with semantic search (category multi-select + date range)
- **Dark theme** with consistent color palette

---

## Implementation Status

### ✅ Tier 1 — Core Functionality (COMPLETE)

- [x] Fix broken recordings — `scripts/fix_recordings.py` + UI reset button
- [x] Full-pipeline sync from UI — ingest → process → index with status dashboard
- [x] Search filters — category + date range, combine with semantic search
- [x] Transcript viewer — collapsible in recording detail

### ✅ Tier 2 — Advanced Features (COMPLETE)

- [x] Knowledge Graph — Cytoscape with 10 entity type styles, 6 layouts, node details
- [x] Analytics Stats — 8 stat cards, sentiment chart, productivity insights
- [x] Real Settings — Plaud/Gemini/Qdrant connectivity checks with latency

### Data Status (March 10, 2026)

| Metric             | Value                        |
| ------------------ | ---------------------------- |
| Events in SQLite   | 1606                         |
| Embedding model    | `gemini-embedding-2-preview` |
| Embedding dim      | 768 (MRL, L2-normalized)     |
| Multimodal support | Text + audio (WAV/MP3 ≤80s)  |
| Tests passing      | 90/90                        |

### ✅ Tier 3 — Automation & Integration (COMPLETE)

- [x] T3.1: Webhook server for auto-processing new recordings (`src/plaud_webhook_server.py`)
- [x] T3.2: USB watcher for local Plaud imports (`src/plaud_usb_watcher.py`, `scripts/import_local_audio.py`)
- [x] T3.3: MCP server — 11 tools via FastMCP (`scripts/mcp_server.py`)
- [x] T3.4: Auto-sync orchestrator — webhook + USB watcher (`scripts/auto_sync.py`)

### ✅ Gemini Embedding 2 Migration (COMPLETE)

- [x] Upgraded from `gemini-embedding-001` to `gemini-embedding-2-preview`
- [x] Multimodal embedding support (text + audio fusion)
- [x] L2 normalization for MRL sub-3072 dimensions
- [x] Explicit task types on all embedding calls
- [x] `--reindex` CLI flag for model migration
- [x] Fixed `delete_by_recording_id()` FilterSelector bug
- [x] Updated `gemini-3-pro-preview` → `gemini-3.1-pro-preview` (old model shut down 2026-03-09)

---

## Developer Notes

### Running the app

```bash
# Option 1: Direct launch (recommended)
python scripts/launch_app.py

# Option 2: Module launch (with debug mode)
PYTHONPATH=. python -m app_v2.main

# Option 3: Background (production-ish)
nohup python scripts/launch_app.py > /tmp/chronos.log 2>&1 &
```

### Running the pipeline

```bash
# Full pipeline (all phases)
python scripts/chronos_pipeline.py --full

# Individual phases
python scripts/chronos_pipeline.py --ingest           # Download from Plaud
python scripts/chronos_pipeline.py --process           # Gemini processing
python scripts/chronos_pipeline.py --index             # Embed + Qdrant upsert
python scripts/chronos_pipeline.py --graph             # Knowledge graph
python scripts/chronos_pipeline.py --process --index   # Process + index combo
python scripts/chronos_pipeline.py --reindex            # Re-embed all events (model migration)
```

### Coding Rules

1. **Environment:** All secrets from `.env` via `python-dotenv`. Never hardcode.
2. **Imports:** Use `from src.X import Y` pattern. All `src/` subdirs have `__init__.py`.
3. **Schemas:** Validate data with Pydantic (`src/models/chronos_schemas.py`).
4. **Database field:** `ChronosRecording.processing_status` (NOT `.status`).
5. **Tests:** Run `pytest tests/` before any commit. Currently 90 tests.

### Don't

- Don't import from `archive/` — that's retired code
- Don't reference Pinecone — we're 100% Qdrant now
- Don't use `gemini-embedding-001` — we're on `gemini-embedding-2-preview` (multimodal)
- Don't use `gemini-3-pro-preview` — shut down 2026-03-09; use `gemini-3.1-pro-preview`
- Don't scatter `load_dotenv()` — use `src/config.py`
- Don't use `ChronosRecording.status` — the field is `processing_status`

### Key Data Service Methods (`app_v2/services/data_service.py`)

| Method                                  | Purpose                                         |
| --------------------------------------- | ----------------------------------------------- |
| `get_recordings()`                      | Load all recordings with events from Qdrant     |
| `search(query, categories, date_range)` | Semantic search with filters                    |
| `get_stats()`                           | Stats with sentiment, insights, pipeline health |
| `get_graph_data()`                      | Load NetworkX graph → Cytoscape elements        |
| `get_transcript(recording_id)`          | Fetch raw transcript from SQLite                |
| `get_recording_db_stats()`              | Pipeline status counts                          |
| `reset_stuck_recordings()`              | Reset processing → pending                      |

### Key Schema Fields (`src/models/chronos_schemas.py`)

- `ChronosEvent` requires: `event_id`, `recording_id`, `start_ts`, `end_ts`, `day_of_week`, `hour_of_day`, `clean_text`, `category`, `sentiment`, `keywords`, `speaker`
- `ChronosEvent` optional: `raw_transcript_snippet`, `gemini_reasoning`
- `TemporalFilter` requires: `hours_of_day` (can be `None`)

---

## GitHub Issues

Tracked via [Issue #1: Chronos System Roadmap](https://github.com/Gunnarguy/PlaudBlender/issues/1)

| Issue | Title                       | Status    |
| ----- | --------------------------- | --------- |
| #1    | Master Roadmap              | Closed    |
| #2    | T2.3: Real Settings checks  | ✅ Closed |
| #3    | T2.2: Analytics Stats       | ✅ Closed |
| #4    | T2.1: Knowledge Graph       | ✅ Closed |
| #5    | T1.1: Fix broken recordings | ✅ Closed |
| #6    | T1.2: Full-pipeline sync    | ✅ Closed |
| #7    | T1.3: Search filters        | ✅ Closed |
| #8    | T1.4: Transcript viewer     | ✅ Closed |
