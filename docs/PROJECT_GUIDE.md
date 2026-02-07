# PlaudBlender — Complete Project Documentation

> **Single source of truth** for architecture, roadmap, implementation status, and next steps.
> 
> *Last updated: February 7, 2026*

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
- A **Dash v2 UI** (`app_v2/`) — recording-centric interface with 7 views and 14 callbacks
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
| Command | Purpose |
|---------|---------|
| `python scripts/chronos_pipeline.py --ingest` | Download recordings from Plaud API |
| `python scripts/chronos_pipeline.py --process` | Process pending through Gemini |
| `python scripts/chronos_pipeline.py --index` | Embed + index to Qdrant |
| `python scripts/chronos_pipeline.py --graph` | Build knowledge graph |
| `python scripts/fix_recordings.py` | Diagnose stuck/failed recordings |
| `python scripts/fix_recordings.py --fix` | Reset stuck recordings to pending |
| `python scripts/index_unindexed.py` | Index events missing from Qdrant |
| `python -m pytest tests/` | Run test suite (74 tests) |

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
- **Qdrant is the vector index** — local-first, 768-dim Gemini embeddings, temporal payload indexes
- **Pydantic enforces contracts** — validated schemas at all boundaries
- **100% local** — no cloud dependencies except Gemini API and Plaud API

---

## Data Flow & Pipeline

### Canonical pipeline (SQL-first)
```
1. INGEST:   Plaud API → validate → SQLite (chronos_recordings, status=pending)
2. PROCESS:  recordings → fetch transcript → Gemini cognitive cleaning → SQLite (chronos_events)
3. INDEX:    events → Gemini embeddings (768-dim) → Qdrant (with temporal metadata)
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
- `gemini-3-flash-preview` — primary processing (fast, cheap)
- `gemini-3-pro-preview` — fallback for complex transcripts
- `gemini-embedding-001` — 768-dimension embeddings

---

## Project Structure

```
PlaudBlender/
├── app_v2/                     # ← MAIN UI (Dash v2)
│   ├── main.py                 # Entry point: python -m app_v2.main
│   ├── layout.py               # 3-column layout (sidebar | content | detail)
│   ├── assets/style.css        # ~2700 lines of dark-theme CSS
│   ├── components/             # UI components
│   │   ├── sidebar.py          # Navigation sidebar (7 views)
│   │   ├── day_view.py         # Date-grouped event timeline
│   │   ├── search.py           # Semantic search + category/date filters
│   │   ├── graph.py            # Cytoscape knowledge graph
│   │   ├── stats.py            # Analytics dashboard (8 cards + charts)
│   │   ├── topics.py           # Category-grouped view
│   │   └── recording_detail.py # Recording detail panel + transcript viewer
│   ├── callbacks/              # Dash interactivity (14 callbacks)
│   │   ├── navigation.py       # Main nav + sync/settings views (~600 lines)
│   │   ├── search.py           # Search + filter callbacks
│   │   ├── day_view.py         # Day view interactions
│   │   └── graph.py            # Graph layout + node click
│   └── services/
│       └── data_service.py     # Data access layer (~1000 lines)
│
├── scripts/                    # CLI tools
│   ├── chronos_pipeline.py     # Full pipeline runner (~688 lines)
│   ├── launch_app.py           # App launcher (debug=False)
│   ├── fix_recordings.py       # Diagnose + repair stuck recordings
│   ├── index_unindexed.py      # Batch index events to Qdrant
│   ├── mcp_server.py           # MCP server (basic)
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
│   │   ├── embedding_service.py     # Gemini embedding batch processing
│   │   ├── ingest_service.py   # Plaud API → SQLite ingestion
│   │   ├── analytics.py        # Day-of-week patterns, heatmaps
│   │   ├── graph_service.py    # Entity extraction, NetworkX graphs
│   │   └── genai_helpers.py    # Gemini client + model selection
│   ├── database/               # SQLAlchemy engine + models
│   │   ├── engine.py           # SessionLocal, init_db
│   │   ├── models.py           # ChronosRecording, ChronosEvent
│   │   └── chronos_repository.py    # CRUD operations
│   └── models/
│       └── chronos_schemas.py  # Pydantic: ChronosEvent, TemporalFilter, etc.
│
├── tests/                      # 74 tests
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
PLAUD_REDIRECT_URI=http://localhost:8080/callback

# Gemini AI
GEMINI_API_KEY=
```

### Processing Models (optional, have defaults)
```bash
CHRONOS_CLEANING_MODEL=gemini-3-flash-preview
CHRONOS_EMBEDDING_MODEL=gemini-embedding-001
CHRONOS_ANALYST_MODEL=gemini-3-pro-preview
```

### Qdrant (optional, defaults shown)
```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=chronos_events
```

### Optional Integrations
```bash
NOTION_TOKEN=               # Notion sync
NOTION_DATABASE_ID=
OPENAI_API_KEY=             # MCP server
```

---

## UI Overview (Dash v2)

The UI runs on **Dash 4.0** at `http://localhost:8050` with a 3-column layout:

### Views (7 total)
| View | Purpose |
|------|---------|
| **Days** | Date-grouped event timeline, click recording → detail panel |
| **Topics** | Events grouped by category (work, meeting, personal, etc.) |
| **Search** | Semantic search with category/date filters, results ranked by score |
| **Graph** | Interactive Cytoscape knowledge graph, 6 layout algorithms, node click → details |
| **Stats** | 8 stat cards, sentiment chart, productivity insights, pipeline health |
| **Sync** | Pipeline dashboard: status counts, Full Sync button, Reset Stuck button |
| **Settings** | Real connectivity checks for Plaud, Gemini, Qdrant with latency |

### Key Features
- **14 callbacks** registered for full interactivity
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

### Data Status (Feb 7, 2026)
| Metric | Value |
|--------|-------|
| Recordings in DB | 35 |
| Completed | 27 |
| Failed (genuinely unfixable) | 7 |
| Pending | 1 |
| Events in SQLite | 469 |
| Events in Qdrant | 469 (100% indexed) |
| Tests passing | 74/74 |

### 🔲 Tier 3 — Automation & Integration (NOT STARTED)
- [ ] T3.1: Webhook server for auto-processing new recordings
- [ ] T3.2: USB watcher for local Plaud imports
- [ ] T3.3: MCP server upgrade — expose Chronos data to Claude/agents

---

## What's Next — Tier 3

### T3.1: Webhook Server
Auto-process new recordings when Plaud sends webhooks.
- `src/plaud_webhook_server.py` exists but is not wired
- Needs: signature verification, processing trigger, status updates

### T3.2: USB Watcher
Auto-import recordings when Plaud device is connected via USB.
- `src/plaud_usb_watcher.py` exists but is not wired
- Needs: file detection, transcript extraction, pipeline trigger

### T3.3: MCP Server Upgrade
Expose Chronos data to Claude and other AI agents.
- `scripts/mcp_server.py` exists (basic, OpenAI-only)
- Needs: Chronos search tools, recording listing, event detail tools

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
```

### Coding Rules
1. **Environment:** All secrets from `.env` via `python-dotenv`. Never hardcode.
2. **Imports:** Use `from src.X import Y` pattern. All `src/` subdirs have `__init__.py`.
3. **Schemas:** Validate data with Pydantic (`src/models/chronos_schemas.py`).
4. **Database field:** `ChronosRecording.processing_status` (NOT `.status`).
5. **Tests:** Run `pytest tests/` before any commit. Currently 74 tests.

### Don't
- Don't import from `archive/` — that's retired code
- Don't reference Pinecone — we're 100% Qdrant now
- Don't scatter `load_dotenv()` — use `src/config.py`
- Don't use `ChronosRecording.status` — the field is `processing_status`

### Key Data Service Methods (`app_v2/services/data_service.py`)
| Method | Purpose |
|--------|---------|
| `get_recordings()` | Load all recordings with events from Qdrant |
| `search(query, categories, date_range)` | Semantic search with filters |
| `get_stats()` | Stats with sentiment, insights, pipeline health |
| `get_graph_data()` | Load NetworkX graph → Cytoscape elements |
| `get_transcript(recording_id)` | Fetch raw transcript from SQLite |
| `get_recording_db_stats()` | Pipeline status counts |
| `reset_stuck_recordings()` | Reset processing → pending |

### Key Schema Fields (`src/models/chronos_schemas.py`)
- `ChronosEvent` requires: `event_id`, `recording_id`, `start_ts`, `end_ts`, `day_of_week`, `hour_of_day`, `clean_text`, `category`, `sentiment`, `keywords`, `speaker`
- `ChronosEvent` optional: `raw_transcript_snippet`, `gemini_reasoning`
- `TemporalFilter` requires: `hours_of_day` (can be `None`)

---

## GitHub Issues

Tracked via [Issue #1: Chronos System Roadmap](https://github.com/Gunnarguy/PlaudBlender/issues/1)

| Issue | Title | Status |
|-------|-------|--------|
| #1 | Master Roadmap | Open (Tier 3 remaining) |
| #2 | T2.3: Real Settings checks | ✅ Closed |
| #3 | T2.2: Analytics Stats | ✅ Closed |
| #4 | T2.1: Knowledge Graph | ✅ Closed |
| #5 | T1.1: Fix broken recordings | ✅ Closed |
| #6 | T1.2: Full-pipeline sync | ✅ Closed |
| #7 | T1.3: Search filters | ✅ Closed |
| #8 | T1.4: Transcript viewer | ✅ Closed |
