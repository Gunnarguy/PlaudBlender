# PlaudBlender — AI Agent Instructions

> **This file is auto-loaded into every Copilot conversation.** It tells you how to navigate and work in this project.

## Quick Orientation

| What            | Where                                                                          |
| --------------- | ------------------------------------------------------------------------------ |
| **Full docs**   | `docs/PROJECT_GUIDE.md` — architecture, structure, and roadmap                 |
| **MVP spec**    | `docs/chronos-mvp.md` — complete Chronos system architecture                   |
| **Entry point** | `python scripts/launch_app.py` — Dash v2 UI on port 8050                       |
| **Pipeline**    | `python scripts/chronos_pipeline.py --full` — ingest → process → index → graph |
| **Tests**       | `python -m pytest tests/` — 74 tests, run before committing                    |

## What This Project Does

**Chronos** transforms **Plaud voice recordings** into a **searchable knowledge timeline**:

- Fetches transcripts from Plaud API (OAuth) and stores locally
- Processes through Gemini AI for cognitive cleaning (removes filler, extracts events)
- Indexes to Qdrant with temporal metadata (day-of-week, hour, category)
- Provides **Dash UI with interactive Knowledge Graph** (Cytoscape)
- Full Plaud integration: devices, workflows, webhooks

## UI Layout (Dash v2 — `app_v2/`)

| View         | Purpose                                                              |
| ------------ | -------------------------------------------------------------------- |
| **Days**     | Date-grouped event timeline, click recording → detail panel          |
| **Topics**   | Events grouped by category (work, meeting, personal, etc.)           |
| **Search**   | Semantic search with category/date filters                           |
| **Graph**    | Interactive Cytoscape knowledge graph, 6 layouts, node click details |
| **Stats**    | 8 stat cards, sentiment chart, productivity insights                 |
| **Sync**     | Pipeline dashboard: status counts, Full Sync, Reset Stuck            |
| **Settings** | Real connectivity checks for Plaud, Gemini, Qdrant                   |

## Project Structure

```
app_v2/                 → MAIN Dash v2 UI (14 callbacks)
  main.py               → Run with: python scripts/launch_app.py
  layout.py             → 3-column layout (sidebar | content | detail)
  assets/style.css      → ~2700 lines dark theme CSS
  components/           → sidebar, day_view, search, graph, stats, topics, recording_detail
  callbacks/            → navigation, search, day_view, graph
  services/data_service.py → Data access layer (~1000 lines)
scripts/                → CLI tools
  chronos_pipeline.py   → Full pipeline runner (~688 lines)
  launch_app.py         → App launcher
  fix_recordings.py     → Diagnose + repair stuck recordings
  index_unindexed.py    → Batch index events to Qdrant
src/chronos/            → Core engine (ingest, process, embed, search, graph)
src/plaud_*.py          → Plaud API clients
src/database/           → SQLAlchemy models & repositories
src/models/             → Pydantic schemas
tests/                  → Pytest suite (74 tests)
docs/                   → PROJECT_GUIDE.md, chronos-mvp.md
```

## Key Services (src/chronos/)

| Service                | Purpose                                               |
| ---------------------- | ----------------------------------------------------- |
| `ingest_service`       | Fetch recordings from Plaud, store metadata in SQLite |
| `transcript_processor` | Process transcripts through Gemini AI                 |
| `qdrant_client`        | Native Qdrant client with temporal payload indexes    |
| `embedding_service`    | Gemini embedding batch processing                     |
| `graph_service`        | Entity extraction and NetworkX graph building         |

## Coding Rules

1. **Environment:** All secrets from `.env` via `python-dotenv`. Never hardcode.
2. **Imports:** Use `from src.X import Y` pattern. All `src/` subdirs have `__init__.py`.
3. **Schemas:** Validate data with Pydantic (`src/models/chronos_schemas.py`)
4. **Qdrant:** Use `src/chronos/qdrant_client.py` — native API with temporal indexes
5. **Tests:** Run `pytest tests/` before any commit.

## User Philosophy

> _"Gunnar loves data, granularity, and depth—the ability to drill down and see what's happening under the hood."_

- **Expose metrics** (latency, scores) in the UI
- **Show command previews** before running pipelines
- **Progressive disclosure** — simple by default, advanced options collapsed
- **Never hide information** that could help debug or understand behavior

## Don't

- Don't import from `archive/` — that's retired code
- Don't reference Pinecone — we're 100% Qdrant now
- Don't scatter `load_dotenv()` — use `src/config.py`
- Don't use `ChronosRecording.status` — the field is `processing_status`
