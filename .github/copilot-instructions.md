# PlaudBlender — AI Agent Instructions

> **This file is auto-loaded into every Copilot conversation.** It tells you how to navigate and work in this project.

## Quick Orientation

| What            | Where                                                                          |
| --------------- | ------------------------------------------------------------------------------ |
| **Full docs**   | `docs/PROJECT_GUIDE.md` — architecture, structure, and roadmap                 |
| **MVP spec**    | `docs/chronos-mvp.md` — complete Chronos system architecture                   |
| **Entry point** | `python scripts/launch_app.py` — Dash v2 UI on port 8050                       |
| **Pipeline**    | `python scripts/chronos_pipeline.py --full` — ingest → process → index → graph |
| **Tests**       | `python -m pytest tests/` — 124 tests, run before committing                   |

## What This Project Does

**Chronos** transforms **Plaud voice recordings** into a **searchable knowledge timeline**:

- Fetches transcripts from Plaud API (OAuth) and stores locally
- Processes through Gemini AI for cognitive cleaning (removes filler, extracts events)
- Indexes to Qdrant with temporal metadata (day-of-week, hour, category)
- Provides **Dash UI with interactive Knowledge Graph** (Cytoscape)
- Full Plaud integration: devices, workflows, webhooks
- **MCP server** for OpenAI/ChatGPT tool access to your knowledge timeline
- **Auto-sync** via webhooks and USB watcher for real-time ingestion

## UI Layout (Dash v2 — `app_v2/`)

| View         | Purpose                                                                   |
| ------------ | ------------------------------------------------------------------------- |
| **Timeline** | Date-grouped event timeline, click recording → detail panel               |
| **Topics**   | Events grouped by category (work, meeting, personal, etc.)                |
| **Search**   | Semantic search with category/date filters + AI answers (GPT-5.4)         |
| **Graph**    | Interactive Cytoscape knowledge graph, 6 layouts, node click details      |
| **Stats**    | 8 stat cards, sentiment chart, productivity insights, API cost tracking   |
| **Notion**   | Notion uplink — OAuth, page filter, channel select, sync to Notion        |
| **Sync**     | Pipeline dashboard: status counts, Full Sync, Reset Stuck                 |
| **Settings** | Connectivity checks for Plaud, Gemini, Qdrant, model pricing in dropdowns |

**X-ray Activity Monitor** — Floating PiP panel showing plain-English telemetry. 12 source categories, filter tabs, incremental polling with sequence IDs. Events persist across page navigations.

## Project Structure

```
app_v2/                 → MAIN Dash v2 UI (50 callbacks)
  main.py               → Run with: python scripts/launch_app.py (+ 9 Flask routes)
  layout.py             → 3-column layout (sidebar | content | detail)
  assets/style.css      → ~5500 lines dark theme CSS
  assets/xray_pip.js    → X-ray Activity Monitor PiP panel (client-side JS)
  components/           → sidebar, day_view, search, graph, stats, topics, recording_detail, notion
  callbacks/            → navigation, search, day_view, graph, recording_detail, notion, xray
  services/data_service.py → Data access layer (~2150 lines)
  services/xray.py      → Telemetry ring buffer, xray_log(), seq IDs
scripts/                → CLI tools
  chronos_pipeline.py   → Full pipeline runner (~688 lines)
  mcp_server.py         → Production MCP server (11 tools, FastMCP)
  auto_sync.py          → Webhook + USB auto-sync orchestrator
  launch_app.py         → App launcher
  fix_recordings.py     → Diagnose + repair stuck recordings
  index_unindexed.py    → Batch index events to Qdrant
src/chronos/            → Core engine (ingest, process, embed, search, graph, cost tracking)
src/plaud_*.py          → Plaud API clients + webhook + USB watcher
src/notion_oauth.py     → Notion OAuth 2.0 client
src/notion_service.py   → Notion API service (page sync)
src/database/           → SQLAlchemy models & repositories
src/models/             → Pydantic schemas
tests/                  → Pytest suite (124 tests, 11 files)
docs/                   → PROJECT_GUIDE.md, chronos-mvp.md
```

## Key Services (src/chronos/)

| Service                | Purpose                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `ingest_service`       | Fetch recordings from Plaud, store metadata in SQLite              |
| `transcript_processor` | Process transcripts through Gemini AI                              |
| `qdrant_client`        | Native Qdrant client with temporal payload indexes                 |
| `embedding_service`    | Gemini embedding — multimodal (text+audio) via embedding-2-preview |
| `graph_service`        | Entity extraction and NetworkX graph building                      |
| `graph_rag`            | Graph-enhanced RAG with community detection + Gemini synthesis     |
| `openai_service`       | OpenAI Responses API — RAG queries via GPT-5.4                     |
| `cost_tracker`         | API cost tracking — session + historical, 12 models with pricing   |
| `notion_bridge`        | Notion integration bridge for page sync                            |

## Coding Rules

1. **Environment:** All secrets from `.env` via `python-dotenv`. Never hardcode.
2. **Imports:** Use `from src.X import Y` pattern. All `src/` subdirs have `__init__.py`.
3. **Schemas:** Validate data with Pydantic (`src/models/chronos_schemas.py`)
4. **Qdrant:** Use `src/chronos/qdrant_client.py` — native API with temporal indexes
5. **Tests:** Run `pytest tests/` before any commit.
6. **X-ray messages:** Use `xray_log(source, operation, message)` for telemetry. Messages must be **plain human English** — no dev jargon. Source is one of: `ingest`, `gemini`, `embed`, `qdrant`, `graph`, `search`, `data`, `nav`, `pipeline`, `detail`, `day`, `sync`.
7. **Cost tracking:** All API calls must use `track_usage(model, call_type, input_tokens, output_tokens)` from `src.chronos.cost_tracker`.

## User Philosophy

> _"Gunnar loves data, granularity, and depth—the ability to drill down and see what's happening under the hood."_

- **Expose metrics** (latency, scores) in the UI
- **Show command previews** before running pipelines
- **Progressive disclosure** — simple by default, advanced options collapsed
- **Never hide information** that could help debug or understand behavior

## Don't

- Don't import from `archive/` — that's retired code
- Don't reference Pinecone — we're 100% Qdrant now
- Don't use `gemini-embedding-001` — we're on `gemini-embedding-2-preview` (multimodal)
- Don't use `gemini-3-pro-preview` — it was shut down 2026-03-09; use `gemini-3.1-pro-preview`
- Don't scatter `load_dotenv()` — use `src/config.py`
- Don't use `ChronosRecording.status` — the field is `processing_status`
- Don't use `gpt-4o` or `gpt-4o-mini` — we're on `gpt-5.4` (OpenAI flagship, 1.05M context)
- Don't make billable API calls without `track_usage()` — every call must be cost-tracked
