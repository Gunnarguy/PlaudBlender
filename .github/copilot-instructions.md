# PlaudBlender — AI Agent Instructions

> **This file is auto-loaded into every Copilot conversation.** It tells you how to navigate and work in this project.

## Quick Orientation

| What            | Where                                                                          |
| --------------- | ------------------------------------------------------------------------------ |
| **Full docs**   | `docs/PROJECT_GUIDE.md` — architecture, structure, and roadmap                 |
| **MVP spec**    | `docs/chronos-mvp.md` — complete Chronos system architecture                   |
| **Entry point** | `streamlit run chronos_app.py` — launches the Streamlit UI                     |
| **Pipeline**    | `python scripts/chronos_pipeline.py --full` — ingest → process → index → graph |
| **Tests**       | `python -m pytest tests/` — 57 tests, run before committing                    |

## What This Project Does

**Chronos** transforms **Plaud voice recordings** into a **searchable knowledge timeline**:

- Fetches transcripts from Plaud API (OAuth) and stores locally
- Processes through Gemini AI for cognitive cleaning (removes filler, extracts events)
- Indexes to Qdrant with temporal metadata (day-of-week, hour, category)
- Provides Streamlit UI with semantic search and temporal filtering
- Full Plaud integration: devices, workflows, webhooks

## UI Pages (Simplified)

| Page            | Purpose                                               |
| --------------- | ----------------------------------------------------- |
| **🏠 Home**     | Quick status, metrics, one-click actions              |
| **🔍 Search**   | Semantic + temporal search with filters               |
| **📚 Library**  | Browse all recordings, view events, manage processing |
| **⚡ Pipeline** | 3-step Fetch → Process → Index workflow               |
| **📱 Plaud**    | Device management, workflows, webhooks (tabs)         |
| **⚙️ Settings** | Configuration, diagnostics, logs                      |

## Project Structure

```
chronos_app.py          → Main Streamlit UI (6 pages, single navigation)
plaud_setup.py          → Setup wizard + OAuth
scripts/                → CLI tools (chronos_pipeline.py, mcp_server.py)
src/chronos/            → Core engine (ingest, process, embed, search)
src/plaud_*.py          → Plaud API clients
src/database/           → SQLite models & repositories
src/models/             → Pydantic schemas
gui/components/         → Reusable UI panels (devices, workflows, webhooks)
tests/                  → Pytest suite (57 tests)
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
