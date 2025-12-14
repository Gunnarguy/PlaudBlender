# PlaudBlender — AI Agent Instructions

> **This file is auto-loaded into every Copilot conversation.** It tells you how to navigate and work in this project.

## Quick Orientation

| What | Where |
|------|-------|
| **Full docs** | `docs/PROJECT_GUIDE.md` — read this first for architecture, structure, and roadmap |
| **Task tracker** | `docs/audit-checklist.md` — live checklist of what's done and what's next |
| **Pinecone rules** | `docs/pinecone-integration-playbook.md` — metadata schema, namespaces, action items |
| **Entry point** | `python gui.py` — launches the Tkinter GUI |
| **Tests** | `python -m pytest tests/` — 57 tests, run before committing |

## What This Project Does

PlaudBlender transforms **Plaud voice recordings** into a **searchable knowledge base**:
- Fetches transcripts from Plaud API (OAuth)
- Stores in SQLite (source of truth) and Pinecone (vector search)
- Provides a GUI for search, visualization, and management
- Optional: Notion sync, OpenAI chat, MCP server

## Project Structure (Simplified)

```
gui.py                  → Entry point (delegates to gui/app.py)
plaud_setup.py          → Setup wizard + OAuth
scripts/                → CLI tools (sync_to_pinecone.py, process_pending.py, etc.)
gui/                    → Modular GUI (views/, services/, components/)
src/                    → Core logic (plaud_client, pinecone_client, database/, processing/)
tests/                  → Pytest suite
docs/                   → PROJECT_GUIDE.md, audit-checklist.md, playbooks
archive/                → Retired code (don't import from here)
```

## User Philosophy

> _"Gunnar loves data, granularity, and depth—the ability to drill down and see what's happening under the hood."_

- **Expose metrics** (latency, read units, scores) in the UI
- **Add tooltips** explaining what every control does
- **Show provenance** — link vectors back to source recordings
- **Never hide information** that could help debug or understand behavior

## Coding Rules

1. **Environment:** All secrets from `.env` via `python-dotenv`. Never hardcode.
2. **Threading:** GUI stays responsive — all I/O in background threads via `gui/utils/async_tasks.py`
3. **Imports:** Use `from src.X import Y` pattern. All `src/` subdirs have `__init__.py`.
4. **Schemas:** Validate data with Pydantic (`src/models/schemas.py`)
5. **Metadata:** Use `src/models/vector_metadata.py::build_metadata()` when upserting to Pinecone
6. **Tests:** Run `pytest tests/` before any commit. Currently 57 tests.

## Before You Code

1. **Read `docs/PROJECT_GUIDE.md`** if you need full context
2. **Check `docs/audit-checklist.md`** to see current priorities
3. **Read `docs/pinecone-integration-playbook.md`** before touching Pinecone/search code
4. **Run tests** after any change: `python -m pytest tests/ -q`

## Key Services (gui/services/)

| Service | Purpose |
|---------|---------|
| `transcripts_service` | Fetch/process Plaud recordings via SQL |
| `pinecone_service` | Index/namespace management, vector ops |
| `search_service` | Semantic search with reranking |
| `embedding_service` | Configurable embeddings (Gemini, OpenAI, Pinecone) |
| `chat_service` | OpenAI Responses API integration |

## Don't

- Don't import from `archive/` — that's retired code
- Don't use `src/dual_store_processor.py` — it's archived
- Don't scatter `load_dotenv()` — use `src/config.py`
- Don't make UI calls from background threads — use `root.after()`
