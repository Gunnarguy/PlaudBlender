# PlaudBlender — Chronos Knowledge Timeline

Transform **Plaud voice recordings** into a **searchable, visual knowledge graph** with AI-powered cognitive processing.

## What It Does

1. **Ingests** transcripts from Plaud API (OAuth) or local audio files
2. **Processes** through Gemini AI — removes filler, extracts discrete events, sentiment, categories
3. **Indexes** to Qdrant vector DB with temporal metadata (day-of-week, hour, category)
4. **Visualizes** via interactive Dash UI with knowledge graph, timeline, semantic search
5. **Exposes** data via MCP server for ChatGPT/OpenAI tool access

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Gunnarguy/PlaudBlender.git
cd PlaudBlender
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env  # Add CHRONOS_GEMINI_API_KEY, PLAUD_CLIENT_ID, PLAUD_CLIENT_SECRET, etc.

# 3. Start Qdrant (Docker)
docker compose up -d

# 4. Authenticate with Plaud (one-time OAuth flow)
python plaud_setup.py

# 5. Run the pipeline
python scripts/chronos_pipeline.py --full

# 6. Launch the UI
python scripts/launch_app.py
# → http://localhost:8050
```

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Plaud API   │───▶│ Ingest       │───▶│ Gemini AI    │───▶│ Qdrant       │
│  (OAuth)     │    │ Service      │    │ Processing   │    │ Vector DB    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                          │                                        │
                    ┌─────▼──────┐                          ┌──────▼──────┐
                    │  SQLite    │                          │  Dash v2 UI │
                    │  brain.db  │                          │  port 8050  │
                    └────────────┘                          └─────────────┘
```

## UI Views

| View         | Description                                                                       |
| ------------ | --------------------------------------------------------------------------------- |
| **Timeline** | Date-grouped event timeline with horizontal strip, heat-map, recording detail     |
| **Topics**   | Events grouped by category (work, meeting, personal, health, etc.)                |
| **Search**   | Semantic vector search with category/date filters + AI answers (GPT-5.4)          |
| **Graph**    | Interactive Cytoscape knowledge graph — 6 layouts, node click details             |
| **Stats**    | 8 stat cards, sentiment trends, productivity insights, **API cost tracking**      |
| **Notion**   | Notion uplink — OAuth, page filter, channel select, sync recordings to Notion     |
| **Sync**     | Pipeline dashboard, Full Sync, Reset Stuck, Plaud workflow status monitoring      |
| **Settings** | 12-section config (34 params), .env save, live connectivity checks, model pricing |

### X-ray Activity Monitor

A floating **Picture-in-Picture panel** that shows what the app is doing in plain English:

- **12 source categories** — Plaud, AI, Embedding, Search DB, Knowledge Graph, Search, Data, Navigation, Pipeline, Recording, Day View, Sync
- **Filter tabs** — Pipeline, Search, Graph, Data, Errors
- **Incremental polling** — events accumulate across page navigations (up to 2,000)
- **Plain English messages** — e.g. "Found 12 matching moments — top match is 94% relevant"
- **Drag, resize, minimize** — stays on screen while you navigate

Powered by a server-side ring buffer with monotonic sequence IDs and a client-side JS panel that polls every 800ms.

## Project Structure

```
app_v2/                → Dash v2 UI (main application)
  main.py              → App entry point + 9 Flask API routes (OAuth, X-ray, costs)
  layout.py            → 3-column layout (sidebar | content | detail)
  assets/style.css     → Dark theme CSS (~5500 lines)
  assets/xray_pip.js   → X-ray Activity Monitor PiP panel (client-side JS)
  components/          → sidebar, day_view, search, graph, stats, topics, recording_detail, notion
  callbacks/           → navigation, search, day_view, graph, recording_detail, notion, xray
  services/            → data_service.py (~2150 lines), xray.py (telemetry ring buffer)

scripts/               → CLI tools
  chronos_pipeline.py  → Full pipeline: ingest → process → index → graph
  mcp_server.py        → Production MCP server (11 tools, FastMCP)
  auto_sync.py         → Webhook + USB auto-sync orchestrator
  launch_app.py        → App launcher

src/chronos/           → Core engine
  ingest_service.py    → Fetch recordings from Plaud, store in SQLite
  transcript_processor → Process transcripts through Gemini AI
  embedding_service.py → Gemini Embedding 2 multimodal (text+audio), L2 norm, MRL
  qdrant_client.py     → Native Qdrant client with temporal payload indexes
  graph_service.py     → Entity extraction and NetworkX graph building
  graph_rag.py         → Graph-enhanced RAG with community detection
  openai_service.py    → OpenAI Responses API wrapper (GPT-5.4 RAG)
  cost_tracker.py      → API usage tracking with Gemini/OpenAI pricing (~300 lines)
  notion_bridge.py     → Notion integration bridge (OAuth + page sync)
  pipeline_progress.py → Pipeline progress tracking

src/plaud_*.py         → Plaud API clients (OAuth, device, webhook, USB watcher, workflows)
src/notion_oauth.py    → Notion OAuth 2.0 client
src/notion_service.py  → Notion API service (page creation, sync)
src/database/          → SQLAlchemy models & repositories
src/models/            → Pydantic schemas (chronos_schemas.py)
tests/                 → 124 tests (pytest, 11 test files)
```

## MCP Server

The MCP server exposes 11 tools for ChatGPT/OpenAI integration:

```
ping, search_events, get_recording, list_recordings, get_timeline,
get_stats, get_topics, get_graph, run_pipeline, system_status, ask_chronos
```

Configure in your MCP client:

```json
{
  "mcpServers": {
    "chronos": {
      "command": "python",
      "args": ["-m", "scripts.mcp_server"],
      "cwd": "/path/to/PlaudBlender"
    }
  }
}
```

If you also want direct access to your Plaud account from the same MCP client,
Plaud's official MCP server can run alongside Chronos. This gives you two layers:

- `plaud` for raw Plaud account access: recent recordings, notes, transcripts, and account auth
- `chronos` for PlaudBlender's local timeline, semantic search, graph, and pipeline tools

Prerequisites for the official Plaud MCP:

- Node.js 20+
- A Plaud account

Recommended setup:

```bash
npx -y @plaud-ai/mcp@latest install
```

Repository helper commands:

```bash
./venv/bin/python scripts/plaud_mcp_doctor.py --status
./venv/bin/python scripts/plaud_mcp_doctor.py --login
```

If your MCP client is using this repository's `mcp.json`, the file now includes
both servers:

```json
{
  "mcpServers": {
    "chronos": {
      "command": "python",
      "args": ["-m", "scripts.mcp_server"],
      "cwd": "/path/to/PlaudBlender",
      "env": {
        "CHRONOS_GEMINI_API_KEY": "${CHRONOS_GEMINI_API_KEY}"
      }
    },
    "plaud": {
      "command": "npx",
      "args": ["-y", "@plaud-ai/mcp@latest"]
    }
  }
}
```

The official Plaud MCP uses OAuth in the client and stores its own local auth
state, so you do not need to inject a Plaud access token into `mcp.json`.

Typical workflow:

- Use `plaud` to browse or fetch a source transcript from Plaud
- Use `chronos` after PlaudBlender has ingested and indexed recordings locally
- Keep `python plaud_setup.py` for PlaudBlender's own ingest pipeline, because the
  app still talks to the Plaud API directly when syncing data into Chronos

## Key Technologies

- **Gemini AI** — `gemini-2.5-flash` (default processing + analysis), `gemini-3.1-pro-preview` (deeper paid fallback), `gemini-embedding-2` (multimodal embeddings)
- **Gemini Embedding 2** — Multimodal (text + audio + image + video + PDF), MRL dims 128–3072, L2-normalized below native dim
- **OpenAI GPT-5.4** — Flagship model for RAG responses (Responses API), 1.05M context, 128K output, reasoning levels
- **Qdrant** — Vector database with temporal metadata indexes
- **Dash + Cytoscape** — Interactive web UI with knowledge graph visualization
- **FastMCP** — Model Context Protocol server for AI tool integration (11 tools)
- **SQLAlchemy + SQLite** — Local metadata storage (`data/brain.db`)

## Environment Variables

| Variable                  | Required | Description                                                                          |
| ------------------------- | -------- | ------------------------------------------------------------------------------------ |
| `CHRONOS_GEMINI_API_KEY`  | Yes      | Dedicated Google Gemini API key for PlaudBlender                                     |
| `CHRONOS_ALLOW_SHARED_GEMINI_KEY` | No | Set to `1` only if you intentionally want Chronos to reuse `GEMINI_API_KEY`         |
| `GEMINI_API_KEY`          | No       | Shared Gemini key used only when `CHRONOS_ALLOW_SHARED_GEMINI_KEY=1`                 |
| `PLAUD_CLIENT_ID`         | Yes      | Plaud OAuth client ID                                                                |
| `PLAUD_CLIENT_SECRET`     | Yes      | Plaud OAuth client secret                                                            |
| `PLAUD_REDIRECT_URI`      | No       | OAuth callback (default: `http://localhost:8050/auth/plaud/callback`)                |
| `QDRANT_URL`              | No       | Qdrant URL (default: `http://localhost:6333`)                                        |
| `QDRANT_COLLECTION_NAME`  | No       | Collection name (default: `chronos_events`)                                          |
| `CHRONOS_PROCESSING_PROVIDER` | No   | Transcript extraction provider (default: `gemini`)                                   |
| `CHRONOS_CLEANING_MODEL`  | No       | Extraction model (default: `gemini-2.5-flash`)                                       |
| `CHRONOS_ANALYST_MODEL`   | No       | Ask Chronos / analysis model (default: `gemini-2.5-flash`)                           |
| `CHRONOS_EMBEDDING_MODEL` | No       | Embedding model (default: `gemini-embedding-2`)                                      |
| `CHRONOS_EMBEDDING_DIM`   | No       | Embedding dim (default: 768, range 128–3072)                                         |
| `GEMINI_BILLING_TIER`     | No       | Gemini pricing mode used for cost estimates (default: `free`)                        |
| `CHRONOS_OPENAI_ENABLED`  | No       | Hard opt-in for OpenAI usage; leave `0` to prevent quota spend even if a key exists   |
| `OPENAI_API_KEY`          | No       | OpenAI API key for paid fallback / Responses API RAG                                 |
| `OPENAI_MODEL`            | No       | OpenAI model (default: `gpt-5.4`)                                                    |
| `CHRONOS_LOCAL_LLM_ENABLED` | No      | Enable local Ollama/llama.cpp degraded helper tasks (default: `0`)                   |
| `CHRONOS_LOCAL_LLM_BASE_URL` | No     | Local sidecar URL, keep bound to localhost (default: `http://127.0.0.1:11434`)       |
| `CHRONOS_LOCAL_LLM_MODEL` | No       | Local model for short fallback tasks (default: `qwen2.5:0.5b`)                       |
| `CHRONOS_LOCAL_LLM_ALLOWED_TASKS` | No | Local task allowlist (default: `json_repair,entity_extract,classify,ask`)           |
| `NOTION_CLIENT_ID`        | No       | Notion OAuth client ID (for Notion uplink)                                           |
| `NOTION_CLIENT_SECRET`    | No       | Notion OAuth client secret                                                           |
| `NOTION_REDIRECT_URI`     | No       | Notion OAuth callback (default: `http://localhost:8000/api/v1/auth/notion/callback`) |

## Commands

```bash
# Full pipeline (ingest + process + index + graph)
python scripts/chronos_pipeline.py --full

# Individual pipeline stages
python scripts/chronos_pipeline.py --ingest
python scripts/chronos_pipeline.py --process
python scripts/chronos_pipeline.py --index
python scripts/chronos_pipeline.py --graph

# Backfill your entire Plaud account history into Chronos
python scripts/chronos_pipeline.py --ingest --all-history

# Re-embed all events (after changing embedding model/dim)
python scripts/chronos_pipeline.py --reindex

# Launch UI
python scripts/launch_app.py

# Run MCP server
python -m scripts.mcp_server

# Auto-sync (webhook + USB watcher)
python scripts/auto_sync.py

# Tests
python -m pytest tests/

# Diagnostics
python scripts/verify_status.py
```

## Pi Auto-Deploy

If Chronos is running on your Raspberry Pi, you can let the Pi poll GitHub and
apply new commits on its own without inbound webhooks or manual SSH deploys.

Enable the timer once on the Pi:

```bash
sudo systemctl enable --now chronos-auto-update.timer
```

What it does:

- Polls the checked-out Git branch on `origin` every minute
- Skips deploys if the Pi has local tracked changes or a diverged branch
- Runs `deploy/update-pi.sh` when GitHub has new commits
- Reuses the hardened update flow that reloads units, restarts services, waits for health, and verifies the stack

Useful checks:

```bash
systemctl status chronos-auto-update.timer
journalctl -u chronos-auto-update.service -f
tail -f logs/auto_update.log
```

## More Documentation

- [docs/chronos-mvp.md](docs/chronos-mvp.md) — Full system architecture
- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) — Complete project reference

## License

MIT
