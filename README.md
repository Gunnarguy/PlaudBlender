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
cp .env.example .env  # Add GEMINI_API_KEY, PLAUD_ACCESS_TOKEN, etc.

# 3. Start Qdrant (Docker)
docker compose up -d

# 4. Authenticate with Plaud (one-time)
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

| View         | Description                                                           |
| ------------ | --------------------------------------------------------------------- |
| **Days**     | Date-grouped event timeline with recording detail panel               |
| **Topics**   | Events grouped by category (work, meeting, personal, health, etc.)    |
| **Search**   | Semantic vector search with category and date filters                 |
| **Graph**    | Interactive Cytoscape knowledge graph — 6 layouts, node click details |
| **Stats**    | 8 stat cards, sentiment trends, productivity insights                 |
| **Sync**     | Pipeline dashboard with status counts, Full Sync, Reset Stuck         |
| **Settings** | Real-time connectivity checks for Plaud, Gemini, Qdrant               |

## Project Structure

```
app_v2/                → Dash v2 UI (main application)
  main.py              → App entry point (python scripts/launch_app.py)
  layout.py            → 3-column layout (sidebar | content | detail)
  assets/style.css     → Dark theme CSS
  components/          → sidebar, day_view, search, graph, stats, topics, recording_detail
  callbacks/           → navigation, search, day_view, graph
  services/            → data_service.py (data access layer)

scripts/               → CLI tools
  chronos_pipeline.py  → Full pipeline: ingest → process → index → graph
  mcp_server.py        → Production MCP server (11 tools, FastMCP)
  auto_sync.py         → Webhook + USB auto-sync orchestrator
  launch_app.py        → App launcher

src/chronos/           → Core engine
  ingest_service.py    → Fetch recordings from Plaud, store in SQLite
  transcript_processor → Process transcripts through Gemini AI
  embedding_service.py → Gemini embedding batch processing
  qdrant_client.py     → Native Qdrant client with temporal payload indexes
  graph_service.py     → Entity extraction and NetworkX graph building

src/plaud_*.py         → Plaud API clients (OAuth, device, webhook, USB watcher)
src/database/          → SQLAlchemy models & repositories
src/models/            → Pydantic schemas (chronos_schemas.py)
tests/                 → 90 tests (pytest)
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

## Key Technologies

- **Gemini AI** — gemini-3-flash-preview (processing), gemini-embedding-001 (embeddings, 768-dim)
- **Qdrant** — Vector database with temporal metadata indexes
- **Dash + Cytoscape** — Interactive web UI with knowledge graph visualization
- **FastMCP** — Model Context Protocol server for AI tool integration
- **SQLAlchemy + SQLite** — Local metadata storage (`data/brain.db`)

## Environment Variables

| Variable              | Required | Description                      |
| --------------------- | -------- | -------------------------------- |
| `GEMINI_API_KEY`      | Yes      | Google Gemini API key            |
| `PLAUD_ACCESS_TOKEN`  | Yes      | Plaud API access token           |
| `QDRANT_HOST`         | No       | Qdrant host (default: localhost) |
| `QDRANT_PORT`         | No       | Qdrant port (default: 6333)      |
| `PLAUD_REFRESH_TOKEN` | No       | For automatic token refresh      |
| `PLAUD_APP_ID`        | No       | Plaud OAuth app ID               |

## Commands

```bash
# Full pipeline (ingest + process + index + graph)
python scripts/chronos_pipeline.py --full

# Individual pipeline stages
python scripts/chronos_pipeline.py --ingest
python scripts/chronos_pipeline.py --process
python scripts/chronos_pipeline.py --index
python scripts/chronos_pipeline.py --graph

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

## More Documentation

- [docs/chronos-mvp.md](docs/chronos-mvp.md) — Full system architecture
- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) — Complete project reference

## License

MIT
