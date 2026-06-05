# PlaudBlender — Chronos Knowledge Timeline & 3D Graph

Public-safe defaults: this repository ships with placeholder configuration only.
Bring your own API keys, OAuth credentials, and endpoints via `.env` and local
overrides.

> **Status:** Developer/power-user project. PlaudBlender is not a polished consumer app yet. It is an open local-first system for experimenting with Plaud recordings, semantic search, knowledge graphs, MCP tools, and personal AI memory workflows.
>
> **Current vector store:** Qdrant is the primary vector database. Older Pinecone references in historical docs are legacy migration history only.
>
> **Notion support:** Optional. Notion exists because early PlaudBlender workflows stored Plaud transcripts in Notion before Chronos became the local source-of-truth system. The Notion bridge can import, dedupe, match, and optionally sync enriched metadata back to Notion.

Transform **Plaud Note voice recordings** into a structured, searchable, local-first knowledge base. PlaudBlender includes an AI-powered processing pipeline, daily timeline UI, Qdrant vector search, graph visualization, MCP integrations, optional Notion import/sync, and a sibling iOS companion client.

### ✨ Key Capabilities (What you get)
* **Ask Your Memory (AI RAG)**: Ask natural language questions like *"What did I do in my meeting last Thursday?"* or *"What was that idea I brainstormed during my commute?"* and get synthesized answers with precise timeline citations.
* **3D Knowledge Graph**: Browse your voice logs as an interactive 3D connection map (arrange by lanes, concentrics, spiral timelines, or physics forces).
* **Observe Under the Hood**: Real-time token expense calculators, session cost stats, and a live WebSocket telemetry monitor (X-Ray).
* **MCP Server Included**: Plug your local timeline database directly into AI tools like Cursor or Claude Desktop to let them query your memories.
* **Headless Pi Server**: Optimized to run on low-resource hardware like a Raspberry Pi and connect securely from your phone via Tailscale VPN.


---

## 📐 Unified Architecture Overview

```
                                ┌─────────────────────────┐
                                │   Plaud Note / Cloud    │
                                └────────────┬────────────┘
                                             │ (Plaud API / OAuth)
                                             ▼
                                ┌─────────────────────────┐
                                │      PlaudBlender       │
                                │   (Ingest & Pipeline)   │
                                └──────┬────────────┬─────┘
                                       │            │
            (SQLite: local metadata)   ▼            ▼   (Qdrant: semantic vectors)
                         ┌──────────────┐          ┌──────────────┐
                         │   brain.db   │          │chronos_events│
                         └──────────────┘          └──────────────┘
                                       │            │
                                       ├────────────┤
                                       ▼            ▼
                        ┌──────────────────────────────┐
                        │    Chronos UI (Dash port)    │◀───┐
                        └──────────────────────────────┘    │
                                       ▲                    │ (REST API)
                  (MCP Stdio)          │                    │
                        ┌──────────────┴───────┐   ┌────────┴─────────────┐
                        │  Chronos MCP Server  │   │  PlaudBlenderiOS    │
                        │ (11 Tools for LLMs)  │   │ (Swift UI iOS Client)│
                        └──────────────────────┘   └──────────────────────┘
```

---

## 🧠 Core Ecosystem Components

1. **PlaudBlender (Backend Pipeline & Dash Web UI)**:
   * **Ingestion**: Fetches voice recordings and transcripts directly from the Plaud Note API using secure OAuth authentication.
   * **Processing**: Leverages Gemini AI to filter conversational noise, extract structured categories (clinical, personal, work, technical), track sentiment, and identify discrete event nodes.
   * **Storage**: Persists metadata in SQLite (`data/brain.db`) and indexes dense vectors in Qdrant.
   * **Dash UI (Port 8050)**: A dark-mode dashboard with search panels, chronological views, stats tickers, and an interactive 3D Knowledge Graph.

2. **PlaudBlenderiOS (Swift Companion App)**:
   * A native SwiftUI client displaying your daily timeline, category breakdowns, and a full-screen interactive **3D Knowledge Graph** optimized for native touch gestures (rotate, pinch-to-zoom, tap-to-select).

3. **Chronos MCP Server (FastMCP)**:
   * Exposes your memory timeline as tools to Model Context Protocol (MCP) clients. Connect your Plaud logs directly into Claude Desktop or Cursor so your LLMs can query your memory database (e.g., *"What did I do in my clinical rounds last Tuesday?"*).

4. **Optional Notion Bridge (Import/Sync Layer)**:
   * Provides migration pathways and ongoing sync capabilities for Notion-based Plaud workflows, enabling import, de-duplication, and metadata synchronization back to your databases.

---

## ⚡ Quick Start: Running PlaudBlender Locally (Mac Setup)

PlaudBlender uses a Python virtual environment to manage dependencies. The launcher scripts automatically detect whether you have a `.venv` (created via `uv`) or `venv` (standard virtualenv) folder.

```bash
# 1. Clone & enter repository
git clone <your-fork-or-clone-url>
cd PlaudBlender

# 2. Initialize virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Setup environment overrides
cp .env.example .env

# 4. Start Qdrant first (required for pipeline vector storage)
docker compose up -d

# 5. Perform Plaud OAuth setup
python plaud_setup.py

# 6. Initialize the database and run the pipeline
python scripts/chronos_pipeline.py --full

# 7. Start the Web UI & API Stack
./chronos start
# Or start UI directly: ./start_chronos.sh --ui-only
# Or start API directly: ./serve.sh
```
* **Dash Web UI**: `http://localhost:8050`
* **FastAPI Backend**: `http://localhost:8000/docs`

---

## 🧹 Database & Vector Store Cleanup

To keep your workspace clean and fast, use the unified cleanup utility. This removes things that don't matter, including legacy tables and failed/unusable records:

```bash
# Run the database cleanup script
.venv/bin/python scripts/db_cleanup.py
```

**What this script does:**
1. **Purges Legacy Tables**: Drops and clears old legacy `recordings` and `segments` tables, keeping only the new `chronos_*` tables.
2. **Purges Failed Runs**: Deletes any recordings with a `failed` processing status (cascading automatically to remove orphaned events).
3. **Vacuums SQLite**: Executes `VACUUM` on SQLite to reclaim unused disk storage space and rebuild the database file.
4. **Purges Vector Orphans**: Queries Qdrant and deletes orphaned points that no longer have corresponding entries in SQLite.

---

## 🕹️ Next-Gen 3D Graph Visualization Modes

PlaudBlender structures your complex memory network into legible 3D arrangements instead of traditional overlapping "hairballs":

* **Lanes (Category Columns)**: Stacks related topic nodes in vertical pillars arranged in a ring on the X-Z plane by category. Ideal for categorizing different parts of your day.
* **Levels (Hierarchical Layers)**: Organizes nodes into flat horizontal layers based on abstraction (Category Hubs top, Topics middle, Entities bottom).
* **Orbit (Concentric Shells)**: Places category hubs in a central cluster, with topics orbiting them in concentric outer shells based on how recently they were captured.
* **Timeline (Chronological Helix)**: Arranges all topics and categories in a 3D spiral climbing up the Y-axis. Vertical height maps directly to chronological time progression.
* **Force (Standard Physics)**: A classic free-form dynamic force-directed simulation.

*Note: In all structured layouts, physics simulation forces are automatically paused (`cooldownTicks(0)`) to lock positions instantly, saving battery on mobile devices.*

---

## ⚙️ Commands Reference

* **Start All Services**: `./chronos start` (Starts Qdrant, API, Web UI, auto-sync webhook, and ngrok)
* **Stop All Services**: `./chronos stop`
* **Check Service Status**: `./chronos status`
* **Run Pipeline**: `./chronos pipeline` or `python scripts/chronos_pipeline.py --full`
* **Sync Recordings Only**: `./chronos sync`
* **Run Chronos MCP Server**: `python -m scripts.mcp_server` (exposes your local timeline as tools to AI agents)
* **Verify Official Plaud MCP**: `python scripts/plaud_mcp_doctor.py --status` (inspects/drives the official Plaud Node MCP server)
* **Clean Database & Vectors**: `python scripts/db_cleanup.py`

* **Diagnostics**: `python scripts/diagnose_failures.py`
* **Audit iOS Backup Drift**: `python scripts/ios_discrepancy_audit.py --backup-ios-root ../backups/<your-ios-backup-folder>/PlaudBlenderiOS`

---

## 🍓 Raspberry Pi Headless Deployment

PlaudBlender is fully optimized and pre-configured to run on low-resource hardware like a **Raspberry Pi**:
* **Bootstrap Script**: Run `deploy/bootstrap-pi.sh` to install packages, setup venv, pull Qdrant, and register systemd services.
* **Tailscale & Remote Access**: Run `deploy/pi-remote-access.sh` to install/configure Tailscale (mesh VPN) and VNC for secure, remote set-and-forget home server access.
* **WAL Mode & Pragmas**: Configured `synchronous=NORMAL` and WAL mode in `src/database/engine.py` to prevent database locks between UI readers and background pipeline writers.
* **Auto-Refresh Short-Circuiting**: Background Dash UI auto-refreshes bypass heavy DOM calculations and return `no_update` to prevent CPU thrashing.
* **Auto-Update Service**: A systemd timer (`chronos-auto-update.timer`) pulls from GitHub and runs `deploy/update-pi.sh` to update when you push commits.


---

## 🦙 Local-First (Free & Offline) Mode via Ollama

PlaudBlender includes native support for running completely offline and free of cloud quotas by routing AI tasks to a local **Ollama** or `llama.cpp` instance:
* **Fully Local Processing**: Set `CHRONOS_PROCESSING_PROVIDER=local` and `CHRONOS_EMBEDDING_MODEL=nomic-embed-text` in your `.env`.
* **Configurable Sidecar**: Enable local models for specific tasks (like classification, JSON repair, and Ask Chronos) by setting `CHRONOS_LOCAL_LLM_ENABLED=1`.
* **Recommended Models**: We recommend `nomic-embed-text` for embeddings and a fast, lightweight model (e.g., `qwen2.5:0.5b`, `llama3.2`) for local reasoning tasks.
* **Extensible & Customizable**: Because local routing uses standard Ollama endpoints, you can easily swap in and experiment with any other models (like larger Qwen, Llama, or Mistral weights if running on a Raspberry Pi 5 or local machine) simply by updating your `.env`.

---

## 🔌 Chronos MCP Server (Model Context Protocol)

Expose your Plaud recordings and timeline memory as tools to any AI client or agent. PlaudBlender includes a standard-compliant stdio MCP server (built with `FastMCP`) that can be integrated into **any MCP host** (including custom agent scripts, command-line tools, other IDEs, or desktop clients).

### Running the Server Standalone
You can start the MCP server locally over stdio transport:
```bash
python -m scripts.mcp_server
```

### Integration Examples

#### 1. Claude Desktop Setup
Open your Claude Desktop config file:
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the following to the `mcpServers` object (adjust the command path to your virtualenv's Python, and `cwd` to your absolute repo path):

```json
{
  "mcpServers": {
    "chronos-mcp": {
      "command": "/Users/your-username/Documents/GitHub/PlaudBlender/.venv/bin/python",
      "args": ["-m", "scripts.mcp_server"],
      "cwd": "/Users/your-username/Documents/GitHub/PlaudBlender",
      "env": {
        "CHRONOS_GEMINI_API_KEY": "your_gemini_api_key_here"
      }
    }
  }
}
```

#### 2. Cursor Setup
1. Go to **Settings > Features > MCP**.
2. Click **+ Add New MCP Server**.
3. Configure:
   * **Name:** `chronos`
   * **Type:** `command`
   * **Command:** `/path/to/PlaudBlender/.venv/bin/python -m scripts.mcp_server`

Now any connected LLM can call tools like `search_events`, `get_timeline`, and `ask_chronos` to query your voice logs!

---

## Current Status

PlaudBlender is active and experimental.

It is suitable for:
- builders
- power users
- Plaud users comfortable with local tooling
- people experimenting with personal AI memory, semantic search, graphs, or MCP

It is not yet:
- a polished consumer app
- a hosted service
- a one-click installer
- guaranteed to work with every Plaud account/workflow without configuration

---

## 📄 License

MIT
