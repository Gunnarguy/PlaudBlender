# PlaudBlender — Chronos Knowledge Timeline & 3D Graph

Transform **Plaud Note voice recordings** into a highly structured, visual local knowledge base. This repository houses the AI-powered cognitive processing pipeline, the daily timeline Web UI, a sibling iOS companion client, and Model Context Protocol (MCP) integrations.

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

---

## ⚡ Quick Start: Running PlaudBlender Locally (Mac Setup)

PlaudBlender uses a Python virtual environment to manage dependencies. The launcher scripts automatically detect whether you have a `.venv` (created via `uv`) or `venv` (standard virtualenv) folder.

```bash
# 1. Clone & enter repository
git clone https://github.com/Gunnarguy/PlaudBlender.git
cd PlaudBlender

# 2. Initialize virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Setup environment overrides
cp .env.example .env

# 4. Perform Plaud OAuth setup
python plaud_setup.py

# 5. Initialize the database and run the pipeline
python scripts/chronos_pipeline.py --full

# 6. Start the Web UI & API Stack
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
* **Run MCP Server**: `python -m scripts.mcp_server`
* **Clean Database & Vectors**: `python scripts/db_cleanup.py`
* **Diagnostics**: `python scripts/diagnose_failures.py`

---

## 🍓 Raspberry Pi Headless Deployment

PlaudBlender is fully optimized to run on low-resource hardware like a **Raspberry Pi**:
* **WAL Mode & Pragmas**: Configured `synchronous=NORMAL` and WAL mode in `src/database/engine.py` to prevent database locks between UI readers and background pipeline writers.
* **Auto-Refresh Short-Circuiting**: Background Dash UI auto-refreshes bypass heavy DOM calculations and return `no_update` to prevent CPU thrashing.
* **Auto-Update Service**: A systemd timer (`chronos-auto-update.timer`) pulls from GitHub and runs `deploy/update-pi.sh` to update when you push commits.

---

## 📄 License

MIT
