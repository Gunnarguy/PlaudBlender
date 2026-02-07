# 🕰️ Chronos (PlaudBlender)

**Transform your Plaud voice recordings into a searchable, queryable knowledge timeline.**

Record with Plaud → Fetch transcripts → Clean with AI → Search semantically

---

## ⚡ Quick Start (3 Steps)

### 1. Install & Configure

```bash
# Clone and install
git clone https://github.com/Gunnarguy/PlaudBlender.git
cd PlaudBlender
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required for Plaud (get from platform.plaud.ai/developer)
PLAUD_CLIENT_ID=your_client_id
PLAUD_CLIENT_SECRET=your_client_secret
PLAUD_REDIRECT_URI=http://localhost:8080/callback

# Required for AI processing (get from ai.google.dev)
GEMINI_API_KEY=your_gemini_key

# Optional (defaults work for local)
QDRANT_URL=http://localhost:6333
```

### 2. Start Services

```bash
# Start Qdrant (vector database)
docker run -p 6333:6333 qdrant/qdrant

# Authenticate with Plaud (one-time)
python plaud_setup.py
```

### 3. Launch Chronos

```bash
# NEW: Glorious Dash UI with Knowledge Graph
python -m app.main

# Legacy Streamlit UI (deprecated)
# streamlit run archive/chronos_app_streamlit.py
```

Open http://localhost:8050 to explore your knowledge graph!

---

## 🎯 What It Does

```
📱 Plaud Recording (5-7 hours of voice notes)
       ↓
🔄 Fetch from Plaud API
       ↓
🧠 Gemini AI cleans transcripts → structured events
       ↓
📤 Index to Qdrant with temporal metadata
       ↓
🔍 Search: "What do I think about on Mondays?"
```

**Key Features:**

- 🔍 **Semantic Search** — Find events by meaning, not just keywords
- 📅 **Temporal Filters** — Query by day of week, hour, or date range
- 🏷️ **Categories** — Events tagged as work, personal, meeting, deep_work, etc.
- 📊 **Full Visibility** — See latency, scores, raw payloads (for power users)

---

## 📖 UI Pages

| Page            | What it does                                          |
| --------------- | ----------------------------------------------------- |
| **🏠 Home**     | Quick status, metrics, one-click actions              |
| **🔍 Search**   | Semantic + temporal search with filters               |
| **📚 Library**  | Browse all recordings, view events, manage processing |
| **⚡ Pipeline** | 3-step Fetch → Process → Index workflow               |
| **📱 Plaud**    | Device management, workflows, webhooks                |
| **⚙️ Settings** | Configuration, diagnostics, logs                      |

---

## 🔧 CLI Commands

```bash
# Full pipeline (fetch + process + index)
python scripts/chronos_pipeline.py --full --limit 25

# Individual steps
python scripts/chronos_pipeline.py --ingest --limit 50    # Fetch from Plaud
python scripts/chronos_pipeline.py --process --limit 10   # Clean with Gemini
python scripts/chronos_pipeline.py --index --limit 100    # Push to Qdrant

# Diagnostics
python scripts/chronos_pipeline.py --preflight           # Check Gemini models
python scripts/plaud_auth_utils.py --check-token         # Validate OAuth

# Tests
python -m pytest tests/ -q
```

---

## 📁 Project Structure

```
chronos_app.py          # Main UI (Streamlit)
plaud_setup.py          # OAuth setup wizard
scripts/
  chronos_pipeline.py   # CLI pipeline runner
  mcp_server.py         # MCP server for AI agents
src/
  chronos/              # Core engine (ingest, process, embed, search)
  plaud_*.py            # Plaud API clients
  database/             # SQLite models & repos
  models/               # Pydantic schemas
data/
  brain.db              # Local SQLite database
  raw/                  # Downloaded audio files
```

---

## 🔑 Environment Variables

| Variable              | Required | Description                         |
| --------------------- | -------- | ----------------------------------- |
| `PLAUD_CLIENT_ID`     | Yes      | From Plaud developer portal         |
| `PLAUD_CLIENT_SECRET` | Yes      | From Plaud developer portal         |
| `PLAUD_REDIRECT_URI`  | Yes      | `http://localhost:8080/callback`    |
| `GEMINI_API_KEY`      | Yes      | From ai.google.dev                  |
| `QDRANT_URL`          | No       | Defaults to `http://localhost:6333` |
| `QDRANT_API_KEY`      | No       | Only for Qdrant Cloud               |

---

## 🐛 Troubleshooting

**Qdrant won't connect:**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Plaud OAuth fails (400 Bad Request):**

- Check that `PLAUD_REDIRECT_URI` in `.env` **exactly** matches your Plaud app registration
- Run `python plaud_setup.py` to re-authenticate

**Gemini model not found:**

```bash
python scripts/chronos_pipeline.py --preflight
```

---

## 📚 More Documentation

- [docs/chronos-mvp.md](docs/chronos-mvp.md) — Full system architecture
- [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) — Complete project reference
- [SIMPLIFICATION.md](SIMPLIFICATION.md) — UI design decisions

---

## 🔐 OAuth & Security

- OAuth 2.0 tokens stored locally in `.plaud_tokens.json`
- Auto-refresh on expiry
- Your Plaud password is never stored

---

## MCP Server (for AI agents)

```bash
python -m scripts.mcp_server
```

Requires `GEMINI_API_KEY` in `.env`.

---

## License

MIT
