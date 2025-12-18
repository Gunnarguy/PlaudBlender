# 🧠 PlaudBlender

**Transform your Plaud voice recordings into a searchable, visual knowledge graph.**

**Canonical docs live in `docs/PROJECT_GUIDE.md`** (this README is a quick-start + pointers only).

Connect directly to Plaud's API to fetch your transcripts, process them with AI to extract themes and insights, and visualize connections as interactive mind maps.

---

## 🎯 What It Does

```
📱 Record with Plaud Device/App
         ↓
🔐 OAuth → Plaud API
         ↓
📝 Fetch Transcripts Automatically  
         ↓
🧠 AI Processing (Gemini)
   • Extract themes & topics
   • Generate summaries
   • Find semantic connections
         ↓
🗄️ Store in Vector DB (Pinecone)
         ↓
🔍 Semantic Search & 🎨 Visual Mind Maps
```

---

## 🚀 Quick Start

### 1. Create Plaud OAuth App

1. Go to [platform.plaud.ai/developer/portal](https://platform.plaud.ai/developer/portal)
2. Click **"New OAuth App"**
3. Fill in:
   - **App Name**: `PlaudBlender`
   - **Homepage URL**: `https://github.com/yourusername/PlaudBlender`
   - **Authorization callback URL**: `http://localhost:8080/callback`
4. Copy your **Client ID** and **Client Secret**

### 2. Install & Configure

```bash
# Clone the repo
git clone https://github.com/yourusername/PlaudBlender.git
cd PlaudBlender

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Add to `.env`:
```bash
PLAUD_CLIENT_ID=your_client_id
PLAUD_CLIENT_SECRET=your_client_secret
PLAUD_REDIRECT_URI=http://localhost:8080/callback

# Optional: For AI processing
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=transcripts
```

### 3. Authenticate & Test

```bash
python plaud_setup.py
```

If the Plaud consent page shows a **400 Bad Request** in the browser console, it’s almost always a **redirect URI mismatch**. Double-check that:
- `PLAUD_REDIRECT_URI` in your `.env` matches the URL registered in the Plaud developer portal **exactly** (scheme/host/port/path).
- During local dev, prefer: `http://localhost:8080/callback` (and register that exact URL).

You may also see Chrome console warnings like **“preload … not used”** or **“credentials mode does not match / crossorigin”** coming from Plaud’s own page assets (e.g. `resource.plaud.ai`). Those warnings are **harmless** and unrelated to PlaudBlender.

This will:
- Check your configuration
- Open browser for Plaud OAuth login
- Test the API connection
- Show your recent recordings

### 4. Diagnostics & Testing

- Print consent URL without running the full wizard:
  ```bash
  python scripts/plaud_auth_utils.py --print-consent-url
  ```
- Validate/auto-refresh your Plaud token and verify the API user:
  ```bash
  python scripts/plaud_auth_utils.py --check-token
  ```
- Run tests (shows skips with reasons):
  ```bash
  python -m pytest -rs
  ```
  Skips: `test_components.py` (legacy live component probes) and `test_gui_import.py` (legacy import probe covered elsewhere).

### 5. OpenAI Responses MCP server (ChatGPT connectors)

Expose PlaudBlender via the Model Context Protocol using OpenAI's Responses API. This runs over stdio and is ready for ChatGPT connectors or any MCP-capable client.

```bash
python -m scripts.mcp_server
```

Environment variables:
- `OPENAI_API_KEY` (required)
- `OPENAI_DEFAULT_MODEL` (optional, defaults to `gpt-4.1`)
- `OPENAI_BASE_URL` (optional, for gateways/proxies)

Available MCP tools:
- `ping` — health probe.
- `list_models` — list accessible OpenAI model IDs for the configured project.
- `respond` — send a prompt through the OpenAI Responses API and return combined text output.

---

## 💻 Usage

### Launch the GUI (recommended)

```bash
python gui.py
```

### Sync Plaud → Pinecone (batch)

```bash
python scripts/sync_to_pinecone.py
```

### Process pending SQL recordings into segments (SQL pipeline)

```bash
python scripts/process_pending.py
```

### Verify advanced feature wiring (developer smoke)

```bash
python verify_integration.py
```

### In-app Chat (OpenAI Responses)

- Set `OPENAI_API_KEY` (and optional `OPENAI_DEFAULT_MODEL`, `OPENAI_BASE_URL`) in `.env`.
- Launch the GUI (`python gui.py`) and open the **💬 Chat** tab.
- Configure model/temperature, optionally set a system prompt, and chat using the OpenAI Responses API.
- Advanced: enable "Advanced overrides" to pass raw JSON overrides to `responses.create` (e.g., `{ "max_output_tokens": 200, "top_p": 0.9, "tools": [...], "tool_choice": "auto" }`).

### Pinecone quick links

See `docs/pinecone-cheatsheet.md` for the most relevant Pinecone API links (query/upsert, namespaces, control-plane, assistant chat options, error handling, and cost/ops).

---

## 📁 Project Structure

```
PlaudBlender/
├── gui.py                      # GUI entry point (calls gui/app.py)
├── plaud_setup.py              # Setup & Plaud OAuth validation
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── data/                       # Local app data (SQLite DB, caches)
├── gui/                        # Tkinter GUI (views/services/components)
├── src/                        # Processing pipeline, DB layer, clients
├── scripts/                    # Batch tools (sync, process, mcp)
└── docs/                       # Roadmap, playbooks, references
```

---

## 🔐 OAuth Flow

PlaudBlender uses OAuth 2.0 to securely access your Plaud data:

1. **You authenticate** in your browser with your Plaud account
2. **Plaud issues** an access token to PlaudBlender
3. **Tokens are stored** locally in `.plaud_tokens.json`
4. **Auto-refresh** when tokens expire

Your Plaud credentials are never stored - only OAuth tokens that can be revoked.

---

## 🧩 Features

### ✅ Direct Plaud Integration
- OAuth 2.0 authentication
- Fetch recordings and transcripts
- Access AI summaries from Plaud

### ✅ AI-Powered Processing
- Theme extraction with Gemini 2.0
- Semantic embeddings for search
- Connection discovery between recordings

### ✅ Interactive Visualizations
- Force-directed knowledge graphs
- Theme-based color coding
- Zoom, pan, and explore connections

### ✅ Semantic Search
- Query your recordings in natural language
- Find related content across all transcripts
- Export results for further analysis

---

## ⚠️ Beta Notes

As per Plaud's beta program:
- **Testing only** - not for production use yet
- **Unlimited Plan** Plaud accounts only
- No user revocation UI yet (coming in ~2 weeks)
- Regional data residency changes coming (may require re-auth)

---

## 🏗️ Architecture

```
📱 Plaud Device/App
  ↓
☁️ Plaud Cloud (recordings + transcripts)
  ↓
🔐 OAuth 2.0 Authentication
  ↓
🗄️ Pinecone Vector Database
  ├── Namespace: full_text (complete transcripts)
  └── Namespace: summaries (AI summaries)
  ↓
💻 Your Computer
  ├── Query tool (search transcripts)
  └── Visualizer (generate mind maps)
```

---

## 📚 Documentation

- **Project guide (single source of truth):** `docs/PROJECT_GUIDE.md`
- Roadmap: `docs/architecture-roadmap.md`
- Live audit/UX checklist: `docs/audit-checklist.md`
- Pinecone playbook: `docs/pinecone-integration-playbook.md`

