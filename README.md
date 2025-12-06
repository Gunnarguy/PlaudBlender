# 🧠 PlaudBlender

**Transform your Plaud voice recordings into a searchable, visual knowledge graph.**

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

### Fetch Your Recordings

```bash
# List recent recordings
python scripts/fetch_from_plaud.py --list-only

# Fetch all with transcripts
python scripts/fetch_from_plaud.py --limit 100

# Fetch only new (last 60 minutes)
python scripts/fetch_from_plaud.py --since 60
```

### Process with AI

```bash
# Process fresh from Plaud
python scripts/process_transcripts.py

# Process from saved JSON
python scripts/process_transcripts.py --input output/plaud_recordings_20251128.json
```

### Query & Visualize

```bash
# View stats
python scripts/query_and_visualize.py --stats

# Search transcripts
python scripts/query_and_visualize.py --full "meeting notes about project X"

# Generate mind map
python scripts/query_and_visualize.py --mindmap "all recordings" --output output/knowledge_graph.html

# Open the visualization
open output/knowledge_graph.html
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
├── plaud_setup.py              # Setup & authentication wizard
├── requirements.txt            # Python dependencies
├── .env                        # Your API credentials (create this)
│
├── src/
│   ├── plaud_oauth.py          # OAuth 2.0 client for Plaud
│   ├── plaud_client.py         # Plaud API client
│   ├── llm_processor.py        # Gemini AI processing
│   ├── pinecone_client.py      # Vector database
│   └── visualizer.py           # Mind map generator
│
├── scripts/
│   ├── fetch_from_plaud.py     # Fetch recordings & transcripts
│   ├── process_transcripts.py  # AI processing pipeline
│   └── query_and_visualize.py  # Search & visualization
│
└── output/                     # Generated files
    └── *.html                  # Mind map visualizations
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

## 💰 Cost Breakdown

- **Zapier**: ~$0.01 per transcript (~$10/month for 1,000 transcripts)
- **OpenAI Embeddings**: ~$0.0001 per transcript (~$0.10/month)
- **Google Gemini**: FREE (generous quota for theme extraction)
- **Pinecone**: ~$3.50/month (1,000 transcripts with dual namespaces)

**Total: ~$13.60/month for 1,000 transcripts**

---

## 🎨 Mind Map Features

- **Nodes**: Each transcript is a node (sized by relevance)
- **Themes**: Color-coded by AI-extracted themes
- **Connections**: Related transcripts are linked
- **Interactive**: Hover for previews, click for details
- **Search**: Filter by keywords or themes

---

## 🐛 Troubleshooting

**Zapier not working?**
- Check environment variables are set
- Verify Pinecone host URL
- Test with sample data

**No query results?**
- Check Pinecone has data: `python3 scripts/query_and_visualize.py --stats`
- Verify API keys in `.env`

**Mind map not generating?**
- Ensure output/ directory exists
- Check query returned results

---

## 📚 Documentation

- **[ZAPIER_SETUP_GUIDE.md](ZAPIER_SETUP_GUIDE.md)** - Complete Zapier configuration
- **Requirements**: Python 3.11+, OpenAI API, Google Gemini API, Pinecone account

---

## 🤝 Contributing

Contributions welcome! This is a personal project but feel free to fork and customize.

---

## 📄 License

MIT License - do whatever you want with this code.

---

**Built with ❤️ to make sense of voice transcripts**

## 🏗️ Architecture

```
PlaudAI → Zapier → Notion Database
                      ↓
            AWS Lambda (every 15min)
                      ↓
            ┌─────────┴─────────┐
            ↓                   ↓
     Gemini API           Pinecone Vector DB
     (Analysis)           (Embeddings)
            ↓                   ↓
         Notion Database
         (Updated with synthesis & connections)
                      ↓
              Mind Map Generator
                      ↓
          Beautiful Interactive HTML
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API Keys (all free tier):
  - Google Gemini API
  - Pinecone account
  - Notion integration
  - AWS account (optional for autonomous mode)

### Installation

1. **Clone and setup:**

```bash
git clone <your-repo>
cd PlaudBlender

# Run setup script
bash scripts/setup.sh
```

2. **Configure environment:**

Your `.env` file should contain:
```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=transcripts
NOTION_TOKEN=your_notion_token
NOTION_DATABASE_ID=your_database_id

# For AWS deployment
AWS_ACCOUNT_ID=your_account_id
AWS_REGION=us-east-1
```

3. **Set up Notion Database:**

Create a Notion database with these properties:
- **Title** (title)
- **Status** (select: "New", "Processing", "Processed")
- **Created** (created time)
- **Synthesis** (text)
- **Connections** (text)
- **Themes** (text)
- **ProcessedAt** (date)

Share the database with your Notion integration.

4. **Test components:**

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python test_components.py

# Test specific components
python test_components.py --notion
python test_components.py --gemini
python test_components.py --pinecone
```

5. **Process transcripts locally:**

```bash
# Process new transcripts
python lambda_function.py

# Generate mind map
python generate_mindmap.py

# Open in browser
open output/knowledge_graph.html
```

## 📊 Usage

### Local Testing

```bash
# Process new transcripts
python lambda_function.py

# Generate mind map with custom output
python generate_mindmap.py -o output/my_mindmap.html

# Test individual components
python test_components.py
```

### AWS Deployment (Autonomous Mode)

```bash
# Deploy to Lambda with EventBridge schedule
bash scripts/deploy_to_lambda.sh

# View logs
aws logs tail /aws/lambda/notion-transcript-processor --follow

# Manual invoke for testing
aws lambda invoke \
  --function-name notion-transcript-processor \
  --payload '{}' \
  output.json
```

## 📁 Project Structure

```
PlaudBlender/
├── src/
│   ├── notion_client.py      # Fetch/update Notion transcripts
│   ├── llm_processor.py      # Gemini + Pinecone processing
│   ├── pinecone_client.py    # Vector operations
│   └── visualizer.py         # Mind map generation
├── scripts/
│   ├── setup.sh             # Initial setup
│   └── deploy_to_lambda.sh  # AWS deployment
├── lambda_function.py        # AWS Lambda handler
├── generate_mindmap.py      # Mind map generator
├── test_components.py       # Component tests
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not in git)
└── output/                  # Generated visualizations
```

## 🎨 Mind Map Features

The generated HTML includes:

- **Interactive Graph**
  - Drag nodes to rearrange
  - Zoom and pan
  - Click to highlight connections
  - Hover for detailed tooltips

- **Statistics Dashboard** (toggle with 📊 button)
  - Total transcripts and connections
  - Average connections per transcript
  - Network density metrics
  - Top 5 themes with counts
  - Most connected transcripts

- **Theme-Based Coloring**
  - Automatic color assignment by theme
  - Consistent colors across sessions
  - Visual clustering of related topics

- **Smart Layout**
  - Physics-based force-directed graph
  - Natural organization of connected concepts
  - Prevents node overlap

## 💰 Cost Breakdown

For 500 transcripts/month:

| Service | Cost |
|---------|------|
| Gemini 2.0 Flash API | $0.00 (free tier: 1,500 req/day) |
| Pinecone Starter | $0.00 (free: 2GB storage) |
| Pinecone Embeddings | ~$0.08 |
| AWS Lambda | $0.00 (free: 1M req/month) |
| AWS EventBridge | $0.00 (free) |
| Notion API | $0.00 (unlimited) |
| **Total** | **~$0.08/month** |

At 1,000 transcripts/month: ~$0.50/month

## 🔧 Configuration

### Processing Frequency

Edit EventBridge schedule in `scripts/deploy_to_lambda.sh`:
```bash
# Every 15 minutes (default)
--schedule-expression "rate(15 minutes)"

# Every hour
--schedule-expression "rate(1 hour)"

# Specific time daily (8 AM UTC)
--schedule-expression "cron(0 8 * * ? *)"
```

### Theme Extraction

Customize in `src/llm_processor.py`:
```python
def extract_themes(self, text):
    prompt = f"""Extract 3-5 key themes...
    # Modify prompt here
    ```

### Visualization Style

Customize colors in `src/visualizer.py`:
```python
self.color_palette = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    # Add your colors
]
```

## 🐛 Troubleshooting

### "No new transcripts found"
- Check Notion database has Status="New" transcripts
- Verify NOTION_DATABASE_ID in .env
- Ensure Notion integration has database access

### "Error initializing clients"
- Verify all API keys in .env
- Test with `python test_components.py`
- Check API key permissions

### "Module not found"
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### AWS Lambda timeout
- Increase timeout in deployment script (default: 900s)
- Check CloudWatch logs for specific errors

### Empty mind map
- Run `python lambda_function.py` to process transcripts first
- Ensure transcripts have Status="Processed" in Notion
- Check logs for processing errors

## 📈 Performance

- **Processing**: ~2-5 seconds per transcript
- **Mind Map Generation**: ~1-3 seconds for 100 transcripts
- **Lambda Memory**: 1024 MB (adjustable)
- **Lambda Timeout**: 900 seconds (15 minutes)

## 🔒 Security

- API keys stored in environment variables
- `.env` excluded from git
- AWS IAM role with minimal permissions
- No data stored outside your accounts

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional visualization types (timeline, cluster view)
- [ ] Export to other formats (PDF, PNG)
- [ ] Advanced analytics (topic modeling, sentiment)
- [ ] Web dashboard for real-time monitoring
- [ ] Mobile-optimized visualizations

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - Document indexing & RAG
- [Google Gemini](https://ai.google.dev/) - LLM analysis
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Pyvis](https://pyvis.readthedocs.io/) - Network visualization
- [Notion API](https://developers.notion.com/) - Database integration

## 📧 Support

- 🐛 Issues: [GitHub Issues](your-repo/issues)
- 💬 Discussions: [GitHub Discussions](your-repo/discussions)
- 📖 Docs: [Full Roadmap](Roadmap.md)

---

**Made with ❤️ for organizing the chaos of daily transcripts into meaningful insights**

