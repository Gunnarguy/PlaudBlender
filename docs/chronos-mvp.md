# Chronos MVP Specification

**Version:** 2.0
**Target Date:** Q1 2025 (original), Q1 2026 (Gemini Embedding 2 migration)
**Status:** Complete — all MVP capabilities shipped, Gemini Embedding 2 multimodal integrated, OpenAI GPT-5.4 RAG, X-ray Activity Monitor, API cost tracking, Notion uplink

## Mission Statement

Transform 5–7 hour Plaud voice recordings from "jumbled mess" into a **clean, queryable, temporally-indexed knowledge timeline** with graph-enhanced retrieval.

---

## Core Capabilities (Must Ship)

### 1. **Local-First Audio Ingestion**

- Pull recordings from Plaud API via OAuth
- Download audio to local storage: `data/raw/YYYY/MM/DD/<recording_id>.opus`
- Store metadata in SQLite (`chronos_recordings` table)
- Detect duplicates (no re-processing)
- Handle network failures with retry/resume logic

**Acceptance Criteria:**

- 100 recordings can be ingested without memory errors
- Audio files are checksummed and verified
- Can recover from interrupted downloads

---

### 2. **Gemini-Driven Event Reconstruction**

- Process raw audio through Gemini 3 Flash (or 3.1 Pro for deep passes)
- Apply "Clean Verbatim" prompt: remove filler, preserve context
- Output structured JSON: array of `ChronosEvent` objects
- Each event has: `timestamp`, `clean_text`, `category`, `sentiment` (optional)

**Acceptance Criteria:**

- 7-hour audio produces 50–200 structured events (not 1 giant blob)
- Events have logical topic boundaries (not mid-sentence cuts)
- No hallucinations (provably tied to audio timestamps)

**Deferrable:**

- Speaker diarization (v2)
- Multi-speaker conversation handling (v2)

---

### 3. **Temporal-First Vector Indexing (Qdrant)**

- Store events in Qdrant with native payload filtering
- Mandatory payload fields:
  - `recording_id`, `event_id`
  - `timestamp` (UTC ISO 8601)
  - `day_of_week` (Monday–Sunday)
  - `hour_of_day` (0–23)
  - `category` (work, personal, meeting, deep_work, break)
  - `clean_text` (the reconstructed narrative)
- Enable **hybrid search**: semantic similarity **AND** time filters

**Acceptance Criteria:**

- Query "all Mondays in October" returns exact matches (100% recall)
- Query "similar to anxiety on Mondays" uses both vector + filter
- Scroll API works for bulk analytics (not just top-k retrieval)

---

### 4. **Graph-Assisted Query Expansion (GraphRAG)**

- Extract entities and relationships from cleaned events
- Build and persist a graph (NetworkX → disk pickle or Neo4j later)
- Use graph for query expansion:
  - User asks: "What's blocking Project Alpha?"
  - System finds: `Project Alpha -> BLOCKED_BY -> Server Migration`
  - System retrieves: chunks about "Server Migration" even if not explicitly linked in text

**Acceptance Criteria:**

- Graph extraction runs on cleaned text (not raw messy transcripts)
- Graph is persisted (not rebuilt on every query)
- At least one "expansion query" succeeds where pure vector search fails

**Deferrable:**

- Community detection (v2)
- Multi-hop reasoning (v2)

---

### 5. **Streamlit Timeline UI (Master-Detail)**

- **Master View:** Interactive timeline (vis.js wrapper)
  - Date range selector
  - Zoom: month view → day view → hour view
  - Color-coded by category
  - Click event → show detail
- **Detail View:** Selected event
  - Clean text
  - Metadata (timestamp, category, sentiment)
  - Provenance link (to raw audio/recording)
- **Mind Map View:** Related events
  - Semantic neighbors (vector similarity)
  - Graph neighbors (entity connections)
  - Expandable nodes

**Acceptance Criteria:**

- Load 1000 events in timeline without UI freeze
- Click-to-detail latency < 300ms
- Timeline aesthetics: "clean as fuck" (minimal clutter, high information density)

**Deferrable:**

- Audio playback widget (v2)
- Inline editing of event text (v2)

---

### 6. **Temporal Pattern Analytics**

- Query: "What happens on Mondays?"
  - Aggregate all Monday events
  - Generate histogram by hour
  - Extract top categories, keywords
  - (Optional) Gemini "analyst mode" synthesis
- Query: "What do I think about on Thursdays?"
  - Same aggregation
  - Sentiment trend over Thursdays
  - Top entities from graph extraction

**Acceptance Criteria:**

- Can retrieve and analyze 50+ events for a given day-of-week in < 2 seconds
- Results include both quantitative (counts, peaks) and qualitative (synthesis) insights

---

## Non-Goals (Explicitly Out of Scope for MVP)

- ❌ Multi-user support (single-user local-first only)
- ❌ Real-time streaming transcription (batch-only)
- ❌ Mobile app (desktop/browser only)
- ❌ Complex desktop GUI (replaced by Dash)
- ❌ Pinecone compatibility shim (Qdrant-only)

---

## Data Model Contract

### **ChronosEvent** (Core Schema)

```json
{
  "event_id": "uuid",
  "recording_id": "plaud_recording_123",
  "start_ts": "2025-10-27T09:15:32Z",
  "end_ts": "2025-10-27T09:18:45Z",
  "day_of_week": "Monday",
  "hour_of_day": 9,
  "category": "work",
  "clean_text": "Reviewed the Sprint planning doc...",
  "sentiment": 0.2,
  "keywords": ["sprint", "planning", "backlog"],
  "speaker": "self_talk"
}
```

### **ChronosRecording** (Metadata Schema)

```json
{
  "recording_id": "plaud_recording_123",
  "created_at": "2025-10-27T07:00:00Z",
  "duration_seconds": 25200,
  "local_audio_path": "data/raw/2025/10/27/plaud_recording_123.opus",
  "source": "plaud",
  "device_id": "plaud_note_001",
  "checksum": "sha256:abc123..."
}
```

---

## Success Metrics

| Metric                     | Target        | Measurement Method                   |
| -------------------------- | ------------- | ------------------------------------ |
| **Ingestion Success Rate** | > 95%         | Recordings fetched without errors    |
| **Event Quality**          | > 90% "clean" | Manual review of 20 samples          |
| **Query Latency (Hybrid)** | < 500ms       | P95 for Monday filter + vector       |
| **Graph Query Success**    | > 80%         | Test suite with 10 expansion queries |
| **UI Responsiveness**      | < 300ms       | Click-to-detail render time          |

---

## Phases

### Phase 1: Foundation (Week 1–2)

- ✅ Chronos schema models (Pydantic)
- ✅ SQLite table for recordings
- ✅ Plaud ingest script (list + download)

### Phase 2: Cognitive Engine (Week 3–4)

- Gemini File API integration
- "Clean Verbatim" prompt engineering
- Event JSON validation + persistence

### Phase 3: Indexing (Week 5)

- Qdrant native client (no shim)
- Payload indexes for temporal fields
- Hybrid search service

### Phase 4: Graph (Week 6)

- Entity extraction from cleaned events
- NetworkX graph builder
- Query expansion service

### Phase 5: UI (Week 7)

- Streamlit app skeleton
- Timeline component (vis.js)
- Detail + mind map views

### Phase 6: Analytics (Week 8)

- Day-of-week aggregation
- Gemini analyst integration
- Pattern insights UI

---

## Open Questions

1. **Gemini Context Window:** Can we truly fit 7 hours in one pass, or do we need sliding windows?
2. **Sentiment Granularity:** Per-event or per-recording aggregate?
3. **Graph Persistence:** Stick with NetworkX pickle or migrate to KuzuDB/Neo4j early?
4. **Audio Playback:** Is this essential for MVP or can we defer?

---

## Dependencies

- Gemini API access (Flash + Pro + Embedding 2)
- OpenAI API access (GPT-5.4 via Responses API for RAG queries)
- Qdrant (local Docker or cloud)
- Plaud OAuth credentials
- Dash + Cytoscape (replaced Streamlit)
- Notion API (optional, for uplink sync)

---

## Post-MVP Features (Shipped)

### OpenAI GPT-5.4 RAG (Complete)

- AI answer panel in Search view — GPT-5.4 synthesizes vector results
- Reasoning levels (none/low/medium/high/xhigh) — adjustable per query
- MCP `ask_chronos` tool uses OpenAI (falls back to Gemini if no key)
- Dedicated service: `src/chronos/openai_service.py` (Responses API)

### X-ray Activity Monitor (Complete)

Floating PiP panel showing real-time telemetry in plain English:

- **Server:** `app_v2/services/xray.py` — ring buffer (200 events), monotonic `seq` IDs
- **API:** `/xray/api/events?since=N` (incremental polling), `/xray/api/clear`
- **Client:** `app_v2/assets/xray_pip.js` — polls 800ms, accumulates ≤2,000 events
- **12 sources:** Plaud, AI, Embedding, Search DB, Knowledge Graph, Search, Data, Nav, Pipeline, Recording, Day View, Sync
- **Filter tabs:** Pipeline, Search, Graph, Data, Errors
- **~87 instrumented messages** across all core services and UI callbacks
- Messages are plain English: "Gemini read it all — wrote 2,400 chars and spotted 8 moments"

### API Cost Tracking (Complete)

Real-time API usage monitoring with per-model pricing:

- **Module:** `src/chronos/cost_tracker.py` — thread-safe singleton, SQLite persistence
- **Pricing table:** 12 models (Gemini + OpenAI) with USD/MTok input/output rates
- **Session ledger:** In-memory tracking since app start
- **Historical ledger:** SQLite `api_usage_log` table with indexes on timestamp + model
- **Instrumentation:** 13 `track_usage()` call sites across 8 files (every billable API call)
- **UI:** Stats view cost section (session costs, 30-day breakdown, per-model, daily chart, pricing reference)
- **API:** `/xray/api/costs` Flask endpoint for live cost data
- **Settings:** Model pricing shown inline in dropdown labels ($X.XX/$Y.YY per MTok)

### Notion Uplink (Complete)

Sync recordings to Notion pages:

- **OAuth:** Full Notion OAuth 2.0 flow (authorize, callback, status check)
- **Service:** `src/notion_service.py` — page creation, property mapping
- **Bridge:** `src/chronos/notion_bridge.py` — integration layer
- **UI:** Notion view with page filter, channel select, sync preview
- **Callbacks:** 20 Dash callbacks for full CRUD + batch sync
- **Flask routes:** `/auth/notion`, `/auth/notion/callback`, `/auth/notion/status`

---

## Model Reference (March 2026)

| Model                        | Purpose                         | Notes                                                    |
| ---------------------------- | ------------------------------- | -------------------------------------------------------- |
| `gemini-3-flash-preview`     | Cognitive cleaning (processing) | Free on standard tier                                    |
| `gemini-3.1-pro-preview`     | Deep analysis fallback          | Replaced `gemini-3-pro-preview` (shut down 2026-03-09)   |
| `gemini-embedding-2-preview` | Multimodal embeddings           | Text + audio + image + video + PDF. MRL 128–3072 dims.   |
| `gpt-5.4`                    | RAG responses (Responses API)   | 1.05M ctx, 128K output, reasoning levels, $2.50/$15 MTok |

### Pricing Reference (March 2026)

| Model                        | Input ($/MTok) | Output ($/MTok) | Notes          |
| ---------------------------- | -------------- | --------------- | -------------- |
| `gemini-3-flash-preview`     | $0.50          | $3.00           | Paid tier      |
| `gemini-3.1-pro-preview`     | $2.00          | $12.00          | Paid tier      |
| `gemini-embedding-2-preview` | $0.20          | —               | Paid tier      |
| `gpt-5.4`                    | $2.50          | $15.00          | Flagship       |
| `gpt-5.4-pro`                | $5.00          | $30.00          | Deep reasoning |
| `gpt-5.4-mini`               | $0.75          | $4.50           | Fast + cheap   |
| `gpt-5.4-nano`               | $0.20          | $1.25           | Lightest       |
| `gpt-5`                      | $2.00          | $10.00          | Balanced       |
| `gpt-4.1`                    | $2.00          | $8.00           | Legacy         |

### Embedding Details

- **Default dim:** 768 (MRL truncation from 3072 native), L2-normalized
- **Multimodal fusion:** Pipeline embeds text + audio together when WAV/MP3 available (≤80s)
- **Task types:** `RETRIEVAL_DOCUMENT` (indexing), `RETRIEVAL_QUERY` (search), `QUESTION_ANSWERING` (RAG)
- **Migration:** `python scripts/chronos_pipeline.py --reindex` to re-embed all events after model change

---

**Document Owner:** Gunnar Hostetler
**Last Updated:** 2026-03-18
