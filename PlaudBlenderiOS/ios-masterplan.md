# Chronos iOS — Native SwiftUI Masterplan

> **Generated:** March 21, 2026
> **Scope:** Full architecture plan to transform PlaudBlender/Chronos from a local Python/Dash app into a cloud-backed native SwiftUI iOS application.
> **Philosophy:** Gunnar loves data, granularity, and depth. Never hide information. Expose metrics, scores, and latency. Progressive disclosure — simple by default, drill-down available.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture (As-Is)](#2-current-architecture-as-is)
3. [Target Architecture (To-Be)](#3-target-architecture-to-be)
4. [Phase 0 — FastAPI Backend](#4-phase-0--fastapi-backend)
5. [Phase 1 — Cloud Infrastructure](#5-phase-1--cloud-infrastructure)
6. [Phase 2 — iOS App Foundation](#6-phase-2--ios-app-foundation)
7. [Phase 3 — Timeline View](#7-phase-3--timeline-view)
8. [Phase 4 — Recording Detail View](#8-phase-4--recording-detail-view)
9. [Phase 5 — Search & AI Answers](#9-phase-5--search--ai-answers)
10. [Phase 6 — Topics View](#10-phase-6--topics-view)
11. [Phase 7 — Knowledge Graph](#11-phase-7--knowledge-graph)
12. [Phase 8 — Stats & Analytics](#12-phase-8--stats--analytics)
13. [Phase 9 — Sync & Pipeline Control](#13-phase-9--sync--pipeline-control)
14. [Phase 10 — Settings](#14-phase-10--settings)
15. [Phase 11 — Notion Integration](#15-phase-11--notion-integration)
16. [Phase 12 — X-ray Activity Monitor](#16-phase-12--x-ray-activity-monitor)
17. [Phase 13 — Authentication (Plaud + Notion OAuth)](#17-phase-13--authentication-plaud--notion-oauth)
18. [Phase 14 — Push Notifications & Background Sync](#18-phase-14--push-notifications--background-sync)
19. [Phase 15 — Offline Support & Caching](#19-phase-15--offline-support--caching)
20. [Phase 16 — Polish, Accessibility & App Store](#20-phase-16--polish-accessibility--app-store)
21. [Complete API Endpoint Specification](#21-complete-api-endpoint-specification)
22. [Data Models — Swift Codable Structs](#22-data-models--swift-codable-structs)
23. [Design System — iOS Color Tokens](#23-design-system--ios-color-tokens)
24. [Dependency Map](#24-dependency-map)
25. [Risk Register](#25-risk-register)
26. [File-by-File Migration Reference](#26-file-by-file-migration-reference)

---

## 1. Executive Summary

### What We're Building

A **native SwiftUI iOS app** called **Chronos** that provides:

- A searchable, AI-powered knowledge timeline of your voice recordings
- Interactive knowledge graph visualization
- Full pipeline control (ingest → process → index → graph)
- Plaud device integration, Notion uplink, and AI search with GPT-5.4

### Core Transformation

| Aspect            | Current (Desktop)                  | Target (iOS)                                 |
| ----------------- | ---------------------------------- | -------------------------------------------- |
| **UI**            | Dash/Plotly (Python web framework) | Native SwiftUI                               |
| **Backend**       | In-process Python services         | FastAPI REST server (cloud-hosted)           |
| **Database**      | Local SQLite (`data/brain.db`)     | Cloud PostgreSQL (Supabase/RDS)              |
| **Vector Store**  | Local Qdrant (Docker)              | Qdrant Cloud                                 |
| **Auth**          | Local callback (localhost:8050)    | Universal Links / ASWebAuthenticationSession |
| **Notifications** | X-ray floating PiP (browser)       | APNs push notifications                      |
| **Offline**       | Always online (local)              | SwiftData cache + sync queue                 |

### Work Estimate (Phases)

| Phase | Scope                            | Dependency        |
| ----- | -------------------------------- | ----------------- |
| 0     | FastAPI Backend                  | None — start here |
| 1     | Cloud Infrastructure             | Phase 0           |
| 2     | iOS Foundation + Navigation      | Phase 1           |
| 3–12  | Feature views (Timeline → X-ray) | Phase 2           |
| 13    | OAuth flows                      | Phase 2           |
| 14    | Push notifications               | Phase 1           |
| 15    | Offline caching                  | Phase 2           |
| 16    | Polish + App Store               | All               |

---

## 2. Current Architecture (As-Is)

### Monolith Stack

```
┌─────────────────────────────────────────────────────┐
│                    User's Mac                        │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Dash UI    │  │   SQLite     │  │   Qdrant   │ │
│  │  (port 8050) │  │ (brain.db)   │  │  (Docker)  │ │
│  │  50 callbacks │  │  6 ORM tables│  │  port 6333 │ │
│  │  9 Flask     │  │              │  │            │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                 │        │
│  ┌──────┴─────────────────┴─────────────────┴──────┐ │
│  │           ChronosDataService (2150 lines)       │ │
│  │         → The single data access layer          │ │
│  └──────────────────────┬──────────────────────────┘ │
│                         │                            │
│  ┌──────────────────────┴──────────────────────────┐ │
│  │              Core Engine (src/chronos/)          │ │
│  │  ingest → transcript_processor → embedding →    │ │
│  │  qdrant_client → graph_service → openai_service │ │
│  └─────────────────────────────────────────────────┘ │
│                         │                            │
│  ┌──────────────────────┴──────────────────────────┐ │
│  │            External APIs                         │ │
│  │  Plaud API · Gemini API · OpenAI · Notion API   │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Existing Service Inventory

| Service File              | Lines | Public Methods | Purpose                                                |
| ------------------------- | ----- | -------------- | ------------------------------------------------------ |
| `data_service.py`         | ~2150 | 30+            | **The** data access layer for all UI views             |
| `ingest_service.py`       | ~420  | 4              | Download recordings from Plaud                         |
| `transcript_processor.py` | ~800  | 4              | Gemini AI event extraction                             |
| `qdrant_client.py`        | ~450  | 10             | Vector search + temporal filtering                     |
| `embedding_service.py`    | ~500  | 5              | Text/audio embeddings (multimodal)                     |
| `graph_service.py`        | ~170  | 4              | Entity extraction, community detection                 |
| `graph_rag.py`            | ~1200 | 15+            | Knowledge graph (entities, relationships, communities) |
| `openai_service.py`       | ~170  | 2              | RAG Q&A via GPT-5.4 Responses API                      |
| `cost_tracker.py`         | ~600  | 6              | API cost tracking (12 models)                          |
| `notion_bridge.py`        | ~1500 | 10+            | Notion ↔ Chronos matching, import, writeback           |
| `engine.py`               | ~350  | 2              | Gemini cognitive audio processing                      |
| `analytics.py`            | ~200  | 4              | Temporal pattern analysis                              |
| `pipeline_progress.py`    | ~250  | —              | Pipeline stage tracking                                |
| `xray.py`                 | ~150  | 5              | Telemetry ring buffer                                  |
| `plaud_client.py`         | ~600  | 20+            | Plaud REST API (recordings, upload, transcripts)       |
| `plaud_oauth.py`          | ~400  | 8              | Plaud OAuth 2.0 (⚠️ requires state in POST body)       |
| `notion_oauth.py`         | ~250  | 5              | Notion OAuth 2.0 (no refresh tokens)                   |
| `notion_service.py`       | ~500  | 10             | Notion data source queries                             |
| `mcp_server.py`           | ~400  | 11             | FastMCP tool server (already REST-like!)               |

### Current UI Views (8 views + X-ray + Settings)

| View                 | Dash Callbacks           | Key Data                                                      |
| -------------------- | ------------------------ | ------------------------------------------------------------- |
| **Timeline**         | 8 + 1 clientside         | Days → Recordings → Events, heat-map calendar                 |
| **Topics**           | Integrated with Timeline | Topic grid, topic timeline, occurrence cards                  |
| **Search**           | 5                        | Semantic search + AI answer + category/date filters           |
| **Graph**            | 3                        | Cytoscape knowledge graph (6 layouts)                         |
| **Stats**            | Integrated               | 8 stat cards, category chart, heatmap, sentiment              |
| **Notion**           | 20                       | OAuth, database picker, page sync, import, writeback          |
| **Sync**             | ~8                       | Pipeline dashboard, workflow monitoring, upload               |
| **Settings**         | Integrated               | 12 sections, service checks, model dropdowns                  |
| **Recording Detail** | 3 + 1 clientside         | 5-tab detail: Overview/Events/Narrative/Transcript/Comparison |
| **X-ray**            | Standalone JS            | Floating PiP, 12 source categories, real-time polling         |

### Current Flask Routes (9 routes)

| Route                   | Method       | Purpose                                      |
| ----------------------- | ------------ | -------------------------------------------- |
| `/auth/plaud`           | GET          | Start Plaud OAuth                            |
| `/auth/plaud/callback`  | GET, OPTIONS | Plaud OAuth callback (triple-hit idempotent) |
| `/auth/plaud/status`    | GET          | Auth status JSON                             |
| `/auth/notion`          | GET          | Start Notion OAuth                           |
| `/auth/notion/callback` | GET, OPTIONS | Notion OAuth callback                        |
| `/auth/notion/status`   | GET          | Auth status JSON                             |
| `/xray/api/events`      | GET          | Telemetry events (since_seq)                 |
| `/xray/api/clear`       | POST         | Clear event buffer                           |
| `/xray/api/costs`       | GET          | Cost tracking data                           |

### Current MCP Tools (11 tools — already REST-shaped!)

The MCP server (`scripts/mcp_server.py`) already defines a JSON-over-function API that maps almost 1:1 to REST endpoints:

| MCP Tool          | → REST Endpoint            |
| ----------------- | -------------------------- |
| `ping`            | `GET /api/health`          |
| `search_events`   | `POST /api/search`         |
| `get_recording`   | `GET /api/recordings/{id}` |
| `list_recordings` | `GET /api/recordings`      |
| `get_timeline`    | `GET /api/timeline`        |
| `get_stats`       | `GET /api/stats`           |
| `get_topics`      | `GET /api/topics`          |
| `get_graph`       | `GET /api/graph`           |
| `run_pipeline`    | `POST /api/pipeline/run`   |
| `system_status`   | `GET /api/system/status`   |
| `ask_chronos`     | `POST /api/ask`            |

---

## 3. Target Architecture (To-Be)

```
┌───────────────────────┐         ┌────────────────────────────────────────┐
│     iOS Device        │         │          Cloud Server                  │
│                       │  HTTPS  │     (Render / Railway / AWS)           │
│  ┌─────────────────┐  │ ◄─────► │  ┌──────────────────────────────────┐ │
│  │  SwiftUI App    │  │  JSON   │  │      FastAPI Backend             │ │
│  │                 │  │         │  │   api/                           │ │
│  │  Views:         │  │         │  │     routes/                      │ │
│  │  - Timeline     │  │         │  │       timeline.py                │ │
│  │  - Search       │  │         │  │       search.py                  │ │
│  │  - Topics       │  │         │  │       recordings.py              │ │
│  │  - Graph        │  │         │  │       graph.py                   │ │
│  │  - Stats        │  │         │  │       stats.py                   │ │
│  │  - Sync         │  │         │  │       sync.py                    │ │
│  │  - Settings     │  │         │  │       notion.py                  │ │
│  │  - Notion       │  │         │  │       auth.py                    │ │
│  │  - X-ray        │  │         │  │       xray.py                    │ │
│  │                 │  │         │  │     services/                    │ │
│  │  Services:      │  │         │  │       (existing src/chronos/)    │ │
│  │  - APIClient    │  │         │  │     models/                      │ │
│  │  - AuthManager  │  │         │  │       (existing src/models/)     │ │
│  │  - CacheManager │  │         │  └──────────┬───────────────────────┘ │
│  │  - SyncEngine   │  │         │             │                        │
│  │                 │  │         │  ┌──────────┴───────────────────────┐ │
│  │  Cache:         │  │         │  │        Data Layer                │ │
│  │  - SwiftData    │  │         │  │  PostgreSQL (Supabase)           │ │
│  │  - URLCache     │  │         │  │  Qdrant Cloud                    │ │
│  │  - Keychain     │  │         │  │  Redis (optional, sessions)      │ │
│  └─────────────────┘  │         │  └──────────────────────────────────┘ │
│                       │         │             │                        │
│  ┌─────────────────┐  │         │  ┌──────────┴───────────────────────┐ │
│  │  Push (APNs)    │◄─┤─────────┤──│  Background Workers              │ │
│  └─────────────────┘  │         │  │  - auto_sync.py (webhook/USB)    │ │
│                       │         │  │  - chronos_pipeline.py            │ │
│                       │         │  │  - Plaud webhook receiver         │ │
│                       │         │  └──────────────────────────────────┘ │
└───────────────────────┘         │             │                        │
                                  │  ┌──────────┴───────────────────────┐ │
                                  │  │        External APIs              │ │
                                  │  │  Plaud · Gemini · OpenAI · Notion │ │
                                  │  └──────────────────────────────────┘ │
                                  └────────────────────────────────────────┘
```

### Key Decisions

| Decision            | Choice                          | Rationale                                                             |
| ------------------- | ------------------------------- | --------------------------------------------------------------------- |
| **API Framework**   | FastAPI                         | Async, auto-docs (OpenAPI), Pydantic native, WebSocket support        |
| **Cloud DB**        | PostgreSQL (Supabase)           | Free tier, auth built-in, real-time subscriptions, row-level security |
| **Vector Store**    | Qdrant Cloud                    | Already using Qdrant locally; cloud version is drop-in                |
| **iOS Min Target**  | iOS 17+                         | SwiftData, NavigationStack, @Observable macro                         |
| **Auth on iOS**     | ASWebAuthenticationSession      | Apple's secure OAuth flow, no WKWebView hacks                         |
| **Offline Cache**   | SwiftData (Core Data successor) | Native Apple, syncs with CloudKit if needed                           |
| **Push**            | APNs via Firebase or direct     | Pipeline completion, new recording alerts                             |
| **Graph Rendering** | WKWebView + Cytoscape.js        | Reuse existing graph stylesheet; native graph libs are immature       |

---

## 4. Phase 0 — FastAPI Backend

### Goal

Create a REST API layer (`api/`) that wraps `ChronosDataService` and all core services, exposing every operation the iOS app needs as JSON endpoints.

### New Directory Structure

```
api/
  __init__.py
  main.py                    ← FastAPI app factory, CORS, middleware
  config.py                  ← Server config (extends src/config.py)
  dependencies.py            ← Dependency injection (DB sessions, services)
  auth/
    __init__.py
    jwt.py                   ← JWT token creation/validation
    middleware.py             ← Bearer token auth middleware
    plaud_oauth.py           ← Plaud OAuth redirect handling
    notion_oauth.py          ← Notion OAuth redirect handling
  routes/
    __init__.py
    timeline.py              ← GET /api/timeline, GET /api/days
    recordings.py            ← GET/POST /api/recordings, GET /api/recordings/{id}
    search.py                ← POST /api/search
    topics.py                ← GET /api/topics, GET /api/topics/{name}
    graph.py                 ← GET /api/graph
    stats.py                 ← GET /api/stats
    sync.py                  ← POST /api/pipeline/run, GET /api/pipeline/status
    settings.py              ← GET/PUT /api/settings
    notion.py                ← GET/POST /api/notion/*
    xray.py                  ← GET /api/xray/events (WebSocket upgrade)
    costs.py                 ← GET /api/costs
    auth.py                  ← POST /api/auth/*, token refresh
    health.py                ← GET /api/health, GET /api/system/status
  schemas/
    __init__.py
    responses.py             ← Pydantic response models (mirrors data_service types)
    requests.py              ← Pydantic request bodies
  websocket/
    __init__.py
    xray_ws.py               ← WebSocket handler for real-time X-ray events
```

### Endpoint → DataService Method Mapping

Every `ChronosDataService` method becomes a REST endpoint:

#### Timeline & Days

```
GET  /api/days                          → get_days(start_date?, end_date?)
GET  /api/days/filled                   → get_days_filled(last_n_days?)
GET  /api/days/{date}                   → get_day_detail(date)
```

#### Recordings

```
GET  /api/recordings                    → list via get_days() + aggregation
GET  /api/recordings/{id}               → get_recording_detail(recording_id)
GET  /api/recordings/{id}/events        → get_events_for_recording(recording_id)
GET  /api/recordings/{id}/transcript    → get_transcript(recording_id)
GET  /api/recordings/{id}/ai-summary    → get_ai_summary(recording_id)
GET  /api/recordings/{id}/extracted     → get_extracted_data(recording_id)
GET  /api/recordings/{id}/workflow      → get_workflow_status_for_recording(recording_id)
GET  /api/recordings/{id}/plaud-transcript → get_plaud_workflow_transcript(recording_id)
POST /api/recordings/{id}/workflow      → submit_single_recording_workflow(recording_id, template_id?, model?)
PUT  /api/recordings/{id}/events/{eid}/category → save_category_override(event_qdrant_id, new_category)
```

#### Search

```
POST /api/search                        → search(query, limit?, categories?, start_date?, end_date?)
POST /api/ask                           → OpenAIResponseService.ask(question, context_events, reasoning?)
```

#### Topics

```
GET  /api/topics                        → get_all_topics()
GET  /api/topics/{name}                 → get_topic_timeline(topic)
```

#### Knowledge Graph

```
GET  /api/graph                         → get_graph_data()
GET  /api/graph/entities?query=&type=   → graph_rag.search_entities_graph()
POST /api/graph/query                   → graph_rag.answer_global_query(query)
```

#### Statistics

```
GET  /api/stats                         → get_stats()
GET  /api/stats/recording-db            → get_recording_db_stats()
GET  /api/stats/workflows               → get_plaud_workflow_stats(days_back?)
```

#### Sync & Pipeline

```
POST /api/pipeline/run                  → run stage (ingest|process|index|graph|full)
GET  /api/pipeline/status               → pipeline_progress.get_progress()
POST /api/pipeline/reset-stuck          → reset_stuck_recordings()
POST /api/pipeline/workflows/submit     → submit_plaud_workflows(days_back?, limit?, template_id?, model?)
POST /api/pipeline/workflows/refresh    → refresh_plaud_workflow_statuses(days_back?, limit?)
GET  /api/pipeline/upload-candidates    → get_upload_candidates()
POST /api/pipeline/upload               → upload_and_process_files(file_paths, template_id?, model?)
```

#### Notion

```
GET  /api/notion/status                 → notion_service.check_connection()
GET  /api/notion/databases              → notion_service.list_databases()
POST /api/notion/databases/select       → notion_service.set_database_id(db_id)
GET  /api/notion/recordings             → notion_service.fetch_recordings(limit?)
POST /api/notion/match                  → notion_bridge.match_notion_to_chronos()
POST /api/notion/import/{page_id}       → notion_bridge.import_notion_recording(page_id)
POST /api/notion/import-all             → notion_bridge.import_all_unmatched()
POST /api/notion/writeback/{page_id}    → notion_bridge.write_back_to_notion(page_id)
POST /api/notion/writeback-all          → notion_bridge.write_back_all_matched()
GET  /api/notion/import-progress        → notion_bridge.get_import_progress()
GET  /api/notion/coverage               → notion_bridge.get_coverage_calendar()
```

#### Cost Tracking

```
GET  /api/costs/session                 → cost_tracker.get_session_cost()
GET  /api/costs/history                 → cost_tracker.get_cost_summary(days?)
GET  /api/costs/pricing                 → cost_tracker.get_model_pricing_table()
```

#### X-ray Telemetry

```
GET  /api/xray/events?since_seq=        → xray.get_recent_events(limit?, since_seq?)
POST /api/xray/clear                    → xray.clear_events()
GET  /api/xray/throughput?buckets=      → xray.get_throughput(buckets?)
WS   /api/xray/ws                       → WebSocket stream of real-time events
```

#### Auth

```
GET  /api/auth/plaud/url                → plaud_oauth.get_authorization_url()
POST /api/auth/plaud/exchange           → plaud_oauth.exchange_code_for_token(code, state)
GET  /api/auth/plaud/status             → plaud_oauth.token_status
GET  /api/auth/notion/url               → notion_oauth.get_authorization_url()
POST /api/auth/notion/exchange          → notion_oauth.exchange_code_for_token(code)
GET  /api/auth/notion/status            → notion_oauth.token_status
```

#### Health

```
GET  /api/health                        → ping
GET  /api/system/status                 → system_status (DB, Qdrant, Gemini, Plaud checks)
```

### Implementation Strategy

1. **Keep all existing `src/` code unchanged** — the FastAPI routes are thin wrappers around `ChronosDataService` and core services.
2. **Reuse `ChronosDataService` as the primary data layer** — inject it via FastAPI dependency injection.
3. **Add JWT authentication** — the iOS app authenticates with a user token; the server validates it on every request.
4. **Add CORS middleware** — allow requests from the iOS app's bundle identifier.
5. **Add WebSocket for X-ray** — replace polling with push-based real-time events.

### Key FastAPI Code Patterns

```python
# api/dependencies.py
from app_v2.services.data_service import get_data_service, ChronosDataService

def get_service() -> ChronosDataService:
    return get_data_service()

# api/routes/timeline.py
from fastapi import APIRouter, Depends, Query
from api.dependencies import get_service

router = APIRouter(prefix="/api", tags=["timeline"])

@router.get("/days")
async def list_days(
    start_date: str | None = None,
    end_date: str | None = None,
    service: ChronosDataService = Depends(get_service)
):
    days = service.get_days(start_date=start_date, end_date=end_date)
    return {"days": [d.__dict__ for d in days]}
```

---

## 5. Phase 1 — Cloud Infrastructure

### Database Migration: SQLite → PostgreSQL

**Current tables to migrate:**

| SQLAlchemy Model       | Table                     | Columns              | Notes                   |
| ---------------------- | ------------------------- | -------------------- | ----------------------- |
| `Recording`            | `recordings`              | 14 cols + JSON extra | Legacy, backward compat |
| `Segment`              | `segments`                | 12 cols              | Legacy segments         |
| `ChronosRecording`     | `chronos_recordings`      | 22 cols              | Primary recording store |
| `ChronosEvent`         | `chronos_events`          | 16 cols              | Primary event store     |
| `ChronosProcessingJob` | `chronos_processing_jobs` | 9 cols               | Job queue               |
| `ChronosWebhookEvent`  | `chronos_webhook_events`  | 8 cols               | Webhook log             |

**Migration steps:**

1. Replace `sqlite:///data/brain.db` with PostgreSQL connection string
2. Replace `_ensure_sqlite_additive_schema()` with Alembic migrations
3. Add connection pooling (`pool_size=5, max_overflow=10`)
4. Add a `users` table for multi-user support (future)

### Qdrant Cloud Setup

1. Create Qdrant Cloud cluster at cloud.qdrant.io
2. Create `chronos_events` collection with:
   - Vector size: 768 (or configured dim, up to 3072)
   - Distance: Cosine
   - Payload indexes: `day_of_week` (keyword), `hour_of_day` (integer), `timestamp` (keyword), `category` (keyword), `recording_id` (keyword), `start_ts_unix` (float)
3. Update `QDRANT_URL` and `QDRANT_API_KEY` in server config

### Server Deployment

**Recommended: Render.com or Railway**

```yaml
# render.yaml
services:
  - type: web
    name: chronos-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: chronos-db
          property: connectionString
      - key: QDRANT_URL
        sync: false
      - key: QDRANT_API_KEY
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: PLAUD_CLIENT_ID
        sync: false
      - key: PLAUD_CLIENT_SECRET
        sync: false
      - key: JWT_SECRET
        generateValue: true

  - type: worker
    name: chronos-pipeline
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/auto_sync.py
    envVars: *shared-env

databases:
  - name: chronos-db
    plan: starter
```

### Audio File Storage

Local audio files (`data/raw/`) need cloud storage:

- **Option A:** AWS S3 / Cloudflare R2 (cheap, durable)
- **Option B:** Supabase Storage (integrated with DB)
- Recordings are served via signed URLs (time-limited)
- iOS app streams audio directly from storage

---

## 6. Phase 2 — iOS App Foundation

### Xcode Project Structure

```
Chronos/
  ChronosApp.swift              ← @main, WindowGroup, app lifecycle
  ContentView.swift             ← Root NavigationSplitView (sidebar + detail)

  Models/
    Recording.swift             ← Codable struct (matches ChronosRecording)
    Event.swift                 ← Codable struct (matches ChronosEvent)
    DaySummary.swift            ← Codable struct
    SearchResult.swift          ← Codable struct
    GraphData.swift             ← Codable struct (nodes + edges)
    Stats.swift                 ← Codable struct
    Topic.swift                 ← Codable struct
    NotionRecording.swift       ← Codable struct
    PipelineStatus.swift        ← Codable struct
    XRayEvent.swift             ← Codable struct
    CostSummary.swift           ← Codable struct
    WorkflowStatus.swift        ← Codable struct

  Services/
    APIClient.swift             ← URLSession-based HTTP client (async/await)
    AuthManager.swift           ← Keychain token storage, JWT refresh
    PlaudAuthService.swift      ← ASWebAuthenticationSession for Plaud OAuth
    NotionAuthService.swift     ← ASWebAuthenticationSession for Notion OAuth
    WebSocketManager.swift      ← URLSession WebSocket for X-ray stream
    CacheManager.swift          ← SwiftData local cache + TTL invalidation
    SyncEngine.swift            ← Background sync coordinator

  ViewModels/
    TimelineViewModel.swift     ← @Observable, drives Timeline view
    RecordingDetailViewModel.swift
    SearchViewModel.swift
    TopicsViewModel.swift
    GraphViewModel.swift
    StatsViewModel.swift
    SyncViewModel.swift
    SettingsViewModel.swift
    NotionViewModel.swift
    XRayViewModel.swift

  Views/
    Sidebar/
      SidebarView.swift         ← Tab icons: Timeline, Topics, Graph, Stats, etc.
    Timeline/
      TimelineView.swift        ← Day list with heat-map calendar
      DayCardView.swift         ← Expandable day card
      TimelineStripView.swift   ← Horizontal hour-colored blocks
      RecordingCardView.swift   ← Quick recording preview
      HeatmapCalendarView.swift ← 30-day density grid
    RecordingDetail/
      RecordingDetailView.swift ← Tabbed: Overview, Events, Narrative, Transcript, Comparison
      EventCardView.swift       ← Time, category picker, confidence, text
      NarrativeView.swift       ← Flowing prose by time-gap grouping
      TranscriptView.swift      ← Raw text with search
    Search/
      SearchView.swift          ← Search bar + filters + results
      SearchResultCardView.swift
      AIAnswerView.swift        ← GPT-5.4 AI answer (markdown)
    Topics/
      TopicsGridView.swift      ← Scrollable topic grid
      TopicCardView.swift       ← Name + count badge
      TopicTimelineView.swift   ← Occurrences for one topic
    Graph/
      GraphContainerView.swift  ← WKWebView wrapper for Cytoscape
      GraphWebView.swift        ← UIViewRepresentable bridge
      GraphNodeDetailView.swift ← Sheet on node tap
    Stats/
      StatsView.swift           ← Stat cards + charts
      CategoryChartView.swift   ← Swift Charts horizontal bars
      HeatmapView.swift         ← Hour × category matrix
      SentimentChartView.swift  ← Sentiment distribution
      CostTickerView.swift      ← API cost session + historical
    Sync/
      SyncDashboardView.swift   ← Pipeline status, actions
      PipelineStatusView.swift  ← Progress bars per stage
      WorkflowMonitorView.swift ← Active workflow cards
      UploadCandidatesView.swift
    Notion/
      NotionView.swift          ← Connection status, database picker
      NotionPageListView.swift  ← Matched/unmatched pages
      NotionSyncEngineView.swift
    Settings/
      SettingsView.swift        ← Grouped Form sections
      ServiceStatusView.swift   ← Connection checks
      ModelConfigView.swift     ← AI model pickers
      CategoryEditorView.swift  ← Custom category definitions
    XRay/
      XRayView.swift            ← Real-time event feed
      XRayEventRow.swift        ← Source, op, message, duration
      XRayFilterBar.swift       ← 12 source category tabs

  Components/
    CategoryPill.swift          ← Color-coded category badge
    ConfidenceBadge.swift       ← high/medium/low badge
    SentimentIndicator.swift    ← -1.0 to 1.0 visual
    DurationLabel.swift         ← "2m 34s" format
    TimeRangeLabel.swift        ← "2:30 PM – 3:15 PM"
    LoadingView.swift           ← Skeleton/spinner states
    EmptyStateView.swift        ← Illustration + message
    ToastView.swift             ← Overlay notification
    MarkdownView.swift          ← Render AI answers

  Extensions/
    Date+Formatting.swift
    Color+Category.swift
    String+Helpers.swift

  Resources/
    Assets.xcassets             ← App icon, color sets (from CSS vars)
    graph.html                  ← Cytoscape.js bundle for WKWebView
    graph.css                   ← Graph stylesheet (from GRAPH_STYLESHEET)
    graph.js                    ← Cytoscape rendering logic

  Preview Content/
    PreviewData.swift           ← Mock data for SwiftUI previews
```

### Root App Structure

```swift
@main
struct ChronosApp: App {
    @State private var authManager = AuthManager()
    @State private var apiClient: APIClient

    init() {
        let auth = AuthManager()
        self._authManager = State(initialValue: auth)
        self._apiClient = State(initialValue: APIClient(authManager: auth))
    }

    var body: some Scene {
        WindowGroup {
            if authManager.isAuthenticated {
                ContentView()
                    .environment(apiClient)
                    .environment(authManager)
            } else {
                OnboardingView()
                    .environment(authManager)
            }
        }
    }
}
```

### Navigation Architecture

```swift
struct ContentView: View {
    @State private var selectedTab: AppTab = .timeline
    @State private var selectedRecording: Recording?

    var body: some View {
        TabView(selection: $selectedTab) {
            TimelineView(selectedRecording: $selectedRecording)
                .tabItem { Label("Timeline", systemImage: "calendar.day.timeline.left") }
                .tag(AppTab.timeline)

            TopicsGridView()
                .tabItem { Label("Topics", systemImage: "tag") }
                .tag(AppTab.topics)

            SearchView(selectedRecording: $selectedRecording)
                .tabItem { Label("Search", systemImage: "magnifyingglass") }
                .tag(AppTab.search)

            GraphContainerView()
                .tabItem { Label("Graph", systemImage: "point.3.connected.trianglepath.dotted") }
                .tag(AppTab.graph)

            StatsView()
                .tabItem { Label("Stats", systemImage: "chart.bar") }
                .tag(AppTab.stats)

            // More tab for Notion, Sync, Settings, X-ray
            MoreView()
                .tabItem { Label("More", systemImage: "ellipsis") }
                .tag(AppTab.more)
        }
        .sheet(item: $selectedRecording) { recording in
            RecordingDetailView(recording: recording)
        }
    }
}

enum AppTab: String, CaseIterable {
    case timeline, topics, search, graph, stats, more
}
```

### APIClient Foundation

```swift
@Observable
class APIClient {
    private let session: URLSession
    private let baseURL: URL
    private let authManager: AuthManager
    private let decoder: JSONDecoder

    init(authManager: AuthManager, baseURL: URL = URL(string: "https://chronos-api.onrender.com")!) {
        self.authManager = authManager
        self.baseURL = baseURL
        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .iso8601

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = true
        self.session = URLSession(configuration: config)
    }

    func get<T: Decodable>(_ path: String, query: [String: String] = [:]) async throws -> T {
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        components.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }

        var request = URLRequest(url: components.url!)
        request.setValue("Bearer \(try authManager.getToken())", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
            throw APIError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        return try decoder.decode(T.self, from: data)
    }

    func post<T: Decodable, B: Encodable>(_ path: String, body: B) async throws -> T {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(try authManager.getToken())", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
            throw APIError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        return try decoder.decode(T.self, from: data)
    }
}
```

---

## 7. Phase 3 — Timeline View

### Mapping: Dash → SwiftUI

| Dash Component                | SwiftUI Equivalent                                   |
| ----------------------------- | ---------------------------------------------------- |
| `create_day_view()`           | `TimelineView` (LazyVStack inside ScrollView)        |
| `create_day_card()`           | `DayCardView` (DisclosureGroup or custom expandable) |
| `create_day_timeline_strip()` | `TimelineStripView` (Canvas or custom Shape drawing) |
| `create_recording_card()`     | `RecordingCardView` (tappable card → sheet)          |
| Heat-map calendar             | `HeatmapCalendarView` (LazyVGrid of colored cells)   |
| Category bar                  | `CategoryBarView` (GeometryReader + HStack)          |
| Sentiment sparkline           | `SentimentSparklineView` (Swift Charts or Canvas)    |
| Timeline range filter         | `Picker` (segmented: 1d, 7d, 14d, 30d, All)          |

### API Calls

```swift
// TimelineViewModel.swift
@Observable
class TimelineViewModel {
    var days: [DaySummary] = []
    var selectedRange: TimeRange = .week
    var isLoading = false

    func loadDays() async {
        isLoading = true
        defer { isLoading = false }

        let response: DaysResponse = try await apiClient.get("/api/days/filled",
            query: ["last_n_days": "\(selectedRange.days)"])
        days = response.days
    }
}
```

### Data Flow

```
TimelineView
  ├── Picker (range: 1d / 7d / 14d / 30d / All)
  ├── HeatmapCalendarView (30-day grid, tap → scroll to day)
  └── LazyVStack {
        ForEach(days) { day in
          DayCardView(day)
            ├── Header: date, event count, AI summary (1-line)
            ├── TimelineStripView (hour-colored blocks)
            ├── CategoryBarView (stacked horizontal)
            └── DisclosureGroup {
                  ForEach(day.recordings) { rec in
                    RecordingCardView(rec)
                      ├── Time range, duration
                      ├── Top category pill
                      ├── Event count
                      └── Top keywords
                  }
                }
        }
      }
```

---

## 8. Phase 4 — Recording Detail View

### Mapping: Dash → SwiftUI

| Dash Component                    | SwiftUI Equivalent                                                          |
| --------------------------------- | --------------------------------------------------------------------------- |
| 5-tab detail panel                | `TabView` with `.tabViewStyle(.page)` or segmented picker                   |
| Overview tab (AI summary)         | `OverviewTab` — Text + Extracted Data cards                                 |
| Events tab (filterable list)      | `EventsTab` — `@Searchable` + `List`                                        |
| Narrative tab (flowing prose)     | `NarrativeTab` — `Text` blocks with `.font(.body)`                          |
| Transcript tab                    | `TranscriptTab` — `ScrollView { Text(transcript).textSelection(.enabled) }` |
| Comparison tab                    | `ComparisonTab` — two-column `HStack` with scrollable text                  |
| Event card with category dropdown | `EventCardView` + `Picker` for category                                     |
| Confidence badge                  | `ConfidenceBadge` (custom view: H/M/L colored pill)                         |
| "Run Plaud AI" button             | `Button` → `POST /api/recordings/{id}/workflow`                             |

### API Calls

```swift
// RecordingDetailViewModel.swift
@Observable
class RecordingDetailViewModel {
    var detail: RecordingDetail?
    var selectedTab: DetailTab = .overview
    var eventFilter: String = ""

    func load(recordingId: String) async {
        detail = try await apiClient.get("/api/recordings/\(recordingId)")
    }

    func saveCategory(eventId: String, category: String) async {
        try await apiClient.put(
            "/api/recordings/\(detail!.summary.recordingId)/events/\(eventId)/category",
            body: CategoryOverride(category: category)
        )
    }

    func submitWorkflow(templateId: String?, model: String) async {
        let result: WorkflowResult = try await apiClient.post(
            "/api/recordings/\(detail!.summary.recordingId)/workflow",
            body: WorkflowRequest(templateId: templateId, model: model)
        )
    }
}
```

### Events Tab — Inline Filtering

```swift
struct EventsTab: View {
    let events: [Event]
    @State private var filterText = ""

    var filteredEvents: [Event] {
        guard !filterText.isEmpty else { return events }
        return events.filter { $0.cleanText.localizedCaseInsensitiveContains(filterText)
            || $0.category.localizedCaseInsensitiveContains(filterText)
            || $0.keywords.joined().localizedCaseInsensitiveContains(filterText)
        }
    }

    var body: some View {
        List {
            ForEach(filteredEvents) { event in
                EventCardView(event: event)
            }
        }
        .searchable(text: $filterText, prompt: "Filter events...")
    }
}
```

---

## 9. Phase 5 — Search & AI Answers

### Mapping: Dash → SwiftUI

| Dash Component        | SwiftUI Equivalent                                              |
| --------------------- | --------------------------------------------------------------- |
| Search bar + button   | `.searchable()` modifier on NavigationStack                     |
| Filter toggle         | `DisclosureGroup("Filters")`                                    |
| Category multi-select | `ForEach(categories)` + `Toggle` or multi-select `Picker`       |
| Date range picker     | Two `DatePicker` (from/to)                                      |
| Search result card    | `SearchResultCardView` (category pill, score %, text snippet)   |
| AI Answer section     | `AIAnswerView` — `MarkdownView` + model/tokens/latency metadata |

### API Calls

```swift
// SearchViewModel.swift
@Observable
class SearchViewModel {
    var query = ""
    var results: [SearchResult] = []
    var aiAnswer: AIAnswer?
    var selectedCategories: Set<String> = []
    var startDate: Date?
    var endDate: Date?
    var isSearching = false

    func search() async {
        isSearching = true
        defer { isSearching = false }

        let body = SearchRequest(
            query: query,
            limit: 20,
            categories: Array(selectedCategories),
            startDate: startDate?.iso8601String,
            endDate: endDate?.iso8601String
        )
        let response: SearchResponse = try await apiClient.post("/api/search", body: body)
        results = response.results
        aiAnswer = response.aiAnswer
    }
}
```

### AI Answer Display

```swift
struct AIAnswerView: View {
    let answer: AIAnswer

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "brain.head.profile")
                Text("AI Answer")
                    .font(.headline)
                Spacer()
                Text(answer.model)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            MarkdownView(answer.text)  // Render markdown from GPT-5.4

            HStack {
                Label("\(answer.tokensUsed) tokens", systemImage: "number")
                Label("\(answer.latencyMs)ms", systemImage: "clock")
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
```

---

## 10. Phase 6 — Topics View

### Mapping: Dash → SwiftUI

| Dash Component            | SwiftUI Equivalent                                        |
| ------------------------- | --------------------------------------------------------- |
| Topics grid (100 topics)  | `LazyVGrid(columns: adaptive(minimum: 140))`              |
| Topic card (name + count) | `TopicCardView` — scaled sizing (hot/warm/normal)         |
| Sort dropdown             | `Picker` (freq-desc, freq-asc, alpha-asc, alpha-desc)     |
| Topic timeline            | `TopicTimelineView` — `NavigationLink` push               |
| Occurrence card           | `OccurrenceCardView` — date, time, category, text snippet |

### Data Flow

```
TopicsGridView
  ├── HStack { searchField, Picker(sort) }
  └── LazyVGrid {
        ForEach(filteredTopics) { topic in
          NavigationLink(value: topic) {
            TopicCardView(topic)
          }
        }
      }
      .navigationDestination(for: Topic.self) { topic in
        TopicTimelineView(topic: topic)  // List of occurrences
      }
```

---

## 11. Phase 7 — Knowledge Graph

### Strategy: WKWebView + Cytoscape.js

The knowledge graph is the most complex view. Native graph libraries on iOS are immature. The best approach is to **embed a WKWebView** that runs the existing Cytoscape.js visualization:

```swift
struct GraphWebView: UIViewRepresentable {
    let graphData: GraphData
    let selectedLayout: String
    var onNodeTap: (GraphNode) -> Void

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.userContentController.add(context.coordinator, name: "nodeTapped")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground

        // Load bundled graph.html
        if let url = Bundle.main.url(forResource: "graph", withExtension: "html") {
            webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        }
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Inject graph data as JSON
        let json = try! JSONEncoder().encode(graphData)
        let jsonStr = String(data: json, encoding: .utf8)!
        webView.evaluateJavaScript("updateGraph(\(jsonStr), '\(selectedLayout)')")
    }
}
```

### Bundled graph.html

```html
<!-- Resources/graph.html -->
<!DOCTYPE html>
<html>
  <head>
    <script src="cytoscape.min.js"></script>
    <script src="cytoscape-cose-bilkent.js"></script>
    <style>
      /* Port existing GRAPH_STYLESHEET from graph.py */
      body {
        margin: 0;
        background: transparent;
      }
      #cy {
        width: 100%;
        height: 100vh;
      }
    </style>
  </head>
  <body>
    <div id="cy"></div>
    <script>
      let cy;
      function updateGraph(data, layout) {
          cy = cytoscape({
              container: document.getElementById('cy'),
              elements: data.elements,
              style: [...],  // Port from GRAPH_STYLESHEET constant
              layout: { name: layout }
          });
          cy.on('tap', 'node', function(evt) {
              webkit.messageHandlers.nodeTapped.postMessage(JSON.stringify(evt.target.data()));
          });
      }
    </script>
  </body>
</html>
```

### Graph Controls (Native)

```swift
struct GraphContainerView: View {
    @State private var viewModel = GraphViewModel()
    @State private var selectedLayout = "cose-bilkent"
    @State private var selectedNode: GraphNode?

    var body: some View {
        ZStack {
            GraphWebView(
                graphData: viewModel.graphData,
                selectedLayout: selectedLayout,
                onNodeTap: { node in selectedNode = node }
            )

            VStack {
                // Layout picker overlay
                Picker("Layout", selection: $selectedLayout) {
                    Text("Force").tag("cose-bilkent")
                    Text("Hierarchy").tag("dagre")
                    Text("Circle").tag("circle")
                    Text("Concentric").tag("concentric")
                }
                .pickerStyle(.segmented)
                .padding()

                Spacer()
            }
        }
        .sheet(item: $selectedNode) { node in
            GraphNodeDetailView(node: node)
        }
    }
}
```

---

## 12. Phase 8 — Stats & Analytics

### Mapping: Dash → SwiftUI

| Dash Component                   | SwiftUI Equivalent                      |
| -------------------------------- | --------------------------------------- |
| 8 stat cards                     | `LazyVGrid` with `StatCardView`         |
| Category chart (horizontal bars) | `Swift Charts` — `BarMark` horizontal   |
| Hour × Category heatmap          | `Grid` or `Canvas` with color intensity |
| Sentiment histogram              | `Swift Charts` — `RectangleMark`        |
| Productivity insights            | Text cards                              |
| API cost ticker                  | `CostTickerView` — session + historical |

### Swift Charts Example

```swift
import Charts

struct CategoryChartView: View {
    let categories: [String: Int]

    var body: some View {
        Chart {
            ForEach(categories.sorted(by: { $0.value > $1.value }), id: \.key) { cat, count in
                BarMark(
                    x: .value("Count", count),
                    y: .value("Category", cat)
                )
                .foregroundStyle(Color.forCategory(cat))
            }
        }
        .chartXAxisLabel("Events")
        .frame(height: CGFloat(categories.count * 44))
    }
}
```

---

## 13. Phase 9 — Sync & Pipeline Control

### Mapping: Dash → SwiftUI

| Dash Section             | SwiftUI Equivalent                                                       |
| ------------------------ | ------------------------------------------------------------------------ |
| Plaud Auth Card          | `PlaudConnectionCard` (status + connect button)                          |
| Pipeline Status (counts) | `PipelineStatusCard` (total/completed/pending/failed)                    |
| Plaud Enrichment         | `EnrichmentCard` (AI summary count, in-flight workflows)                 |
| Full Sync button         | `Button("Full Sync")` → `POST /api/pipeline/run` body: `{stage: "full"}` |
| Reset Stuck              | `Button("Reset Stuck")` → `POST /api/pipeline/reset-stuck`               |
| Submit Workflows         | Configurable: days slider, batch size, template, model                   |
| Active Workflows grid    | `LazyVStack` of `WorkflowCardView`                                       |
| Upload Candidates        | `List` of local files (future: share sheet → API upload)                 |
| Sync History             | `List` of recent jobs (trigger, timestamp, status)                       |

### Pipeline Progress Polling

```swift
@Observable
class SyncViewModel {
    var dbStats: RecordingDbStats?
    var workflowStats: WorkflowStats?
    var pipelineRunning = false
    var pipelineProgress: PipelineProgress?

    func runPipeline(stage: String = "full") async {
        pipelineRunning = true
        let _: PipelineResult = try await apiClient.post("/api/pipeline/run", body: ["stage": stage])

        // Poll for progress
        while pipelineRunning {
            try await Task.sleep(for: .seconds(2))
            pipelineProgress = try await apiClient.get("/api/pipeline/status")
            if pipelineProgress?.status == "complete" || pipelineProgress?.status == "failed" {
                pipelineRunning = false
            }
        }

        // Refresh stats
        await loadStats()
    }
}
```

---

## 14. Phase 10 — Settings

### Mapping: 12 Dash Sections → SwiftUI Form

```swift
struct SettingsView: View {
    @State private var viewModel = SettingsViewModel()

    var body: some View {
        Form {
            Section("Service Connections") {
                ServiceStatusRow("Plaud", status: viewModel.plaudStatus)
                ServiceStatusRow("Notion", status: viewModel.notionStatus)
                ServiceStatusRow("Gemini", status: viewModel.geminiStatus)
                ServiceStatusRow("OpenAI", status: viewModel.openaiStatus)
                ServiceStatusRow("Qdrant", status: viewModel.qdrantStatus)
                ServiceStatusRow("Database", status: viewModel.dbStatus)
            }

            Section("AI Models") {
                Picker("Cleaning Model", selection: $viewModel.cleaningModel) {
                    Text("gemini-3-flash-preview").tag("gemini-3-flash-preview")
                    // ...more models
                }
                Picker("Analyst Model", selection: $viewModel.analystModel) { ... }
                Picker("OpenAI Model", selection: $viewModel.openaiModel) {
                    Text("gpt-5.4").tag("gpt-5.4")
                    Text("gpt-5.4-pro").tag("gpt-5.4-pro")
                    Text("gpt-5-mini").tag("gpt-5-mini")
                    Text("gpt-5-nano").tag("gpt-5-nano")
                }
                Picker("Thinking Level", selection: $viewModel.thinkingLevel) { ... }
                Slider(value: $viewModel.temperature, in: 0...2, step: 0.1)
            }

            Section("Embedding") {
                Picker("Dimensions", selection: $viewModel.embeddingDim) {
                    ForEach([128, 256, 512, 768, 1024, 1536, 3072], id: \.self) { dim in
                        Text("\(dim)").tag(dim)
                    }
                }
            }

            Section("Categories") {
                // Color grid of built-in categories
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))]) {
                    ForEach(EventCategory.allCases, id: \.self) { cat in
                        CategoryPill(category: cat.rawValue)
                    }
                }
            }

            Section("API Costs") {
                NavigationLink("View Cost Dashboard") {
                    CostDashboardView()
                }
            }

            Section("About") {
                LabeledContent("Version", value: Bundle.main.appVersion)
                LabeledContent("Backend", value: viewModel.serverVersion)
            }
        }
        .navigationTitle("Settings")
    }
}
```

---

## 15. Phase 11 — Notion Integration

### Mapping: 20 Dash Callbacks → SwiftUI

| Dash Callback             | SwiftUI Action                         |
| ------------------------- | -------------------------------------- |
| `notion-fetch-btn`        | `GET /api/notion/recordings`           |
| `notion-discover-dbs-btn` | `GET /api/notion/databases`            |
| `notion-select-db`        | `POST /api/notion/databases/select`    |
| Import single             | `POST /api/notion/import/{page_id}`    |
| Import all                | `POST /api/notion/import-all`          |
| Write back single         | `POST /api/notion/writeback/{page_id}` |
| Write back all            | `POST /api/notion/writeback-all`       |
| Coverage calendar         | `GET /api/notion/coverage`             |

### View Structure

```
NotionView
  ├── NotionHeroCard (connected?, database title, page counts)
  ├── Section("Databases") {
  │     DatabasePickerView (list of databases, tap to select)
  │   }
  ├── Section("Pages") {
  │     LazyVStack {
  │       ForEach(pages) { page in
  │         NotionPageRow(page, matchStatus)
  │           ├── Title, date, source badge
  │           ├── Match status (✅ Matched / ⬜ Unique)
  │           └── Swipe actions: Import, Write Back
  │       }
  │     }
  │   }
  ├── Section("Sync Actions") {
  │     Button("Import All Unmatched")
  │     Button("Push All to Notion")
  │   }
  └── Section("Coverage Calendar") {
        CoverageCalendarView (grid: date → has_chronos, has_notion)
      }
```

---

## 16. Phase 12 — X-ray Activity Monitor

### Strategy: WebSocket-Powered Live Feed

Instead of polling (like the current JS PiP), use a WebSocket connection for real-time streaming:

```swift
@Observable
class XRayViewModel {
    var events: [XRayEvent] = []
    var selectedSource: String? = nil  // nil = all
    var isConnected = false
    private var wsTask: URLSessionWebSocketTask?

    var filteredEvents: [XRayEvent] {
        guard let source = selectedSource else { return events }
        return events.filter { $0.source == source }
    }

    func connect() {
        let url = URL(string: "wss://chronos-api.onrender.com/api/xray/ws")!
        wsTask = URLSession.shared.webSocketTask(with: url)
        wsTask?.resume()
        isConnected = true
        receiveLoop()
    }

    private func receiveLoop() {
        wsTask?.receive { [weak self] result in
            if case .success(let message) = result,
               case .string(let text) = message,
               let data = text.data(using: .utf8),
               let event = try? JSONDecoder().decode(XRayEvent.self, from: data) {
                Task { @MainActor in
                    self?.events.insert(event, at: 0)
                    if (self?.events.count ?? 0) > 500 {
                        self?.events.removeLast()
                    }
                }
            }
            self?.receiveLoop()
        }
    }
}
```

### View

```swift
struct XRayView: View {
    @State private var viewModel = XRayViewModel()

    let sources = ["ingest", "gemini", "embed", "qdrant", "graph", "search",
                   "data", "nav", "pipeline", "detail", "day", "sync"]

    var body: some View {
        VStack(spacing: 0) {
            // Source filter tabs
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    FilterChip("All", selected: viewModel.selectedSource == nil) {
                        viewModel.selectedSource = nil
                    }
                    ForEach(sources, id: \.self) { source in
                        FilterChip(source, selected: viewModel.selectedSource == source) {
                            viewModel.selectedSource = source
                        }
                    }
                }
                .padding(.horizontal)
            }

            // Event feed
            List(viewModel.filteredEvents) { event in
                XRayEventRow(event: event)
            }
        }
        .onAppear { viewModel.connect() }
    }
}
```

---

## 17. Phase 13 — Authentication (Plaud + Notion OAuth)

### Critical: Plaud OAuth Quirks

⚠️ **Plaud's OAuth is non-standard.** Key gotchas from the existing codebase:

1. **State parameter required in POST body** — Unlike standard OAuth 2.0, Plaud requires `state` in the token exchange POST body. Without it: `401 AUTH_STATE_INVALID`.
2. **Basic Auth required** — Token exchange uses `Authorization: Basic base64(client_id:client_secret)`.
3. **Triple callback hit** — Plaud hits the callback URL 3 times: OPTIONS preflight, XHR GET (real), browser redirect GET (duplicate). Handler must be idempotent.
4. **No `grant_type` parameter** — Unlike standard OAuth, Plaud doesn't want `grant_type=authorization_code`.

### iOS OAuth Flow (ASWebAuthenticationSession)

```swift
class PlaudAuthService {
    func authenticate() async throws -> PlaudTokens {
        // 1. Get auth URL from our backend (which generates state)
        let urlResponse: AuthURLResponse = try await apiClient.get("/api/auth/plaud/url")
        let authURL = URL(string: urlResponse.authUrl)!

        // 2. Present OAuth in system browser
        let callbackURL = try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(
                url: authURL,
                callbackURLScheme: "chronos"  // chronos://auth/plaud/callback
            ) { url, error in
                if let url { continuation.resume(returning: url) }
                else { continuation.resume(throwing: error ?? AuthError.cancelled) }
            }
            session.presentationContextProvider = self
            session.prefersEphemeralWebBrowserSession = true
            session.start()
        }

        // 3. Extract code from callback URL
        let components = URLComponents(url: callbackURL, resolvingAgainstBaseURL: false)!
        let code = components.queryItems?.first(where: { $0.name == "code" })?.value ?? ""
        let state = components.queryItems?.first(where: { $0.name == "state" })?.value ?? ""

        // 4. Exchange code via our backend (which handles Plaud's quirky requirements)
        let tokens: PlaudTokens = try await apiClient.post("/api/auth/plaud/exchange",
            body: ["code": code, "state": state])

        // 5. Store in Keychain
        try authManager.storePlaudTokens(tokens)
        return tokens
    }
}
```

### Notion OAuth Flow

Simpler than Plaud (standard OAuth 2.0, no refresh tokens needed):

```swift
class NotionAuthService {
    func authenticate() async throws {
        let urlResponse: AuthURLResponse = try await apiClient.get("/api/auth/notion/url")
        let authURL = URL(string: urlResponse.authUrl)!

        let callbackURL = try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(
                url: authURL,
                callbackURLScheme: "chronos"
            ) { url, error in
                if let url { continuation.resume(returning: url) }
                else { continuation.resume(throwing: error ?? AuthError.cancelled) }
            }
            session.presentationContextProvider = self
            session.prefersEphemeralWebBrowserSession = true
            session.start()
        }

        let code = URLComponents(url: callbackURL, resolvingAgainstBaseURL: false)!
            .queryItems?.first(where: { $0.name == "code" })?.value ?? ""

        let _: NotionTokens = try await apiClient.post("/api/auth/notion/exchange",
            body: ["code": code])
    }
}
```

### Backend Redirect Configuration

On the server, configure redirect URIs that Plaud/Notion will accept:

- **Plaud:** `https://chronos-api.onrender.com/auth/plaud/callback` — server receives the code, then the iOS app polls `/api/auth/plaud/status` or the server returns via deep link.
- **Alternative:** Use `chronos://auth/plaud/callback` as redirect URI if Plaud allows custom schemes.

---

## 18. Phase 14 — Push Notifications & Background Sync

### APNs Integration

When the pipeline finishes processing a recording, or a Plaud webhook fires, the server sends a push notification:

```python
# api/push.py (server-side)
import httpx

async def send_push(device_token: str, title: str, body: str, data: dict = {}):
    """Send APNs push notification via HTTP/2."""
    # Use PyAPNs2 or direct HTTP/2 to api.push.apple.com
    payload = {
        "aps": {
            "alert": {"title": title, "body": body},
            "sound": "default",
            "badge": 1
        },
        "data": data
    }
    # ... send via APNs
```

### Notification Types

| Trigger                           | Title              | Body                                      | Data             |
| --------------------------------- | ------------------ | ----------------------------------------- | ---------------- |
| Pipeline: new recording processed | "New Recording"    | "March 21, 2:30 PM — 15 events extracted" | `{recording_id}` |
| Plaud workflow complete           | "AI Summary Ready" | "Recording 'March 21 Meeting' enriched"   | `{recording_id}` |
| Notion sync complete              | "Notion Sync"      | "Imported 3 recordings from Notion"       | `{}`             |
| Pipeline error                    | "Pipeline Error"   | "Failed to process recording: {error}"    | `{recording_id}` |

### Background App Refresh

```swift
// ChronosApp.swift
struct ChronosApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup { ContentView() }
            .backgroundTask(.appRefresh("chronos.sync")) {
                await SyncEngine.shared.backgroundSync()
            }
    }
}

class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { _, _ in }
        application.registerForRemoteNotifications()
        return true
    }

    func application(_ application: UIApplication,
                     didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let token = deviceToken.map { String(format: "%02.2hhx", $0) }.joined()
        Task { try await APIClient.shared.post("/api/push/register", body: ["token": token]) }
    }
}
```

---

## 19. Phase 15 — Offline Support & Caching

### SwiftData Cache Layer

```swift
import SwiftData

@Model
class CachedDay {
    @Attribute(.unique) var date: String
    var summary: String
    var eventCount: Int
    var categoryPercentages: [String: Double]
    var topKeywords: [String]
    var lastFetched: Date

    var isStale: Bool { Date().timeIntervalSince(lastFetched) > 300 } // 5 min TTL
}

@Model
class CachedRecording {
    @Attribute(.unique) var recordingId: String
    var title: String?
    var date: Date
    var durationSeconds: Int
    var topCategory: String
    var eventCount: Int
    var transcript: String?
    var aiSummary: String?
    var lastFetched: Date
}

@Model
class CachedSearchResult {
    var query: String
    var resultJson: Data  // Encoded SearchResponse
    var timestamp: Date
}
```

### Cache Strategy

| Data Type        | TTL              | Strategy                                     |
| ---------------- | ---------------- | -------------------------------------------- |
| Day summaries    | 5 min            | Fetch → cache → show cached while refreshing |
| Recording detail | 30 min           | Cache on first view, refresh on demand       |
| Search results   | 10 min per query | Cache last 20 queries                        |
| Graph data       | 1 hour           | Heavy payload, cache aggressively            |
| Stats            | 5 min            | Background refresh                           |
| Topics           | 15 min           | Cache full list                              |
| X-ray events     | No cache         | Real-time only                               |

### Offline Queue

```swift
@Observable
class SyncEngine {
    static let shared = SyncEngine()

    /// Queue of mutations made offline (category overrides, etc.)
    @Model class PendingMutation {
        var endpoint: String
        var method: String
        var bodyJson: Data
        var createdAt: Date
    }

    func enqueue(endpoint: String, method: String, body: Encodable) {
        // Store in SwiftData when offline
    }

    func flushQueue() async {
        // Replay all pending mutations when back online
    }
}
```

---

## 20. Phase 16 — Polish, Accessibility & App Store

### Accessibility

- All views support **Dynamic Type** (no hardcoded font sizes)
- `accessibilityLabel` on category pills, confidence badges, sentiment indicators
- VoiceOver descriptions for graph nodes
- **Reduce Motion**: Skip animations when enabled
- High contrast: category colors pass WCAG AA on dark/light backgrounds

### Dark Mode

The current Dash app has CSS variables for a light theme. SwiftUI automatically supports dark mode via semantic colors:

```swift
extension Color {
    static let chronosBg = Color(.systemBackground)
    static let chronosCard = Color(.secondarySystemBackground)
    static let chronosText = Color(.label)
    static let chronosSecondary = Color(.secondaryLabel)
}
```

### Keyboard Shortcuts (iPad)

| Shortcut | Action                       |
| -------- | ---------------------------- |
| ⌘F       | Focus search                 |
| Esc      | Close detail / dismiss modal |
| ⌘1-5     | Switch tabs                  |
| ⌘R       | Refresh current view         |

### App Store Checklist

- [ ] App icon (1024×1024)
- [ ] Screenshots (6.7", 6.5", 5.5", iPad Pro)
- [ ] Privacy policy URL
- [ ] App description & keywords
- [ ] Age rating
- [ ] Privacy labels (data types: voice recordings, usage data, analytics)
- [ ] Review notes (test account credentials)
- [ ] In-app purchases? (probably not for v1)

---

## 21. Complete API Endpoint Specification

### Full Route Table (58 endpoints)

| #   | Method | Path                                         | Params                                                 | Returns                                     | Source Function                       |
| --- | ------ | -------------------------------------------- | ------------------------------------------------------ | ------------------------------------------- | ------------------------------------- |
| 1   | `GET`  | `/api/health`                                | —                                                      | `{status, version}`                         | ping                                  |
| 2   | `GET`  | `/api/system/status`                         | —                                                      | `{database, qdrant, gemini, plaud, openai}` | system_status                         |
| 3   | `GET`  | `/api/days`                                  | `start_date?, end_date?`                               | `{days: DaySummary[]}`                      | `get_days()`                          |
| 4   | `GET`  | `/api/days/filled`                           | `last_n_days?`                                         | `{days: DaySummary[]}`                      | `get_days_filled()`                   |
| 5   | `GET`  | `/api/days/{date}`                           | —                                                      | `DaySummary`                                | `get_day_detail()`                    |
| 6   | `GET`  | `/api/recordings`                            | `status?, limit?, offset?`                             | `{recordings[], count}`                     | list_recordings                       |
| 7   | `GET`  | `/api/recordings/{id}`                       | —                                                      | `RecordingDetail`                           | `get_recording_detail()`              |
| 8   | `GET`  | `/api/recordings/{id}/events`                | —                                                      | `{events: Event[]}`                         | `get_events_for_recording()`          |
| 9   | `GET`  | `/api/recordings/{id}/transcript`            | —                                                      | `{transcript}`                              | `get_transcript()`                    |
| 10  | `GET`  | `/api/recordings/{id}/ai-summary`            | —                                                      | `{summary}`                                 | `get_ai_summary()`                    |
| 11  | `GET`  | `/api/recordings/{id}/extracted`             | —                                                      | `{data}`                                    | `get_extracted_data()`                |
| 12  | `GET`  | `/api/recordings/{id}/workflow`              | —                                                      | `WorkflowStatus`                            | `get_workflow_status_for_recording()` |
| 13  | `GET`  | `/api/recordings/{id}/plaud-transcript`      | —                                                      | `{transcript}`                              | `get_plaud_workflow_transcript()`     |
| 14  | `POST` | `/api/recordings/{id}/workflow`              | `{template_id?, model?}`                               | `WorkflowResult`                            | `submit_single_recording_workflow()`  |
| 15  | `PUT`  | `/api/recordings/{id}/events/{eid}/category` | `{category}`                                           | `{success}`                                 | `save_category_override()`            |
| 16  | `POST` | `/api/search`                                | `{query, limit?, categories?, start_date?, end_date?}` | `{results[], aiAnswer?}`                    | `search()` + `ask()`                  |
| 17  | `POST` | `/api/ask`                                   | `{question, reasoning?}`                               | `{answer, sources[], model, tokens}`        | `ask_chronos()`                       |
| 18  | `GET`  | `/api/topics`                                | —                                                      | `{topics: [{name, count}]}`                 | `get_all_topics()`                    |
| 19  | `GET`  | `/api/topics/{name}`                         | —                                                      | `TopicTimeline`                             | `get_topic_timeline()`                |
| 20  | `GET`  | `/api/graph`                                 | `max_nodes?, entity_types?`                            | `GraphData`                                 | `get_graph_data()`                    |
| 21  | `GET`  | `/api/graph/entities`                        | `query?, type?`                                        | `{entities[]}`                              | `search_entities_graph()`             |
| 22  | `POST` | `/api/graph/query`                           | `{query}`                                              | `{answer, communities[]}`                   | `answer_global_query()`               |
| 23  | `GET`  | `/api/stats`                                 | —                                                      | `Stats`                                     | `get_stats()`                         |
| 24  | `GET`  | `/api/stats/recording-db`                    | —                                                      | `{pending, processing, failed, completed}`  | `get_recording_db_stats()`            |
| 25  | `GET`  | `/api/stats/workflows`                       | `days_back?`                                           | `WorkflowStats`                             | `get_plaud_workflow_stats()`          |
| 26  | `POST` | `/api/pipeline/run`                          | `{stage}`                                              | `{job_id, status}`                          | run_pipeline                          |
| 27  | `GET`  | `/api/pipeline/status`                       | —                                                      | `PipelineProgress`                          | pipeline_progress                     |
| 28  | `POST` | `/api/pipeline/reset-stuck`                  | —                                                      | `{count}`                                   | `reset_stuck_recordings()`            |
| 29  | `POST` | `/api/pipeline/workflows/submit`             | `{days_back?, limit?, template_id?, model?}`           | `{submitted[], skipped[], errors[]}`        | `submit_plaud_workflows()`            |
| 30  | `POST` | `/api/pipeline/workflows/refresh`            | `{days_back?, limit?}`                                 | `{pending, completed, failed}`              | `refresh_plaud_workflow_statuses()`   |
| 31  | `GET`  | `/api/pipeline/upload-candidates`            | —                                                      | `{files[]}`                                 | `get_upload_candidates()`             |
| 32  | `POST` | `/api/pipeline/upload`                       | `{file_paths, template_id?, model?}`                   | `{uploaded[], errors[]}`                    | `upload_and_process_files()`          |
| 33  | `GET`  | `/api/notion/status`                         | `quick?`                                               | `NotionSyncStatus`                          | `check_connection()`                  |
| 34  | `GET`  | `/api/notion/databases`                      | —                                                      | `{databases[]}`                             | `list_databases()`                    |
| 35  | `POST` | `/api/notion/databases/select`               | `{db_id}`                                              | `{success}`                                 | `set_database_id()`                   |
| 36  | `GET`  | `/api/notion/recordings`                     | `limit?`                                               | `{recordings: NotionRecording[]}`           | `fetch_recordings()`                  |
| 37  | `POST` | `/api/notion/match`                          | —                                                      | `{matches: {page_id: recording_id?}}`       | `match_notion_to_chronos()`           |
| 38  | `POST` | `/api/notion/import/{page_id}`               | `{process?, index?}`                                   | `{success, message}`                        | `import_notion_recording()`           |
| 39  | `POST` | `/api/notion/import-all`                     | `{process?, index?}`                                   | `{success, failed, errors[]}`               | `import_all_unmatched()`              |
| 40  | `POST` | `/api/notion/writeback/{page_id}`            | —                                                      | `{success, message}`                        | `write_back_to_notion()`              |
| 41  | `POST` | `/api/notion/writeback-all`                  | `{match_map}`                                          | `{success, failed, errors[]}`               | `write_back_all_matched()`            |
| 42  | `GET`  | `/api/notion/import-progress`                | —                                                      | `ImportProgress`                            | `get_import_progress()`               |
| 43  | `GET`  | `/api/notion/coverage`                       | `days?`                                                | `{calendar[]}`                              | `get_coverage_calendar()`             |
| 44  | `GET`  | `/api/costs/session`                         | —                                                      | `SessionCost`                               | `get_session_cost()`                  |
| 45  | `GET`  | `/api/costs/history`                         | `days?`                                                | `CostSummary`                               | `get_cost_summary()`                  |
| 46  | `GET`  | `/api/costs/pricing`                         | —                                                      | `{models[]}`                                | `get_model_pricing_table()`           |
| 47  | `GET`  | `/api/xray/events`                           | `since_seq?, limit?`                                   | `{events[]}`                                | `get_recent_events()`                 |
| 48  | `POST` | `/api/xray/clear`                            | —                                                      | `{success}`                                 | `clear_events()`                      |
| 49  | `GET`  | `/api/xray/throughput`                       | `buckets?`                                             | `{buckets: int[]}`                          | `get_throughput()`                    |
| 50  | `WS`   | `/api/xray/ws`                               | —                                                      | Stream of XRayEvent                         | WebSocket                             |
| 51  | `GET`  | `/api/auth/plaud/url`                        | —                                                      | `{auth_url, state}`                         | `get_authorization_url()`             |
| 52  | `POST` | `/api/auth/plaud/exchange`                   | `{code, state}`                                        | `{access_token, expires_at}`                | `exchange_code_for_token()`           |
| 53  | `GET`  | `/api/auth/plaud/status`                     | —                                                      | `TokenStatus`                               | `token_status`                        |
| 54  | `GET`  | `/api/auth/notion/url`                       | —                                                      | `{auth_url, state}`                         | `get_authorization_url()`             |
| 55  | `POST` | `/api/auth/notion/exchange`                  | `{code}`                                               | `{access_token, workspace}`                 | `exchange_code_for_token()`           |
| 56  | `GET`  | `/api/auth/notion/status`                    | —                                                      | `TokenStatus`                               | `token_status`                        |
| 57  | `POST` | `/api/push/register`                         | `{device_token}`                                       | `{success}`                                 | APNs registration                     |
| 58  | `GET`  | `/api/settings`                              | —                                                      | `Settings`                                  | Server config (safe subset)           |

---

## 22. Data Models — Swift Codable Structs

Every Python Pydantic model / dataclass maps to a Swift `Codable` struct:

```swift
// MARK: - Enums

enum EventCategory: String, Codable, CaseIterable {
    case work, personal, meeting, deepWork = "deep_work",
         breakTime = "break", reflection, idea, unknown
}

enum SpeakerMode: String, Codable {
    case selfTalk = "self_talk", conversation, unknown
}

enum DayOfWeek: String, Codable, CaseIterable {
    case monday = "Monday", tuesday = "Tuesday", wednesday = "Wednesday",
         thursday = "Thursday", friday = "Friday", saturday = "Saturday",
         sunday = "Sunday"
}

// MARK: - Core Models

struct Event: Codable, Identifiable, Hashable {
    let id: String              // event_id / qdrant_point_id
    let recordingId: String
    let startTs: Date
    let endTs: Date
    let dayOfWeek: String
    let hourOfDay: Int
    let cleanText: String
    let category: String
    let categoryConfidence: Double?
    let sentiment: Double?
    let keywords: [String]
    let speaker: String
    let durationSeconds: Double
    let durationCapped: Bool?

    enum CodingKeys: String, CodingKey {
        case id, startTs = "start_ts", endTs = "end_ts", dayOfWeek = "day_of_week",
             hourOfDay = "hour_of_day", cleanText = "clean_text", category,
             categoryConfidence = "category_confidence", sentiment, keywords, speaker,
             recordingId = "recording_id", durationSeconds = "duration_seconds",
             durationCapped = "duration_capped"
    }
}

struct RecordingSummary: Codable, Identifiable {
    let recordingId: String
    let startTime: Date?
    let endTime: Date?
    let durationSeconds: Int
    let topCategory: String
    let eventCount: Int
    let timeRangeFormatted: String?
    let durationFormatted: String?
    let timeIsEstimated: Bool?
    let timeEstimateReason: String?
    let title: String?
    let plaudAiSummary: String?
    let cloudStatus: String?    // cloud / local / ai

    var id: String { recordingId }

    enum CodingKeys: String, CodingKey {
        case recordingId = "recording_id", startTime = "start_time", endTime = "end_time",
             durationSeconds = "duration_seconds", topCategory = "top_category",
             eventCount = "event_count", timeRangeFormatted = "time_range_formatted",
             durationFormatted = "duration_formatted", timeIsEstimated = "time_is_estimated",
             timeEstimateReason = "time_estimate_reason", title,
             plaudAiSummary = "plaud_ai_summary", cloudStatus = "cloud_status"
    }
}

struct DaySummary: Codable, Identifiable {
    let date: String
    let summary: String?
    let eventCount: Int
    let categoryPercentages: [String: Double]?
    let topKeywords: [String]?
    let sentimentTrend: [Double]?
    let recordings: [RecordingSummary]?

    var id: String { date }

    enum CodingKeys: String, CodingKey {
        case date, summary, eventCount = "event_count",
             categoryPercentages = "category_percentages",
             topKeywords = "top_keywords", sentimentTrend = "sentiment_trend",
             recordings
    }
}

struct RecordingDetail: Codable {
    let summary: RecordingSummary
    let events: [Event]
    let categoryPercentages: [String: Double]?
    let transcript: String?
    let aiSummary: String?
    let extractedData: [String: AnyCodable]?
    let workflowStatus: WorkflowStatus?
    let plaudTranscript: String?

    enum CodingKeys: String, CodingKey {
        case summary, events, categoryPercentages = "category_percentages",
             transcript, aiSummary = "ai_summary", extractedData = "extracted_data",
             workflowStatus = "workflow_status", plaudTranscript = "plaud_transcript"
    }
}

struct SearchResult: Codable, Identifiable {
    let event: Event
    let score: Double
    var id: String { event.id }
}

struct AIAnswer: Codable {
    let text: String
    let model: String
    let tokensUsed: Int
    let latencyMs: Int
    let responseId: String?

    enum CodingKeys: String, CodingKey {
        case text, model, tokensUsed = "tokens_used", latencyMs = "latency_ms",
             responseId = "response_id"
    }
}

struct Topic: Codable, Identifiable, Hashable {
    let name: String
    let count: Int
    var id: String { name }
}

struct TopicOccurrence: Codable, Identifiable {
    let eventId: String
    let recordingId: String
    let timestamp: Date
    let category: String
    let textSnippet: String
    var id: String { eventId }

    enum CodingKeys: String, CodingKey {
        case eventId = "event_id", recordingId = "recording_id",
             timestamp, category, textSnippet = "text_snippet"
    }
}

struct GraphData: Codable {
    let nodes: [GraphNode]
    let edges: [GraphEdge]
}

struct GraphNode: Codable, Identifiable {
    let id: String
    let label: String
    let fullLabel: String?
    let type: String          // category, topic, person, project, org, location, date
    let size: Double?
    let color: String?
    let count: Int?
    let mentionCount: Int?
    let categories: [String]?
    let sentiment: Double?
    let relatedKeywords: [String]?
    let recordingCount: Int?

    enum CodingKeys: String, CodingKey {
        case id, label, fullLabel = "full_label", type, size, color, count,
             mentionCount = "mention_count", categories, sentiment,
             relatedKeywords = "related_keywords", recordingCount = "recording_count"
    }
}

struct GraphEdge: Codable {
    let source: String
    let target: String
    let label: String?
    let weight: Double?
}

struct Stats: Codable {
    let totalEvents: Int
    let totalDays: Int
    let totalDurationHours: Double
    let categories: [String: Int]
    let categoriesByHour: [String: [String: Int]]?
    let sentimentDistribution: [String: Int]?
    let plaudCloudStats: [String: AnyCodable]?

    enum CodingKeys: String, CodingKey {
        case totalEvents = "total_events", totalDays = "total_days",
             totalDurationHours = "total_duration_hours", categories,
             categoriesByHour = "categories_by_hour",
             sentimentDistribution = "sentiment_distribution",
             plaudCloudStats = "plaud_cloud_stats"
    }
}

struct WorkflowStatus: Codable {
    let workflowId: String?
    let status: String?       // PENDING, PROCESSING, SUCCESS, FAILED
    let submittedAt: Date?
    let completedAt: Date?
    let templateId: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case workflowId = "workflow_id", status, submittedAt = "submitted_at",
             completedAt = "completed_at", templateId = "template_id", error
    }
}

struct XRayEvent: Codable, Identifiable {
    let seq: Int
    let ts: Double
    let source: String
    let op: String
    let message: String
    let durationMs: Double?
    let detail: String?
    let level: String
    var id: Int { seq }

    enum CodingKeys: String, CodingKey {
        case seq, ts, source, op, message, durationMs = "duration_ms", detail, level
    }
}

struct CostSummary: Codable {
    let totalCostUsd: Double
    let totalCalls: Int
    let byModel: [String: ModelCost]
    let byType: [String: Double]
    let recent: [UsageRecord]?

    enum CodingKeys: String, CodingKey {
        case totalCostUsd = "total_cost_usd", totalCalls = "total_calls",
             byModel = "by_model", byType = "by_type", recent
    }
}

struct ModelCost: Codable {
    let calls: Int
    let costUsd: Double
    let inputTokens: Int
    let outputTokens: Int

    enum CodingKeys: String, CodingKey {
        case calls, costUsd = "cost_usd", inputTokens = "input_tokens",
             outputTokens = "output_tokens"
    }
}

struct NotionRecordingDTO: Codable, Identifiable {
    let pageId: String
    let title: String
    let createdTime: String
    let lastEditedTime: String
    let url: String
    let transcript: String?
    let summary: String?
    let date: String?
    let duration: String?
    let tags: [String]?
    let category: String?
    let source: String
    let matchedRecordingId: String?

    var id: String { pageId }

    enum CodingKeys: String, CodingKey {
        case pageId = "page_id", title, createdTime = "created_time",
             lastEditedTime = "last_edited_time", url, transcript, summary,
             date, duration, tags, category, source,
             matchedRecordingId = "matched_recording_id"
    }
}

struct PipelineProgress: Codable {
    let status: String        // idle, running, complete, failed
    let currentPhase: String?
    let phases: [PhaseProgress]?
    let startedAt: Date?
    let completedAt: Date?

    enum CodingKeys: String, CodingKey {
        case status, currentPhase = "current_phase", phases,
             startedAt = "started_at", completedAt = "completed_at"
    }
}

struct PhaseProgress: Codable {
    let name: String
    let status: String
    let totalItems: Int
    let completedItems: Int
    let currentStep: String?

    enum CodingKeys: String, CodingKey {
        case name, status, totalItems = "total_items",
             completedItems = "completed_items", currentStep = "current_step"
    }
}
```

---

## 23. Design System — iOS Color Tokens

### Porting CSS Variables to SwiftUI

```swift
// Extensions/Color+Chronos.swift

import SwiftUI

extension Color {
    // MARK: - Core Palette (from CSS :root)
    static let chronosBgPrimary = Color(.systemBackground)          // --bg-primary: #f8f9fb
    static let chronosBgSecondary = Color(.secondarySystemBackground) // --bg-secondary: #ffffff
    static let chronosBgTertiary = Color(.tertiarySystemBackground)   // --bg-tertiary: #f0f2f5
    static let chronosText = Color(.label)                            // --text-primary: #1a1d21
    static let chronosTextSecondary = Color(.secondaryLabel)          // --text-secondary: #57606a
    static let chronosTextMuted = Color(.tertiaryLabel)               // --text-muted: #8b949e

    // MARK: - Accent Colors (from CSS)
    static let accentPrimary = Color(hex: "0969da")     // --accent-primary
    static let accentGreen = Color(hex: "1a7f37")       // --accent-green
    static let accentYellow = Color(hex: "9a6700")      // --accent-yellow
    static let accentRed = Color(hex: "cf222e")         // --accent-red
    static let accentPurple = Color(hex: "8250df")      // --accent-purple
    static let accentOrange = Color(hex: "bc4c00")      // --accent-orange
    static let accentPink = Color(hex: "bf3989")        // --accent-pink
    static let accentCyan = Color(hex: "0891b2")        // --accent-cyan

    // MARK: - Category Colors (from --cat-* CSS variables)
    static let catWork = Color(hex: "0969da")            // --cat-work (blue)
    static let catPersonal = Color(hex: "8250df")        // --cat-personal (purple)
    static let catMeeting = Color(hex: "1a7f37")         // --cat-meeting (green)
    static let catReflection = Color(hex: "9a6700")      // --cat-reflection (gold)
    static let catIdea = Color(hex: "bf3989")            // --cat-idea (pink)
    static let catBreak = Color(hex: "6e7781")           // --cat-break (gray)
    static let catDeepWork = Color(hex: "116329")        // --cat-deep-work (dark green)
    static let catUnknown = Color(hex: "8b949e")         // (gray muted)

    /// Get category color by name string
    static func forCategory(_ category: String) -> Color {
        switch category.lowercased().replacingOccurrences(of: "_", with: "") {
        case "work": return .catWork
        case "personal": return .catPersonal
        case "meeting": return .catMeeting
        case "reflection": return .catReflection
        case "idea": return .catIdea
        case "break": return .catBreak
        case "deepwork": return .catDeepWork
        default: return .catUnknown
        }
    }

    // MARK: - Hex Initializer
    init(hex: String) {
        let scanner = Scanner(string: hex)
        var rgb: UInt64 = 0
        scanner.scanHexInt64(&rgb)
        self.init(
            red: Double((rgb >> 16) & 0xFF) / 255,
            green: Double((rgb >> 8) & 0xFF) / 255,
            blue: Double(rgb & 0xFF) / 255
        )
    }
}
```

### Typography

```swift
// Use system styles — matches Apple HIG, supports Dynamic Type automatically
.font(.largeTitle)     // View titles
.font(.title2)         // Section headers
.font(.headline)       // Card titles
.font(.subheadline)    // Secondary info
.font(.body)           // Event text, narratives
.font(.caption)        // Metadata (timestamps, scores)
.font(.caption2)       // Tertiary labels (tokens, latency)
```

### Spacing

```swift
// Map CSS spacing tokens to SwiftUI
// --sp-4:  4px  →  .padding(4) or .padding(.small) custom
// --sp-8:  8px  →  .padding(8)
// --sp-12: 12px →  .padding(12)
// --sp-16: 16px →  .padding() default
// --sp-24: 24px →  .padding(24)
// --sp-32: 32px →  .padding(32)
```

---

## 24. Dependency Map

### Swift Package Dependencies

```swift
// Package.swift or Xcode SPM
dependencies: [
    // Networking - none needed (URLSession built-in)

    // Markdown rendering for AI answers
    .package(url: "https://github.com/gonzalezreal/swift-markdown-ui", from: "2.0.0"),

    // Keychain access for token storage
    .package(url: "https://github.com/kishikawakatsumi/KeychainAccess", from: "4.2.2"),

    // SwiftUI Charts - built into iOS 16+ (no package needed)

    // Cytoscape.js - bundled as local HTML/JS in Resources/
]
```

### Python Backend Dependencies (additions to requirements.txt)

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-jose[cryptography]>=3.3.0    # JWT
python-multipart>=0.0.18            # Form data
websockets>=13.0                     # WebSocket support
httpx>=0.27.0                        # Async HTTP (for APNs)
alembic>=1.14.0                      # Database migrations (PostgreSQL)
psycopg2-binary>=2.9.0              # PostgreSQL driver
```

---

## 25. Risk Register

| Risk                                             | Severity | Mitigation                                                           |
| ------------------------------------------------ | -------- | -------------------------------------------------------------------- |
| **Plaud OAuth triple-callback** breaks on mobile | High     | All OAuth handled server-side; iOS only gets final token             |
| **Gemini rate limits** affect mobile users       | Medium   | Server queueing; show "Processing..." with push notification on done |
| **Large graph data** slow on mobile              | Medium   | Server-side pagination (max_nodes param); lazy loading               |
| **Audio file streaming** bandwidth               | Low      | Audio stays server-side; iOS streams via signed URL on demand        |
| **Offline mode** creates merge conflicts         | Medium   | Only allow read cache offline; queue writes are idempotent           |
| **Qdrant Cloud latency**                         | Low      | Already fast; add server-side response caching (Redis optional)      |
| **App Store review** push notification policy    | Low      | Notifications are opt-in, relevant to user data                      |
| **SwiftData migration** on schema changes        | Medium   | Version SwiftData models; lightweight migration                      |
| **Cost tracking accuracy** across devices        | Low      | Cost tracked server-side (already in SQLite); iOS is read-only       |
| **WebSocket drops** on bad network               | Medium   | Auto-reconnect with exponential backoff; fallback to polling         |

---

## 26. File-by-File Migration Reference

### What Stays Server-Side (Unchanged)

| File                                  | Reason                               |
| ------------------------------------- | ------------------------------------ |
| `src/chronos/ingest_service.py`       | Heavy I/O, Plaud API calls           |
| `src/chronos/transcript_processor.py` | Gemini API calls, JSON repair        |
| `src/chronos/engine.py`               | Gemini audio processing              |
| `src/chronos/embedding_service.py`    | Gemini embedding API                 |
| `src/chronos/qdrant_client.py`        | Vector store operations              |
| `src/chronos/graph_service.py`        | Entity extraction (LLM)              |
| `src/chronos/graph_rag.py`            | Knowledge graph + communities        |
| `src/chronos/openai_service.py`       | GPT-5.4 RAG queries                  |
| `src/chronos/cost_tracker.py`         | Server-side cost ledger              |
| `src/chronos/notion_bridge.py`        | Notion API operations                |
| `src/plaud_client.py`                 | Plaud REST API                       |
| `src/plaud_oauth.py`                  | OAuth token management               |
| `src/notion_oauth.py`                 | OAuth token management               |
| `src/notion_service.py`               | Notion data queries                  |
| `src/database/*`                      | All DB access                        |
| `src/models/*`                        | Pydantic schemas (server validation) |
| `scripts/chronos_pipeline.py`         | Pipeline orchestration               |
| `scripts/auto_sync.py`                | Background workers                   |
| `scripts/mcp_server.py`               | MCP tools (keep alongside REST)      |

### What Gets a New Layer (FastAPI routes wrapping existing code)

| Python File                       | New FastAPI Route File                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| `app_v2/services/data_service.py` | `api/routes/timeline.py`, `recordings.py`, `search.py`, `topics.py`, `graph.py`, `stats.py` |
| `app_v2/services/xray.py`         | `api/routes/xray.py` + `api/websocket/xray_ws.py`                                           |

### What Gets Replaced (Dash → SwiftUI)

| Dash File                               | SwiftUI Replacement                                 |
| --------------------------------------- | --------------------------------------------------- |
| `app_v2/layout.py`                      | `ContentView.swift` (TabView + NavigationSplitView) |
| `app_v2/components/day_view.py`         | `Views/Timeline/TimelineView.swift` + subviews      |
| `app_v2/components/search.py`           | `Views/Search/SearchView.swift` + subviews          |
| `app_v2/components/graph.py`            | `Views/Graph/GraphContainerView.swift` + WKWebView  |
| `app_v2/components/stats.py`            | `Views/Stats/StatsView.swift` + Swift Charts        |
| `app_v2/components/topics.py`           | `Views/Topics/TopicsGridView.swift` + subviews      |
| `app_v2/components/recording_detail.py` | `Views/RecordingDetail/RecordingDetailView.swift`   |
| `app_v2/components/notion.py`           | `Views/Notion/NotionView.swift` + subviews          |
| `app_v2/components/sidebar.py`          | `Views/Sidebar/SidebarView.swift` (→ TabBar)        |
| `app_v2/callbacks/day_view.py`          | `ViewModels/TimelineViewModel.swift`                |
| `app_v2/callbacks/search.py`            | `ViewModels/SearchViewModel.swift`                  |
| `app_v2/callbacks/graph.py`             | `ViewModels/GraphViewModel.swift`                   |
| `app_v2/callbacks/recording_detail.py`  | `ViewModels/RecordingDetailViewModel.swift`         |
| `app_v2/callbacks/notion.py`            | `ViewModels/NotionViewModel.swift`                  |
| `app_v2/callbacks/navigation.py`        | Navigation handled by SwiftUI's `NavigationStack`   |
| `app_v2/assets/style.css` (8527 lines)  | `Color+Chronos.swift` (50 lines) + system styles    |
| `app_v2/assets/xray_pip.js` (788 lines) | `Views/XRay/XRayView.swift` + WebSocket             |
| `app_v2/main.py` (Flask routes)         | `api/routes/auth.py`, `api/routes/xray.py`          |

### What Gets Bundled in iOS App (Resources)

| Source                                            | Destination                           | Purpose                    |
| ------------------------------------------------- | ------------------------------------- | -------------------------- |
| `app_v2/components/graph.py` → `GRAPH_STYLESHEET` | `Resources/graph.css`                 | Cytoscape node/edge styles |
| Cytoscape.js library                              | `Resources/cytoscape.min.js`          | Graph rendering engine     |
| cose-bilkent layout                               | `Resources/cytoscape-cose-bilkent.js` | Force-directed layout      |
| New `graph.html`                                  | `Resources/graph.html`                | WKWebView graph container  |

---

## Summary: Build Order

```
Phase 0: FastAPI Backend                    ← START HERE
  └── Wrap data_service.py in REST routes
  └── Add JWT auth
  └── Add WebSocket for X-ray

Phase 1: Cloud Infrastructure
  └── PostgreSQL (Supabase)
  └── Qdrant Cloud
  └── Deploy to Render/Railway

Phase 2: iOS Foundation
  └── Xcode project structure
  └── APIClient + AuthManager
  └── Tab navigation + routing

Phase 3-6: Core Views (parallel)
  ├── Timeline + Recording Detail
  ├── Search + AI Answers
  ├── Topics
  └── Graph (WKWebView + Cytoscape)

Phase 7-8: Supporting Views
  ├── Stats + Charts
  └── Sync Dashboard

Phase 9-11: Integrations
  ├── Settings
  ├── Notion
  └── X-ray (WebSocket)

Phase 12-13: Auth + Notifications
  ├── Plaud OAuth (ASWebAuthenticationSession)
  ├── Notion OAuth
  └── APNs push notifications

Phase 14-15: Polish
  ├── Offline caching (SwiftData)
  └── Accessibility + App Store prep
```

---

_This document covers every service, model, callback, route, schema, color token, and data flow in PlaudBlender — providing a complete blueprint for the native iOS transformation._
