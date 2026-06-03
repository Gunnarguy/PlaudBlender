# Notion Integration — Architecture & Usage

> PlaudBlender ↔ Notion two-way sync for Plaud voice recordings.

## Overview

The Notion integration connects your **Notion database of transcriptions** to Chronos. It performs:

1. **OAuth authentication** — Secure Notion workspace access
2. **Fetching** — Pull recordings from your Notion database
3. **Cross-referencing** — Fuzzy-match Notion pages to existing Chronos recordings
4. **Import** — Bring Notion-only recordings into the Chronos pipeline (process → embed → index)
5. **Write-back** — Push Chronos enrichments (categories, sentiment, keywords, AI summaries, transcripts) back to Notion
6. **Change detection** — Flag Notion pages edited after import (stale badge)

**Important:** Notion pages are **never deleted** during any operation. Import is additive and idempotent.

---

## Architecture

```
┌──────────────────┐      OAuth 2.0       ┌─────────────────────┐
│  Notion Database  │◄────────────────────►│  NotionOAuthClient   │
│  (PlaudAI Trans.) │   fetch / write-back │  src/notion_oauth.py │
└────────┬─────────┘                       └──────────────────────┘
         │
    Notion API
         │
         ▼
┌────────────────────┐
│  NotionService     │  List databases, fetch recordings, page content
│  src/notion_service│  Auto-detects schema (title, date, body properties)
└────────┬───────────┘
         │ List[NotionRecording]
         ▼
┌────────────────────────┐
│  notion_bridge.py      │  Match → Import → Write-back → Stale detect
│  src/chronos/          │  Core orchestration layer
└────────┬───────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ SQLite │ │ Qdrant       │
│ (DB)   │ │ (Vectors)    │
└────────┘ └──────────────┘
         │
         ▼
┌───────────────────────────────┐
│  Dash UI — Notion Tab         │
│  app_v2/components/notion.py  │
│  app_v2/callbacks/notion.py   │
└───────────────────────────────┘
```

---

## File Map

| File                               | Purpose                                                      |
| ---------------------------------- | ------------------------------------------------------------ |
| `src/notion_oauth.py`              | OAuth 2.0 client — auth URL, code exchange, token validation |
| `src/notion_service.py`            | Notion API access — list DBs, fetch recordings, page content |
| `src/chronos/notion_bridge.py`     | Matching, import, write-back, stale detection, coverage      |
| `app_v2/components/notion.py`      | UI layout — hero, toolbar, recording list, detail panel      |
| `app_v2/callbacks/notion.py`       | All 15 Dash callbacks for Notion tab interactivity           |
| `app_v2/assets/style.css`          | Notion-specific CSS (badges, stale pulse animation)          |
| `data/notion_import_progress.json` | Runtime import progress (auto-deleted on completion)         |

---

## Data Model

### NotionRecording (dataclass)

```python
@dataclass
class NotionRecording:
    page_id: str                      # Notion page UUID
    title: str                        # Page title
    created_time: str                 # ISO timestamp
    last_edited_time: str             # ISO timestamp (used for stale detection)
    url: str                          # Notion page URL
    transcript: str                   # Body text content
    summary: str                      # Summary property (if present)
    date: str                         # Extracted date (YYYY-MM-DD)
    duration: Optional[str]           # Duration property
    tags: List[str]                   # Tags property
    category: str                     # Category property
    source: str                       # Source property
    properties: Dict[str, Any]        # Raw Notion properties
    body_text: str                    # Full page body text
    matched_recording_id: Optional[str]  # Chronos recording ID (post-match)
```

### Chronos Storage

Imported Notion recordings are stored as:

- `ChronosRecording` with `source = "notion"` and `recording_id = "notion:{page_id}"`
- `ChronosEvent` records for processed events
- Qdrant vectors for semantic search

---

## Features

### 1. OAuth Authentication

- **Flow:** OAuth 2.0 authorization code
- **Token storage:** `.notion_tokens.json` (local, never expires unless user revokes)
- **Setup:** Enter OAuth Client ID + Secret in Settings → Notion section, click Connect

### 2. Database Discovery & Selection

- Lists all accessible Notion databases in the workspace
- Persists selected database ID to `.env` as `NOTION_DATABASE_ID`
- Auto-detects database schema (title column, date columns, body properties)

### 3. Cross-Reference Matching

Matching algorithm (`match_notion_to_chronos`):

1. **Extract date from title** — parses `MM-DD` and `YYYY-MM-DD` prefixes
2. **SequenceMatcher fuzzy matching** — title similarity scoring
3. **Greedy 1:1 assignment** — each Notion page maps to at most one Chronos recording
4. **Result:** `{notion_page_id → chronos_recording_id}` or `None` for unmatched

### 4. Import Pipeline

**Single import** (`import_notion_recording`):

- Idempotent — skips if already imported
- Creates `ChronosRecording` with `source="notion"`
- Runs Gemini processing (clean, extract events, categorize)
- Embeds and indexes to Qdrant

**Batch import** (`import_all_unmatched`):

- Imports all Notion-only recordings not yet in Chronos
- **Newest-first priority** — sorted by date descending before batch limit
- Background thread with progress tracking (`data/notion_import_progress.json`)
- Resume support — picks up where it left off after interruption
- Configurable batch size

### 5. Write-Back to Notion

Pushes Chronos enrichments back to Notion pages (`write_back_to_notion`):

| Property                               | Source                                           |
| -------------------------------------- | ------------------------------------------------ |
| Category                               | Most common event category                       |
| Sentiment                              | Average event sentiment score                    |
| Keywords                               | Aggregated event keywords                        |
| Event Count                            | Number of extracted events                       |
| Summary / ChatGPT Summary / AI Summary | `plaud_ai_summary` or generated from event texts |
| Transcript / Text                      | Cleaned transcript                               |

- Only writes to properties that exist in the Notion database schema
- Rich text capped at 2000 characters (Notion API limit)
- **Batch write-back** (`write_back_all_matched`) — one-click for all matched pages

### 6. Change Detection (Stale Imports)

`detect_stale_imports` compares each Notion page's `last_edited_time` against the Chronos `ingested_at` timestamp. Pages edited after import show a yellow pulsing **🔄 Updated in Notion** badge.

### 7. Coverage Calendar

30-day heatmap showing Notion recording density per day, rendered in the Notion tab hero section.

---

## UI — Notion Tab

### Views

| Section             | Description                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| **Hero**            | Connection status, database info, recording counts, interplay card, coverage calendar, batch buttons |
| **Database Picker** | Dropdown to select or change Notion database                                                         |
| **Search Toolbar**  | Text search + sort dropdown + category filter buttons                                                |
| **Recording List**  | All Notion recordings with badges (In Chronos / Not in Chronos / Stale)                              |
| **Detail Panel**    | Full page content, properties, action buttons                                                        |

### Badges

| Badge                | Meaning                                  |
| -------------------- | ---------------------------------------- |
| ✅ In Chronos        | Matched to an existing Chronos recording |
| 📥 Not in Chronos    | No match — eligible for import           |
| 🔄 Updated in Notion | Page edited after Chronos import (stale) |

### Category Filters

Category filter buttons in the toolbar provide instant filtering. Active category is highlighted. Click "All" to reset.

### Actions

| Button                | Location     | Effect                                         |
| --------------------- | ------------ | ---------------------------------------------- |
| Import to Chronos     | Detail panel | Import single Notion page                      |
| Import All to Chronos | Hero section | Batch import all unmatched                     |
| Resume Import         | Hero section | Continue interrupted import                    |
| Write Back            | Detail panel | Push enrichments to one Notion page            |
| Write Back All        | Hero section | Push enrichments to all matched pages          |
| View in Timeline      | Detail panel | Navigate to matched recording in Timeline view |

---

## Callbacks Reference

| Callback                     | Trigger                         | Effect                                 |
| ---------------------------- | ------------------------------- | -------------------------------------- |
| `auto_fetch_on_load`         | Tab switch to Notion            | Pre-populates store from cache         |
| `discover_databases`         | Click discover                  | Lists available Notion databases       |
| `select_database`            | Click database card             | Sets active database, persists to .env |
| `fetch_notion_recordings`    | Click fetch / auto              | Fetches recordings from Notion API     |
| `refresh_notion_view`        | Store update                    | Re-renders the full Notion view        |
| `filter_recordings`          | Search / sort / category filter | Filters and sorts displayed recordings |
| `show_notion_page_detail`    | Click recording row             | Shows detail panel with page content   |
| `import_one_to_chronos`      | Click import button             | Imports single recording               |
| `import_all_to_chronos`      | Click import all                | Starts background batch import         |
| `resume_import`              | Click resume                    | Resumes interrupted import             |
| `poll_import_progress`       | Interval timer                  | Updates progress bar during import     |
| `write_back_to_notion`       | Click write-back                | Pushes enrichments to one page         |
| `write_back_all_to_notion`   | Click write-back all            | Pushes enrichments to all matched      |
| `goto_timeline_after_import` | Click go-to-timeline            | Navigates to imported recording        |
| `goto_timeline_from_detail`  | Click view-in-timeline          | Navigates to matched recording         |

---

## Configuration

### Environment Variables

| Variable               | Required | Description                                                                       |
| ---------------------- | -------- | --------------------------------------------------------------------------------- |
| `NOTION_CLIENT_ID`     | Yes      | OAuth application client ID                                                       |
| `NOTION_CLIENT_SECRET` | Yes      | OAuth application client secret                                                   |
| `NOTION_DATABASE_ID`   | Auto-set | Selected database UUID (persisted on selection)                                   |
| `NOTION_REDIRECT_URI`  | No       | OAuth callback URL (default: `http://localhost:8000/api/v1/auth/notion/callback`) |

### Notion Database Schema

The integration auto-detects your database schema. Recommended properties:

| Property             | Type         | Used For                       |
| -------------------- | ------------ | ------------------------------ |
| Title (any name)     | Title        | Recording name, matching       |
| Date                 | Date         | Temporal sorting, matching     |
| Category             | Select       | Write-back: category           |
| Sentiment            | Number       | Write-back: sentiment score    |
| Keywords             | Multi-select | Write-back: extracted keywords |
| Event Count          | Number       | Write-back: event count        |
| Summary / AI Summary | Rich text    | Write-back: AI summary         |
| Transcript / Text    | Rich text    | Write-back: cleaned transcript |

Properties not present in your database are silently skipped during write-back.

---

## Design Principles

1. **Non-destructive** — Notion pages are never deleted or modified without explicit user action (write-back button)
2. **Idempotent** — Re-importing the same page is safe (skips if already present)
3. **Resumable** — Batch imports track progress to JSON and can resume after failure
4. **Schema-tolerant** — Works with any Notion database structure, auto-detects available properties
5. **Newest-first** — Import prioritizes recent recordings when batch size is limited
6. **Stale-aware** — Flags pages edited in Notion after Chronos import for re-processing consideration
