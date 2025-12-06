# PlaudBlender Audit & UX Checklist

_Comprehensive, end-to-end checklist to track UX, functionality, and platform hardening (UI/logic/infra). Use this as the single source of truth for cleanup and future SQL integration._

**Progress log (2025-12-05)**
- Added stdio MCP server powered by OpenAI Responses (ping/list_models/respond tools)
- Added in-app Chat tab using OpenAI Responses API (model/temp controls, system prompt)

**Progress log (2025-12-04)**
- Added Pinecone dialog validation (JSON checks, empty-input warnings, manual embedding parsing)
- Transcripts view: empty-state message; selection handling for actions; details popup
- Search view: friendly placeholder when no results
- Settings view redesigned with tabs (Connections, Embedding & Pinecone) and clearer status/danger zone
- Pinecone view reorganized into grouped toolbars (Search/Inspect, Write/Edit, Bulk/Export, Namespaces) with clearer selectors/stats and filter card
- Added dimension-safe "Re-embed all" workflow (auto creates/switches matching index) from Pinecone Bulk row
- Global UI: panel labelframe styling for consistent cards; dashboard/transcripts/search/logs/pinecone now use coherent hero + card layouts

## Legend
- [ ] Not started
- [~] In progress
- [x] Done

## 1) Foundations & Environment
- [ ] Verify `.env` creation via `plaud_setup.py` (all keys present, no hardcoded secrets)
- [ ] Confirm `requirements.txt` is installable on clean macOS (Tkinter, Pinecone, dotenv, etc.)
- [x] Add `SQLAlchemy` dependency when SQL layer work begins
- [x] Ensure `data/` folder exists (for future `brain.db`) and is gitignored
- [x] Standardize logging level via env (`PB_LOG_LEVEL`, `PB_VERBOSE`) via `src/utils/logger.py`

## 2) Navigation & UX Framework
- [ ] Sidebar nav highlights current view; ensure keyboard shortcuts (optional)
- [ ] Consistent theming across views (no default Tk styles leaking)
- [ ] Status bar: busy/idle messages updated for every async op; show failures
- [x] Empty states for each view (no silent blanks)

## 3) Dashboard
- [ ] Cards show: Auth status, Recording count, Pinecone vector count, Last sync
- [ ] Quick actions wired: Sync all, Mind map, Semantic search, Settings
- [ ] Last sync timestamp updated after any sync/import

## 4) Transcripts View
- [ ] Fetch transcripts populates table with meaningful columns (name/date/time/duration/id)
- [ ] Filter box debounced and applied consistently
- [x] Row selection enables actions (sync/delete/view/details/export)
- [x] View/Details open transcript viewer with text + metadata
- [ ] Sync/Delete/Export actions fully implemented (currently stubs)
- [ ] Pagination or virtualization if transcript list is large
- [ ] Error toasts when Plaud API fails

## 5) Pinecone View (Vector Workspace)
- [ ] Index + namespace selectors populate on first load; defaults are sensible
- [ ] Stats (vectors/dimension/metric) reflect current index
- [ ] Table sorting works for all sortable columns and preserves selection
- [ ] Pagination respects page size selector; page label accurate
- [ ] Local filter box filters rendered rows without mutating source data
- [x] Preview pane shows metadata/text with readable formatting
- [x] Similarity search dialog: validates JSON filter, shows errors
- [x] Metadata filter dialog: validates JSON, limit enforced
- [x] Cross-namespace search dialog: handles empty namespace list gracefully
- [x] Fetch-by-ID dialog: handles comma-separated IDs, shows not-found feedback
- [x] Upsert dialog: supports auto-embed (from 'text' field) and manual vector; validates metadata JSON and manual vector length
- [x] Re-embed all workflow wires to dimension-safe index selection/creation
- [ ] Edit metadata dialog: persists changes and refreshes table
- [ ] Edit metadata dialog: persists changes and refreshes table
- [x] Delete dialog: selected / by-filter / delete-all paths confirm and succeed
- [x] Namespace mgmt: create/delete/list flows refresh selectors + stats
- [ ] Bulk import dialog: CSV/JSON selection, error handling, progress feedback
- [ ] Export dialog: exports selected/all to CSV/JSON with metadata
- [ ] Right-click context menu (if present) matches toolbar actions
- [ ] Graceful handling of Pinecone errors (auth, network, 4xx/5xx)

## 6) Search View
- [ ] Input accepts Enter to run default search
- [ ] Cross-namespace search wired to backend action with limit and filter options
- [ ] Full-text vs summaries namespace routing correct
- [ ] Results rendered with clear formatting (title, score, namespace, snippet)
- [x] Handles empty/no-result cases with friendly message
- [x] Errors surfaced in UI (not just logs)

## 11) Legacy GUI Parity (keep & mine useful features)
- [x] Identify legacy-only features to port (e.g., saved searches)
- [x] Extract reusable logic into services (saved searches service added)
- [ ] Remove duplication after extraction while preserving legacy file for reference
- [ ] Document feature gaps between legacy and new GUI

## 16) SQL Layer (source of truth)
- [x] Create `src/database/` scaffold (engine/models) and `data/` folder (gitignored)
- [x] Persist Plaud recordings via `PlaudClient.fetch_and_store_recordings` (status=`raw`)
- [x] Add chunking engine to emit `segments` (status=`pending`) and mark recordings `processed`
- [x] Add indexer scaffold to embed/upsert via injected functions; updates `pinecone_id` and status=`indexed`
- [ ] Wire embedding + Pinecone upsert, tagging metadata with `recording_id`/`segment_id` in production pipeline
- [ ] Plan UI affordances for SQL-backed queries (filters, provenance display)

## 7) Settings View
- [ ] Settings load current config on show
- [ ] Save/Apply persists to disk/env as intended
- [ ] Validation and error states surfaced to user

## 8) Logs View
- [ ] Logs stream or refresh button works; clear/log-level filter if applicable
- [ ] Copy-to-clipboard or export option (optional)

## 9) Async/Threading & Responsiveness
- [ ] All network/IO work uses background threads (`run_async`/threading) to keep UI responsive
- [ ] Busy cursors/spinners for long ops; cancel where feasible
- [ ] No Tk calls from non-main threads

## 10) Error Handling & Messaging
- [ ] All service calls wrapped with try/except and surface messagebox/log entries
- [ ] Status bar shows failures with clear text; logs include stack traces
- [ ] Input validation on all dialogs (IDs, JSON, numeric fields)

## 11) Legacy GUI Parity (keep & mine useful features)
- [ ] Identify legacy-only features to port (e.g., saved searches, menu actions, richer export flows, namespace stats caching, analytics/wordcloud hooks)
- [ ] Extract reusable logic into services (Plaud/Pinecone helpers, caching, formatting)
- [ ] Remove duplication after extraction while preserving legacy file for reference
- [ ] Document feature gaps between legacy and new GUI

## 12) Data & Processing Pipeline (pre-SQL)
- [ ] Plaud fetch: robust pagination, retries, rate-limit handling
- [ ] Transcription normalization (timestamps, speaker labels if available)
- [ ] Chunking/LLM processing path well-defined (current `dual_store_processor`)
- [ ] Embedding service: single source of truth for model/config; error handling
- [ ] Pinecone upsert: consistent metadata schema (`source`, `recording_id`, etc.)
- [ ] Mind-map generation path verified
- [~] Re-embed/re-sync workflows: detect stale embeddings when model/settings change (manual re-embed path added)

## 13) Configuration & Secrets
- [ ] No secrets in repo; all loaded via dotenv
- [ ] Clear setup doc for obtaining Plaud tokens and Pinecone keys
- [ ] Optional: schema for settings (types/defaults) validated on load

## 14) Testing & QA
- [ ] Add `tests/` package; relocate any ad-hoc tests
- [ ] Smoke test: Plaud auth + transcript fetch + one Pinecone query
- [ ] UI smoke (headless where possible) for view switching and key dialogs
- [ ] Service-unit tests for embedding + Pinecone service logic

## 15) Documentation & Onboarding
- [ ] Update README with refined run instructions (GUI entry, optional scripts)
- [ ] Add short GIF/screens for main flows (transcripts, Pinecone search/upsert, search view)
- [ ] Developer guide: architecture, services, threading rules, theming

## 17) Direct Notion Integration (replace Zapier; no MCP middleman unless justified)
- [ ] Inventory Notion databases you rely on (Plaud transcriptions, Arsenal, Projects, Ideas) and map required properties
- [ ] Implement direct Notion API client (rate limits, pagination, retries); avoid Zapier
- [ ] One-way ingest: push new/updated recordings, summaries, themes, links to audio into Notion
- [ ] Two-way enrich: pull Notion edits back into SQL/Pinecone (e.g., corrected titles, tags/themes)
- [ ] Idempotency: use stable IDs (recording_id/segment_id) to upsert and avoid dupes
- [ ] Backoff & failure logging for Notion writes; surface UI errors
- [ ] Privacy/ACL review: ensure no sensitive fields leak to public pages

## 18) MCP (optional, only if it adds value)
- [ ] Define exact MCP use-cases (e.g., uniform tool access to Pinecone/Notion) before building
- [x] If adopted, keep MCP as a thin facade over existing services; do not gate core flows on MCP availability
- [ ] Auth & scopes: ensure MCP tokens are scoped minimally; store secrets via env
- [ ] Robust fallbacks: GUI and services must work without MCP running

## 19) Integrations & Data Sources (beyond core Plaud)
- [ ] Email ingest (optional): watch mailbox for Plaud-delivered audio/transcripts; de-dup via recording_id
- [ ] GitHub projects: optional semantic indexing; tag vectors with repo/project; keep separate namespaces
- [ ] Future streams (bank statements, etc.): design normalized schema first; avoid ad-hoc ingestion
- [ ] Mind-map/export pipelines stay consistent across sources

## 20) Accuracy, Themes, and Metadata Quality
- [ ] Theme extraction: per recording + per segment; store in SQL; tag Pinecone metadata (`theme`, `recording_id`, `segment_id`, `source`)
- [ ] Confidence/quality flags: track LLM confidence, ASR quality, length, and processing status
- [ ] Re-embed/re-sync workflows: detect stale embeddings when model/settings change
- [ ] Validation: enforce metadata schema before upsert/search; reject malformed filters early

## 21) Observability & Ops
- [ ] Centralized logging with levels; user-visible toasts for failures
- [ ] Metrics: counts of ingested recordings/segments, Pinecone upserts, Notion sync successes/failures
- [ ] Backups: periodic export of SQL (`brain.db`) and Notion-linked IDs; document restore steps
- [ ] Health checks for Plaud, Pinecone, Notion connectivity surfaced in UI

## 22) Execution Phases (runnable plan)
- [ ] Phase 0: UX wiring/validation (Transcripts actions, Search formatting, Pinecone dialog validation/toasts)
- [ ] Phase 1: Mine legacy GUI for reusable features (saved searches, exports, namespace stats caching) and port into services
- [~] Phase 2: Stand up SQLAlchemy layer (Recording/Segment), route Plaud ingestion through SQL → Pinecone with provenance (ingest + chunking done; embedding/upsert pending)
- [ ] Phase 3: Direct Notion integration (replace Zapier), two-way enrichment with idempotent upserts
- [ ] Phase 4: MCP optional layer (only if it provides net value), plus additional data sources (email/GitHub) guarded by schema/namespace separation

---
_Updated: 2025-12-04_
