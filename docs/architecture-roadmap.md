# PlaudBlender Architecture Roadmap (Control Plane / Data Plane)

## Current Reality (snapshot)
- GUI is modular (views/services/components) and Pinecone coverage is rich.
- Processing logic is fragmented (dual_store_processor vs llm_processor; scripts in root).
- No deterministic store between Plaud API and Pinecone vectors; data contracts are loose dicts.

## North Star
A platform where the GUI is a control plane and a worker/engine (data plane) runs ingestion → validation → storage → enrichment → indexing. SQLite/SQLAlchemy is the source of truth; Pinecone is the serving index; Pydantic enforces contracts; Pytest guards regressions.

## Target topology
- `src/config.py` — single settings loader (.env once) ✔
- `src/utils/logger.py` — single logger setup ✔
- `src/models/schemas.py` — Pydantic ingress contracts ✔
- `src/database/` — SQLAlchemy Base, engine, models (Recording, Segment) ✔ (scaffold present)
- `src/processing/engine.py` — unified processor (rename dual_store_processor, remove llm_processor)
- `src/clients/` — Plaud, Pinecone clients (shared by GUI and workers)
- `workers/` or `scripts/` — CLI/worker entrypoints (e.g., ingest, process, reindex)
- `tests/` — pytest suite with mocks (schemas ✔, extend to processors/services)

## Immediate cleanup (high-impact, low-risk)
- Remove/retire legacy: `gui_legacy.py`, `test_gui_import.py`, `test_components.py` (migrate any useful cases into `tests/`).
- Remove `src/llm_processor.py` after merging any unique logic into the unified processor.
- Ensure `.env` is read only via `src/config.py`; stop scattered `load_dotenv` calls.
- Ensure logging only via `src/utils/logger.py`; remove duplicated `basicConfig`.

## SQL as source of truth (flow)
1) **Ingest Plaud**
   - Plaud client fetches recordings → validate with `RecordingSchema` → write to `recordings` table (status=`raw`).
2) **Process**
   - Worker pulls `recordings.status == 'raw'`
   - Run LLM synthesis/theme extraction and chunking
   - Write `segments` rows; store per-segment metadata and provenance (recording_id, start/end ms)
3) **Index**
   - Embed segments; upsert to Pinecone with metadata `{sql_segment_id, recording_id, namespace}`
   - Mark `segments.status = indexed`, `recordings.status = synced`
4) **Serve**
   - GUI reads from Pinecone for fast search; can fall back to SQL for provenance/detail

## Refactor steps (concrete)
1. **Unify processor**
   - Move `src/dual_store_processor.py` → `src/processing/engine.py`; delete `llm_processor.py` post-merge.
   - Inject deps (LLM, Pinecone client, DB session) instead of hardcoding env.
2. **DB models**
   - Extend `src/database/models.py` to include fields: recordings(id, title, duration_ms, created_at, transcript, status); segments(id uuid, recording_id FK, start_ms, end_ms, text, namespace, pinecone_id, themes/json, status).
   - Add convenience CRUD helpers in `src/database/engine.py` or a `repos.py`.
3. **Plaud ingest**
   - Update `src/plaud_client.py` (or a new ingest service) to validate with `RecordingSchema` and persist immediately.
4. **Processing pipeline**
   - Engine reads pending recordings from DB, runs theme/synthesis/chunking, writes segments, embeds, upserts, updates statuses.
5. **GUI wiring**
   - Actions (sync, search provenance) call the engine/service instead of direct Pinecone. Use SQL ids in metadata to deep-link.
6. **Testing**
   - Add pytest suites: ingest (schema + repo), processor (chunking, status transitions), Pinecone service (formatting, metadata tagging) with mocks.

## Notes on data contracts
- Pydantic at ingress to catch drift early.
- SQL as durable store; Pinecone as serving layer with `sql_segment_id` for provenance.
- Namespace strategy: `full_text` for chunks, `summaries` for synths; both carry recording_id.

## Suggested task ordering
1) Legacy removal + processor rename; enforce config/logger usage.
2) Flesh out SQL models + repos; integrate ingest write-through.
3) Refactor processor to consume DB, emit segments, and tag Pinecone metadata with SQL IDs.
4) Add pytest coverage for schemas/repos/processor with mocks.
5) Optional: add a background worker CLI (e.g., `python scripts/worker.py --loop`).
