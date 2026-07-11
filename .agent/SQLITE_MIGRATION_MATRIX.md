# SQLITE MIGRATION MATRIX

## Database Table Matrix
| Table | Row Count (Pi) | Column Schema | Purpose |
| :--- | :--- | :--- | :--- |
| `chronos_recordings` | 613 | recording_id, title, transcript, processing_status, ingested_at, ... | Core Plaud recordings metadata and text transcripts. |
| `chronos_events` | 10725 | event_id, recording_id, start_ts, end_ts, clean_text, category, sentiment, user_category_override, ... | Granular temporal event entries extracted by AI. |
| `api_usage_log` | 213039 | id, timestamp, model, call_type, input_tokens, output_tokens, cost_usd, ... | AI API calls cost tracking log. |
| `chronos_execution_runs` | 14056 | run_id, trigger, source, status, started_at, ended_at, ... | Process execution metadata log. |
| `chronos_execution_spans` | 20376 | span_id, run_id, parent_span_id, stage, input_tokens, cost_usd, ... | Granular trace spans of pipeline execution. |
| `recordings` | 0 | id, title, filename, transcript, duration_ms, ... | Legacy/unimplemented table structure. |
| `segments` | 0 | id, recording_id, text, start_ms, ... | Empty vector segments cache table. |
| `chronos_processing_jobs` | 0 | job_id, recording_id, job_type, status | Background processing job queue. |
| `chronos_webhook_events` | 0 | event_id, webhook_id, event_type, payload | Webhook event logs. |

## Schema Upgrades Strategy
* Schema updates (such as adding the SQL table for manual Notion matches in PR #87) must be additive and version-checked.
* Before running migrations, copy the database to `/home/gunnarhostetler/PlaudBlender/data/backups/brain_pre_migration.db`.
* Implement transactional migrations. In case of failure, restore the backup copy.
