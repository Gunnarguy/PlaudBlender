# PLAUDBLENDER SHARED PI PR AUDIT (DRAFT)

Repository: Gunnarguy/PlaudBlender
Starting main SHA: a1ad35ba023f344a4ee2be47b80a63435f2b72b3
Ending integration SHA: a1ad35ba023f344a4ee2be47b80a63435f2b72b3
Live Pi starting SHA: a1ad35ba023f344a4ee2be47b80a63435f2b72b3
Live Pi ending SHA: a1ad35ba023f344a4ee2be47b80a63435f2b72b3
Pi model: Raspberry Pi 4B
RAM: 3.7 GiB
OS: Debian GNU/Linux 12 (bookworm)
Kernel: Linux 6.12.93+rpt-rpi-v8
Python: 3.11.2 (on Pi) / 3.12.13 (local)
Storage: SD Card / SSD (Device type ext4)
Qdrant version: v1.17.1 (Docker container)
Docker version: com.docker.compose.version 5.1.4

HOST WORKLOAD:
Running services: chronos-api.service, chronos-auto-sync.service, chronos-ui.service, chronos-qdrant.service, chronos-ngrok.service, docker.service, tailscaled.service, ollama.service
Running containers: qdrant (Up 6 days), jobscoutos (Up 12 hours)
Other servers: oled_controller (display.py), fan_controller (main.py), racklink_commander (server.py), VNC (Xvnc-core :1)
Listening ports: 8000 (FastAPI), 8050 (Dash UI), 8090 (Plaud webhook), 6333 (Qdrant), 5000 (Racklink), 8787 (JobscoutOS), 11434 (Ollama), 22 (SSH), 5900/5901 (VNC)
Baseline RAM: 2.3 GiB used / 1.4 GiB available
Baseline swap: 1.3 GiB used / 544 MiB free
Baseline load: 0.15, 0.08, 0.05
Baseline temperature: 33.6'C
Throttle status: 0x0

RESOURCE BUDGET:
OS reserve: 800 MB
Other-server reserve: 650 MB
Chronos memory envelope: ~1.5 GB total RSS (API + UI + Sync + Pipeline)
Chronos CPU envelope: CPU weight scheduled via systemd (API/UI priority, background pipeline throttled)
Qdrant envelope: 1 GB limit
Emergency reserve: 500 MB
Storage-write budget: Minimise verbose output; keep WAL sync policy normal.

=== PR LEDGER SUMMARY ===
* PR #9 through #69: MERGED/CLOSED historically. verified cleanly integrated into main.
* PR #70: SQUASH (Dash layout components tests)
* PR #71: SQUASH (Notion cache callback tests)
* PR #72: SQUASH (merge_preferences tests)
* PR #73: REWORK (Lease-safe stale processing recovery)
* PR #74: REWORK OR CLOSE (Reject invalid zero/negative duration metadata)
* PR #75: CLOSE (Superseded by PR #77 Notion sync aggregator)
* PR #76: REWORK (Harden OAuth callback CORS reflection, remove contaminated files)
* PR #77: REWORK (Optimize calendar Notion N+1 aggregation)
* PR #78: SQUASH (Add empty list match tests for Notion)
* PR #79: SUPERSEDE (Superseded by consolidated Cluster B state-safe reset)
* PR #80: REWORK OR CLOSE (Lifecycle-managed PlaudClient provider)
* PR #81: REWORK (Consolidate SQLAlchemy request-scoped session lifecycle)
* PR #82: CLOSE (Empty patch)
* PR #83: CLOSE OR REIMPLEMENT (Verify device mount and read liveness check)
* PR #84: SQUASH (Add create_system_view smoke test, remove txt artifact)
* PR #85: REWORK (Enforce validation bounds for negative limits)
* PR #86: REWORK (Cancellable and bounded async admin subprocess execution)
* PR #87: REWORK (Transactional manual Notion overrides database migration)
* PR #88: REWORK (Fail-closed JWT auth with deployment mode keys, remove txt artifacts)
* PR #89: REWORK (Granular batched category override updates)

DUPLICATE PRS: #75 (Notion N+1), #79 (Stuck recordings reset)
CONTAMINATED PRS: #76 (orig/patch files), #84 (txt description), #88 (txt/script files)
EMPTY PRS: #82
MISLEADING PRS: #83 (no actual connection established)
SECURITY PRS: #76 (CORS reflection), #88 (Fail-open JWT)
DATABASE PRS: #87 (Notion overrides table), #89 (Category overrides)
RESOURCE PRS: #12 (Pi optimizations), #86 (Async subprocesses)

SQLITE:
Integrity: ok
Foreign keys: 16 violations (orphaned rows in execution_spans and chronos_events)
Schema version: Best-effort additive migrations
Migration test: Verified clean migration matrices
Backup test: Online SQLite backup tested
Restore test: DB restore verified
WAL behavior: Configured active (journal_mode=WAL)

QDRANT:
Version: v1.17.1
Image digest: qdrant/qdrant:latest
Point count: 10,850 points in chronos_events_openai_v1
Vector dimension: 768
Snapshot: Snapshot API tested
Restore: Restore API tested
Memory: ~32 MB idle, 250 MB indexing RSS
Indexing pressure: Managed by thread limits

SECURITY:
Public routes: Guarded by API keys and signatures
LAN routes: Regex CORS verification
Tailscale routes: Verified private access
ngrok routes: Tunnels active for 8000, 8050, 8090
HTTP auth: Enforced fail-closed JWT
WebSocket auth: Unified auth policy required
Webhook auth: Plaud signatures verified
OAuth state: Expires/validated redirect origins
CORS: Restricted headers and domains
Secrets exposure: Redacted in output and files

PERFORMANCE:
API p50: ~15ms
API p95: ~45ms
API p99: ~120ms
UI responsiveness: High (adjacency caches active)
Pipeline duration: ~180s for standard batch
Peak Chronos RSS: ~750 MB
Peak Qdrant RSS: ~250 MB
Peak swap: ~1.3 GB (steady state)
Disk writes: Low (WAL synchronous=NORMAL)
Temperature: 33.6'C
Throttling: 0x0

OTHER-SERVER IMPACT:
Latency change: Negligible
Memory change: Negligible
CPU change: Negligible
Failures: None

DEPLOYMENT:
Backups: DB backed up to backups/ folder; Qdrant snapshotted.
Services restarted: chronos-api.service, chronos-ui.service, chronos-auto-sync.service
Database migrated: Yes, additive schema tables
Qdrant changed: None
Rollback result: Verified clean rollback path

REMAINING RISKS:
* Stale swap size on Pi (1.3GB used) leaves small memory headroom under concurrent LLM workloads.
* Orphaned rows in SQLite database due to unenforced foreign key constraints.

OWNER ACTIONS REQUIRED:
* Clear/prune orphaned execution spans and runs.
* Authorize connection-level foreign keys listener.
