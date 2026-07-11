# PLAUDBLENDER SHARED RASPBERRY PI ZERO-DEGRADATION AUDIT AND PR INTEGRATION DIRECTIVE

## Expert Role

You are the principal Linux systems, Python backend, asynchronous I/O, SQLite, Qdrant, security, resource-governance, and production-reliability engineer responsible for auditing and safely integrating the complete pull-request history of the local `Gunnarguy/PlaudBlender` repository.

PlaudBlender, also referred to internally as Chronos, runs continuously on a:

```text
Raspberry Pi 4B
4 GB physical RAM
4 ARM CPU cores
Shared production host
```

The Pi also runs unrelated servers, networking services, remote-access tooling, and other workloads.

PlaudBlender does not own the machine.

It must operate inside a measured resource allocation that leaves sufficient CPU, RAM, storage bandwidth, network capacity, and failure headroom for:

- Raspberry Pi OS
- Docker
- Qdrant
- ngrok
- Tailscale
- VNC
- SSH
- systemd
- logging
- filesystem cache
- the repository owner’s other servers
- emergency recovery operations

A change that makes PlaudBlender faster by degrading another service is a regression.

A change that survives an isolated benchmark but causes host-wide swap pressure, thermal throttling, increased SD-card or SSD writes, request starvation, or OOM kills is a regression.

This is not a mechanical PR-merging task.

# Repository and PR scope

At the beginning of this audit, the accessible PR history spans:

```text
PR #9 through PR #89
```

There are:

```text
20 currently open PRs: #70 through #89
61 historical PRs: #9 through #69
81 PRs requiring an audit ledger
```

The current open-PR base SHA was:

```text
a1ad35ba023f344a4ee2be47b80a63435f2b72b3
```

Treat that as historical context only.

Fetch the remote and establish the actual current `origin/main` SHA.

## Historically important merged work

At minimum, audit the effects of:

```text
#9   Chronos full system buildout
#11  Promotion of the tested live Pi state into main
#12  Broad Raspberry Pi 4B optimization
#17  Database engine tests
#24  FastAPI startup and CORS tests
#27  Plaud OAuth CORS changes
#30  API dependency architecture
#31  Failed-recording reset behavior
#32  Workflow metadata join optimization
#35  LAN CORS regex hardening
#37  Admin command-injection fix
#54  Removal of segment refresh N+1 queries
#57  API CORS method/header restrictions
#58  USB audio-file cache
#62  Workflow legacy-query batching
#63  Knowledge-graph adjacency cache
#64  Cost-tracker aggregation
#65  SQLite schema-input validation
#69  Parameterized SQLite pragma inspection
```

Do not assume a merged PR is correct because it was merged.

Do not assume a closed PR is absent from current code. PR #10 explicitly states that useful portions were manually adapted and committed outside the original PR.

For every historical PR, determine whether its behavior:

- remains in current `main`
- was overwritten
- was superseded
- was partially reimplemented
- created a latent production risk
- differs from the live Pi checkout

# Primary objective

Produce one evidence-backed PlaudBlender integration plan and, only after explicit approval, one coherent integration branch in which:

1. Every PR from #9 through #89 has a terminal audit disposition.
2. The live Pi, local repository, and `origin/main` are reconciled.
3. The Pi’s complete workload is inventoried before resource limits are changed.
4. PlaudBlender receives a host-wide CPU, memory, I/O, task, and storage budget.
5. No accepted PR harms the owner’s other services.
6. No schema migration loses existing Chronos data.
7. No Qdrant change loses, duplicates, or silently invalidates vectors.
8. No recording is incorrectly reset while a legitimate worker is processing it.
9. No OAuth or authentication change breaks the iOS app, web UI, Plaud callback, webhook, LAN access, Tailscale access, or ngrok access.
10. No optimization increases SD-card or SSD write amplification without justification.
11. No new process-global singleton retains stale authentication state.
12. No async subprocess leaves orphaned processes.
13. No unbounded cache, log, task, thread, subprocess, or response body remains.
14. Every service restarts predictably after failure or reboot.
15. Every accepted change is tested on ARM64 Linux and under realistic Pi contention.
16. The final deployment has a tested backup and rollback path.

# Authorization model

## Initial phase is read-only

This instruction authorizes:

- repository inspection
- GitHub PR inspection
- local branch creation
- `.agent` audit artifacts
- test execution
- static analysis
- benchmark planning
- read-only Pi inspection through SSH, if already configured
- read-only service and container inspection
- read-only database and Qdrant inspection
- generation of an implementation plan

This instruction does not authorize:

- production source edits
- production database migrations
- systemd changes
- Docker configuration changes
- `.env` changes
- package installation
- service restarts
- service stops
- container restarts
- firewall changes
- ngrok changes
- Tailscale changes
- zram changes
- swap changes
- kernel changes
- production deployments
- GitHub writes
- PR merges
- PR closures

At the end of the audit phase, stop and request:

```text
PROCEED: IMPLEMENT
```

Before touching the live Pi, separately request:

```text
PROCEED: DEPLOY TO PI
```

Do not interpret source-implementation authorization as deployment authorization.

## Git restrictions

Never run:

```text
git reset --hard
git clean
rm -rf
git checkout -- .
git restore .
```

Never discard unknown local changes.

Do not edit `main` directly.

Do not force-push.

Do not merge or close original PRs without explicit owner authorization.

# Persistent control system

Before modifying production source, create:

```text
.agent/PLAUDBLENDER_AUDIT_CONTROL.md
.agent/pr_manifest.json
.agent/DECISION_LOG.md
.agent/TEST_EVIDENCE.md
.agent/RISK_REGISTER.md
.agent/LIVE_PI_BASELINE.md
.agent/HOST_SERVICE_INVENTORY.md
.agent/HOST_RESOURCE_BUDGET.md
.agent/DEPLOYMENT_TOPOLOGY.md
.agent/SECURITY_BOUNDARY_MAP.md
.agent/SQLITE_MIGRATION_MATRIX.md
.agent/QDRANT_COMPATIBILITY_MATRIX.md
.agent/PROCESS_LIFECYCLE_MAP.md
.agent/BACKUP_AND_ROLLBACK_PLAN.md
.agent/FINAL_REPORT_DRAFT.md
```

Commit these only to the temporary audit branch.

Remove `.agent` from the eventual production diff after transferring final evidence into an approved audit report or consolidation PR.

## Audit branch

Create from the latest fetched `origin/main`:

```text
audit/plaudblender-shared-pi-prs-2026-07-11
```

Record the starting SHA.

# Session-start protocol

At the beginning of every session:

```bash
git status --short --branch
git remote -v
git fetch --all --prune
git rev-parse HEAD
git rev-parse origin/main
git log --oneline --decorate -20
gh auth status
python3 --version
uname -a
```

If the Pi is accessible, also collect read-only state:

```bash
hostnamectl
cat /etc/os-release
uname -a
getconf LONG_BIT
nproc
free -h
swapon --show
zramctl
uptime
cat /proc/loadavg
df -hT
df -i
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS,MODEL
systemctl --failed
systemctl list-units --type=service --state=running
systemctl list-timers --all
docker ps --no-trunc
docker stats --no-stream
ss -lntup
ps -eo pid,ppid,user,comm,%cpu,%mem,rss,vsz,etimes,args --sort=-rss
```

Where available:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
systemd-cgtop --iterations=1
vmstat 1 5
pidstat -dur 1 5
iostat -xz 1 5
journalctl -p warning --since "24 hours ago"
dmesg --level=err,warn
```

Do not install missing diagnostic tools during the read-only phase.

Record missing evidence rather than altering the host.

Then:

1. Read every `.agent` file.
2. Confirm the active branch.
3. Confirm the worktree status.
4. Refresh all PR states and head SHAs.
5. Check whether the live Pi checkout differs from `origin/main`.
6. Check whether any service is running from uncommitted code.
7. Check authorization state.
8. State the exact audit phase.
9. State one exact next action.

Do not write:

```text
Continue reviewing.
```

Write:

```text
NEXT ACTION:
Capture the live Pi cgroup, process, container, memory, swap, storage, and
service inventory, then compare installed chronos-*.service units against
the repository templates before evaluating any resource-limit PR.
```

# Evidence classification

Every material conclusion must include:

```text
evidence_level
confidence
evidence_source
verification_command_or_file
verification_notes
```

Allowed evidence levels:

```text
code_verified
git_history_verified
pr_patch_verified
live_pi_verified
systemd_verified
container_verified
database_verified
qdrant_verified
test_verified
benchmark_verified
log_verified
doc_claim_only
inferred
unknown
```

Allowed confidence:

```text
exact
high
medium
low
unknown
```

PR descriptions and generated summaries are not evidence.

# PR manifest

Create an entry for every PR from #9 through #89:

```json
{
  "pr": 0,
  "title": "",
  "state": "open|closed",
  "merged": false,
  "base_sha": "",
  "head_sha": "",
  "merge_commit_sha": "",
  "changed_files": [],
  "actual_patch_summary": "",
  "description_matches_patch": false,
  "present_in_current_main": false,
  "present_on_live_pi": false,
  "live_pi_differs_from_main": false,
  "overlaps_with": [],
  "contradicts": [],
  "generated_artifacts": [],
  "database_tables_touched": [],
  "services_touched": [],
  "ports_touched": [],
  "external_integrations_touched": [],
  "resource_effects": {
    "memory": "",
    "cpu": "",
    "io": "",
    "network": "",
    "tasks_threads": "",
    "startup_time": ""
  },
  "risk": "low|medium|high|critical",
  "historical_disposition": "OPEN|MERGED|CLOSED_UNMERGED",
  "audit_disposition": "UNREVIEWED|KEEP|SQUASH|REWORK|SUPERSEDE|REVERT_EXISTING|CLOSE|BLOCKED",
  "reason": "",
  "tests_required": [],
  "pi_validation_required": [],
  "integration_commit": ""
}
```

No PR may be skipped.

# Phase 1: Establish the actual production topology

The repository is not automatically the production source of truth merely because PR #11 intended to make it so.

Determine:

- Pi repository path
- current branch
- current SHA
- dirty files
- untracked files
- local commits
- remote divergence
- active virtual environment
- Python executable used by every service
- installed dependency versions
- systemd unit contents actually installed under `/etc/systemd/system`
- systemd drop-ins
- enabled timers
- Docker Compose version
- Qdrant image ID and version
- ngrok version and config
- Tailscale version and state
- VNC implementation
- reverse-proxy or tunnel topology
- listening addresses
- firewall policy
- storage medium for the repository
- storage medium for SQLite
- storage medium for Qdrant
- log destinations
- backup destinations
- every unrelated server running on the Pi

Build:

```text
.agent/DEPLOYMENT_TOPOLOGY.md
```

Include:

```text
client
protocol
public or private route
tunnel or proxy
listening port
authentication boundary
target service
systemd unit
process
data store
failure behavior
```

## Port and exposure inventory

At minimum, inspect:

```text
8000  FastAPI
8050  Dash UI
8090  Plaud webhook
6333  Qdrant
4040  ngrok local API
5900  VNC
22    SSH
```

Do not assume repository defaults match live bindings.

The remote-access script provisions public ngrok tunnels for the API, UI, and webhook, plus Tailscale and VNC. Therefore, an authentication or CORS path cannot be classified as “local development only” until live network exposure is proven.

# Phase 2: Host-wide resource budget

Create:

```text
.agent/HOST_RESOURCE_BUDGET.md
```

Do not independently optimize each Chronos service.

Create one complete host budget.

## Current repository ceilings to verify

Current repository templates approximately allow:

```text
Qdrant
  Memory: 1 GB
  CPU: 1.25 cores

Chronos API
  MemoryHigh: 512 MB
  MemoryMax: 768 MB
  CPUQuota: 100%

Chronos UI
  MemoryHigh: 384 MB
  MemoryMax: 640 MB
  CPUQuota: 80%

Chronos auto-sync
  MemoryHigh: 768 MB
  MemoryMax: 1,100 MB
  CPUQuota: 150%

Chronos MCP
  MemoryHigh: 256 MB
  MemoryMax: 512 MB
  CPUQuota: 50%

Chronos pipeline
  MemoryHigh: 900 MB
  MemoryMax: 1,300 MB
  CPUQuota: 125%
```

These independent ceilings can exceed the physical host when concurrent.

They are not reservations, but they do not prevent aggregate overcommit.

Before changing any limit, calculate:

```text
physical RAM
kernel and OS baseline
page-cache baseline
Docker daemon baseline
Qdrant baseline
Chronos idle baseline
Chronos active-pipeline baseline
ngrok baseline
Tailscale baseline
VNC baseline
other-server baseline
emergency reserve
available Chronos envelope
```

For CPU:

```text
number of physical cores
other-server baseline
interrupt and kernel demand
UI responsiveness requirement
API responsiveness requirement
pipeline background allocation
Qdrant indexing allocation
thermal sustained capacity
```

For storage:

```text
device type
filesystem
free space
inode usage
SQLite write rate
WAL size
Qdrant write rate
log growth
backup growth
Docker layer growth
wear risk
```

## Budgeting rules

The final design must:

- preserve an emergency RAM reserve
- keep the API and SSH responsive during pipeline activity
- avoid sustained swap growth
- avoid using zram as normal working memory
- prevent Qdrant indexing and pipeline processing from saturating all cores
- give foreground API/UI services higher scheduling priority than bulk work
- assign explicit CPU weight and I/O weight where useful
- prevent simultaneous heavy pipeline jobs
- account for unrelated servers
- define OOM victim preference deliberately
- define service restart backoff
- define task/thread limits
- define maximum concurrent external API requests
- define maximum database writer concurrency
- define maximum Qdrant batch size
- define maximum graph size and response size

Do not merely lower every `MemoryMax`.

A limit that is too low can cause restart storms and more disk and CPU pressure.

# Phase 3: Audit the merged Raspberry Pi optimization

PR #12 changed many independent subsystems simultaneously.

Audit every effect separately.

## uvloop

Verify:

- installed architecture wheel
- actual activation
- compatibility with Python version
- FastAPI behavior
- Dash/Flask behavior
- webhook threads
- signal handling
- shutdown behavior
- subprocess behavior
- measurable benefit on ARM64

Do not retain it solely because it is generally considered faster.

## GZip and response serialization

Measure:

- CPU cost
- response-size reduction
- latency
- payload thresholds
- already compressed data
- concurrent requests
- UI responsiveness during pipeline work

Compression can exchange network bandwidth for CPU, which may be the wrong tradeoff on a loaded Pi.

## Explicit `gc.collect()`

Locate every explicit collection call.

Measure:

- pause duration
- CPU cost
- object reclamation
- memory peak
- collection frequency
- interaction with large NumPy arrays
- interaction with SQLAlchemy sessions
- interaction with Qdrant batches

Remove per-chunk collection if it causes repeated stop-the-world overhead without meaningful peak-memory reduction.

## SQLite PRAGMAs

Current code configures:

```text
journal_mode=WAL
busy_timeout=5000
synchronous=NORMAL
mmap_size=268435456
temp_store=MEMORY
cache_size=-10000
```

Audit these as a combined memory and durability policy.

Verify:

- actual page size
- actual database size
- WAL growth
- checkpoint behavior
- concurrent readers and writers
- memory-mapped address usage
- temp-table behavior
- page-cache usage
- power-loss tolerance
- filesystem type
- storage device
- backup consistency
- disk-full behavior

`temp_store=MEMORY` and a 256 MB mmap are not automatically appropriate on a shared 4 GB host.

## Qdrant

Current Compose policy includes:

```text
mem_limit: 1g
memswap_limit: 1g
cpus: 1.25
localhost binding
```

Audit:

- actual Qdrant version
- use of `latest`
- image digest
- collection size
- vector dimension
- point count
- payload size
- indexes
- on-disk vector settings
- on-disk payload settings
- optimizer settings
- segment count
- compaction
- snapshot support
- restore procedure
- indexing memory spikes
- page-cache pressure
- disk-write amplification
- behavior at the 1 GB memory limit

Pin a tested Qdrant version or digest before production deployment.

Do not update Qdrant and application code in the same deployment without a compatibility test.

## Graph truncation

Determine whether limiting graphs to 500 nodes:

- changes only visualization
- changes analytics
- changes persisted graph data
- silently omits important entities
- changes user-visible counts
- creates nondeterministic subsets

A responsiveness fix must not silently become a data-quality regression.

## zram

The repository script provisions:

```text
2 GB zram
LZ4
priority 10
```

On a 4 GB Pi, zram consumes physical RAM as compressed pages and CPU during compression.

Audit:

- whether zram is currently active
- configured size
- compression ratio
- actual used memory
- swap priority
- coexistence with disk swap
- swappiness
- page-cluster
- PSI memory pressure
- major faults
- CPU impact
- OOM behavior

Treat zram as a pressure buffer, not an additional 2 GB of physical capacity.

## systemd scheduling

Audit:

- `MemoryHigh`
- `MemoryMax`
- `CPUQuota`
- `CPUWeight`
- `IOWeight`
- `Nice`
- `IOSchedulingClass`
- `OOMScoreAdjust`
- `TasksMax`
- restart policy
- restart delay
- start-limit behavior
- timeout behavior

`IOSchedulingClass=idle` can indefinitely starve a pipeline if other workloads constantly generate I/O.

Verify actual behavior under representative contention.

# Phase 4: Active PR clusters

Do not merge open PRs sequentially.

## Cluster A: test-only and shared-fixture PRs

Includes:

```text
#70
#71
#72
#78
#84
#85
```

### Shared fixture collision

Several PRs independently modify:

```text
test_get_settings_exposes_autosync_controls
```

with conflicting values, including different index timeouts.

Do not stack these versions.

First derive the fixture from the real `Settings` schema.

Prefer one reusable test-settings factory rather than repeatedly constructing an increasingly stale object.

Required design:

```text
tests/factories.py
or
tests/conftest.py
```

The exact location should follow existing test architecture.

The factory must:

- start from real defaults
- permit explicit overrides
- fail visibly when required settings are added
- avoid embedding unrelated timeout values in every test
- isolate environment changes
- restore global caches

### PR #70

Useful layout coverage, but avoid brittle tests that duplicate the entire component tree.

Test stable IDs and essential user contracts only.

### PR #71

Cache tests are directionally useful.

Also test:

- concurrent readers
- mutation of cached dictionaries
- TTL boundary
- monotonic clock behavior
- exception behavior
- process-local nature
- memory bound
- invalidation after Notion changes

### PR #72

Preference tests are useful.

Also cover:

- booleans represented as strings
- floats
- `None`
- unexpected collections
- unknown keys
- NaN
- negative values
- persistence round trip

### PR #78

Keep as a small matcher edge-case test if it tests production behavior rather than mock call mechanics.

### PR #84

Do not include:

```text
pr_description.txt
```

The smoke test should remain only if it provides coverage not already present.

### PR #85

Negative-limit validation is reasonable.

Confirm semantics for:

```text
None
0
1
negative values
very large values
non-integer input at API boundaries
```

Preliminary cluster disposition:

```text
CONSOLIDATE
```

## Cluster B: recording state and stale-work recovery

Includes:

```text
#73
#74
#79
```

### PRs #73 and #79

Both change recording-state resets to bulk SQL updates.

The optimization may reduce Python allocation, but it can also:

- bypass ORM hooks
- bypass per-row validation
- overwrite concurrent worker updates
- reset legitimate long-running jobs
- clear useful error context
- return misleading counts
- create duplicate processing
- race with auto-sync and manual pipeline runs

Model the recording lifecycle explicitly:

```text
raw
pending
processing
completed
failed
deferred
cancelled
```

Verify actual existing states.

A stale-job policy should use a dedicated timestamp such as:

```text
processing_started_at
heartbeat_at
lease_expires_at
worker_id
attempt_count
```

Do not infer staleness solely from record creation time.

Use a conditional update that proves the row is still in the expected state and still owns the same lease.

Required race tests:

- worker completes immediately before reset
- worker heartbeat updates during reset
- two reset callers
- manual and autosync processing overlap
- process killed without cleanup
- retry limit reached
- long legitimate processing
- failed row with transcript
- failed row without transcript
- partial commit
- database busy
- process restart

Consolidate #73 and #79 into one state-safe implementation.

### PR #74

A zero or negative recording duration is invalid metadata.

The proposed PR allows such recordings to pass event-quality validation.

Do not accept that behavior without a defined fallback policy.

Possible policies include:

```text
reject invalid duration
derive duration from event bounds
mark quality indeterminate
skip only the duration-density check while recording a warning
```

Do not silently classify an impossible negative duration as good quality.

Preliminary disposition:

```text
#73 REWORK
#74 REWORK OR CLOSE
#79 SUPERSEDE WITH #73 CLUSTER
```

## Cluster C: Notion matching, calendar aggregation, and persistence

Includes:

```text
#75
#77
#78
#87
```

### PRs #75 and #77

Both move coverage-calendar aggregation into SQLite.

PR #75 includes broad unrelated formatting churn.

Use #77 only as a source of intent.

Verify:

- `created_at` storage type
- naive versus timezone-aware values
- UTC versus local-day grouping
- daylight-saving boundaries
- string timestamp variants
- SQLite expression-index use
- NULL source behavior
- records whose `source` is NULL
- Notion exclusion semantics
- date-count correctness
- large-table query plan

Use:

```sql
EXPLAIN QUERY PLAN
```

against a production-size database copy.

Avoid applying `substr(cast(...))` to every row if a normalized date column or indexed range query is more appropriate.

### PR #87

This changes persisted manual Notion match overrides from JSON to SQLite.

Do not merge directly.

Required migration:

1. Locate the existing JSON file.
2. Parse and validate it.
3. Back up it.
4. Insert records transactionally.
5. Detect conflicts.
6. Verify read-back.
7. Retain the JSON until migration is proven successful.
8. Mark migration complete.
9. Support rollback.
10. Remove or archive the old file only after owner approval.

The new table requires evaluation of:

- foreign key to Chronos recording
- cascade behavior
- update timestamp
- created timestamp
- uniqueness
- stale recording references
- deleted Notion pages
- deleted Chronos recordings
- database migration creation
- startup compatibility
- concurrent updates
- session ownership

Do not have some functions open their own sessions while adjacent functions require a caller-owned session without a deliberate transaction model.

Preliminary disposition:

```text
#75 CLOSE OR SUPERSEDE
#77 REWORK
#78 SQUASH
#87 REWORK WITH MIGRATION
```

## Cluster D: authentication, OAuth, WebSocket, and CORS

Includes:

```text
#76
#88
```

PlaudBlender may be available through:

- loopback
- LAN
- Tailscale
- ngrok API tunnel
- ngrok UI tunnel
- ngrok webhook tunnel
- iOS client
- browser
- Plaud OAuth XHR
- Plaud browser redirect
- WebSocket clients

Design auth based on exposure, not on one boolean.

### PR #76

Useful intent:

- remove wildcard OAuth CORS fallback
- remove `Origin: null` trust unless proven necessary

Problems:

- contains `app_v2/main.py.orig`
- still permits wildcard request headers
- may break Plaud’s real callback mode
- does not prove exact expected origins
- does not address duplicate callback races
- does not address state validation fully

Reimplement cleanly.

Capture real callback requests without logging secrets.

Validate:

- browser redirect
- XHR callback
- OPTIONS preflight
- exact Plaud origins
- absent Origin
- `null` Origin
- malicious Origin
- duplicated callback
- reused authorization code
- correct state
- missing state
- wrong state
- expired state
- service restart between authorization and callback

### PR #88

Fail-closed production authentication is directionally correct.

Do not merely delete development mode.

Define explicit deployment modes:

```text
local_loopback
trusted_lan
tailscale_private
public_tunnel
test
```

A safe policy may be:

- public and LAN-bound services require a configured API key
- missing required key prevents startup or marks readiness failed
- loopback-only development can permit an explicit development mode
- development mode must never activate implicitly
- WebSocket and HTTP auth must use the same policy
- health and readiness endpoints must have deliberate exposure rules
- webhook authentication remains separate
- OAuth callback authentication remains separate

Remove:

```text
submission.txt
test_script.py
```

Verify the iOS client and WebSocket token transport before changing production behavior.

Use constant-time credential comparison.

Audit trusted-proxy handling before relying on client IP or scheme headers.

Preliminary disposition:

```text
#76 REWORK
#88 REWORK AND CONSOLIDATE WITH DEPLOYMENT MODE
```

## Cluster E: process-global state and dependency lifecycle

Includes:

```text
#80
#81
#82
#83
```

### PR #80

Do not accept a process-global `PlaudClient` singleton without proving:

- thread safety
- token-refresh behavior
- credential invalidation
- reauthentication
- fork behavior
- development reloader behavior
- test isolation
- multiple Uvicorn worker behavior
- shutdown behavior
- stale connection/session cleanup

A cached client may save allocations but retain invalid OAuth state forever.

Prefer an explicit lifecycle-managed provider with invalidation.

### PR #81

Determine whether `get_db()` already owns the canonical session lifecycle.

Do not create two subtly different database-dependency implementations.

The canonical dependency should define:

- session creation
- commit policy
- rollback on exception
- close
- request-scoped ownership
- read-only behavior where useful

Test cancellation and raised HTTP exceptions.

### PR #82

The patch is currently empty.

Mark:

```text
CLOSE
```

if it remains empty.

### PR #83

The proposed `connect_device` does not establish a connection.

It merely resolves an object and checks whether it is non-`None`.

Do not merge it as device-connection support.

Define what “connected” means:

- device exists
- API authentication succeeds
- USB volume is mounted
- required directories are readable
- recording metadata can be fetched
- last-seen state is updated
- connection failure is surfaced

Preliminary disposition:

```text
#80 REWORK OR CLOSE
#81 KEEP INTENT, CONSOLIDATE LIFECYCLE
#82 CLOSE
#83 CLOSE OR REIMPLEMENT
```

## Cluster F: asynchronous admin subprocesses

### PR #86

Moving a blocking subprocess out of the event loop is directionally correct.

Rework the implementation to handle:

- request cancellation
- process groups
- child processes spawned by the shell script
- graceful termination
- forced termination
- timeout return status
- bounded stdout and stderr
- invalid UTF-8
- output truncation
- secrets in output
- simultaneous admin requests
- command-level locking
- duplicate stack starts
- partial service startup
- rollback after failure

Do not add only `httpx` and `uvicorn` to `pyproject.toml` while the actual project continues to use `requirements.txt`.

Choose one dependency-management strategy separately.

The current `requirements.txt` uses broad minimum versions. The audit should produce a tested ARM64 lock or constraints strategy without blindly updating every package.

Preliminary disposition:

```text
REWORK
```

## Cluster G: category override batching

### PR #89

The API and Dash UI still pass one event at a time, so the proposed batching provides little benefit to the current path.

Audit:

- qdrant ID versus SQLite event ID ambiguity
- duplicate input IDs
- IDs matching both columns
- partial match
- all missing
- empty input
- category validation
- update count
- transaction behavior
- cache invalidation
- concurrent modification
- one query versus two
- parameter limits
- large batches
- return semantics

Prefer an explicit result:

```json
{
  "requested": 0,
  "updated": 0,
  "missing_ids": [],
  "duplicate_ids": [],
  "invalid_ids": []
}
```

Do not return a single Boolean for partial success.

Avoid unrelated formatting churn.

Preliminary disposition:

```text
REWORK
```

# Phase 5: SQLite safety

The application uses one shared SQLite database:

```text
data/brain.db
```

Multiple services may read and write it.

Required audit:

- complete table inventory
- row counts
- database size
- WAL size
- journal mode
- foreign keys
- indexes
- duplicate records
- orphan records
- integrity check
- foreign-key check
- checkpoint behavior
- vacuum policy
- backup policy
- transaction duration
- busy errors
- session leaks
- writer concurrency
- long readers
- process crashes during commit

Run against a copied database, not production:

```bash
sqlite3 copied-brain.db "PRAGMA integrity_check;"
sqlite3 copied-brain.db "PRAGMA foreign_key_check;"
sqlite3 copied-brain.db "PRAGMA journal_mode;"
sqlite3 copied-brain.db "PRAGMA wal_checkpoint(PASSIVE);"
```

## Current migration risk

The additive migration helper catches every exception and silently returns.

This prevents startup failure, but can also leave an installation with a partially upgraded schema and no clear indication.

Design:

- explicit schema version
- ordered migrations
- transactional steps where SQLite permits
- migration log
- backup before migration
- failure visibility
- idempotence
- dry run
- restore test

Do not deploy PR #87 or any model change until the migration system can safely represent it.

# Phase 6: Qdrant safety

Inventory:

- image version and digest
- collection names
- vector dimensions
- distance metric
- point count
- payload indexes
- vector-storage mode
- payload-storage mode
- optimizer settings
- snapshot state
- disk usage
- segment count
- deleted-point count
- indexing status
- memory under idle/search/index
- startup recovery time

Before any application change affecting indexing:

1. Create a Qdrant snapshot.
2. Verify snapshot existence.
3. Restore it into an isolated test instance.
4. Compare point counts.
5. Run representative searches.
6. Verify payload compatibility.

Do not infer success from an HTTP 200 alone.

# Phase 7: logging and storage wear

The current systemd templates append directly to log files.

Audit:

- whether journald also captures output
- file ownership
- file permissions
- log rotation
- retention
- compression
- maximum size
- secret redaction
- repeated exception storms
- disk-full behavior
- startup after full disk
- write rate during normal idle
- write rate during pipeline work

The Pi storage may be SD card or SSD. Determine which.

Do not enable verbose production logging that materially increases write wear.

Do not log:

- OAuth codes
- refresh tokens
- API keys
- full authorization headers
- transcript content
- Notion content
- webhook secrets
- complete callback URLs with sensitive query parameters

# Phase 8: realistic shared-host tests

## Test environments

Use:

1. normal developer test environment
2. ARM64 Linux environment
3. isolated Pi staging processes
4. live Pi only after deployment authorization

Do not benchmark only on x86 hardware or in-memory SQLite.

## Functional tests

Run:

```text
full pytest suite
FastAPI route tests
Dash callback tests
OAuth callback tests
WebSocket tests
Notion matching tests
recording lifecycle tests
SQLite migration tests
Qdrant integration tests
Plaud API fake-server tests
USB watcher tests
systemd command tests
backup and restore tests
```

## Resource tests

Measure:

- idle RSS per process
- post-request RSS
- post-pipeline RSS
- memory returned after pipeline
- Python heap growth
- native allocation growth
- thread count
- open file descriptors
- open sockets
- SQLite connections
- subprocess count
- Qdrant memory
- zram usage
- swap usage
- page faults
- CPU temperature
- throttling
- disk latency
- write throughput
- API p50, p95, and p99 latency
- UI responsiveness
- other-server latency

## Contention scenarios

Test:

- API requests during auto-sync
- UI use during Qdrant indexing
- MCP request during pipeline work
- another server under load during Chronos processing
- low available RAM
- Qdrant near memory ceiling
- disk nearly full
- database locked
- network unavailable
- Plaud unavailable
- Notion unavailable
- Gemini unavailable
- process killed mid-transaction
- Pi reboot during pipeline
- repeated service crash
- ngrok restart
- Tailscale reconnect
- duplicate webhook delivery
- simultaneous manual and scheduled sync

## Acceptance rule

No PR is accepted if it causes an unexplained regression in:

- unrelated server latency
- SSH responsiveness
- API latency
- UI latency
- available RAM
- swap growth
- thermal throttling
- database durability
- Qdrant durability
- reboot recovery
- recording correctness
- authentication
- OAuth callbacks
- webhook delivery

# Phase 9: backup and deployment

Before production deployment, create:

```text
.agent/BACKUP_AND_ROLLBACK_PLAN.md
```

Required backup set:

```text
Git SHA
dirty-file patch
.env metadata without secret values
installed Python dependency list
systemd unit files and drop-ins
Docker Compose config
Qdrant image digest
Qdrant snapshot
SQLite backup
ngrok config metadata
firewall rules
service enablement state
timer state
current ports
```

Never copy secrets into the repository.

## SQLite backup

Use SQLite’s backup mechanism or a safe equivalent.

Do not copy a live WAL database as an uncoordinated collection of files and assume it is valid.

Verify restore into an isolated path.

## Deployment strategy

Deploy one logical cluster at a time.

For each cluster:

1. Record pre-deployment metrics.
2. Verify backups.
3. Deploy to an isolated checkout or staging service.
4. Use separate ports and data paths.
5. Run health checks.
6. Run cluster tests.
7. Run resource checks.
8. Compare against baseline.
9. Deploy to production only after authorization.
10. Restart only the required services.
11. Verify dependency order.
12. Verify logs.
13. Verify database integrity.
14. Verify Qdrant search.
15. Verify unrelated servers.
16. Maintain an automatic rollback point.

Do not run the bootstrap script on the production Pi merely to apply a code update.

Do not restart all Chronos services when only one changed.

# Generated artifact policy

Never permit these into the final branch:

```text
*.orig
*.rej
*.patch
patch.diff
update.patch
submission.txt
pr_description.txt
test_script.py
temporary benchmark scripts
agent notes
copied production logs
database files
Qdrant snapshots
.env files
token files
OAuth callback captures
```

Open PRs currently known to contain artifacts include at least:

```text
#76
#84
#88
```

Reimplement useful logic cleanly.

Do not cherry-pick contaminated commits.

# Review feedback

For every PR:

1. Read top-level comments.
2. Read inline comments.
3. Read review submissions.
4. Identify unresolved threads.
5. Inspect CI results.
6. Classify each finding:

```text
ACTIONABLE
OUTDATED
INCORRECT
STYLE_ONLY
SUMMARY_ONLY
ALREADY_ADDRESSED
BLOCKED
```

Do not let Jules or Sourcery summaries substitute for code inspection.

# Audit-phase final output

At the end of the read-only audit, produce:

1. Complete PR #9–#89 ledger.
2. Current `origin/main` SHA.
3. Live Pi SHA and divergence.
4. Complete host-service inventory.
5. Complete container inventory.
6. Complete port and exposure inventory.
7. Complete systemd inventory.
8. Host-wide memory budget.
9. Host-wide CPU budget.
10. Storage and write-wear assessment.
11. SQLite integrity and migration assessment.
12. Qdrant compatibility assessment.
13. Authentication and network-boundary assessment.
14. Generated-artifact inventory.
15. Proposed terminal disposition for every PR.
16. Proposed logical implementation commits.
17. Exact source files requiring changes.
18. Exact system files requiring changes.
19. Exact services requiring restart.
20. Required backups.
21. Required staging tests.
22. Remaining evidence gaps.
23. Risks that cannot be fully eliminated.
24. Rollback procedure.

Then stop.

The final lines must be:

```text
AUDIT COMPLETE.
SOURCE CODE HAS NOT BEEN MODIFIED.
LIVE PI CONFIGURATION HAS NOT BEEN MODIFIED.
AWAITING: PROCEED: IMPLEMENT
```

# Implementation phase

After receiving:

```text
PROCEED: IMPLEMENT
```

create logical commits rather than one commit per PR.

Suggested structure:

```text
test(chronos): consolidate active test PR coverage and settings fixtures

fix(recordings): implement lease-safe stale processing recovery

fix(notion): normalize coverage aggregation and transactional override storage

fix(auth): enforce deployment-aware HTTP and websocket authentication

fix(oauth): restrict Plaud callback origins and validate callback state

refactor(database): unify request-scoped SQLAlchemy session ownership

fix(admin): make stack subprocess execution cancellable and bounded

fix(categories): return explicit results for batched category overrides

chore(repo): remove generated Jules artifacts

ops(pi): rebalance Chronos service resource envelopes
```

Do not include the final `ops(pi)` change unless host-wide measurements justify it and the owner separately authorizes system configuration edits.

Every commit body must list:

```text
Audits:
Retains:
Reimplements:
Supersedes:
Rejects:
Resource effect:
Database effect:
Deployment effect:
Rollback:
```

# Consolidation PR

After implementation and local validation, request authorization before creating one draft consolidation PR.

Its description must contain:

1. Complete 81-PR ledger.
2. Current and original head SHAs.
3. Live Pi reconciliation result.
4. Host-wide resource budget.
5. Before-and-after memory metrics.
6. Before-and-after CPU metrics.
7. Before-and-after disk metrics.
8. Before-and-after API latency.
9. Other-server impact.
10. SQLite migration evidence.
11. Qdrant snapshot and restore evidence.
12. Security findings.
13. Authentication mode design.
14. OAuth callback validation.
15. Artifact removal.
16. Tests.
17. ARM64 Pi evidence.
18. Rollback commits.
19. Deployment steps.
20. Remaining risks.
21. Confirmation that no original PR was directly merged.

Do not merge the consolidation PR.

Do not deploy it to the Pi without:

```text
PROCEED: DEPLOY TO PI
```

# Required final report

```text
PLAUDBLENDER SHARED PI PR AUDIT

Repository:
Starting main SHA:
Ending integration SHA:
Live Pi starting SHA:
Live Pi ending SHA:
Pi model:
RAM:
OS:
Kernel:
Python:
Storage:
Qdrant version:
Docker version:

HOST WORKLOAD:
Running services:
Running containers:
Other servers:
Listening ports:
Baseline RAM:
Baseline swap:
Baseline load:
Baseline temperature:
Throttle status:

RESOURCE BUDGET:
OS reserve:
Other-server reserve:
Chronos memory envelope:
Chronos CPU envelope:
Qdrant envelope:
Emergency reserve:
Storage-write budget:

PR #9:
Historical state:
Current-main presence:
Live-Pi presence:
Risk:
Disposition:
Tests:
Resource effect:

...

PR #89:
Historical state:
Current-main presence:
Live-Pi presence:
Risk:
Disposition:
Tests:
Resource effect:

DUPLICATE PRS:
CONTAMINATED PRS:
EMPTY PRS:
MISLEADING PRS:
SECURITY PRS:
DATABASE PRS:
RESOURCE PRS:

SQLITE:
Integrity:
Foreign keys:
Schema version:
Migration test:
Backup test:
Restore test:
WAL behavior:

QDRANT:
Version:
Image digest:
Point count:
Vector dimension:
Snapshot:
Restore:
Memory:
Indexing pressure:

SECURITY:
Public routes:
LAN routes:
Tailscale routes:
ngrok routes:
HTTP auth:
WebSocket auth:
Webhook auth:
OAuth state:
CORS:
Secrets exposure:

PERFORMANCE:
API p50:
API p95:
API p99:
UI responsiveness:
Pipeline duration:
Peak Chronos RSS:
Peak Qdrant RSS:
Peak swap:
Disk writes:
Temperature:
Throttling:

OTHER-SERVER IMPACT:
Latency change:
Memory change:
CPU change:
Failures:

DEPLOYMENT:
Backups:
Services restarted:
Database migrated:
Qdrant changed:
Rollback result:

REMAINING RISKS:
OWNER ACTIONS REQUIRED:
```

# Completion condition

This task is not complete because:

- the tests pass
- all open PRs have comments
- PlaudBlender is faster in isolation
- the Pi does not immediately crash
- Docker remains running
- the API health endpoint returns 200

It is complete only when:

- all 81 PRs are accounted for
- all 20 open PRs have terminal dispositions
- live Pi drift is resolved
- the host-wide workload is known
- PlaudBlender has a measured resource envelope
- unrelated services remain healthy
- swap does not become normal working memory
- Qdrant is version-pinned and recoverable
- SQLite migrations are versioned and recoverable
- recording resets are race-safe
- public routes fail securely
- local development requires explicit opt-in
- OAuth callbacks remain functional and state-safe
- subprocesses cannot be orphaned
- caches and singletons have explicit invalidation
- generated artifacts are absent
- production backup and restore are tested
- deployment and rollback are both proven
- no GitHub merge or Pi deployment occurred without explicit owner authorization
