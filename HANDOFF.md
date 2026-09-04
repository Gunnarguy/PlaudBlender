# Handoff — PlaudBlender, updated 2026-08-29

## System shape

Chronos/PlaudBlender is a Python pipeline that ingests Plaud voice recordings,
processes them into events, indexes them into Qdrant, and pushes to Notion.

- **Production runs on a Raspberry Pi**, not the Mac. `ssh raspberry-pi`
  (Tailscale, `100.76.130.109`), repo at `~/PlaudBlender`, venv at `venv/bin/python`.
- The Pi auto-deploys `origin/main` every 10 min (`chronos-auto-update.timer`).
  Push to main = deploy.
- The Mac checkout is a dev copy. Diagnose production from the Pi, never from the
  Mac's `data/brain.db`.
- Real data lives in the Pi's `~/PlaudBlender/data/brain.db` (733 recordings).
- The Pi has no `sqlite3` CLI. Query it with `venv/bin/python -c` and `sqlite3`
  the module.

## Six commits, pushed and deployed

All six are on `origin/main` as of 2026-08-29 and live on the Pi
(`chronos-auto-update` picked them up at 19:14 PDT; Pi HEAD is `393a909`).

Verified after deploy: all five `chronos-*` services active, no errors in
`chronos-api` or `chronos-auto-sync` logs, `/api/v1/health` ok, `/api/v1/xray/runs`
serving real data. The database is untouched — 733 recordings, 40 tombstones, 13
tables, and the hand-made `janitor_tombstones` DDL byte-identical, confirming
`create_all` skipped it. The ingest hot-path tombstone query was exercised
read-only against production: an unknown id returns None, a known tombstoned id
is still blocked, and the new ORM model maps cleanly onto the live table.

One cleanup is still outstanding — a leftover git worktree registration pointing
at a scratch dir that no longer matters. `git worktree remove --force` is blocked
by the auto-mode classifier, so run it yourself if `git worktree list` still
shows it:

```
git worktree remove --force "/private/tmp/claude-501/-Users-gunnarhostetler-Documents-GitHub-PlaudBlender/1e252227-4a2f-4ead-a4ba-18ae03c736e8/scratchpad/wt"
```

### What the six commits do

| commit | what |
|---|---|
| `80d9c84` | declares `janitor_tombstones` so a fresh database can ingest |
| `934477a` | fixes a stale OpenAI contract test and a lifespan startup race |
| `c3af69b` | keeps the test suite off the real database |
| `76f6070` | removes `plaud_device.py` (562 lines against a 404 API) |
| `ad1124f` | logs trace persistence failures; covers the X-Ray trace endpoints |
| `393a909` | pins the untested route surface so it cannot grow silently |

**Suite went 403 passed / 2 failed / 1 flake → 422 passed.** Five consecutive
full runs leave `data/brain.db` byte-for-byte identical; it used to gain 6 rows
per run.

**`80d9c84` is the one that matters.** `janitor_tombstones` is read on every
ingest but was created nowhere in the repo — the Pi has it only because it was
made by hand, so production works by accident. Any fresh install or rebuilt
database fails its first upsert. Validated by rebuilding the Pi's exact schema
locally (from its live `sqlite_master` plus its 40 real tombstone rows) and
running the real `init_db()` over it: zero schema objects changed, zero row
counts changed, tombstone DDL byte-identical.

`c3af69b` makes the engine honor `$DATABASE_URL`, which `src/config.py` has
always exposed and `/api/health` reports but the engine ignored. Verified unset
on the Pi — absent from the systemd units and the environment, and present in
`.env` only as a commented line, which `EnvironmentFile` ignores.

## Corrections to the previous handoff

Three things it said were wrong. Verify before acting on old notes.

1. **The 3 `TestChronosPipeline` tests were never broken.** Commit `4da468c`
   fixed them in the same commit that caused them (35 lines in
   `tests/test_tier3.py`). All 7 passed on HEAD before any work started.
2. **`test_lifespan_startup_success` was not order-dependent.** It is a race:
   `api/main.py:72` fires `run_startup_tasks()` as a fire-and-forget
   `create_task` and yields immediately. Fixed in `934477a`; it now passes in
   isolation, which it never did.
3. **`test_resolve_openai_api_key` was not broken code.** `59f7224` deliberately
   moved the `CHRONOS_OPENAI_ENABLED` kill switch out of the resolver to the call
   sites. The test was stale.

## Open items

**1. The Plaud One clip — unresolved, and upstream of this codebase.** A 10s clip
recorded on a new Plaud One never appeared. MCP and REST both fully authenticated
and agreeing exactly: 20 files, one serial (`888317281808436884`), 0 clips ≤60s.
A 20-minute poller found nothing. The clip is not in Plaud's cloud. Next step is
the phone app — if it is not visible there either, the One is not linked to this
account (`96904da518ecc246db916fa8f7ac0aa7`, `apple-001535...`) or has not
finished uploading. Note `list_files(page_size=50)` still returns 20 with
`total: None`, so deep history needs pagination.

**2. Spans for not-yet-imported recordings are silently dropped.** Found this
session, pinned by test, deliberately not fixed.
`chronos_execution_spans.recording_id` has a foreign key to `chronos_recordings`
and SQLite runs `foreign_keys=ON`, so tracing a recording that has not been
imported yet violates it and the span disappears. On the Pi, **3 of 33,901 spans
carry a recording_id**, all three process-stage. Scheduled runs pass
`recording_id=None` and are fine; the case that loses data is
`chronos_pipeline.py --ingest --recording-id <new-id>` — the run you would most
want traced. Fixing it means dropping the constraint, which on SQLite means
rebuilding a live 33k-row table. That is a real decision with real risk, so it is
recorded rather than done. `ad1124f` at least makes future failures visible in
the log. `tests/test_api_xray_traces.py::TestSpansRequireTheirRecording` pins the
behavior and tells you what to delete when you fix it.

**3. 24 uncovered API routes, now tracked instead of claimed.** Of 87
non-framework routes, 24 have no test naming them —
`tests/test_api_route_coverage.py::KNOWN_UNCOVERED` lists them, grouped by area,
and fails if the set grows or goes stale. Highest risk first: the admin backup
and stack-restart routes (6), and the Notion match-override writes
(`POST /notion/match/override` and `/override/bulk`). The X-Ray trace routes were
the first group covered.

**4. Duplicated Swift.** `AcousticAnalyzer.swift` and `AcousticReport.swift` exist
in both `PlaudBlenderiOS/` and `~/Documents/GitHub/PlaudAPIConsole`, same origin,
already diverged ~75 and ~66 lines. Four more shared filenames
(`TranscriptResolver`, `XRayView`, `SettingsView`, `JSONValue`) were never
compared — reading files in that repo triggers slow iCloud downloads.

## Gotchas

- The **Mac** `.env` has 5 keys and no Plaud credentials, so the Mac cannot
  ingest. The **Pi** is authenticated and working. Don't confuse the two.
- Plaud's official MCP exposes exactly 7 tools — `login`, `list_files`,
  `get_file`, `get_note`, `get_transcript`, `get_current_user`, `logout`. **No
  device tool**, and the REST device endpoints 404 for this account. That is why
  `plaud_device.py` was deleted; don't rewrite it from the same premise.
- Writes to the Pi over SSH, `git push`, and `git worktree remove` may all be
  blocked by the auto-mode classifier. Hand the user the command instead of
  routing around it.
- Never run bare `agy -p` with the `delegate` skill — it resolves its workspace
  from `~/.gemini/config/projects/default-cli-project.json` and will confidently
  analyze the wrong repo. Only `--project` scopes it; the wrapper handles this.
- zsh does not word-split unquoted `$VAR`. A `for n in $NAMES` loop runs **once**
  with everything as one pattern and reports zero hits — a false pass. Run such
  checks under `bash -c`.
- `grep --include=*.py` fails in zsh with "no matches found"; quote it or use
  `bash -c`.
