# PlaudBlender - Chronos: Plaud recordings to a searchable timeline (Python Dash, SQLite, Qdrant) plus the PlaudBlenderiOS Swift client

## Verify

Before any commit (copilot-instructions.md, Coding Rules 5):

    .venv/bin/python -m pytest tests/    # 419 test functions in 44 files; 422 passed on 2026-08-29 (HANDOFF.md); fits a 2 min Bash timeout, once exceeded it (brief section 11)

For one change, run the matching file first: `.venv/bin/python -m pytest tests/test_<module>.py`.
Do not report done until this exits 0 and you have read the output.
The suite stays off the real database since commit `c3af69b`; five full runs leave `data/brain.db` byte-identical (HANDOFF.md).

Production check after a deploy: on the Pi (SSH alias in HANDOFF.md) run `deploy/verify-pi.sh`, which prints PASS, WARN or FAIL
per service, and confirm `/api/v1/health` answers. Diagnose from the Pi, never from the Mac.

## Never

- Never push to `main` unless Gunnar asked for a deploy in this session. The Pi pulls `origin/main` every 10 minutes (`deploy/systemd/chronos-auto-update.timer`, `OnUnitActiveSec=10min`) with no review gate: push is deploy. Until 2026-08-29 only an auto-memory note encoded this (brief section 12.2).
- Never diagnose production from the Mac's `data/brain.db`. It is a dev copy; the Mac `.env` has no Plaud credentials and cannot ingest. Real data is the Pi's `~/PlaudBlender/data/brain.db` (HANDOFF.md, Gotchas).
- Never rewrite a Plaud device API client. The REST device endpoints 404 for this account and the official MCP has no device tool; `plaud_device.py` (562 lines) was deleted for that reason in `76f6070` (HANDOFF.md).
- Never make a billable model call without `track_usage()` from `src.chronos.cost_tracker`, and never scatter `load_dotenv()`; secrets arrive through `src/config.py` (copilot-instructions.md, Coding Rules 1 and 7).
- Never run a pattern check under zsh. Unquoted `$VAR` does not word-split and `--include=*.py` errors, and both produced false passes (HANDOFF.md, Gotchas). Use `bash -c`.

## Where things live

| you need | it is in |
|---|---|
| the Dash UI (entry point) | `python scripts/launch_app.py`, port 8050; code in `app_v2/` (main, layout, components, callbacks, services) |
| the pipeline | `python scripts/chronos_pipeline.py --full`: ingest, process, index, graph |
| the REST API the iOS app calls | `api/`; the untested routes are pinned in `tests/test_api_route_coverage.py::KNOWN_UNCOVERED` (24) and the test fails if the set grows |
| the core engine | `src/chronos/`: ingest, transcript_processor, embedding, qdrant_client, graph, graph_rag, openai_service, cost_tracker, notion_bridge |
| Plaud API clients, webhook, USB watcher | `src/plaud_*.py`, `src/plaud_integrations/`; capabilities in `plaud-capability-manifest.json` |
| models, database, config | `src/models/chronos_schemas.py` (Pydantic), `src/database/` (SQLAlchemy), `src/config.py` |
| the MCP server for ChatGPT and other clients | `scripts/mcp_server.py` (11 tools). Root `mcp.json` is not read by Claude; see NOTES.md |
| the Pi deployment | `deploy/` (bootstrap-pi, update-pi, auto-update, verify-pi, watchdog) and `deploy/systemd/*.service` and `*.timer` |
| the iOS client | `PlaudBlenderiOS/` (own README, ARCHITECTURE.md, ROADMAP.md, xcodeproj); Swift files duplicated with `../PlaudAPIConsole` (HANDOFF.md open item 4) |
| tests | `tests/`; `pyproject.toml` sets `testpaths` and `-v --tb=short` |
| docs | `docs/PROJECT_GUIDE.md`, `docs/chronos-mvp.md`, `HANDOFF.md` (2026-08-29), `README.md` |

## Pointers

- `.github/copilot-instructions.md`: orientation table, UI layout, project structure, key services, coding rules, the Don't list. Not imported: its test count ("124 tests, 11 files") is stale and it does not mention the Pi deployment or the iOS subproject. Fix those three and import it.
- `HANDOFF.md`: the Pi model, the six 2026-08-29 commits, open items and gotchas. Read it before touching ingest, the API or the Pi.
- `SECURITY.md`, `docs/PUBLIC_RELEASE_CHECKLIST.md`: the repo is public-safe by design; secrets only through `.env`, which is never read.
- `Explore` and `Plan` subagents do not load this file; restate the push rule in any subagent prompt.
