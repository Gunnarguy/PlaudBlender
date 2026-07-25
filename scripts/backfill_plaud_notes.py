#!/usr/bin/env python
"""Backfill chronos_recordings.plaud_ai_summary from Plaud's MCP get_note tool.

Plaud already generates a structured summary for every recording. The column
has readers (notion_bridge, ask_context, transcript_processor) but nothing ever
wrote to it, so the Plaud AI tab renders empty for all recordings.

Runs against the shared MCP adapter, which keeps one npx subprocess alive, so
this is deliberately serial — spawning parallel adapters would start a stdio
server per worker.

    venv/bin/python scripts/backfill_plaud_notes.py --limit 25
    venv/bin/python scripts/backfill_plaud_notes.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.database.engine import SessionLocal  # noqa: E402
from src.plaud_integrations.mcp_account import PlaudMCPAccountAdapter  # noqa: E402


def pending_recordings(session, limit: int | None) -> list[tuple[str, str]]:
    # Only Plaud-sourced rows carry a real Plaud file_id; notion:* and
    # local_import ids 500 against the MCP server.
    sql = (
        "SELECT recording_id, COALESCE(title, '') FROM chronos_recordings "
        "WHERE (plaud_ai_summary IS NULL OR plaud_ai_summary = '') "
        "AND source = 'plaud' "
        "ORDER BY created_at DESC"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [(row[0], row[1]) for row in session.execute(text(sql)).all()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--limit", type=int, help="Process at most N recordings")
    group.add_argument("--all", action="store_true", help="Process every pending recording")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between calls")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write")
    args = parser.parse_args()

    adapter = PlaudMCPAccountAdapter()
    filled = skipped = failed = 0

    with SessionLocal() as session:
        targets = pending_recordings(session, None if args.all else args.limit)
        print(f"{len(targets)} recordings without a Plaud summary")

        for index, (recording_id, title) in enumerate(targets, 1):
            label = (title or recording_id)[:48]
            try:
                note = adapter.get_note(recording_id)
                markdown = (note.markdown or "").strip()
            except Exception as exc:  # noqa: BLE001 - one bad recording must not stop the run
                failed += 1
                print(f"  [{index}/{len(targets)}] FAIL {label}: {str(exc)[:80]}")
                continue

            if not markdown:
                skipped += 1
                print(f"  [{index}/{len(targets)}] none {label}")
                continue

            if not args.dry_run:
                session.execute(
                    text(
                        "UPDATE chronos_recordings SET plaud_ai_summary = :md "
                        "WHERE recording_id = :rid"
                    ),
                    {"md": markdown, "rid": recording_id},
                )
                session.commit()

            filled += 1
            print(f"  [{index}/{len(targets)}] ok   {label} ({len(markdown):,} chars)")
            time.sleep(args.delay)

    verb = "would fill" if args.dry_run else "filled"
    print(f"\n{verb} {filled} | no summary {skipped} | failed {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
