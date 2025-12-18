#!/usr/bin/env python
"""CLI to process pending recordings stored in SQLite into chunked segments.

Usage:
    python scripts/process_pending.py

This will:
- Initialize the database if needed
- Fetch recordings with status="raw"
- Chunk transcripts into segments
- Persist segments with status="pending"
- Mark recordings as "processed"

Embedding + Pinecone upsert can be layered separately.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path so 'src' imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.engine import init_db, SessionLocal
from src.processing.engine import ChunkingConfig, process_pending_recordings


def main():
    parser = argparse.ArgumentParser(
        description="Process pending recordings into segments"
    )
    parser.add_argument(
        "--max-words", type=int, default=200, help="Max words per chunk"
    )
    parser.add_argument(
        "--overlap-words", type=int, default=20, help="Overlap words between chunks"
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    args = parser.parse_args()

    init_db()
    session = SessionLocal()
    try:
        cfg = ChunkingConfig(max_words=args.max_words, overlap_words=args.overlap_words)
        summary = process_pending_recordings(session, cfg=cfg, dry_run=args.dry_run)
        print(summary)
    finally:
        session.close()


if __name__ == "__main__":
    main()
