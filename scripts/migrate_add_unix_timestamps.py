#!/usr/bin/env python3
"""One-time migration: add start_ts_unix field to existing Qdrant points.

This updates all existing points in the chronos_events collection to include
the new start_ts_unix field (Unix timestamp as float) for efficient range filtering.

Run once after pulling the latest code:
    python scripts/migrate_add_unix_timestamps.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chronos.qdrant_client import ChronosQdrantClient
from qdrant_client.models import UpdateStatus


def migrate_add_unix_timestamps():
    """Add start_ts_unix field to all existing points."""
    client = ChronosQdrantClient()

    print("Fetching all points from collection...")

    # Scroll through all points
    offset = None
    total_updated = 0
    batch_size = 100

    while True:
        # Fetch batch
        result, next_offset = client.client.scroll(
            collection_name=client.collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not result:
            break

        print(f"Processing batch of {len(result)} points...")

        # Update each point
        for point in result:
            payload = point.payload

            # Skip if already has start_ts_unix
            if "start_ts_unix" in payload:
                continue

            # Parse ISO timestamp and convert to Unix
            start_ts_iso = payload.get("start_ts")
            if not start_ts_iso:
                print(f"Warning: Point {point.id} has no start_ts, skipping")
                continue

            try:
                dt = datetime.fromisoformat(start_ts_iso.replace("Z", "+00:00"))
                start_ts_unix = dt.timestamp()

                # Update the point with new field
                client.client.set_payload(
                    collection_name=client.collection_name,
                    payload={"start_ts_unix": start_ts_unix},
                    points=[point.id],
                )

                total_updated += 1

            except Exception as e:
                print(f"Error processing point {point.id}: {e}")
                continue

        print(f"  ... {total_updated} points updated so far")

        # Check if we're done
        if next_offset is None:
            break

        offset = next_offset

    print(f"\n✅ Migration complete! Updated {total_updated} points.")
    print("\nYou can now use the Timeline feature without errors.")


if __name__ == "__main__":
    migrate_add_unix_timestamps()
