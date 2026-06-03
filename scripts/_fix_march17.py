"""Repair the March 17 Notion import so stored DB/Qdrant times match the UI."""

import os
import sys
from datetime import timezone as tz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.chronos.notion_bridge import (
    _estimate_local_start_from_date,
    _local_naive_to_utc_naive,
    _normalize_relative_event_times,
    _parse_transcript_duration,
)
from src.chronos.qdrant_client import ChronosQdrantClient
from src.database.engine import SessionLocal, init_db
from src.database.models import ChronosEvent, ChronosRecording


RECORDING_ID = "notion:32749a74-d54f-81df-a2b1-f3c745f64c37"
RECORDING_DATE = "2026-03-17"


def main() -> int:
    init_db()
    session = SessionLocal()

    rec = session.query(ChronosRecording).filter_by(recording_id=RECORDING_ID).first()
    if not rec:
        print("Recording not found")
        return 1

    transcript = str(rec.transcript or "")
    duration_seconds = _parse_transcript_duration(transcript) or int(
        rec.duration_seconds
    )
    local_start = _estimate_local_start_from_date(RECORDING_DATE)
    utc_start = _local_naive_to_utc_naive(local_start)

    print("BEFORE:")
    print(f"  title:      {rec.title}")
    print(f"  created_at: {rec.created_at}")
    print(f"  duration:   {rec.duration_seconds}s")

    rec.created_at = utc_start
    rec.duration_seconds = duration_seconds
    rec.time_is_estimated = True
    rec.time_estimate_reason = (
        "Estimated from Notion title date and configured fallback start time"
    )

    events = (
        session.query(ChronosEvent)
        .filter_by(recording_id=RECORDING_ID)
        .order_by(ChronosEvent.start_ts)
        .all()
    )
    _normalize_relative_event_times(events, local_start, duration_seconds)
    session.commit()

    qdrant = ChronosQdrantClient()
    points, _ = qdrant.client.scroll(
        collection_name=qdrant.collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="recording_id", match=MatchValue(value=RECORDING_ID))
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    payload_by_id = {str(point.id): point for point in points}
    repaired = 0
    for event in events:
        point_id = str(event.qdrant_point_id or event.event_id)
        if point_id not in payload_by_id:
            continue
        qdrant.client.set_payload(
            collection_name=qdrant.collection_name,
            payload={
                "start_ts": event.start_ts.isoformat(),
                "end_ts": event.end_ts.isoformat(),
                "timestamp": event.start_ts.isoformat(),
                "start_ts_unix": event.start_ts.timestamp(),
                "day_of_week": str(event.day_of_week),
                "hour_of_day": int(event.hour_of_day),
                "duration_seconds": max(
                    0.0, (event.end_ts - event.start_ts).total_seconds()
                ),
            },
            points=[point_id],
        )
        repaired += 1

    print("\nAFTER:")
    print(f"  created_at (UTC): {rec.created_at}")
    print(f"  displays as:      {local_start.strftime('%Y-%m-%d %I:%M %p')} local")
    print(f"  duration:         {rec.duration_seconds}s")
    print(f"  events updated:   {len(events)} SQLite rows, {repaired} Qdrant payloads")

    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
