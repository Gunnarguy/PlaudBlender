#!/usr/bin/env python3
"""Safely rebuild a Plaud MCP recording batch under the current OpenAI path.

Replaces existing Chronos events and Qdrant vectors only after a new extraction
successfully completes, so previously processed recordings do not lose data on a
transient model/API failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import bindparam, func, select, text

from src.chronos.embedding_service import ChronosEmbeddingService
from src.chronos.qdrant_client import ChronosQdrantClient
from src.chronos.transcript_processor import TranscriptProcessor
from src.database.chronos_repository import (
    add_chronos_events,
    delete_chronos_events_by_recording,
    mark_chronos_recording_status,
    set_chronos_recording_transcript,
)
from src.database.engine import SessionLocal, init_db
from src.database.models import ChronosEvent as ChronosEventDB, ChronosRecording
from src.models.chronos_schemas import (
    ChronosEvent as ChronosEventSchema,
    DayOfWeek,
    EventCategory,
    SpeakerMode,
)

logger = logging.getLogger(__name__)


def _load_ids(path: Path) -> list[str]:
    items = json.loads(path.read_text())
    ids: list[str] = []
    for item in items:
        for key in ("id", "file_id", "recording_id"):
            value = item.get(key)
            if value:
                ids.append(str(value))
                break
    return list(dict.fromkeys(ids))


def _select_ids_needing_work(session, ids: list[str]) -> list[str]:
    ids_param = bindparam("ids", expanding=True)
    sql = text(
        """
        with event_counts as (
            select
                e.recording_id as recording_id,
                count(*) as total_events,
                sum(case when e.qdrant_point_id is null then 1 else 0 end) as unindexed_events
            from chronos_events e
            where e.recording_id in :ids
            group by e.recording_id
        )
        select r.recording_id
        from chronos_recordings r
        left join event_counts ec on ec.recording_id = r.recording_id
        where r.recording_id in :ids
          and (
              r.processing_status != 'completed'
              or coalesce(ec.total_events, 0) = 0
              or coalesce(ec.unindexed_events, 0) > 0
          )
        order by r.created_at desc
        """
    ).bindparams(ids_param)
    return [str(row[0]) for row in session.execute(sql, {"ids": ids}).fetchall()]


def _build_plaud_context(rec: ChronosRecording) -> str | None:
    parts: list[str] = []
    ai_summary = getattr(rec, "plaud_ai_summary", None)
    if ai_summary:
        parts.append(f"AI Summary: {ai_summary}")

    extracted = getattr(rec, "plaud_extracted_data", None)
    if extracted and isinstance(extracted, dict):
        parts.append(
            "Extracted Data: " + json.dumps(extracted, default=str)[:2000]
        )

    return "\n\n".join(parts) if parts else None


def _recording_date(rec: ChronosRecording) -> str:
    created_at = getattr(rec, "created_at", None)
    if created_at is None:
        return ""
    try:
        if isinstance(created_at, str):
            return created_at[:10]
        return created_at.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _build_db_events(output) -> list[ChronosEventDB]:
    return [
        ChronosEventDB(
            event_id=str(uuid.uuid4()),
            recording_id=e.recording_id,
            start_ts=e.start_ts,
            end_ts=e.end_ts,
            day_of_week=str(getattr(e.day_of_week, "value", e.day_of_week)),
            hour_of_day=e.hour_of_day,
            clean_text=e.clean_text,
            category=str(getattr(e.category, "value", e.category)),
            sentiment=e.sentiment,
            keywords=e.keywords,
            speaker=str(getattr(e.speaker, "value", e.speaker)),
            raw_transcript_snippet=e.raw_transcript_snippet,
            gemini_reasoning=e.gemini_reasoning,
        )
        for e in output.events
    ]


def _fetch_transcript(processor: TranscriptProcessor, session, recording_id: str) -> str:
    file_details = processor.plaud.get_recording(recording_id)
    transcript_text = processor._extract_transcript(file_details) or ""
    if len(transcript_text.strip()) >= 100:
        set_chronos_recording_transcript(session, recording_id, transcript_text)
    return transcript_text


def _index_recordings(session, recording_ids: list[str]) -> int:
    qdrant = ChronosQdrantClient()
    embedder = ChronosEmbeddingService()

    qdrant.create_collection(force_recreate=False)

    events_to_index = (
        session.query(ChronosEventDB)
        .filter(ChronosEventDB.recording_id.in_(recording_ids))
        .filter(ChronosEventDB.qdrant_point_id.is_(None))
        .all()
    )
    if not events_to_index:
        return 0

    pydantic_events: list[ChronosEventSchema] = []
    valid_db_events: list[ChronosEventDB] = []
    for db_event in events_to_index:
        try:
            pydantic_events.append(
                ChronosEventSchema(
                    event_id=db_event.event_id,
                    recording_id=db_event.recording_id,
                    start_ts=db_event.start_ts,
                    end_ts=db_event.end_ts,
                    day_of_week=DayOfWeek(db_event.day_of_week),
                    hour_of_day=db_event.hour_of_day,
                    clean_text=db_event.clean_text,
                    category=EventCategory(db_event.category),
                    category_confidence=getattr(db_event, "category_confidence", None),
                    sentiment=db_event.sentiment,
                    keywords=db_event.keywords or [],
                    speaker=(
                        SpeakerMode(db_event.speaker)
                        if db_event.speaker
                        else SpeakerMode.SELF_TALK
                    ),
                    raw_transcript_snippet=db_event.raw_transcript_snippet,
                    gemini_reasoning=db_event.gemini_reasoning,
                )
            )
            valid_db_events.append(db_event)
        except Exception as exc:
            logger.warning("Skipping invalid event %s: %s", db_event.event_id, exc)

    if not pydantic_events:
        return 0

    texts = [event.clean_text for event in pydantic_events]
    embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        embeddings.extend(
            embedder.embed_batch(
                texts[i : i + batch_size],
                task_type="RETRIEVAL_DOCUMENT",
                batch_size=batch_size,
            )
        )

    indexed_count = qdrant.upsert_events_batch(pydantic_events, embeddings)

    for db_event, event in zip(valid_db_events, pydantic_events):
        db_event.qdrant_point_id = event.event_id
    session.commit()
    return indexed_count


def reprocess_batch(
    input_path: Path,
    *,
    limit: int | None = None,
    skip_index: bool = False,
    needs_work_only: bool = False,
    index_only: bool = False,
) -> dict:
    ids = _load_ids(input_path)

    session = SessionLocal()
    if needs_work_only:
        ids = _select_ids_needing_work(session, ids)
    if limit is not None:
        ids = ids[:limit]

    processor = TranscriptProcessor(db_session=session)
    qdrant = ChronosQdrantClient()

    success = 0
    failed = 0
    preserved = 0
    fetched = 0
    replaced_events = 0
    removed_qdrant_points = 0
    index_error: str | None = None
    started = time.time()

    try:
        print(
            f"Rebuilding {len(ids)} Plaud MCP recordings with current OpenAI settings...",
            flush=True,
        )

        if not index_only:
            for idx, recording_id in enumerate(ids, 1):
                rec = session.get(ChronosRecording, recording_id)
                if rec is None:
                    failed += 1
                    print(f"[{idx}/{len(ids)}] {recording_id[:12]} missing local row", flush=True)
                    continue

                old_status = str(rec.processing_status or "pending")
                old_event_count = session.execute(
                    select(func.count())
                    .select_from(ChronosEventDB)
                    .where(ChronosEventDB.recording_id == recording_id)
                ).scalar_one()

                transcript_text = str(rec.transcript or "").strip()
                transcript_source = "cached" if len(transcript_text) >= 100 else "plaud"
                print(
                    f"[{idx}/{len(ids)}] {recording_id[:12]} status={old_status} old_events={old_event_count} transcript={transcript_source}",
                    flush=True,
                )
                t0 = time.time()

                if len(transcript_text) < 100:
                    try:
                        transcript_text = _fetch_transcript(processor, session, recording_id)
                        if len(transcript_text.strip()) >= 100:
                            fetched += 1
                    except Exception as exc:
                        err = f"Plaud transcript fetch failed: {str(exc)[:120]}"
                        if old_event_count == 0:
                            mark_chronos_recording_status(
                                session, recording_id, "failed", error_message=err
                            )
                            failed += 1
                            print(f"   FAIL {err}", flush=True)
                        else:
                            preserved += 1
                            print(f"   KEEP {err}", flush=True)
                        continue

                if len(transcript_text) < 100:
                    err = "Transcript missing or too short for reprocessing"
                    if old_event_count == 0:
                        mark_chronos_recording_status(
                            session, recording_id, "failed", error_message=err
                        )
                        failed += 1
                        print(f"   FAIL {err}", flush=True)
                    else:
                        preserved += 1
                        print(f"   KEEP {err}", flush=True)
                    continue

                output = processor.process_transcript_text(
                    transcript_text,
                    recording_id,
                    verbose=False,
                    recording_date=_recording_date(rec),
                    plaud_context=_build_plaud_context(rec),
                )

                if not output or not output.events:
                    err = processor._last_processing_error or "No events extracted"
                    if old_event_count == 0:
                        mark_chronos_recording_status(
                            session, recording_id, "failed", error_message=err
                        )
                        failed += 1
                        print(f"   FAIL {err[:140]}", flush=True)
                    else:
                        preserved += 1
                        print(f"   KEEP {err[:140]}", flush=True)
                    continue

                deleted_points = qdrant.delete_by_recording_id(recording_id)
                deleted_events = delete_chronos_events_by_recording(session, recording_id)

                add_chronos_events(session, _build_db_events(output))
                mark_chronos_recording_status(
                    session, recording_id, "completed", error_message=None
                )

                success += 1
                replaced_events += deleted_events
                removed_qdrant_points += deleted_points
                print(
                    f"   OK {len(output.events)} events in {time.time() - t0:.1f}s "
                    f"(replaced {deleted_events}, deleted {deleted_points} qdrant)",
                    flush=True,
                )

        indexed_count = 0
        if not skip_index:
            print("Indexing rebuilt events into Qdrant...", flush=True)
            try:
                indexed_count = _index_recordings(session, ids)
            except Exception as exc:
                index_error = str(exc)
                print(f"Index step failed: {index_error[:200]}", flush=True)

        ids_param = bindparam("ids", expanding=True)
        latest_status = text(
            """
            select r.processing_status, count(*)
            from chronos_recordings r
            where r.recording_id in :ids
            group by r.processing_status
            order by r.processing_status
            """
        ).bindparams(ids_param)
        latest_no_events = text(
            """
            select count(*)
            from chronos_recordings r
            left join chronos_events e on e.recording_id = r.recording_id
            where r.recording_id in :ids and e.recording_id is null
            """
        ).bindparams(ids_param)
        latest_unindexed = text(
            """
            select count(*)
            from chronos_events e
            where e.recording_id in :ids and e.qdrant_point_id is null
            """
        ).bindparams(ids_param)

        status_rows = session.execute(latest_status, {"ids": ids}).fetchall()
        without_events = session.execute(latest_no_events, {"ids": ids}).scalar_one()
        unindexed_events = session.execute(latest_unindexed, {"ids": ids}).scalar_one()

        return {
            "latest_batch_total": len(ids),
            "rebuilt_successfully": success,
            "failed_without_old_data": failed,
            "preserved_old_data_on_failure": preserved,
            "transcripts_fetched_from_plaud": fetched,
            "replaced_old_events": replaced_events,
            "deleted_old_qdrant_points": removed_qdrant_points,
            "indexed_events": indexed_count,
            "index_error": index_error,
            "latest_batch_status_counts": [tuple(r) for r in status_rows],
            "latest_batch_without_events": without_events,
            "latest_batch_unindexed_events": unindexed_events,
            "elapsed_seconds": round(time.time() - started, 1),
        }
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely rebuild Plaud MCP recordings")
    parser.add_argument(
        "--input",
        default="data/plaud_latest_200.json",
        help="JSON file containing Plaud recording metadata",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N recordings from the input set",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Rebuild events only; skip the Qdrant indexing pass",
    )
    parser.add_argument(
        "--needs-work-only",
        action="store_true",
        help="Only operate on recordings that are failed, have no events, or still have unindexed events",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip transcript reprocessing and only index unindexed events for the selected recordings",
    )
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="Use a cheaper extraction model while keeping high-fidelity embeddings (gpt-5.4-mini + text-embedding-3-large)",
    )
    parser.add_argument(
        "--cleaning-model",
        default=None,
        help="Override the extraction model for this run only",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the embedding model for this run only",
    )
    args = parser.parse_args()

    if args.cheap and not args.cleaning_model:
        args.cleaning_model = "gpt-5.4-mini"
    if args.cheap and not args.embedding_model:
        args.embedding_model = "text-embedding-3-large"

    if args.cleaning_model:
        os.environ["CHRONOS_CLEANING_MODEL"] = args.cleaning_model
        os.environ["OPENAI_MODEL"] = args.cleaning_model
    if args.embedding_model:
        os.environ["CHRONOS_EMBEDDING_MODEL"] = args.embedding_model

    logging.basicConfig(level=logging.INFO)
    for name in [
        "httpx",
        "openai",
        "src.chronos.qdrant_client",
        "src.chronos.embedding_service",
        "src.chronos.transcript_processor",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    selected_cleaning_model = os.getenv("CHRONOS_CLEANING_MODEL", "gpt-5.5")
    selected_embedding_model = os.getenv(
        "CHRONOS_EMBEDDING_MODEL", "text-embedding-3-large"
    )
    print(
        json.dumps(
            {
                "selected_cleaning_model": selected_cleaning_model,
                "selected_embedding_model": selected_embedding_model,
            }
        ),
        flush=True,
    )

    init_db()
    result = reprocess_batch(
        Path(args.input),
        limit=args.limit,
        skip_index=bool(args.skip_index),
        needs_work_only=bool(args.needs_work_only),
        index_only=bool(args.index_only),
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
