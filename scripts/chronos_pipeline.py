#!/usr/bin/env python3
"""Chronos Pipeline Runner

End-to-end pipeline for ingesting, processing, and indexing Plaud recordings.

Usage:
    python scripts/chronos_pipeline.py --ingest    # Download from Plaud
    python scripts/chronos_pipeline.py --process   # Process pending recordings
    python scripts/chronos_pipeline.py --full      # Run full pipeline
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SessionLocal, init_db
from src.database.chronos_repository import (
    get_pending_chronos_recordings,
    mark_chronos_recording_status,
    add_chronos_events,
)
from src.chronos.ingest_service import ChronosIngestService
from src.chronos.engine import ChronosEngine, validate_event_quality
from src.models.chronos_schemas import ChronosEvent as ChronosEventSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingest(session, limit: int = 100) -> int:
    """Run ingestion phase: download recordings from Plaud.

    Returns:
        int: Number of recordings ingested
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: INGEST")
    logger.info("=" * 60)

    service = ChronosIngestService(db_session=session)
    success_count, failure_count = service.ingest_recent_recordings(limit=limit)

    logger.info(
        f"Ingestion complete: {success_count} success, {failure_count} failures"
    )
    return success_count


def run_process(session, limit: int = 10) -> int:
    """Run processing phase: Gemini cleaning + event extraction.

    Returns:
        int: Number of recordings processed
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: PROCESS")
    logger.info("=" * 60)

    # Fetch pending recordings
    pending = get_pending_chronos_recordings(session, limit=limit)
    logger.info(f"Found {len(pending)} pending recordings")

    if not pending:
        logger.info("No pending recordings to process")
        return 0

    # Initialize Gemini engine
    engine = ChronosEngine()

    processed_count = 0

    for recording in pending:
        logger.info(f"Processing recording: {recording.recording_id}")
        logger.info(f"  Duration: {recording.duration_seconds}s")
        logger.info(f"  Audio path: {recording.local_audio_path}")

        # Mark as processing
        mark_chronos_recording_status(
            session=session,
            recording_id=recording.recording_id,
            status="processing",
        )

        try:
            # Process audio through Gemini
            events = engine.process_audio_to_events(
                audio_path=recording.local_audio_path,
                recording_id=recording.recording_id,
            )

            if not events:
                logger.error(f"Failed to extract events from {recording.recording_id}")
                mark_chronos_recording_status(
                    session=session,
                    recording_id=recording.recording_id,
                    status="failed",
                    error_message="Gemini processing returned no events",
                )
                continue

            # Validate quality
            if not validate_event_quality(events, recording.duration_seconds):
                logger.warning(
                    f"Event quality check failed for {recording.recording_id}"
                )
                # Still save them, but log the warning

            # Convert Pydantic to SQLAlchemy models
            from src.database.models import ChronosEvent as ChronosEventDB

            db_events = []
            for event in events:
                db_event = ChronosEventDB(
                    event_id=event.event_id,
                    recording_id=event.recording_id,
                    start_ts=event.start_ts,
                    end_ts=event.end_ts,
                    day_of_week=event.day_of_week.value,
                    hour_of_day=event.hour_of_day,
                    clean_text=event.clean_text,
                    category=event.category.value,
                    sentiment=event.sentiment,
                    keywords=event.keywords,
                    speaker=event.speaker.value,
                    raw_transcript_snippet=event.raw_transcript_snippet,
                    gemini_reasoning=event.gemini_reasoning,
                )
                db_events.append(db_event)

            # Save to database
            add_chronos_events(session, db_events)
            logger.info(f"Saved {len(db_events)} events to database")

            # Mark as completed
            mark_chronos_recording_status(
                session=session,
                recording_id=recording.recording_id,
                status="completed",
            )

            processed_count += 1
            logger.info(f"Successfully processed {recording.recording_id}")

        except Exception as e:
            logger.error(
                f"Error processing {recording.recording_id}: {e}", exc_info=True
            )
            mark_chronos_recording_status(
                session=session,
                recording_id=recording.recording_id,
                status="failed",
                error_message=str(e),
            )

    logger.info(f"Processing complete: {processed_count}/{len(pending)} recordings")
    return processed_count


def run_index(session, limit: int = 10) -> int:
    """Run indexing phase: push events to Qdrant.

    Returns:
        int: Number of events indexed
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: INDEX (Qdrant)")
    logger.info("=" * 60)

    from src.database.models import ChronosEvent as ChronosEventDB
    from src.chronos.qdrant_client import ChronosQdrantClient
    from src.chronos.embedding_service import ChronosEmbeddingService

    # Initialize clients
    qdrant = ChronosQdrantClient()
    embedder = ChronosEmbeddingService()

    # Ensure collection exists
    try:
        qdrant.create_collection(vector_size=768, force_recreate=False)
    except Exception as e:
        logger.warning(f"Collection may already exist: {e}")

    # Fetch events that need indexing (those without qdrant_point_id)
    events_to_index = (
        session.query(ChronosEventDB)
        .filter(ChronosEventDB.qdrant_point_id.is_(None))
        .limit(limit * 10)  # Get more events (multiple per recording)
        .all()
    )

    if not events_to_index:
        logger.info("No events to index")
        return 0

    logger.info(f"Found {len(events_to_index)} events to index")

    # Convert to Pydantic for validation
    from src.models.chronos_schemas import (
        ChronosEvent as ChronosEventSchema,
        DayOfWeek,
        EventCategory,
        SpeakerMode,
    )

    pydantic_events = []
    for db_event in events_to_index:
        try:
            pydantic_event = ChronosEventSchema(
                event_id=db_event.event_id,
                recording_id=db_event.recording_id,
                start_ts=db_event.start_ts,
                end_ts=db_event.end_ts,
                day_of_week=DayOfWeek(db_event.day_of_week),
                hour_of_day=db_event.hour_of_day,
                clean_text=db_event.clean_text,
                category=EventCategory(db_event.category),
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
            pydantic_events.append(pydantic_event)
        except Exception as e:
            logger.error(f"Failed to convert event {db_event.event_id}: {e}")
            continue

    if not pydantic_events:
        logger.error("No valid events to index")
        return 0

    # Generate embeddings
    logger.info("Generating embeddings...")
    texts = [event.clean_text for event in pydantic_events]
    embeddings = embedder.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    # Upsert to Qdrant
    logger.info("Upserting to Qdrant...")
    indexed_count = qdrant.upsert_events_batch(pydantic_events, embeddings)

    # Update database with qdrant_point_id
    for event in pydantic_events:
        db_event = (
            session.query(ChronosEventDB).filter_by(event_id=event.event_id).first()
        )
        if db_event:
            db_event.qdrant_point_id = event.event_id

    session.commit()

    logger.info(f"Successfully indexed {indexed_count} events")
    return indexed_count


def run_graph(session, limit: int = 10) -> int:
    """Run graph extraction phase: build knowledge graph from events.

    Returns:
        int: Number of events processed for graph
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: GRAPH EXTRACTION")
    logger.info("=" * 60)

    from src.database.models import ChronosEvent as ChronosEventDB
    from src.chronos.graph_service import ChronosGraphExtractor
    from src.models.chronos_schemas import (
        ChronosEvent as ChronosEventSchema,
        DayOfWeek,
        EventCategory,
        SpeakerMode,
    )
    import pickle
    from pathlib import Path
    from src.config import get_settings

    settings = get_settings()
    graph_extractor = ChronosGraphExtractor()

    # Fetch events that have been indexed
    events_to_process = (
        session.query(ChronosEventDB)
        .filter(ChronosEventDB.qdrant_point_id.isnot(None))
        .limit(limit * 10)
        .all()
    )

    if not events_to_process:
        logger.info("No events to process for graph extraction")
        return 0

    logger.info(f"Processing {len(events_to_process)} events for graph extraction")

    # Convert to Pydantic
    pydantic_events = []
    for db_event in events_to_process:
        try:
            pydantic_event = ChronosEventSchema(
                event_id=db_event.event_id,
                recording_id=db_event.recording_id,
                start_ts=db_event.start_ts,
                end_ts=db_event.end_ts,
                day_of_week=DayOfWeek(db_event.day_of_week),
                hour_of_day=db_event.hour_of_day,
                clean_text=db_event.clean_text,
                category=EventCategory(db_event.category),
                sentiment=db_event.sentiment,
                keywords=db_event.keywords or [],
                speaker=(
                    SpeakerMode(db_event.speaker)
                    if db_event.speaker
                    else SpeakerMode.SELF_TALK
                ),
            )
            pydantic_events.append(pydantic_event)
        except Exception as e:
            logger.error(f"Failed to convert event {db_event.event_id}: {e}")
            continue

    # Extract entities and build graph
    entities, graph = graph_extractor.extract_from_events(pydantic_events)

    # Detect communities
    communities = graph_extractor.detect_communities(graph)

    # Save graph to cache
    graph_cache_dir = Path(settings.chronos_graph_cache_dir)
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_cache_dir / "knowledge_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(
            {
                "graph": graph,
                "entities": entities,
                "communities": communities,
            },
            f,
        )

    logger.info(f"Saved graph to {graph_path}")
    logger.info(
        f"Graph stats: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    return len(pydantic_events)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chronos Pipeline Runner")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
    parser.add_argument("--process", action="store_true", help="Run processing phase")
    parser.add_argument("--index", action="store_true", help="Run indexing phase")
    parser.add_argument(
        "--graph", action="store_true", help="Run graph extraction phase"
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--limit", type=int, default=10, help="Max items per phase")

    args = parser.parse_args()

    # If no specific phase, show help
    if not any([args.ingest, args.process, args.index, args.graph, args.full]):
        parser.print_help()
        return

    # Initialize database
    init_db()
    session = SessionLocal()

    try:
        if args.full or args.ingest:
            run_ingest(session, limit=args.limit)

        if args.full or args.process:
            run_process(session, limit=args.limit)

        if args.full or args.index:
            run_index(session, limit=args.limit)

        if args.full or args.graph:
            run_graph(session, limit=args.limit)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()
