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
import time
import threading
from pathlib import Path

# Increase Python's integer string conversion limit (default 4300 digits).
# This prevents "Exceeds the limit (4300 digits)" errors when parsing JSON
# responses from Gemini that contain very large numbers (e.g., token counts).
# See: https://docs.python.org/3/library/sys.html#sys.set_int_max_str_digits
if sys.version_info >= (3, 11):
    sys.set_int_max_str_digits(
        0
    )  # 0 = no limit (use with caution, but safe for local processing)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import SessionLocal, init_db
from src.database.chronos_repository import (
    get_pending_chronos_recordings,
    mark_chronos_recording_status,
    add_chronos_events,
    delete_chronos_events_by_recording,
)
from src.chronos.ingest_service import ChronosIngestService
from src.chronos.transcript_processor import TranscriptProcessor
from src.chronos.engine import ChronosEngine, validate_event_quality
from src.models.chronos_schemas import ChronosEvent as ChronosEventSchema
from src.config import get_settings
from src.chronos.genai_helpers import (
    get_genai_client,
    list_model_names,
    pick_first_available,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Progress Display Helpers
# ═══════════════════════════════════════════════════════════════════


def progress_bar(current: int, total: int, width: int = 40, prefix: str = "") -> str:
    """Generate ASCII progress bar string."""
    if total == 0:
        pct = 100
    else:
        pct = int(100 * current / total)
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}[{bar}] {current}/{total} ({pct}%)"


def print_progress(
    phase: str, current: int, total: int, item: str = "", elapsed: float = 0
):
    """Print progress update with phase info (newline for subprocess compatibility)."""
    bar = progress_bar(current, total, width=30)
    elapsed_str = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
    item_str = (
        f" → {item[:40]}..."
        if item and len(item) > 40
        else f" → {item}" if item else ""
    )
    # Use newline (not \r) so subprocess output streams properly to Streamlit
    print(f"⏳ {phase}: {bar}{item_str}{elapsed_str}", flush=True)


def print_phase_header(phase: str, icon: str = "▶"):
    """Print a visible phase header."""
    print(f"\n{'═' * 60}")
    print(f"{icon} {phase}")
    print(f"{'═' * 60}")


def print_phase_complete(phase: str, count: int, elapsed: float):
    """Print phase completion summary."""
    print(f"\n✅ {phase} complete: {count} items in {elapsed:.1f}s")


def run_preflight(*, smoke_call: bool = False) -> int:
    """Validate Gemini configuration and show available models.

    This is the fastest way to debug "model not found / not supported" issues.
    It lists models accessible to your API key and checks whether the configured
    Chronos models are present.

    Args:
        smoke_call: If True, performs a tiny embed call to verify connectivity.

    Returns:
        int: 0 if ok, non-zero if configuration is missing or models unavailable.
    """
    logger.info("=" * 60)
    logger.info("PREFLIGHT: GEMINI MODELS")
    logger.info("=" * 60)

    settings = get_settings()
    if not settings.gemini_api_key:
        logger.error("GEMINI_API_KEY is not set. Update your .env and retry.")
        return 2

    logger.info(
        f"Gemini API version: {getattr(settings, 'gemini_api_version', 'v1beta')}"
    )
    logger.info("Listing models available to this API key...")
    try:
        names = list_model_names()
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return 3

    if not names:
        logger.error("No models were returned by models.list().")
        return 4

    # Show a short, helpful subset (full list is often huge).
    preview = [
        n
        for n in names
        if any(
            k in n
            for k in (
                "gemini-3-",
                "gemini-2.5-",
                "gemini-embedding-",
                "text-embedding",
            )
        )
    ]

    logger.info(f"Found {len(names)} models. Relevant subset ({len(preview)}):")
    for n in preview[:40]:
        logger.info(f"  - {n}")
    if len(preview) > 40:
        logger.info(f"  ... (+{len(preview) - 40} more)")

    configured_clean = (settings.chronos_cleaning_model or "").strip()
    configured_analyst = (settings.chronos_analyst_model or "").strip()
    configured_embed = (settings.chronos_embedding_model or "").strip()

    chosen_clean = pick_first_available(
        configured_clean,
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash",
    )
    chosen_analyst = pick_first_available(
        configured_analyst,
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    )
    chosen_embed = pick_first_available(
        configured_embed,
        "gemini-embedding-001",
    )

    def _present(label: str, configured: str, chosen: str | None) -> None:
        ok = (configured in names) if configured else True
        chosen_ok = (chosen in names) if chosen else False
        logger.info(
            f"{label}: configured={configured or '(default)'} (present={ok}), chosen={chosen or '(none)'} (present={chosen_ok})"
        )

    _present("Chronos cleaning model", configured_clean, chosen_clean)
    _present("Chronos analyst model", configured_analyst, chosen_analyst)
    _present("Chronos embedding model", configured_embed, chosen_embed)

    if smoke_call:
        # A tiny call that should be cheap and fast.
        from google.genai import types

        client = get_genai_client()
        model = chosen_embed or configured_embed or "gemini-embedding-001"
        logger.info(f"Running embed smoke call with model={model!r}...")
        try:
            client.models.embed_content(
                model=model,
                contents="ping",
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=8
                ),
            )
            logger.info("Embed smoke call succeeded.")
        except Exception as e:
            logger.error(f"Embed smoke call failed: {e}")
            return 5

    logger.info("Preflight OK")
    return 0


def run_ingest(session, limit: int = 100, *, fetch_all_pages: bool = False) -> int:
    """Run ingestion phase: download recordings from Plaud.

    Returns:
        int: Number of recordings ingested
    """
    print_phase_header("PHASE 1: INGEST (Plaud API)", "📥")
    print("Fetching recordings from Plaud...")

    start_time = time.time()

    service = ChronosIngestService(db_session=session)

    # Add progress callback to the service
    def on_progress(current, total, recording_id):
        print_progress("Plaud", current, total, recording_id, time.time() - start_time)

    success_count, failure_count = service.ingest_recent_recordings(
        limit=limit, fetch_all_pages=fetch_all_pages
    )

    elapsed = time.time() - start_time
    print_phase_complete("Ingest", success_count, elapsed)

    if failure_count > 0:
        print(f"⚠️  {failure_count} recordings failed to ingest")

    return success_count


def run_process(
    session,
    limit: int = 10,
    *,
    recording_id: str | None = None,
    force: bool = False,
) -> int:
    """Process pending recordings through Gemini using transcripts."""
    print_phase_header("PHASE 2: PROCESS (Gemini AI)", "🧠")

    start_time = time.time()
    processor = TranscriptProcessor(db_session=session)

    if recording_id:
        print(f"Processing single recording: {recording_id[:20]}...")
        print_progress("Gemini", 0, 1, recording_id[:30])
        ok = processor.process_recording_id(
            recording_id,
            delete_existing_events=bool(force),
        )
        print_progress("Gemini", 1, 1, recording_id[:30], time.time() - start_time)
        success_count, failure_count = (1, 0) if ok else (0, 1)
    else:
        # Get pending recordings
        pending = get_pending_chronos_recordings(session, limit=limit)
        total = len(pending)

        print(f"Found {total} pending recordings")

        if not pending:
            print("✓ No pending recordings to process")
            return 0

        # Process with progress tracking
        success_count = 0
        failure_count = 0

        for i, rec in enumerate(pending):
            rec_id = rec.recording_id
            duration_mins = (rec.duration_seconds or 0) // 60

            # Show which recording we're starting
            print(
                f"\n📄 Recording {i+1}/{total}: {rec_id[:20]}... ({duration_mins}m)",
                flush=True,
            )
            print(f"   🔄 Fetching transcript from Plaud...", flush=True)

            try:
                # This is the slow part - show we're calling Gemini
                print(f"   🧠 Sending to Gemini AI...", flush=True)
                proc_start = time.time()

                # Heartbeat thread to show we're still alive
                stop_heartbeat = threading.Event()

                def heartbeat():
                    dots = 0
                    while not stop_heartbeat.is_set():
                        elapsed = time.time() - proc_start
                        dots = (dots % 3) + 1
                        print(
                            f"   ⏳ Still processing{'.' * dots} ({elapsed:.0f}s elapsed)",
                            flush=True,
                        )
                        stop_heartbeat.wait(5)  # Print every 5 seconds

                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()

                try:
                    ok = processor.process_recording_id(rec_id)
                finally:
                    stop_heartbeat.set()
                    heartbeat_thread.join(timeout=1)

                proc_time = time.time() - proc_start
                if ok:
                    success_count += 1
                    print(
                        f"   ✅ Done! Extracted events in {proc_time:.1f}s", flush=True
                    )
                else:
                    failure_count += 1
                    print(f"   ❌ Failed after {proc_time:.1f}s", flush=True)
                    session.rollback()  # Clear any failed transaction state
            except Exception as e:
                logger.error(f"Error processing {rec_id}: {e}")
                failure_count += 1
                print(f"   ❌ Error: {str(e)[:60]}", flush=True)
                session.rollback()  # Clear any failed transaction state

            # Overall progress bar
            print_progress(
                "Gemini",
                i + 1,
                total,
                f"{success_count} ok, {failure_count} failed",
                time.time() - start_time,
            )

    elapsed = time.time() - start_time
    print_phase_complete("Process", success_count, elapsed)

    if failure_count > 0:
        print(f"⚠️  {failure_count} recordings failed")

    return success_count


def run_index(
    session,
    limit: int = 10,
    *,
    recording_id: str | None = None,
) -> int:
    """Run indexing phase: push events to Qdrant.

    Returns:
        int: Number of events indexed
    """
    print_phase_header("PHASE 3: INDEX (Qdrant)", "📤")

    start_time = time.time()

    from src.database.models import ChronosEvent as ChronosEventDB
    from src.chronos.qdrant_client import ChronosQdrantClient
    from src.chronos.embedding_service import ChronosEmbeddingService

    # Initialize clients
    print("Connecting to Qdrant...")
    qdrant = ChronosQdrantClient()
    embedder = ChronosEmbeddingService()

    # Ensure collection exists
    print("Ensuring collection exists...")
    try:
        qdrant.create_collection(vector_size=768, force_recreate=False)
    except Exception as e:
        logger.warning(f"Collection may already exist: {e}")

    # Fetch events that need indexing (those without qdrant_point_id)
    q = session.query(ChronosEventDB).filter(ChronosEventDB.qdrant_point_id.is_(None))
    if recording_id:
        q = q.filter(ChronosEventDB.recording_id == recording_id)

    events_to_index = q.limit(limit * 10).all()  # multiple events per recording

    if not events_to_index:
        print("✓ No events to index")
        return 0

    total = len(events_to_index)
    print(f"Found {total} events to index")

    # Convert to Pydantic for validation
    from src.models.chronos_schemas import (
        ChronosEvent as ChronosEventSchema,
        DayOfWeek,
        EventCategory,
        SpeakerMode,
    )

    print_progress("Validate", 0, total, "converting events...")
    pydantic_events = []
    for i, db_event in enumerate(events_to_index):
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
            if (i + 1) % 10 == 0:
                print_progress("Validate", i + 1, total, f"{i + 1} events validated")
        except Exception as e:
            logger.error(f"Failed to convert event {db_event.event_id}: {e}")
            continue

    print_progress(
        "Validate", total, total, "validation complete", time.time() - start_time
    )

    if not pydantic_events:
        print("\n❌ No valid events to index")
        return 0

    # Generate embeddings with progress
    print(f"\n\n🔮 Generating embeddings for {len(pydantic_events)} events...")
    embed_start = time.time()
    texts = [event.clean_text for event in pydantic_events]

    # Batch embedding with progress
    batch_size = 20
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print_progress(
            "Embed",
            i,
            len(texts),
            f"batch {i//batch_size + 1}",
            time.time() - embed_start,
        )
        batch_embeddings = embedder.embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
        embeddings.extend(batch_embeddings)

    print_progress(
        "Embed", len(texts), len(texts), "complete", time.time() - embed_start
    )

    # Upsert to Qdrant
    print(f"\n\n📤 Upserting {len(pydantic_events)} events to Qdrant...")
    upsert_start = time.time()
    indexed_count = qdrant.upsert_events_batch(pydantic_events, embeddings)
    print_progress(
        "Qdrant",
        indexed_count,
        indexed_count,
        "upsert complete",
        time.time() - upsert_start,
    )

    # Update database with qdrant_point_id
    print("\n\n📝 Updating database references...")
    for i, event in enumerate(pydantic_events):
        db_event = (
            session.query(ChronosEventDB).filter_by(event_id=event.event_id).first()
        )
        if db_event:
            db_event.qdrant_point_id = event.event_id

    session.commit()

    elapsed = time.time() - start_time
    print_phase_complete("Index", indexed_count, elapsed)
    return indexed_count


def run_graph(
    session,
    limit: int = 10,
    *,
    recording_id: str | None = None,
) -> int:
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
    q = session.query(ChronosEventDB).filter(ChronosEventDB.qdrant_point_id.isnot(None))
    if recording_id:
        q = q.filter(ChronosEventDB.recording_id == recording_id)

    events_to_process = q.limit(limit * 10).all()

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
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="List available Gemini models and validate configured model IDs",
    )
    parser.add_argument(
        "--preflight-smoke",
        action="store_true",
        help="Run preflight plus a tiny embed call (verifies connectivity)",
    )
    parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
    parser.add_argument("--process", action="store_true", help="Run processing phase")
    parser.add_argument("--index", action="store_true", help="Run indexing phase")
    parser.add_argument(
        "--graph", action="store_true", help="Run graph extraction phase"
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--limit", type=int, default=10, help="Max items per phase")
    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
        help="Operate on a single recording_id (applies to --process/--index/--graph)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For --process with --recording-id: delete existing DB events first",
    )
    parser.add_argument(
        "--fetch-all",
        action="store_true",
        help="For --ingest: paginate through ALL recordings in Plaud account (not just most recent 100)",
    )

    args = parser.parse_args()

    # If no specific phase, show help
    if not any(
        [
            args.preflight,
            args.preflight_smoke,
            args.ingest,
            args.process,
            args.index,
            args.graph,
            args.full,
        ]
    ):
        parser.print_help()
        return

    if args.preflight or args.preflight_smoke:
        code = run_preflight(smoke_call=bool(args.preflight_smoke))
        # If the user asked ONLY for preflight, exit early.
        if not any([args.ingest, args.process, args.index, args.graph, args.full]):
            raise SystemExit(code)
        # If preflight failed but the user asked for other phases, stop early.
        if code != 0:
            raise SystemExit(code)

    # Initialize database
    init_db()
    session = SessionLocal()

    try:
        if args.full or args.ingest:
            run_ingest(session, limit=args.limit, fetch_all_pages=bool(args.fetch_all))

        if args.full or args.process:
            run_process(
                session,
                limit=args.limit,
                recording_id=args.recording_id,
                force=bool(args.force),
            )

        if args.full or args.index:
            run_index(session, limit=args.limit, recording_id=args.recording_id)

        if args.full or args.graph:
            run_graph(session, limit=args.limit, recording_id=args.recording_id)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

    finally:
        session.close()


if __name__ == "__main__":
    main()
