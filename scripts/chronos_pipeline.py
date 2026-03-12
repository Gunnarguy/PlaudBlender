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
from src.chronos.pipeline_progress import progress as pipeline_progress

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
    pipeline_progress.start_phase("ingest")
    pipeline_progress.update(step="Fetching recording list from Plaud…")

    service = ChronosIngestService(db_session=session)

    # Add progress callback to the service
    def on_progress(current, total, recording_id):
        print_progress("Plaud", current, total, recording_id, time.time() - start_time)

    try:
        success_count, failure_count = service.ingest_recent_recordings(
            limit=limit, fetch_all_pages=fetch_all_pages
        )
    except Exception as e:
        # Auth failure or network error — log warning but don't crash
        # so process/index phases can still run on existing data.
        print(f"⚠️  Ingest failed: {e}")
        logger.warning(f"Ingest phase failed (non-fatal): {e}")
        success_count, failure_count = 0, 0

    elapsed = time.time() - start_time
    print_phase_complete("Ingest", success_count, elapsed)
    pipeline_progress.finish_phase(
        summary=f"{success_count} ingested, {failure_count} failed"
    )

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
    pipeline_progress.start_phase("process")

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
        pipeline_progress.update(total=total, step=f"Found {total} pending recordings")

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
            pipeline_progress.update(
                step=f"Recording {i+1}/{total}: fetching transcript…",
                item=rec_id[:20],
            )

            try:
                # This is the slow part - show we're calling Gemini
                print(f"   🧠 Sending to Gemini AI...", flush=True)
                pipeline_progress.update(step=f"Recording {i+1}/{total}: Gemini AI…")
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
                    pipeline_progress.advance(
                        item=rec_id[:20], step=f"✅ {success_count} done"
                    )
                else:
                    failure_count += 1
                    print(f"   ❌ Failed after {proc_time:.1f}s", flush=True)
                    pipeline_progress.advance(item=rec_id[:20], step=f"❌ failed")
                    session.rollback()  # Clear any failed transaction state
            except Exception as e:
                logger.error(f"Error processing {rec_id}: {e}")
                failure_count += 1
                print(f"   ❌ Error: {str(e)[:60]}", flush=True)
                pipeline_progress.advance(item=rec_id[:20])
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
    pipeline_progress.finish_phase(
        summary=f"{success_count} processed, {failure_count} failed"
    )

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
    pipeline_progress.start_phase("index")

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
        qdrant.create_collection(force_recreate=False)
    except Exception as e:
        logger.warning(f"Collection may already exist: {e}")

    # Fetch events that need indexing (those without qdrant_point_id)
    q = session.query(ChronosEventDB).filter(ChronosEventDB.qdrant_point_id.is_(None))
    if recording_id:
        q = q.filter(ChronosEventDB.recording_id == recording_id)

    events_to_index = q.limit(limit * 10).all()  # multiple events per recording

    if not events_to_index:
        print("✓ No events to index")
        pipeline_progress.finish_phase(summary="No events to index")
        return 0

    total = len(events_to_index)
    print(f"Found {total} events to index")
    pipeline_progress.update(total=total, step=f"Validating {total} events…")

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
    #
    # When using gemini-embedding-2-preview with audio files available,
    # we create fused text+audio embeddings per-event. Otherwise we
    # fall back to the fast text-only batch path.
    print(f"\n\n🔮 Generating embeddings for {len(pydantic_events)} events...")
    pipeline_progress.update(
        step=f"Generating embeddings for {len(pydantic_events)} events…"
    )
    embed_start = time.time()

    use_multimodal = embedder.supports_multimodal
    if use_multimodal:
        # Build a map of recording_id → local_audio_path for audio lookup
        from src.database.models import (
            ChronosRecording as ChronosRecordingDB,
        )

        rec_ids = {e.recording_id for e in pydantic_events}
        audio_map: dict[str, str] = {}
        for rec in (
            session.query(ChronosRecordingDB)
            .filter(ChronosRecordingDB.recording_id.in_(rec_ids))
            .all()
        ):
            if rec.local_audio_path:
                audio_map[rec.recording_id] = rec.local_audio_path

        audio_count = sum(1 for e in pydantic_events if e.recording_id in audio_map)
        print(
            f"  Multimodal mode: {audio_count}/{len(pydantic_events)} events "
            f"have audio → fused text+audio embeddings"
        )

        embeddings = []
        for i, event in enumerate(pydantic_events):
            if (i + 1) % 5 == 0 or i == 0:
                print_progress(
                    "Embed",
                    i,
                    len(pydantic_events),
                    "multimodal" if event.recording_id in audio_map else "text",
                    time.time() - embed_start,
                )
            audio_path = audio_map.get(event.recording_id, "")
            vec = embedder.embed_text_with_audio(
                text=event.clean_text,
                audio_path=audio_path,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.append(vec)
    else:
        # Text-only batch path (fast, works with embedding-001 too)
        texts = [event.clean_text for event in pydantic_events]
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
            batch_embeddings = embedder.embed_batch(
                batch, task_type="RETRIEVAL_DOCUMENT"
            )
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
    pipeline_progress.finish_phase(summary=f"{indexed_count} events indexed")
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

    pipeline_progress.start_phase("graph", total_items=0)

    if not events_to_process:
        logger.info("No events to process for graph extraction")
        pipeline_progress.finish_phase(summary="No events to process")
        return 0

    pipeline_progress.update(total=len(events_to_process), step="Converting events")
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
    pipeline_progress.update(
        step="Extracting entities", item=f"{len(pydantic_events)} events"
    )
    entities, graph = graph_extractor.extract_from_events(pydantic_events)

    # Detect communities
    pipeline_progress.update(step="Detecting communities")
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
    pipeline_progress.finish_phase(
        summary=f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    return len(pydantic_events)


def run_refresh_workflows(session, *, days_back: int = 30, limit: int = 10) -> int:
    """Refresh pending Plaud workflow statuses and persist artifacts.

    Returns:
        int: Number of workflows checked
    """
    print_phase_header("PLAUD WORKFLOW REFRESH", "🔄")
    print(f"Checking workflow statuses (last {days_back} days, limit {limit})...")
    pipeline_progress.start_phase("refresh-workflows")
    pipeline_progress.update(step="Checking workflow statuses")

    start_time = time.time()

    from app_v2.services.data_service import ChronosDataService

    service = ChronosDataService()
    result = service.refresh_plaud_workflow_statuses(days_back=days_back, limit=limit)

    completed = len(result.get("completed", []))
    pending = len(result.get("pending", []))
    failed = len(result.get("failed", []))
    total = completed + pending + failed

    elapsed = time.time() - start_time
    print(f"  ✅ Completed: {completed} | ⏳ Pending: {pending} | ❌ Failed: {failed}")
    print(f"  ⏱️  {elapsed:.1f}s")
    pipeline_progress.finish_phase(
        summary=f"{completed} completed, {pending} pending, {failed} failed"
    )

    return total


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
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Recreate Qdrant collection and re-embed ALL events. "
        "Required when switching embedding models (e.g. embedding-001 → embedding-2-preview).",
    )
    parser.add_argument(
        "--refresh-workflows",
        action="store_true",
        help="Check pending Plaud cloud workflow statuses and persist completed results.",
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
            args.reindex,
            args.refresh_workflows,
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

    # Determine which phases will run
    phases = []
    if args.reindex:
        phases.append("index")
    if args.full or args.ingest:
        phases.append("ingest")
    if args.full or args.process:
        phases.append("process")
    if args.full or args.index:
        phases.append("index")
    if args.full or args.graph:
        phases.append("graph")
    if args.full or args.refresh_workflows:
        phases.append("refresh-workflows")
    # Deduplicate while preserving order
    phases = list(dict.fromkeys(phases))
    pipeline_progress.start_run(phases=phases, trigger="cli")

    # Initialize database
    init_db()
    session = SessionLocal()

    try:
        if args.reindex:
            from src.database.models import ChronosEvent as ChronosEventDB

            print_phase_header("REINDEX: Rebuild Qdrant with new embedding model", "🔄")
            print(
                "  Clearing all qdrant_point_id values so every event gets re-embedded."
            )
            count = (
                session.query(ChronosEventDB)
                .filter(ChronosEventDB.qdrant_point_id.isnot(None))
                .update({ChronosEventDB.qdrant_point_id: None})
            )
            session.commit()
            print(f"  Cleared {count} events — they will be re-indexed.")

            from src.chronos.qdrant_client import ChronosQdrantClient

            qdrant = ChronosQdrantClient()
            qdrant.create_collection(force_recreate=True)
            print("  Qdrant collection recreated. Running index phase now...\n")
            run_index(session, limit=999_999)

        if args.full or args.ingest:
            # --full always fetches all pages; --ingest alone respects --fetch-all flag
            fetch_all = True if args.full else bool(args.fetch_all)
            run_ingest(session, limit=args.limit, fetch_all_pages=fetch_all)

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

        if args.full or args.refresh_workflows:
            run_refresh_workflows(session, days_back=30, limit=args.limit)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        pipeline_progress.finish_run()

    except Exception as e:
        pipeline_progress.finish_run(error=str(e))
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
