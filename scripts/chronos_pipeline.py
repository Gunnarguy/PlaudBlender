#!/usr/bin/env python3
"""Chronos Pipeline Runner

End-to-end pipeline for ingesting, processing, and indexing Plaud recordings.

Usage:
    python scripts/chronos_pipeline.py --ingest    # Download from Plaud
    python scripts/chronos_pipeline.py --process   # Process pending recordings
    python scripts/chronos_pipeline.py --backfill  # Run full-history pipeline
    python scripts/chronos_pipeline.py --full      # Run full pipeline
"""
import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional, cast

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
    set_chronos_recording_transcript,
)
from src.chronos.ingest_service import ChronosIngestService
from src.chronos.transcript_processor import TranscriptProcessor
from src.chronos.engine import ChronosEngine, validate_event_quality
from src.models.chronos_schemas import ChronosEvent as ChronosEventSchema
from src.config import get_settings
from src.chronos.pipeline_progress import progress as pipeline_progress
from src.chronos.trace_service import finish_trace_run, start_trace_run, trace_span

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseResult:
    """Result for a pipeline phase with optional hard-failure context."""

    processed_count: int
    failure_count: int = 0
    error_message: Optional[str] = None
    warnings: tuple[str, ...] = ()


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


def print_phase_failed(phase: str, error: str, elapsed: float):
    """Print phase failure summary."""
    print(f"\n❌ {phase} failed after {elapsed:.1f}s")
    print(f"   {error}")


def _meminfo_mb() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if parts:
                    values[key] = int(parts[0]) // 1024
    except Exception:
        return {}
    return values


def _pipeline_already_running() -> bool:
    current_pid = os.getpid()
    parent_pid = os.getppid()
    proc_root = Path("/proc")
    try:
        proc_dirs = list(proc_root.iterdir())
    except Exception:
        return False

    for proc_dir in proc_dirs:
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == current_pid or pid == parent_pid:
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        except Exception:
            continue
        if b"scripts/chronos_pipeline.py" in cmdline:
            return True
    return False


def _host_pressure_reason() -> Optional[str]:
    if os.environ.get("CHRONOS_PIPELINE_IGNORE_PRESSURE", "0") == "1":
        return None

    settings = get_settings()
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        load_1m = 0.0
    if load_1m > settings.chronos_autosync_max_load_avg:
        return f"load {load_1m:.2f} > {settings.chronos_autosync_max_load_avg:.2f}"

    meminfo = _meminfo_mb()
    available_mb = meminfo.get("MemAvailable", 0)
    if available_mb and available_mb < settings.chronos_autosync_min_available_mb:
        return (
            f"available RAM {available_mb}MB < "
            f"{settings.chronos_autosync_min_available_mb}MB"
        )

    swap_total_mb = meminfo.get("SwapTotal", 0)
    swap_free_mb = meminfo.get("SwapFree", 0)
    swap_used_mb = max(0, swap_total_mb - swap_free_mb)
    recovery_headroom_mb = max(256, settings.chronos_autosync_min_available_mb // 3)
    if swap_used_mb > settings.chronos_autosync_max_swap_used_mb and (
        not available_mb
        or available_mb
        <= settings.chronos_autosync_min_available_mb + recovery_headroom_mb
    ):
        return (
            f"swap used {swap_used_mb}MB > {settings.chronos_autosync_max_swap_used_mb}MB"
            + (
                f" with only {available_mb}MB RAM available"
                if available_mb
                else ""
            )
        )

    if _pipeline_already_running():
        return "another Chronos pipeline is already running"

    return None


def _requested_heavy_work(args: argparse.Namespace) -> bool:
    return bool(
        args.full
        or args.backfill
        or args.process
        or args.index
        or args.graph
        or args.reindex
        or args.refresh_workflows
        or args.repair_recent
    )


def run_preflight(*, smoke_call: bool = False) -> int:
    """Validate configured provider routing and optional embedding connectivity.

    Args:
        smoke_call: If True, performs a tiny embed call to verify connectivity.

    Returns:
        int: 0 if ok, non-zero if configuration is missing.
    """
    logger.info("=" * 60)
    logger.info("PREFLIGHT: CHRONOS PROVIDERS")
    logger.info("=" * 60)

    settings = get_settings()
    configured_clean = (settings.chronos_cleaning_model or "").strip()
    configured_analyst = (settings.chronos_analyst_model or "").strip()
    configured_embed = (settings.chronos_embedding_model or "").strip()
    provider = (settings.chronos_processing_provider or "gemini").strip()

    logger.info(f"Processing provider: {provider}")
    logger.info(f"Cleaning model:  {configured_clean}")
    logger.info(f"Analyst model:   {configured_analyst}")
    logger.info(f"Embedding model: {configured_embed}")
    logger.info(f"Embedding dim:   {settings.chronos_embedding_dim}")
    logger.info(
        "OpenAI enabled:  %s",
        "yes" if getattr(settings, "chronos_openai_enabled", False) else "no",
    )

    if provider == "gemini" and not settings.gemini_api_key:
        logger.warning("CHRONOS_GEMINI_API_KEY is not set; Gemini processing will be unavailable")
    if provider == "local" and not getattr(settings, "chronos_local_llm_enabled", False):
        logger.error("CHRONOS_PROCESSING_PROVIDER=local requires CHRONOS_LOCAL_LLM_ENABLED=1")
        return 2
    if provider == "openai" and not settings.openai_api_key:
        logger.error("OpenAI provider requested but disabled or missing key; set CHRONOS_OPENAI_ENABLED=1 to opt in")
        return 2

    if smoke_call:
        from src.chronos.embedding_service import ChronosEmbeddingService

        logger.info(f"Running embed smoke call with model={configured_embed!r}...")
        try:
            vector = ChronosEmbeddingService().embed_text("ping")
            logger.info(f"Embed smoke call succeeded — {len(vector)} dims")
        except Exception as e:
            logger.error(f"Embed smoke call failed: {e}")
            return 5

    logger.info("Preflight OK")
    return 0


def run_ingest(
    session,
    limit: int = 100,
    *,
    days_back: int = 7,
    fetch_all_pages: bool = False,
    all_history: bool = False,
    recording_id: str | None = None,
    phase_name: str = "ingest",
) -> PhaseResult:
    """Run ingestion phase: download recordings from Plaud.

    Returns:
        int: Number of recordings ingested
    """
    phase_label = "Backfill" if phase_name == "backfill" else "Ingest"
    phase_title = "BACKFILL" if phase_name == "backfill" else "INGEST"

    print_phase_header(f"PHASE 1: {phase_title} (Plaud API)", "📥")
    print(
        "Fetching full Plaud history from Plaud..."
        if all_history
        else "Fetching recordings from Plaud..."
    )

    start_time = time.time()
    pipeline_progress.start_phase(phase_name)
    pipeline_progress.update(
        step=(
            "Fetching full Plaud history from Plaud…"
            if all_history
            else "Fetching recording list from Plaud…"
        )
    )

    service = ChronosIngestService(db_session=session)

    # Add progress callback to the service
    def on_progress(current, total, recording_id):
        print_progress("Plaud", current, total, recording_id, time.time() - start_time)

    error_message = None
    warnings: tuple[str, ...] = ()
    partial_success = False

    try:
        success_count, failure_count = service.ingest_recent_recordings(
            limit=limit,
            days_back=days_back,
            fetch_all_pages=fetch_all_pages,
            all_history=all_history,
            recording_id=recording_id,
        )
        warnings = tuple(getattr(service, "last_batch_warnings", []) or ())
        partial_success = bool(getattr(service, "last_batch_partial_success", False))
        if all_history and success_count > 0 and failure_count > 0 and not warnings:
            warnings = ("Partial Plaud backfill preserved earlier pages.",)
            partial_success = True
    except Exception as e:
        error_message = str(e)
        logger.warning(f"Ingest phase failed: {e}")
        success_count, failure_count = 0, 0

    elapsed = time.time() - start_time
    if error_message:
        print_phase_failed(phase_label, error_message, elapsed)
        pipeline_progress.finish_phase(summary=f"FAILED: {error_message[:120]}")
        return PhaseResult(
            processed_count=success_count,
            failure_count=failure_count,
            error_message=error_message,
            warnings=warnings,
        )

    if warnings:
        pipeline_progress.set_phase_warnings(phase_name, list(warnings))
        pipeline_progress.set_run_context(
            partial_success=partial_success,
            warning=warnings[0],
            warnings=list(warnings),
        )

    print_phase_complete(phase_label, success_count, elapsed)
    pipeline_progress.finish_phase(
        summary=f"{success_count} ingested, {failure_count} failed"
    )

    if failure_count > 0:
        print(f"⚠️  {failure_count} recordings failed to ingest")

    return PhaseResult(
        processed_count=success_count,
        failure_count=failure_count,
        warnings=warnings,
    )


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

    try:
        from app_v2.services.xray import xray_log
    except Exception:
        xray_log = None

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
            rec_id = str(rec.recording_id)
            duration_seconds = getattr(rec, "duration_seconds", 0) or 0
            duration_mins = int(cast(Any, duration_seconds)) // 60

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
                last_xray_heartbeat = 0.0

                def heartbeat():
                    nonlocal last_xray_heartbeat
                    dots = 0
                    while not stop_heartbeat.is_set():
                        elapsed = time.time() - proc_start
                        dots = (dots % 3) + 1
                        print(
                            f"   ⏳ Still processing{'.' * dots} ({elapsed:.0f}s elapsed)",
                            flush=True,
                        )

                        pipeline_progress.update(
                            total=total,
                            step=f"Recording {i+1}/{total}: Gemini AI… ({elapsed:.0f}s elapsed)",
                            item=rec_id[:20],
                        )

                        if xray_log and elapsed - last_xray_heartbeat >= 10:
                            xray_log(
                                "gemini",
                                "heartbeat",
                                f"Still processing recording {i+1}/{total}",
                                detail=f"{rec_id[:20]} · {elapsed:.0f}s elapsed",
                            )
                            last_xray_heartbeat = elapsed

                        stop_heartbeat.wait(5)  # Print every 5 seconds

                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()

                try:

                    def progress_callback(step: str, detail: str = ""):
                        detail_suffix = f" — {detail}" if detail else ""
                        pipeline_progress.update(
                            total=total,
                            step=f"Recording {i+1}/{total}: {step}{detail_suffix}",
                            item=rec_id[:20],
                        )

                    ok = processor.process_recording_id(
                        rec_id,
                        progress_callback=progress_callback,
                    )
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
    # Uses OpenAI text-embedding-3-large (or text-embedding-3-small) via the
    # embedding service. Text-only (OpenAI embeddings don't support audio).
    # Dimensionality controlled by CHRONOS_EMBEDDING_DIM (default 768).
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
        batch_size = max(1, get_settings().chronos_embed_batch_size)
        if embedder.provider == "ollama":
            # On local CPU (Pi), sequential/large batches cause HTTP timeouts.
            # Limit the batch slice size to local_embed_batch_size to ensure fast responses
            # and publish granular progress updates to the UI.
            batch_size = min(batch_size, embedder.local_embed_batch_size)

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            print_progress(
                "Embed",
                i,
                len(texts),
                f"batch {current_batch_num}",
                time.time() - embed_start,
            )
            pipeline_progress.update(
                step=f"Generating embeddings: batch {current_batch_num}/{total_batches} ({i}/{len(texts)})",
                completed=i,
                total=len(texts),
            )
            batch_embeddings = embedder.embed_batch(
                batch, task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.extend(batch_embeddings)
        pipeline_progress.update(completed=len(texts), total=len(texts))


    total_embedded = len(pydantic_events)
    print_progress(
        "Embed", total_embedded, total_embedded, "complete", time.time() - embed_start
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

    pipeline_progress.start_phase("graph", total_items=len(events_to_process))

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
                category_confidence=getattr(db_event, "category_confidence", None),
                sentiment=db_event.sentiment,
                keywords=db_event.keywords or [],
                speaker=(
                    SpeakerMode(db_event.speaker)
                    if db_event.speaker
                    else SpeakerMode.SELF_TALK
                ),
                raw_transcript_snippet=getattr(
                    db_event, "raw_transcript_snippet", None
                ),
                gemini_reasoning=getattr(db_event, "gemini_reasoning", None),
            )
            pydantic_events.append(pydantic_event)
        except Exception as e:
            logger.error(f"Failed to convert event {db_event.event_id}: {e}")
            continue

    # Extract entities and build graph
    pipeline_progress.update(
        step="Extracting entities", item=f"{len(pydantic_events)} events"
    )

    def _graph_progress(event_id: str) -> None:
        pipeline_progress.advance(item=event_id, step="Extracting entities")

    entities, graph = graph_extractor.extract_from_events(
        pydantic_events, progress_callback=_graph_progress
    )

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


def run_repair_recent(
    session,
    *,
    days_back: int = 45,
    limit: int = 10,
    stale_after_minutes: int = 90,
) -> int:
    """Audit recent Plaud recordings and repair missing or retryable rows.

    Repairs applied here feed directly into the normal process/index phases so
    the always-on poll loop can self-heal missed or failed recent recordings.
    """
    print_phase_header("RECENT SELF-HEAL", "🩹")
    print(
        f"Auditing Plaud recordings from the last {days_back} days (repair budget: {limit})..."
    )
    pipeline_progress.start_phase("repair")
    pipeline_progress.update(step="Scanning Plaud for recent recordings")

    start_time = time.time()

    from app_v2.services import data_service as data_service_module
    from src.database.models import ChronosRecording as ChronosRecordingDB

    ingest_service = ChronosIngestService(db_session=session)
    recent_records, pages_fetched, stop_reason = ingest_service._fetch_recent_recordings_window(
        days_back=max(int(days_back), 1),
        page_size=20,
    )

    recent_ids = [
        str(rec.get("id") or "").strip()
        for rec in recent_records
        if str(rec.get("id") or "").strip()
    ]

    existing_rows = {}
    if recent_ids:
        existing_rows = {
            str(rec.recording_id): rec
            for rec in session.query(ChronosRecordingDB)
            .filter(ChronosRecordingDB.recording_id.in_(recent_ids))
            .all()
        }

    repairs_applied = 0
    missing_ingested = 0
    failures_reset = 0
    stale_reset = 0
    workflows_submitted = 0
    archived_skipped = 0
    errors = 0
    workflow_service = None
    transcript_probe = None

    # Use the same stale heuristic as the UI service: rows marked processing long
    # after creation are considered interrupted and are safe to retry.
    stale_cutoff = datetime.utcnow() - timedelta(
        minutes=max(int(stale_after_minutes), 1)
    )

    for rec_data in recent_records:
        if repairs_applied >= max(int(limit), 1):
            break

        recording_id = str(rec_data.get("id") or "").strip()
        if not recording_id:
            continue

        db_rec = existing_rows.get(recording_id)

        if db_rec is None:
            pipeline_progress.update(
                step="Repairing missing recording",
                item=recording_id[:20],
            )
            ok, error = ingest_service.ingest_recording_by_id(recording_id)
            if ok:
                repairs_applied += 1
                missing_ingested += 1
            else:
                errors += 1
                logger.warning(
                    "Recent self-heal could not ingest missing Plaud recording %s: %s",
                    recording_id,
                    error,
                )
            continue

        status = str(db_rec.processing_status or "").strip().lower()
        if status == "failed":
            failure_text = str(getattr(db_rec, "error_message", "") or "")
            workflow_id = str(getattr(db_rec, "plaud_workflow_id", "") or "").strip()
            workflow_status = str(getattr(db_rec, "plaud_workflow_status", "") or "").strip().upper()

            if "No transcript available in Plaud source_list" in failure_text:
                if transcript_probe is None:
                    from src.chronos.transcript_processor import TranscriptProcessor

                    transcript_probe = TranscriptProcessor(
                        session,
                        plaud_client=ingest_service.plaud,
                    )
                try:
                    live_details = ingest_service.plaud.get_recording(recording_id)
                    live_transcript = transcript_probe._extract_transcript(live_details)
                except Exception as exc:
                    live_transcript = None
                    logger.warning(
                        "Recent self-heal could not re-check transcript availability for %s: %s",
                        recording_id,
                        exc,
                    )

                if live_transcript and live_transcript.strip():
                    pipeline_progress.update(
                        step="Transcript recovered from Plaud",
                        item=recording_id[:20],
                    )
                    try:
                        set_chronos_recording_transcript(
                            session,
                            recording_id,
                            live_transcript,
                        )
                    except Exception:
                        logger.debug(
                            "Could not cache recovered transcript for %s",
                            recording_id,
                            exc_info=True,
                        )
                    mark_chronos_recording_status(
                        session,
                        recording_id,
                        "pending",
                        error_message=None,
                    )
                    repairs_applied += 1
                    failures_reset += 1
                    continue

                if workflow_status in {"PENDING", "PROCESSING", "RUNNING"}:
                    archived_skipped += 1
                    continue
                if not workflow_id:
                    pipeline_progress.update(
                        step="Submitting Plaud recovery workflow",
                        item=recording_id[:20],
                    )
                    if workflow_service is None:
                        workflow_service = data_service_module.ChronosDataService()
                    workflow_result = workflow_service.submit_single_recording_workflow(
                        recording_id
                    )
                    if workflow_result.get("workflow_id"):
                        repairs_applied += 1
                        workflows_submitted += 1
                    else:
                        errors += 1
                        logger.warning(
                            "Recent self-heal could not submit Plaud workflow for %s: %s",
                            recording_id,
                            workflow_result.get("error"),
                        )
                    continue

            bucket, _reason = data_service_module.classify_sync_failure_row(db_rec)
            if bucket == "actionable":
                pipeline_progress.update(
                    step="Re-queueing failed recording",
                    item=recording_id[:20],
                )
                mark_chronos_recording_status(
                    session,
                    recording_id,
                    "pending",
                    error_message=None,
                )
                repairs_applied += 1
                failures_reset += 1
            else:
                archived_skipped += 1
            continue

        if (
            status == "processing"
            and getattr(db_rec, "processed_at", None) is None
            and getattr(db_rec, "created_at", None) is not None
            and db_rec.created_at < stale_cutoff
        ):
            pipeline_progress.update(
                step="Resetting stale processing row",
                item=recording_id[:20],
            )
            mark_chronos_recording_status(
                session,
                recording_id,
                "pending",
                error_message=None,
            )
            repairs_applied += 1
            stale_reset += 1

    elapsed = time.time() - start_time
    print(
        f"  🔎 Audited {len(recent_records)} Plaud recordings across {pages_fetched} page(s)"
    )
    print(
        f"  🩹 Repairs: {repairs_applied} | missing ingested: {missing_ingested} | failed reset: {failures_reset} | stale reset: {stale_reset} | workflows submitted: {workflows_submitted}"
    )
    if archived_skipped:
        print(f"  📦 Archived failures skipped: {archived_skipped}")
    if errors:
        print(f"  ❌ Repair errors: {errors}")
    print(f"  🛑 Window stop reason: {stop_reason}")
    print(f"  ⏱️  {elapsed:.1f}s")

    pipeline_progress.finish_phase(
        summary=(
            f"{repairs_applied} repaired, {workflows_submitted} workflows, {archived_skipped} archived, {errors} errors"
        )
    )

    return repairs_applied


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
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run the full pipeline with a page-by-page Plaud history backfill",
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--limit", type=int, default=10, help="Max items per phase")
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Only ingest Plaud recordings from the last N days when paginating",
    )
    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
        help="Operate on a single recording_id (applies to --ingest/--process/--index/--graph)",
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
        "--all-history",
        action="store_true",
        help="For --ingest/--full: backfill the entire Plaud history instead of stopping at --days-back",
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
    parser.add_argument(
        "--repair-recent",
        action="store_true",
        help="Audit recent Plaud recordings and repair missing or retryable rows before processing.",
    )

    args = parser.parse_args()

    if _requested_heavy_work(args):
        pressure_reason = _host_pressure_reason()
        if pressure_reason:
            logger.warning("Chronos pipeline deferred: %s", pressure_reason)
            print(f"⏸ Chronos pipeline deferred: {pressure_reason}")
            print("   It will retry on the next timer/webhook/poll cycle.")
            return

    # If no specific phase, show help
    if not any(
        [
            args.preflight,
            args.preflight_smoke,
            args.ingest,
            args.process,
            args.index,
            args.graph,
            args.backfill,
            args.full,
            args.reindex,
            args.refresh_workflows,
            args.repair_recent,
        ]
    ):
        parser.print_help()
        return

    if args.preflight or args.preflight_smoke:
        code = run_preflight(smoke_call=bool(args.preflight_smoke))
        # If the user asked ONLY for preflight, exit early.
        if not any(
            [
                args.ingest,
                args.process,
                args.index,
                args.graph,
                args.backfill,
                args.full,
            ]
        ):
            raise SystemExit(code)
        # If preflight failed but the user asked for other phases, stop early.
        if code != 0:
            raise SystemExit(code)

    # Determine which phases will run
    run_full_pipeline = bool(args.full or args.backfill)
    ingest_phase_name = "backfill" if args.backfill else "ingest"
    all_history = bool(args.all_history or args.backfill)
    if args.backfill:
        sync_mode = "backfill"
    elif args.full:
        sync_mode = "full"
    elif args.ingest:
        sync_mode = "ingest"
    elif args.process:
        sync_mode = "process"
    elif args.index:
        sync_mode = "index"
    elif args.graph:
        sync_mode = "graph"
    elif args.reindex:
        sync_mode = "reindex"
    else:
        sync_mode = ""

    phases = []
    if args.reindex:
        phases.append("index")
    if args.backfill:
        phases.append("backfill")
    elif args.full or args.ingest:
        phases.append("ingest")
    if run_full_pipeline or args.refresh_workflows:
        phases.append("refresh-workflows")
    if run_full_pipeline or args.repair_recent:
        phases.append("repair")
    if run_full_pipeline or args.process:
        phases.append("process")
    if run_full_pipeline or args.index:
        phases.append("index")
    if run_full_pipeline or args.graph:
        phases.append("graph")
    # Deduplicate while preserving order
    phases = list(dict.fromkeys(phases))
    pipeline_progress.start_run(phases=phases, trigger="cli")
    pipeline_progress.set_run_context(sync_mode=sync_mode)

    # Initialize database
    init_db()
    trace_run_id = start_trace_run(
        run_id=os.environ.get("CHRONOS_TRACE_RUN_ID") or None,
        trigger=os.environ.get("CHRONOS_TRACE_TRIGGER", "cli"),
        source="pipeline",
        title="Chronos pipeline",
        entrypoint="scripts/chronos_pipeline.py",
        metadata={
            "phases": phases,
            "recording_id": args.recording_id,
            "days_back": args.days_back,
            "full": bool(args.full),
            "backfill": bool(args.backfill),
            "all_history": all_history,
            "sync_mode": sync_mode or "stage",
        },
    )
    session = SessionLocal()

    exit_code = 0
    phase_errors: list[str] = []

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
            with trace_span(
                operation="phase-reindex",
                source="pipeline",
                stage="index",
                message="Reindex all Chronos events",
            ):
                run_index(session, limit=999_999)

        if args.backfill or args.full or args.ingest:
            # --full and --backfill always fetch all pages within their scopes.
            # --all-history overrides the ingest window and pulls the entire Plaud history.
            fetch_all = True if run_full_pipeline else bool(args.fetch_all)
            with trace_span(
                operation=f"phase-{ingest_phase_name}",
                source="pipeline",
                stage=ingest_phase_name,
                message=(
                    "Fetch full Plaud history from Plaud"
                    if ingest_phase_name == "backfill"
                    else "Fetch recordings from Plaud"
                ),
                recording_id=args.recording_id,
                metadata={
                    "days_back": args.days_back,
                    "fetch_all_pages": fetch_all,
                    "all_history": all_history,
                    "sync_mode": ingest_phase_name,
                },
            ):
                ingest_result = run_ingest(
                    session,
                    limit=args.limit,
                    days_back=args.days_back,
                    fetch_all_pages=fetch_all,
                    all_history=all_history,
                    recording_id=args.recording_id,
                    phase_name=ingest_phase_name,
                )
            if ingest_result.error_message:
                phase_errors.append(f"{ingest_phase_name}: {ingest_result.error_message}")

        if run_full_pipeline or args.refresh_workflows:
            with trace_span(
                operation="phase-refresh-workflows",
                source="pipeline",
                stage="refresh-workflows",
                provider="plaud",
                message="Refresh Plaud workflow statuses",
            ):
                run_refresh_workflows(session, days_back=max(args.days_back, 30), limit=args.limit)

        if run_full_pipeline or args.repair_recent:
            with trace_span(
                operation="phase-repair-recent",
                source="pipeline",
                stage="repair",
                message="Repair recent Chronos rows",
            ):
                run_repair_recent(
                    session,
                    days_back=args.days_back,
                    limit=args.limit,
                )

        if run_full_pipeline or args.process:
            with trace_span(
                operation="phase-process",
                source="pipeline",
                stage="process",
                provider="chronos",
                message="Extract Chronos events from transcripts",
                recording_id=args.recording_id,
            ):
                run_process(
                    session,
                    limit=args.limit,
                    recording_id=args.recording_id,
                    force=bool(args.force),
                )

        if run_full_pipeline or args.index:
            with trace_span(
                operation="phase-index",
                source="pipeline",
                stage="index",
                provider="qdrant",
                message="Embed and index Chronos events",
                recording_id=args.recording_id,
            ):
                run_index(session, limit=args.limit, recording_id=args.recording_id)

        if run_full_pipeline or args.graph:
            with trace_span(
                operation="phase-graph",
                source="pipeline",
                stage="graph",
                message="Extract knowledge graph",
                recording_id=args.recording_id,
            ):
                run_graph(session, limit=args.limit, recording_id=args.recording_id)

        if phase_errors:
            logger.warning("=" * 60)
            logger.warning("PIPELINE COMPLETE WITH WARNINGS")
            for phase_error in phase_errors:
                logger.warning(phase_error)
            logger.warning("=" * 60)
            pipeline_progress.finish_run(error="; ".join(phase_errors))
            finish_trace_run(
                trace_run_id,
                status="failed",
                summary={"phase_errors": phase_errors},
                error_message="; ".join(phase_errors),
            )
            exit_code = 1
        else:
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            pipeline_progress.finish_run()
            finish_trace_run(trace_run_id, status="completed", summary={"phases": phases})

    except Exception as e:
        pipeline_progress.finish_run(error=str(e))
        finish_trace_run(trace_run_id, status="failed", error_message=str(e))
        raise
    finally:
        session.close()

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
