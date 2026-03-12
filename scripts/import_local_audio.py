#!/usr/bin/env python3
"""Import local audio files directly into Chronos pipeline.

Bypasses Plaud API entirely - processes audio through Gemini AI
and indexes events to Qdrant.

Usage:
    python scripts/import_local_audio.py /path/to/audio.wav
    python scripts/import_local_audio.py --usb  # Import from connected Plaud device
"""

import argparse
import hashlib
import logging
import struct
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.database.engine import SessionLocal
from src.database.chronos_repository import (
    upsert_chronos_recording,
    get_chronos_recording,
    add_chronos_events,
    mark_chronos_recording_status,
)
from src.database.models import ChronosEvent as ChronosEventModel
from src.chronos.engine import ChronosEngine
from src.chronos.qdrant_client import ChronosQdrantClient
from src.chronos.embedding_service import ChronosEmbeddingService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Supported audio formats
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".opus", ".aac", ".flac", ".ogg"}


def compute_checksum(file_path: Path, quick: bool = False) -> str:
    """Compute checksum for deduplication.

    Args:
        file_path: Path to file
        quick: If True, use first+last 1MB for speed (good for large WAV files)
    """
    sha256 = hashlib.sha256()

    if quick:
        # Quick mode: hash first 1MB + last 1MB + file size
        size = file_path.stat().st_size
        sha256.update(str(size).encode())
        with open(file_path, "rb") as f:
            sha256.update(f.read(1024 * 1024))  # First 1MB
            if size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)  # Last 1MB
                sha256.update(f.read(1024 * 1024))
        return f"sha256q:{sha256.hexdigest()}"

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def recording_id_from_path(file_path: Path) -> str:
    """Generate a stable recording ID from a Plaud USB file.

    Uses the filename (Unix timestamp) which is unique per recording.
    Falls back to checksum-based ID for non-Plaud files.
    """
    stem = file_path.stem
    try:
        # Plaud files use Unix timestamp as filename - this IS the unique ID
        int(stem)
        return hashlib.md5(f"plaud_usb:{stem}".encode()).hexdigest()
    except ValueError:
        # Non-Plaud file - use quick checksum
        checksum = compute_checksum(file_path, quick=True)
        return hashlib.md5(checksum.encode()).hexdigest()


def parse_plaud_filename(filename: str) -> Optional[datetime]:
    """Parse timestamp from Plaud filename format (Unix timestamp)."""
    try:
        stem = Path(filename).stem
        timestamp = int(stem)
        return datetime.fromtimestamp(timestamp)
    except (ValueError, OSError):
        return None


def get_wav_duration(file_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(file_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / rate
    except Exception:
        pass
    # Fallback: estimate from file size (16-bit mono 16kHz typical for Plaud)
    try:
        size = file_path.stat().st_size
        return size / (16000 * 2)  # 16kHz, 16-bit
    except Exception:
        return 0.0


def find_audio_files(path: Path) -> List[Path]:
    """Find all audio files in a directory."""
    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return [path]
        return []

    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))
        files.extend(path.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def find_plaud_usb_device() -> Optional[Path]:
    """Find connected Plaud USB device."""
    volumes = Path("/Volumes")
    plaud_patterns = ["PLAUD", "NOTE", "PIN"]

    for vol in volumes.iterdir():
        if not vol.is_dir():
            continue
        name_upper = vol.name.upper()
        if any(p in name_upper for p in plaud_patterns):
            for folder in ["NOTES", "RECORD", "CALLS", "VOICE"]:
                if (vol / folder).exists():
                    logger.info(f"Found Plaud device: {vol}")
                    return vol
    return None


def process_audio_file(
    audio_path: Path,
    session,
    engine: ChronosEngine,
    qdrant: ChronosQdrantClient,
    embedder: ChronosEmbeddingService,
    force: bool = False,
) -> bool:
    """Process a single audio file through the full pipeline."""
    # Generate recording ID - fast for Plaud USB files (filename-based)
    recording_id = recording_id_from_path(audio_path)

    # Check if already processed
    existing = get_chronos_recording(session, recording_id)
    if existing and existing.processing_status == "completed" and not force:  # type: ignore[truthy-bool]
        logger.info(f"⏭️ Already processed: {audio_path.name}")
        return True

    # Parse timestamp from filename or parent folder (Plaud USB: NOTES/YYYYMMDD/timestamp.WAV)
    created_at = parse_plaud_filename(audio_path.name)
    if not created_at:
        # Try parent folder name as YYYYMMDD
        try:
            folder_name = audio_path.parent.name
            if len(folder_name) == 8 and folder_name.isdigit():
                folder_date = datetime.strptime(folder_name, "%Y%m%d")
                # Use file modification time for the time-of-day
                file_mtime = datetime.fromtimestamp(audio_path.stat().st_mtime)
                created_at = folder_date.replace(
                    hour=file_mtime.hour,
                    minute=file_mtime.minute,
                    second=file_mtime.second,
                )
        except (ValueError, OSError):
            pass
    if not created_at:
        created_at = datetime.fromtimestamp(audio_path.stat().st_mtime)

    # Get actual audio duration
    audio_duration = (
        get_wav_duration(audio_path) if audio_path.suffix.upper() == ".WAV" else 0.0
    )

    # Compute checksum for integrity (quick mode for large files)
    file_size = audio_path.stat().st_size
    checksum = compute_checksum(audio_path, quick=(file_size > 100 * 1024 * 1024))

    logger.info(f"📥 Processing: {audio_path.name}")
    logger.info(f"   Size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"   Date: {created_at}")
    logger.info(f"   Duration: {audio_duration/60:.1f} min")

    # Create/update recording in database
    rec = upsert_chronos_recording(
        session=session,
        recording_id=recording_id,
        title=audio_path.stem,
        created_at=created_at,
        duration_seconds=int(audio_duration),
        local_audio_path=str(audio_path),
        source="usb_import",
        checksum=checksum,
    )
    mark_chronos_recording_status(session, recording_id, "processing")

    try:
        # Process through Gemini
        logger.info("🧠 Processing audio through Gemini AI...")
        result = engine.process_audio(
            str(audio_path),
            recording_id,
            recording_date=created_at.strftime("%Y-%m-%d"),
        )

        if not result or not result.events:
            logger.warning(f"⚠️ No events extracted from {audio_path.name}")
            mark_chronos_recording_status(session, recording_id, "completed")
            return True

        logger.info(f"✅ Extracted {len(result.events)} events")

        # Store events in database
        db_events = []
        for event in result.events:
            # Use event time for temporal fields (not recording time)
            evt_ts = event.start_ts or created_at
            db_event = ChronosEventModel(
                event_id=str(uuid.uuid4()),
                recording_id=recording_id,
                start_ts=event.start_ts,
                end_ts=event.end_ts,
                day_of_week=evt_ts.strftime("%A"),
                hour_of_day=evt_ts.hour,
                clean_text=event.clean_text,
                category=(
                    event.category.value
                    if hasattr(event.category, "value")
                    else str(event.category)
                ),
                sentiment=event.sentiment,
                keywords=",".join(event.keywords) if event.keywords else "",
                speaker=(
                    event.speaker.value
                    if hasattr(event.speaker, "value")
                    else str(event.speaker)
                ),
                raw_transcript_snippet=event.raw_transcript_snippet,
                gemini_reasoning=event.gemini_reasoning,
            )
            db_events.append(db_event)

        add_chronos_events(session, db_events)
        logger.info(f"💾 Saved {len(db_events)} events to database")

        # Generate embeddings and index to Qdrant
        logger.info("🔮 Generating embeddings...")
        texts = [e.clean_text for e in db_events]
        if embedder.supports_multimodal and str(audio_path):
            # Multimodal: fuse text + audio for each event
            logger.info(f"  Multimodal mode — embedding text+audio from {audio_path}")
            embeddings = [
                embedder.embed_text_with_audio(
                    text=t,
                    audio_path=str(audio_path),
                    task_type="RETRIEVAL_DOCUMENT",
                )
                for t in texts
            ]
        else:
            embeddings = embedder.embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

        # Prepare points for Qdrant
        from qdrant_client.models import PointStruct

        points = []
        for event, embedding in zip(db_events, embeddings):
            point_id = str(uuid.uuid4())
            event.qdrant_point_id = point_id

            # Compute duration_seconds from start/end
            evt_duration = 0.0
            if event.start_ts and event.end_ts:
                evt_duration = max((event.end_ts - event.start_ts).total_seconds(), 0)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "event_id": event.event_id,
                        "recording_id": recording_id,
                        "clean_text": event.clean_text,
                        "category": event.category,  # already .value from above
                        "sentiment": event.sentiment,
                        "keywords": (
                            event.keywords.split(",")
                            if isinstance(event.keywords, str) and event.keywords
                            else (
                                event.keywords
                                if isinstance(event.keywords, list)
                                else []
                            )
                        ),
                        "day_of_week": event.day_of_week,
                        "hour_of_day": event.hour_of_day,
                        "start_ts": (
                            event.start_ts.isoformat() if event.start_ts else None
                        ),
                        "end_ts": event.end_ts.isoformat() if event.end_ts else None,
                        "timestamp": (
                            event.start_ts.isoformat() if event.start_ts else None
                        ),
                        "start_ts_unix": (
                            event.start_ts.timestamp() if event.start_ts else 0.0
                        ),
                        "duration_seconds": evt_duration,
                        "speaker": event.speaker,  # already .value from above
                    },
                )
            )

        # Upsert to Qdrant
        qdrant.client.upsert(
            collection_name=qdrant.collection_name,
            points=points,
        )
        logger.info(f"📤 Indexed {len(points)} events to Qdrant")

        # Update recording status
        mark_chronos_recording_status(session, recording_id, "completed")
        return True

    except Exception as e:
        logger.error(f"❌ Error processing {audio_path.name}: {e}")
        import traceback

        traceback.print_exc()
        mark_chronos_recording_status(session, recording_id, "failed", str(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Import local audio files into Chronos pipeline"
    )
    parser.add_argument("path", nargs="?", help="Path to audio file or directory")
    parser.add_argument(
        "--usb", action="store_true", help="Import from connected Plaud USB device"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-process already completed recordings",
    )
    parser.add_argument(
        "--limit", "-l", type=int, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and list files without processing (dry run)",
    )

    args = parser.parse_args()

    # Determine input path
    if args.usb:
        device = find_plaud_usb_device()
        if not device:
            logger.error("❌ No Plaud USB device found")
            sys.exit(1)
        input_path = device
    elif args.path:
        input_path = Path(args.path)
    else:
        parser.print_help()
        sys.exit(1)

    if not input_path.exists():
        logger.error(f"❌ Path not found: {input_path}")
        sys.exit(1)

    # Find audio files
    audio_files = find_audio_files(input_path)
    if not audio_files:
        logger.error(f"❌ No audio files found in {input_path}")
        sys.exit(1)

    if args.limit:
        audio_files = audio_files[: args.limit]

    logger.info(f"📂 Found {len(audio_files)} audio file(s) to process")

    # Initialize services
    session = SessionLocal()

    if args.scan:
        # Dry-run: list all files and check which are already in DB
        from collections import defaultdict

        by_date = defaultdict(list)
        already_in_db = 0
        new_files = 0

        for af in audio_files:
            rid = recording_id_from_path(af)
            existing = get_chronos_recording(session, rid)
            ts = parse_plaud_filename(af.name)
            if not ts:
                try:
                    folder_name = af.parent.name
                    if len(folder_name) == 8 and folder_name.isdigit():
                        ts = datetime.strptime(folder_name, "%Y%m%d")
                except Exception:
                    pass
            if not ts:
                ts = datetime.fromtimestamp(af.stat().st_mtime)

            status = "IN_DB" if existing else "NEW"
            if existing:
                already_in_db += 1
            else:
                new_files += 1

            dur = get_wav_duration(af) if af.suffix.upper() == ".WAV" else 0
            date_key = ts.strftime("%Y-%m-%d")
            by_date[date_key].append((af.name, dur, status))

        print(f"\n{'='*60}")
        print(f"SCAN RESULTS: {len(audio_files)} files")
        print(f"  Already in DB: {already_in_db}")
        print(f"  New (to import): {new_files}")
        print(f"{'='*60}")
        for d in sorted(by_date.keys()):
            items = by_date[d]
            new_count = sum(1 for _, _, s in items if s == "NEW")
            print(f"\n  {d}: {len(items)} files ({new_count} new)")
            for name, dur, status in items:
                print(f"    [{status:5s}] {name}  ({dur/60:.0f}m)")
        print()
        session.close()
        return

    engine = ChronosEngine()
    qdrant = ChronosQdrantClient()
    embedder = ChronosEmbeddingService()

    # Ensure Qdrant collection exists
    qdrant.create_collection()

    # Process each file
    success = 0
    failed = 0

    for audio_path in audio_files:
        try:
            if process_audio_file(
                audio_path, session, engine, qdrant, embedder, args.force
            ):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            failed += 1

    session.close()

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"✅ Successfully processed: {success}")
    logger.info(f"❌ Failed: {failed}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
