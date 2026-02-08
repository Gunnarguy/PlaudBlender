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
import sys
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


def compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def parse_plaud_filename(filename: str) -> Optional[datetime]:
    """Parse timestamp from Plaud filename format (Unix timestamp)."""
    try:
        stem = Path(filename).stem
        timestamp = int(stem)
        return datetime.fromtimestamp(timestamp)
    except (ValueError, OSError):
        return None


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
    # Generate recording ID from checksum for deduplication
    checksum = compute_checksum(audio_path)
    recording_id = hashlib.md5(checksum.encode()).hexdigest()

    # Check if already processed
    existing = get_chronos_recording(session, recording_id)
    if existing and existing.processing_status == "completed" and not force:  # type: ignore[truthy-bool]
        logger.info(f"⏭️ Already processed: {audio_path.name}")
        return True

    # Parse timestamp from filename or use file modification time
    created_at = parse_plaud_filename(audio_path.name)
    if not created_at:
        created_at = datetime.fromtimestamp(audio_path.stat().st_mtime)

    logger.info(f"📥 Processing: {audio_path.name}")
    logger.info(f"   Size: {audio_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"   Date: {created_at}")

    # Create/update recording in database
    rec = upsert_chronos_recording(
        session=session,
        recording_id=recording_id,
        title=audio_path.stem,
        created_at=created_at,
        duration_seconds=0,
        local_audio_path=str(audio_path),
        source="local_import",
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
            db_event = ChronosEventModel(
                event_id=str(uuid.uuid4()),
                recording_id=recording_id,
                start_ts=event.start_ts,
                end_ts=event.end_ts,
                day_of_week=created_at.strftime("%A"),
                hour_of_day=created_at.hour,
                clean_text=event.clean_text,
                category=event.category,
                sentiment=event.sentiment,
                keywords=",".join(event.keywords) if event.keywords else "",
                speaker=event.speaker,
                raw_transcript_snippet=event.raw_transcript_snippet,
                gemini_reasoning=event.gemini_reasoning,
            )
            db_events.append(db_event)

        add_chronos_events(session, db_events)
        logger.info(f"💾 Saved {len(db_events)} events to database")

        # Generate embeddings and index to Qdrant
        logger.info("🔮 Generating embeddings...")
        texts = [e.clean_text for e in db_events]
        embeddings = embedder.embed_batch(texts)

        # Prepare points for Qdrant
        from qdrant_client.models import PointStruct

        points = []
        for event, embedding in zip(db_events, embeddings):
            point_id = str(uuid.uuid4())
            event.qdrant_point_id = point_id

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "event_id": event.event_id,
                        "recording_id": recording_id,
                        "clean_text": event.clean_text,
                        "category": event.category,
                        "sentiment": event.sentiment,
                        "keywords": event.keywords.split(",") if event.keywords else [],
                        "day_of_week": event.day_of_week,
                        "hour_of_day": event.hour_of_day,
                        "start_ts": (
                            event.start_ts.isoformat() if event.start_ts else None
                        ),
                        "end_ts": event.end_ts.isoformat() if event.end_ts else None,
                        "speaker": event.speaker,
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
