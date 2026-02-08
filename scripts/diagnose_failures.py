"""Diagnose all non-completed recordings."""

import sys

sys.path.insert(0, ".")

from src.database.engine import SessionLocal
from src.database.models import ChronosRecording

db = SessionLocal()
failed = (
    db.query(ChronosRecording)
    .filter(ChronosRecording.processing_status == "failed")
    .all()
)
pending = (
    db.query(ChronosRecording)
    .filter(ChronosRecording.processing_status == "pending")
    .all()
)

print("=" * 70)
print(f"FAILED RECORDINGS: {len(failed)}")
print("=" * 70)
for r in failed:
    t_len = len(r.transcript) if r.transcript else 0
    print(f"  ID: {r.recording_id}")
    print(f"    Duration: {r.duration_seconds}s ({r.duration_seconds // 60}m)")
    print(f"    Created: {r.created_at}")
    print(f"    Source: {r.source}")
    print(f"    Title: {r.title}")
    print(f"    Transcript: {t_len} chars")
    print(f"    Audio path: {r.local_audio_path}")
    print(f"    ERROR: {r.error_message}")
    if r.transcript:
        preview = r.transcript[:200].replace("\n", " ")
        print(f"    Preview: {preview}...")
    print()

print("=" * 70)
print(f"PENDING RECORDINGS: {len(pending)}")
print("=" * 70)
for r in pending:
    t_len = len(r.transcript) if r.transcript else 0
    print(f"  ID: {r.recording_id}")
    print(f"    Duration: {r.duration_seconds}s ({r.duration_seconds // 60}m)")
    print(f"    Created: {r.created_at}")
    print(f"    Source: {r.source}")
    print(f"    Transcript: {t_len} chars")
    print(f"    ERROR: {r.error_message}")
    print()

db.close()
