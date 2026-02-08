#!/usr/bin/env python3
"""
Recover ALL failed/pending recordings — zero data gaps.

Strategy:
  A) Re-fetch transcripts from Plaud (may have become available since ingestion)
  B) Trigger Plaud Workflow transcription for recordings still missing transcripts
  C) Simple retry for transient errors (broken pipe, timeouts)
  D) Relaxed Gemini prompt for sparse/casual transcripts
  E) Mark genuinely empty recordings (≤5s, no transcript) as completed-empty

Usage:
    python scripts/recover_failures.py           # Full recovery
    python scripts/recover_failures.py --dry-run  # Show plan only
"""

import sys
import os
import json
import uuid
import time
import argparse
from datetime import datetime
from typing import Optional

sys.path.insert(0, ".")

from src.config import get_settings
from src.plaud_client import PlaudClient
from src.plaud_workflow import PlaudWorkflowClient
from src.database.engine import SessionLocal
from src.database.models import ChronosRecording, ChronosEvent as ChronosEventModel
from src.database.chronos_repository import (
    mark_chronos_recording_status,
    add_chronos_events,
    set_chronos_recording_transcript,
    delete_chronos_events_by_recording,
)
from src.chronos.transcript_processor import TranscriptProcessor
from src.chronos.engine import ChronosEngine, GeminiEventOutput
from src.chronos.genai_helpers import normalize_thinking_level
from src.models.chronos_schemas import ChronosEvent
from google.genai import types


# ── Relaxed prompt for sparse / casual transcripts ──────────────────────
SPARSE_TRANSCRIPT_PROMPT = """
**INPUT:** A transcript from a voice recording. This may be short, casual,
contain profanity, personal conversation, or seemingly random thoughts.
Unlike a structured workday, this could be ANY type of audio — a quick
voice memo, background chatter, a brief thought, or a casual conversation.

**YOUR TASK:** Extract EVERY distinct piece of content as an event.
For very short transcripts, even a single event is acceptable.

**RULES:**

1. **PRESERVE EVERYTHING** — even if it seems trivial, casual, or contains
   profanity. Convert each distinct thought or statement into an event.
2. **NO MINIMUM EVENT COUNT** — if there's only one thing said, return one event.
3. **ACCEPT ALL CONTENT** — profanity, slang, casual speech, half-thoughts.
   Clean up filler words (um, uh) but keep the substance intact.
4. **CATEGORIZE HONESTLY:**
   - personal: most casual speech, thoughts, life stuff
   - work: anything work-related
   - meeting: conversation with others
   - reflection: thinking about things
   - idea: brainstorming
   - unknown: unclear content
   - break: silence or pauses

5. **TEMPORAL ACCURACY:**
   - Recording date: {{RECORDING_DATE}}
   - Use this date for all timestamps
   - Distribute events across a reasonable time window

**OUTPUT FORMAT:** Return a JSON object:

```json
{
  "events": [
    {
      "event_id": "uuid-string",
      "recording_id": "{{RECORDING_ID}}",
      "start_ts": "{{RECORDING_DATE}}T12:00:00Z",
      "end_ts": "{{RECORDING_DATE}}T12:05:00Z",
      "day_of_week": "Monday",
      "hour_of_day": 12,
      "clean_text": "The actual content, cleaned up but preserving meaning",
      "category": "personal",
      "sentiment": 0.0,
      "keywords": ["relevant", "keywords"],
      "speaker": "self_talk"
    }
  ],
  "total_events": 1,
  "processing_metadata": {
    "audio_duration_seconds": 0,
    "total_silence_seconds": 0,
    "quality_notes": "Short/sparse recording"
  }
}
```

**CONSTRAINTS:**
- clean_text must be at least 10 characters
- day_of_week must be: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- category must be: work, personal, meeting, deep_work, break, reflection, idea, unknown
- speaker must be: self_talk, conversation, unknown
- Timestamps must use the RECORDING DATE above
- Return at least 1 event if there's ANY content at all

**REMEMBER:** Your job is to preserve what was said, not judge it. Even a
1-sentence recording should become 1 event. Do NOT return empty events array
unless the transcript is literally empty.
"""


def get_recording_date_str(rec: ChronosRecording) -> str:
    """Extract date string from recording."""
    if rec.created_at:
        try:
            if isinstance(rec.created_at, str):
                return rec.created_at[:10]
            return rec.created_at.strftime("%Y-%m-%d")
        except Exception:
            pass
    return "2025-01-01"


def get_day_of_week(date_str: str) -> str:
    """Get day of week from date string."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A")
    except Exception:
        return "Monday"


def extract_transcript_from_plaud(
    plaud: PlaudClient, recording_id: str
) -> Optional[str]:
    """Try to fetch transcript from Plaud API."""
    try:
        file_details = plaud.get_recording(recording_id)
        source_list = file_details.get("source_list", [])

        for source in source_list:
            if source.get("data_type") == "transaction":
                segments = json.loads(source.get("data_content", "[]"))
                texts = [seg.get("content", "") for seg in segments]
                text = " ".join(texts).strip()
                if text:
                    return text

        return None
    except Exception as e:
        print(f"      ⚠️  Plaud API error: {e}")
        return None


def process_with_sparse_prompt(
    engine: ChronosEngine,
    transcript: str,
    recording_id: str,
    recording_date: str,
    max_retries: int = 3,
) -> Optional[GeminiEventOutput]:
    """Process a sparse transcript with the relaxed prompt."""
    day_of_week = get_day_of_week(recording_date)

    prompt = SPARSE_TRANSCRIPT_PROMPT.replace("{{RECORDING_ID}}", recording_id)
    prompt = prompt.replace("{{RECORDING_DATE}}", recording_date)

    full_prompt = f"""{prompt}

**RAW TRANSCRIPT:**

{transcript}

Extract events from this transcript. Remember: even 1 event is acceptable.
Return valid JSON matching the schema above."""

    print(f"      📤 Sending to Gemini with relaxed prompt...")
    print(
        f"      📊 Transcript: {len(transcript):,} chars, {len(transcript.split()):,} words"
    )

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"      🔄 Retry {attempt + 1}/{max_retries}...")

            config = {
                "response_mime_type": "application/json",
                "response_json_schema": GeminiEventOutput.model_json_schema(),
                "temperature": 0.3,  # Slightly more creative for sparse content
            }
            thinking_level = normalize_thinking_level(
                getattr(
                    (
                        engine._settings
                        if hasattr(engine, "_settings")
                        else get_settings()
                    ),
                    "chronos_thinking_level",
                    "",
                )
            )
            if thinking_level is not None:
                config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )

            response_text = ""
            start_time = time.time()

            stream = engine.client.models.generate_content_stream(
                model=engine.model_name,
                contents=full_prompt,
                config=config,
            )

            for chunk in stream:
                chunk_text = chunk.text or ""
                response_text += chunk_text

            elapsed = time.time() - start_time
            events_found = response_text.count('"event_id"')
            print(
                f"      📝 Response: {len(response_text):,} chars | {events_found} events | {elapsed:.0f}s"
            )

            # Parse response
            raw_text = response_text.strip()
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                if len(parts) >= 2:
                    raw_text = parts[1].strip()
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()

            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object in Gemini response")

            output_data = json.loads(raw_text[start : end + 1])
            validated = GeminiEventOutput(**output_data)

            print(f"      ✅ Extracted {validated.total_events} events")
            return validated

        except Exception as e:
            print(f"      ❌ Attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue

    return None


def parse_ts(ts_str: str) -> datetime:
    """Parse an ISO timestamp string into a datetime object."""
    return datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")


def create_empty_marker_event(
    recording_id: str, recording_date: str, reason: str
) -> ChronosEventModel:
    """Create a single marker event for genuinely empty recordings."""
    day_of_week = get_day_of_week(recording_date)
    return ChronosEventModel(
        event_id=str(uuid.uuid4()),
        recording_id=recording_id,
        start_ts=parse_ts(f"{recording_date}T12:00:00Z"),
        end_ts=parse_ts(f"{recording_date}T12:00:05Z"),
        day_of_week=day_of_week,
        hour_of_day=12,
        clean_text=f"[Recording marker] {reason}",
        category="unknown",
        sentiment=0.0,
        keywords=["marker", "empty"],
        speaker="unknown",
        raw_transcript_snippet=None,
        gemini_reasoning=None,
    )


def store_events(db, recording_id: str, output: GeminiEventOutput) -> int:
    """Store extracted events to database. Returns count."""
    # Delete any existing events first
    delete_chronos_events_by_recording(db, recording_id)

    db_events = [
        ChronosEventModel(
            event_id=str(uuid.uuid4()),
            recording_id=e.recording_id,
            start_ts=e.start_ts,
            end_ts=e.end_ts,
            day_of_week=(
                e.day_of_week.value
                if hasattr(e.day_of_week, "value")
                else str(e.day_of_week)
            ),
            hour_of_day=e.hour_of_day,
            clean_text=e.clean_text,
            category=(
                e.category.value if hasattr(e.category, "value") else str(e.category)
            ),
            sentiment=e.sentiment,
            keywords=e.keywords,
            speaker=e.speaker.value if hasattr(e.speaker, "value") else str(e.speaker),
            raw_transcript_snippet=getattr(e, "raw_transcript_snippet", None),
            gemini_reasoning=getattr(e, "gemini_reasoning", None),
        )
        for e in output.events
    ]
    add_chronos_events(db, db_events)
    return len(db_events)


def main():
    parser = argparse.ArgumentParser(description="Recover failed recordings")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show plan only, don't execute"
    )
    parser.add_argument(
        "--skip-workflow", action="store_true", help="Skip Plaud Workflow API attempts"
    )
    args = parser.parse_args()

    db = SessionLocal()
    plaud = PlaudClient()
    engine = ChronosEngine()

    # Get all non-completed recordings
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
    all_broken = failed + pending

    print("=" * 70)
    print(f"🔧  RECOVERY PLAN — {len(all_broken)} recordings to fix")
    print("=" * 70)

    # Classify each recording
    categories = {
        "no_transcript_short": [],  # ≤5s with no transcript (empty markers)
        "no_transcript_long": [],  # >5s with no transcript (need Plaud re-fetch)
        "has_transcript_transient": [],  # Has transcript, transient error (retry)
        "has_transcript_noevents": [],  # Has transcript, Gemini returned no events (sparse prompt)
        "zero_duration_local": [],  # 0s local_import (edge case)
    }

    for rec in all_broken:
        t_len = len(rec.transcript) if rec.transcript else 0
        dur = rec.duration_seconds or 0
        err = rec.error_message or ""

        if dur == 0 and rec.source == "local_import":
            categories["zero_duration_local"].append(rec)
        elif t_len == 0 and dur <= 5:
            categories["no_transcript_short"].append(rec)
        elif t_len == 0:
            categories["no_transcript_long"].append(rec)
        elif (
            "Broken pipe" in err
            or "timeout" in err.lower()
            or "connection" in err.lower()
        ):
            categories["has_transcript_transient"].append(rec)
        else:
            categories["has_transcript_noevents"].append(rec)

    # Print plan
    print()
    for cat_name, recs in categories.items():
        if not recs:
            continue
        strategies = {
            "no_transcript_short": "Mark as completed-empty (too short for content)",
            "no_transcript_long": "Re-fetch from Plaud → Workflow API → retry",
            "has_transcript_transient": "Simple retry (transient network error)",
            "has_transcript_noevents": "Retry with RELAXED sparse-content prompt",
            "zero_duration_local": "Mark as completed-empty (no source data)",
        }
        print(f"  [{cat_name.upper()}] {len(recs)} recordings — {strategies[cat_name]}")
        for r in recs:
            t_len = len(r.transcript) if r.transcript else 0
            print(
                f"    • {r.recording_id[:12]}... | {r.duration_seconds or 0}s | {t_len} chars | {r.error_message or 'pending'}"
            )
    print()

    if args.dry_run:
        print("  [DRY RUN] — no changes made")
        db.close()
        return

    results = {"success": 0, "failed": 0, "skipped": 0}

    # ── Phase 1: Short empties (≤5s, no transcript) ─────────────────────
    phase1 = categories["no_transcript_short"] + categories["zero_duration_local"]
    if phase1:
        print("=" * 70)
        print(f"📌 PHASE 1: Marking {len(phase1)} genuinely empty recordings")
        print("=" * 70)

        for rec in phase1:
            recording_date = get_recording_date_str(rec)
            dur = rec.duration_seconds or 0
            reason = f"Recording too short ({dur}s) — likely accidental activation"

            # Still try Plaud re-fetch even for short ones
            print(f"\n  [{rec.recording_id[:12]}...] {dur}s recording")
            print(f"      🔄 Trying Plaud API re-fetch...")
            transcript = extract_transcript_from_plaud(plaud, rec.recording_id)

            if transcript and len(transcript.strip()) > 10:
                print(f"      🎉 Found transcript! {len(transcript)} chars")
                set_chronos_recording_transcript(db, rec.recording_id, transcript)

                # Process through Gemini with sparse prompt
                output = process_with_sparse_prompt(
                    engine, transcript, rec.recording_id, recording_date
                )
                if output and output.events:
                    count = store_events(db, rec.recording_id, output)
                    mark_chronos_recording_status(
                        db, rec.recording_id, "completed", error_message=None
                    )
                    print(f"      ✅ RECOVERED: {count} events!")
                    results["success"] += 1
                    continue

            # Genuinely empty — create marker event
            print(f"      📌 Creating marker event: {reason}")
            delete_chronos_events_by_recording(db, rec.recording_id)
            marker = create_empty_marker_event(rec.recording_id, recording_date, reason)
            add_chronos_events(db, [marker])
            mark_chronos_recording_status(
                db,
                rec.recording_id,
                "completed",
                error_message=None,
            )
            results["success"] += 1
            print(f"      ✅ Marked as completed (empty)")

    # ── Phase 2: Long recordings missing transcripts ────────────────────
    phase2 = categories["no_transcript_long"]
    if phase2:
        print()
        print("=" * 70)
        print(f"🔍 PHASE 2: Recovering transcripts for {len(phase2)} long recordings")
        print("=" * 70)

        for rec in phase2:
            recording_date = get_recording_date_str(rec)
            dur = rec.duration_seconds or 0
            print(f"\n  [{rec.recording_id[:12]}...] {dur}s ({dur // 60}m) recording")

            # Step A: Re-fetch from Plaud API
            print(f"      🔄 Step A: Re-fetching from Plaud API...")
            transcript = extract_transcript_from_plaud(plaud, rec.recording_id)

            if transcript and len(transcript.strip()) > 10:
                print(f"      🎉 Transcript found! {len(transcript):,} chars")
                set_chronos_recording_transcript(db, rec.recording_id, transcript)

                # Decide prompt strategy based on transcript density
                chars_per_min = len(transcript) / max(dur / 60, 1)
                if chars_per_min < 100:  # Very sparse — use relaxed prompt
                    print(
                        f"      ℹ️  Sparse transcript ({chars_per_min:.0f} chars/min) — using relaxed prompt"
                    )
                    output = process_with_sparse_prompt(
                        engine, transcript, rec.recording_id, recording_date
                    )
                else:
                    print(
                        f"      ℹ️  Normal density ({chars_per_min:.0f} chars/min) — using standard prompt"
                    )
                    processor = TranscriptProcessor(db, plaud, engine)
                    output = processor.process_transcript_text(
                        transcript, rec.recording_id, recording_date=recording_date
                    )

                if output and output.events:
                    count = store_events(db, rec.recording_id, output)
                    mark_chronos_recording_status(
                        db, rec.recording_id, "completed", error_message=None
                    )
                    print(f"      ✅ RECOVERED: {count} events!")
                    results["success"] += 1
                    continue
                else:
                    print(f"      ⚠️  Gemini still returned no events after re-fetch")

            # Step B: Try Plaud Workflow API to trigger transcription
            if not args.skip_workflow:
                print(f"      🔄 Step B: Triggering Plaud Workflow transcription...")
                try:
                    workflow = PlaudWorkflowClient()
                    workflow_result = workflow.process_recording(
                        file_id=rec.recording_id,
                        language="en",
                        wait_for_result=True,
                        timeout=300,
                    )
                    if (
                        workflow_result.transcript
                        and len(workflow_result.transcript.strip()) > 10
                    ):
                        transcript = workflow_result.transcript
                        print(
                            f"      🎉 Workflow transcript! {len(transcript):,} chars"
                        )
                        set_chronos_recording_transcript(
                            db, rec.recording_id, transcript
                        )

                        output = process_with_sparse_prompt(
                            engine, transcript, rec.recording_id, recording_date
                        )
                        if output and output.events:
                            count = store_events(db, rec.recording_id, output)
                            mark_chronos_recording_status(
                                db, rec.recording_id, "completed", error_message=None
                            )
                            print(f"      ✅ RECOVERED via Workflow: {count} events!")
                            results["success"] += 1
                            continue
                    else:
                        print(f"      ⚠️  Workflow completed but no transcript returned")
                        if workflow_result.error_message:
                            print(f"          Error: {workflow_result.error_message}")
                except Exception as e:
                    print(f"      ⚠️  Workflow API failed: {str(e)[:100]}")
                    print(
                        f"          (This is expected if Plaud doesn't support workflow for this file)"
                    )

            # Step C: Last resort — check if there's a cached transcript in DB
            if rec.transcript and len(rec.transcript.strip()) > 10:
                print(
                    f"      🔄 Step C: Using cached DB transcript ({len(rec.transcript)} chars)"
                )
                output = process_with_sparse_prompt(
                    engine, rec.transcript, rec.recording_id, recording_date
                )
                if output and output.events:
                    count = store_events(db, rec.recording_id, output)
                    mark_chronos_recording_status(
                        db, rec.recording_id, "completed", error_message=None
                    )
                    print(f"      ✅ RECOVERED from cache: {count} events!")
                    results["success"] += 1
                    continue

            # Still no transcript — mark with detailed error
            print(f"      ❌ No transcript available from any source")
            mark_chronos_recording_status(
                db,
                rec.recording_id,
                "failed",
                error_message="No transcript: Plaud API has no transcript, Workflow API unavailable",
            )
            results["failed"] += 1

    # ── Phase 3: Transient errors (broken pipe, timeout) ────────────────
    phase3 = categories["has_transcript_transient"]
    if phase3:
        print()
        print("=" * 70)
        print(f"🔄 PHASE 3: Retrying {len(phase3)} transient failures")
        print("=" * 70)

        processor = TranscriptProcessor(db, plaud, engine)

        for rec in phase3:
            recording_date = get_recording_date_str(rec)
            dur = rec.duration_seconds or 0
            t_len = len(rec.transcript) if rec.transcript else 0
            print(
                f"\n  [{rec.recording_id[:12]}...] {dur}s ({dur // 60}m) | {t_len:,} chars | was: {rec.error_message}"
            )

            # First try Plaud re-fetch for freshest transcript
            print(f"      🔄 Re-fetching transcript from Plaud...")
            transcript = extract_transcript_from_plaud(plaud, rec.recording_id)
            if transcript and len(transcript) > len(rec.transcript or ""):
                print(
                    f"      📥 Got fresher transcript: {len(transcript):,} chars (was {t_len:,})"
                )
                set_chronos_recording_transcript(db, rec.recording_id, transcript)
            else:
                transcript = rec.transcript

            if not transcript:
                print(f"      ❌ No transcript available!")
                results["failed"] += 1
                continue

            # Retry with standard processor
            print(f"      📤 Retrying Gemini processing...")
            mark_chronos_recording_status(
                db, rec.recording_id, "processing", error_message=None
            )
            output = processor.process_transcript_text(
                transcript, rec.recording_id, recording_date=recording_date
            )

            if output and output.events:
                count = store_events(db, rec.recording_id, output)
                mark_chronos_recording_status(
                    db, rec.recording_id, "completed", error_message=None
                )
                print(f"      ✅ RECOVERED: {count} events!")
                results["success"] += 1
            else:
                # Fallback: try sparse prompt
                print(f"      ⚠️  Standard prompt failed, trying sparse prompt...")
                output = process_with_sparse_prompt(
                    engine, transcript, rec.recording_id, recording_date
                )
                if output and output.events:
                    count = store_events(db, rec.recording_id, output)
                    mark_chronos_recording_status(
                        db, rec.recording_id, "completed", error_message=None
                    )
                    print(f"      ✅ RECOVERED with sparse prompt: {count} events!")
                    results["success"] += 1
                else:
                    mark_chronos_recording_status(
                        db,
                        rec.recording_id,
                        "failed",
                        error_message="Gemini still returned no events after retry",
                    )
                    results["failed"] += 1

    # ── Phase 4: Sparse transcripts (Gemini returned no events) ─────────
    phase4 = categories["has_transcript_noevents"]
    if phase4:
        print()
        print("=" * 70)
        print(
            f"📝 PHASE 4: Re-processing {len(phase4)} sparse transcripts with relaxed prompt"
        )
        print("=" * 70)

        for rec in phase4:
            recording_date = get_recording_date_str(rec)
            dur = rec.duration_seconds or 0
            t_len = len(rec.transcript) if rec.transcript else 0
            print(
                f"\n  [{rec.recording_id[:12]}...] {dur}s ({dur // 60}m) | {t_len:,} chars"
            )

            # Re-fetch from Plaud for freshest transcript
            print(f"      🔄 Re-fetching transcript from Plaud...")
            transcript = extract_transcript_from_plaud(plaud, rec.recording_id)
            if transcript and len(transcript) >= len(rec.transcript or ""):
                if len(transcript) > t_len:
                    print(
                        f"      📥 Got longer transcript: {len(transcript):,} chars (was {t_len:,})"
                    )
                set_chronos_recording_transcript(db, rec.recording_id, transcript)
            else:
                transcript = rec.transcript

            if not transcript or len(transcript.strip()) < 5:
                print(f"      ❌ No usable transcript")
                # Create marker event
                marker = create_empty_marker_event(
                    rec.recording_id,
                    recording_date,
                    f"Recording has {t_len} char transcript but no extractable content",
                )
                delete_chronos_events_by_recording(db, rec.recording_id)
                add_chronos_events(db, [marker])
                mark_chronos_recording_status(
                    db, rec.recording_id, "completed", error_message=None
                )
                results["success"] += 1
                continue

            # Show transcript preview
            preview = transcript[:200].replace("\n", " ")
            print(f"      📋 Preview: {preview}...")

            # Process with relaxed prompt
            mark_chronos_recording_status(
                db, rec.recording_id, "processing", error_message=None
            )
            output = process_with_sparse_prompt(
                engine, transcript, rec.recording_id, recording_date
            )

            if output and output.events:
                count = store_events(db, rec.recording_id, output)
                mark_chronos_recording_status(
                    db, rec.recording_id, "completed", error_message=None
                )
                print(f"      ✅ RECOVERED: {count} events!")
                results["success"] += 1
            else:
                # Nuclear option: create a single event with the full transcript as content
                print(
                    f"      ⚠️  Gemini still returned nothing — creating manual event from transcript"
                )
                day_of_week = get_day_of_week(recording_date)
                manual_event = ChronosEventModel(
                    event_id=str(uuid.uuid4()),
                    recording_id=rec.recording_id,
                    start_ts=parse_ts(f"{recording_date}T12:00:00Z"),
                    end_ts=parse_ts(f"{recording_date}T12:30:00Z"),
                    day_of_week=day_of_week,
                    hour_of_day=12,
                    clean_text=transcript[:5000].strip(),  # Cap at 5k chars
                    category="personal",
                    sentiment=0.0,
                    keywords=transcript.split()[:10],
                    speaker="unknown",
                    raw_transcript_snippet=transcript[:500],
                    gemini_reasoning="Raw transcript — Gemini could not extract structured events",
                )
                delete_chronos_events_by_recording(db, rec.recording_id)
                add_chronos_events(db, [manual_event])
                mark_chronos_recording_status(
                    db, rec.recording_id, "completed", error_message=None
                )
                print(f"      ✅ Created manual event from raw transcript")
                results["success"] += 1

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"🏁 RECOVERY COMPLETE")
    print(f"   ✅ Recovered: {results['success']}")
    print(f"   ❌ Still failed: {results['failed']}")
    print(f"   ⏭️  Skipped: {results['skipped']}")
    print("=" * 70)

    # Final status check
    remaining_failed = (
        db.query(ChronosRecording)
        .filter(ChronosRecording.processing_status.in_(["failed", "pending"]))
        .count()
    )
    total = db.query(ChronosRecording).count()
    completed = (
        db.query(ChronosRecording)
        .filter(ChronosRecording.processing_status == "completed")
        .count()
    )

    print(
        f"\n   📊 Final Status: {completed}/{total} completed, {remaining_failed} remaining"
    )

    if remaining_failed == 0:
        print(f"   🎉 ZERO DATA GAPS — all recordings accounted for!")
    else:
        remaining = (
            db.query(ChronosRecording)
            .filter(ChronosRecording.processing_status.in_(["failed", "pending"]))
            .all()
        )
        print(f"\n   Still broken:")
        for r in remaining:
            print(
                f"     • {r.recording_id[:12]}... | {r.duration_seconds or 0}s | {r.error_message}"
            )

    db.close()


if __name__ == "__main__":
    main()
