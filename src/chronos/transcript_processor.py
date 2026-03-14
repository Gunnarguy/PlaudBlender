"""
Chronos Transcript Processor - Process Plaud transcripts without audio.

Since Plaud API doesn't provide audio downloads (presigned_url is null),
this module processes transcripts directly through Gemini for event extraction.
"""

import json
import logging
import time as _time
import uuid
from typing import Optional

from sqlalchemy.orm import Session
from pydantic import ValidationError
from google.genai import types

from src.config import get_settings
from src.plaud_client import PlaudClient
from src.database.chronos_repository import (
    get_pending_chronos_recordings,
    mark_chronos_recording_status,
    add_chronos_events,
    set_chronos_recording_transcript,
    upsert_chronos_recording,
    get_chronos_recording,
    delete_chronos_events_by_recording,
)
from src.database.models import ChronosEvent as ChronosEventModel
from src.models.chronos_schemas import ChronosEvent
from src.chronos.engine import ChronosEngine, GeminiEventOutput
from src.chronos.genai_helpers import normalize_thinking_level

logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Process Plaud transcripts through Chronos engine."""

    def __init__(
        self,
        db_session: Session,
        plaud_client: Optional[PlaudClient] = None,
        engine: Optional[ChronosEngine] = None,
    ):
        self.db = db_session
        self.plaud = plaud_client or PlaudClient()
        self.engine = engine or ChronosEngine()
        self.settings = get_settings()

    def _extract_transcript(self, file_details: dict) -> Optional[str]:
        """Extract transcript text from Plaud file details.

        Args:
            file_details: Response from get_recording API call

        Returns:
            Combined transcript text, or None if not available
        """
        source_list = file_details.get("source_list", [])

        # Find transaction (transcript) data
        for source in source_list:
            if source.get("data_type") == "transaction":
                try:
                    # data_content is JSON string with transcript segments
                    segments = json.loads(source.get("data_content", "[]"))
                    # Combine all segment content
                    texts = [seg.get("content", "") for seg in segments]
                    return " ".join(texts).strip()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse transcript JSON: {e}")
                    return None

        return None

    def _repair_json_with_gemini(self, broken_json: str, recording_id: str) -> str:
        """Repair a near-JSON Gemini response into strict JSON.

        We only call this after a JSONDecodeError. The goal is to transform
        almost-valid output (e.g., trailing commas, single quotes, unescaped
        newlines) into strict RFC8259 JSON.

        NOTE: This is a best-effort fallback. If it fails, we re-raise.
        """
        # Avoid sending extremely large payloads back to the model.
        # (Also helps keep cost/latency sane.)
        max_chars = 40_000
        snippet = broken_json[:max_chars]

        repair_prompt = f"""You are a JSON repair tool.

Fix the following content into STRICT valid JSON (RFC8259). Requirements:
- Output ONLY the repaired JSON object (no markdown, no commentary)
- Use double quotes for all keys/strings
- No trailing commas
- Preserve all fields/structure as-is unless required to make JSON valid

RECORDING_ID: {recording_id}

BROKEN_JSON:
{snippet}
"""

        model_name = (
            getattr(self.settings, "chronos_analyst_model", None)
            or self.engine.model_name
        )
        thinking_level = normalize_thinking_level(
            getattr(self.settings, "chronos_thinking_level", "")
        )
        from app_v2.services.xray import xray_log
        xray_log("gemini", "json-repair",
                 f"Gemini's answer was garbled — asking it to try again ({len(snippet):,} chars)")
        _t0 = _time.perf_counter()
        try:
            config: dict = {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
            if thinking_level is not None:
                config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )

            resp = self.engine.client.models.generate_content(
                model=model_name,
                contents=repair_prompt,
                config=config,
            )
            _ms = (_time.perf_counter() - _t0) * 1000
            repaired = (resp.text or "").strip()
            xray_log("gemini", "json-repair",
                     f"Gemini cleaned up its own mess ({len(repaired):,} chars)",
                     duration_ms=round(_ms, 1))
            return repaired
        except Exception as e:
            _ms = (_time.perf_counter() - _t0) * 1000
            xray_log("gemini", "json-repair",
                     f"Couldn't salvage it: {str(e)[:60]}",
                     duration_ms=round(_ms, 1), level="error")
            logger.error(f"JSON repair call failed (model={model_name}): {e}")
            return broken_json

    def process_transcript_text(
        self,
        transcript_text: str,
        recording_id: str,
        max_retries: int = 3,
        verbose: bool = True,
        recording_date: str = "",
        plaud_context: Optional[str] = None,
    ) -> Optional[GeminiEventOutput]:
        """Process transcript text through Gemini (modified for text input).

        Args:
            transcript_text: Raw transcript from Plaud
            recording_id: Recording ID
            max_retries: Number of retries on failure
            verbose: Print detailed progress to stdout
            recording_date: ISO date for temporal anchoring
            plaud_context: Optional Plaud AI Summary/ETL output to guide extraction

        Returns:
            GeminiEventOutput with extracted events
        """
        # Skip transcripts that are too short to extract meaningful events
        MIN_TRANSCRIPT_CHARS = 100
        if len(transcript_text.strip()) < MIN_TRANSCRIPT_CHARS:
            if verbose:
                print(
                    f"      ⚠️ Transcript too short ({len(transcript_text.strip())} chars < {MIN_TRANSCRIPT_CHARS}), skipping",
                    flush=True,
                )
            from app_v2.services.xray import xray_log
            xray_log("gemini", "skip",
                     f"Only {len(transcript_text.strip())} characters — too short for Gemini to work with")
            logger.warning(
                f"Skipping {recording_id}: transcript too short ({len(transcript_text.strip())} chars)"
            )
            return None

        # Build prompt (same as audio version)
        prompt = self.engine._build_prompt(recording_id, recording_date)

        # Inject Plaud AI context if available (improves categorization accuracy)
        plaud_section = ""
        if plaud_context:
            plaud_section = (
                "\n\n**PLAUD AI CONTEXT** (use this to guide categorization, "
                "sentiment, and structure — but always extract events from the "
                "raw transcript below):\n\n"
                f"{plaud_context}\n"
            )

        # Combine prompt with transcript
        full_prompt = f"""{prompt}{plaud_section}

**RAW TRANSCRIPT:**

{transcript_text}

Extract events from this transcript following the schema exactly."""

        # Show what we're sending to Gemini
        if verbose:
            transcript_words = len(transcript_text.split())
            transcript_chars = len(transcript_text)
            print(
                f"      📊 Transcript: {transcript_words:,} words, {transcript_chars:,} chars",
                flush=True,
            )
            print(f"      🤖 Model: {self.engine.model_name}", flush=True)
            print(f"      📤 Sending to Gemini API...", flush=True)

        from app_v2.services.xray import xray_log
        _prompt_words = len(transcript_text.split())
        _prompt_chars = len(transcript_text)
        xray_log("gemini", "prompt",
                 f"Feeding {_prompt_words:,} words to Gemini — 'read this and tell me what happened'")

        for attempt in range(max_retries):
            try:
                if verbose and attempt > 0:
                    print(
                        f"      🔄 Retry attempt {attempt + 1}/{max_retries}...",
                        flush=True,
                    )

                logger.info(
                    f"Processing transcript for {recording_id} (attempt {attempt + 1}/{max_retries})..."
                )

                config: dict = {
                    "response_mime_type": "application/json",
                    "response_json_schema": GeminiEventOutput.model_json_schema(),
                    "temperature": 0.2,
                }
                if self.engine._thinking_level is not None:
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.engine._thinking_level
                    )

                # Use streaming to show real-time progress
                STREAM_TIMEOUT = 600  # 10 min max for any single Gemini call
                if verbose:
                    response_text = ""
                    token_count = 0
                    events_found = 0
                    last_print_len = 0
                    start_time = __import__("time").time()
                    last_chunk_time = start_time

                    stream = self.engine.client.models.generate_content_stream(
                        model=self.engine.model_name,
                        contents=full_prompt,
                        config=config,
                    )

                    for chunk in stream:
                        now = __import__("time").time()
                        # Timeout: if total elapsed exceeds limit, abort
                        if now - start_time > STREAM_TIMEOUT:
                            print(
                                f"\n      ⚠️ Stream timeout ({STREAM_TIMEOUT}s), using partial response",
                                flush=True,
                            )
                            break
                        last_chunk_time = now
                        chunk_text = chunk.text or ""
                        response_text += chunk_text
                        token_count += len(chunk_text.split())  # Rough estimate

                        # Count events as we see them in the stream
                        events_found = response_text.count('"event_id"')

                        # Update progress every ~500 chars
                        if len(response_text) - last_print_len > 500:
                            elapsed = __import__("time").time() - start_time
                            # Show streaming progress with event count
                            print(
                                f"\r      📝 Streaming: {len(response_text):,} chars | {events_found} events | {elapsed:.0f}s",
                                end="",
                                flush=True,
                            )
                            last_print_len = len(response_text)

                    # Final newline after streaming
                    elapsed = __import__("time").time() - start_time
                    print(
                        f"\r      📝 Streaming: {len(response_text):,} chars | {events_found} events | {elapsed:.0f}s ✓",
                        flush=True,
                    )

                    xray_log("gemini", "stream",
                             f"Gemini read it all — wrote {len(response_text):,} chars and spotted {events_found} moments",
                             duration_ms=round(elapsed * 1000, 1))

                    # Get final usage from last chunk if available
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        _in_tok = getattr(usage, 'prompt_token_count', 0)
                        _out_tok = getattr(usage, 'candidates_token_count', 0)
                        print(
                            f"      📊 Tokens - Input: {_in_tok:,} | Output: {_out_tok:,}",
                            flush=True,
                        )
                        xray_log("gemini", "tokens",
                                 f"Gemini used {_in_tok:,} words reading + {_out_tok:,} words writing = {_in_tok + _out_tok:,} total")

                    # Parse the accumulated response
                    raw_text = response_text.strip()
                else:
                    # Non-verbose: use regular call
                    response = self.engine.client.models.generate_content(
                        model=self.engine.model_name,
                        contents=full_prompt,
                        config=config,
                    )
                    raw_text = (response.text or "").strip()

                    # Check for structured output in non-verbose mode
                    parsed = getattr(response, "parsed", None)
                    if parsed is not None:
                        validated = GeminiEventOutput(**parsed)
                        logger.info(
                            f"Extracted {validated.total_events} events from transcript"
                        )
                        return validated

                if verbose:
                    print(
                        f"      🔍 Parsing JSON ({len(raw_text):,} chars)...",
                        flush=True,
                    )

                # Handle markdown code fences (Gemini sometimes wraps JSON)
                if raw_text.startswith("```"):
                    parts = raw_text.split("```")
                    if len(parts) >= 2:
                        raw_text = parts[1].strip()
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:].strip()

                # Best-effort: extract the first JSON object from the response.
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError("Gemini response did not contain a JSON object")

                raw_json = raw_text[start : end + 1]

                try:
                    output_data = json.loads(raw_json)
                except json.JSONDecodeError:
                    # One more attempt: ask a "thinking" model to repair the JSON.
                    repaired = self._repair_json_with_gemini(raw_json, recording_id)

                    # The repair response may still contain fences; reuse the same extraction.
                    repaired_text = repaired.strip()
                    if repaired_text.startswith("```"):
                        parts = repaired_text.split("```")
                        if len(parts) >= 2:
                            repaired_text = parts[1].strip()
                            if repaired_text.startswith("json"):
                                repaired_text = repaired_text[4:].strip()

                    rs = repaired_text.find("{")
                    re = repaired_text.rfind("}")
                    if rs == -1 or re == -1 or re <= rs:
                        raise ValueError(
                            "JSON repair response did not contain a JSON object"
                        )

                    output_data = json.loads(repaired_text[rs : re + 1])

                validated = GeminiEventOutput(**output_data)

                if verbose:
                    print(
                        f"      ✅ Extracted {validated.total_events} events",
                        flush=True,
                    )
                    self._print_event_summary(validated.events)

                # Log per-event category breakdown
                from collections import Counter as _Counter
                _cats = _Counter()
                for _ev in validated.events:
                    _c = getattr(_ev, 'category', 'unknown')
                    _cats[_c.value if hasattr(_c, 'value') else str(_c)] += 1
                _cat_str = ', '.join(f"{c}:{n}" for c, n in _cats.most_common(5))
                xray_log("gemini", "extract",
                         f"Pulled out {validated.total_events} moments — {_cat_str}")

                logger.info(
                    f"Extracted {validated.total_events} events from transcript"
                )
                return validated

            except ValidationError as e:
                if verbose:
                    print(f"      ❌ Validation error: {str(e)[:100]}", flush=True)
                xray_log("gemini", "error",
                         f"Gemini's answer didn't make sense — couldn't use it",
                         level="error")
                logger.error(f"Pydantic validation failed: {e}")
                return None

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"      ⚠️ JSON parse error, retrying...", flush=True)
                xray_log("gemini", "retry",
                         f"Gemini gave a weird answer, trying again ({attempt + 1}/{max_retries})",
                         level="warn")
                logger.error(f"JSON parse error: {e}")
                if attempt < max_retries - 1:
                    continue
                return None

            except Exception as e:
                if verbose:
                    print(f"      ❌ Error: {str(e)[:80]}", flush=True)
                xray_log("gemini", "error",
                         f"Something went wrong with Gemini: {str(e)[:60]}",
                         level="error")
                logger.error(f"Failed to process transcript: {e}")
                if attempt < max_retries - 1:
                    continue
                return None

        return None

    def _print_event_summary(self, events: list) -> None:
        """Print a summary of extracted events."""
        if not events:
            return

        # Show first 3 events as preview
        print(f"      📋 Event preview:", flush=True)
        for i, e in enumerate(events[:3]):
            category = getattr(e, "category", "unknown")
            if hasattr(category, "value"):
                category = category.value
            clean_text = getattr(e, "clean_text", "")[:60]
            sentiment = getattr(e, "sentiment", 0)
            print(
                f"         {i+1}. [{category}] {clean_text}... (sentiment: {sentiment:.1f})",
                flush=True,
            )

        if len(events) > 3:
            print(f"         ... and {len(events) - 3} more events", flush=True)

        # Category breakdown
        from collections import Counter

        categories = Counter()
        for e in events:
            cat = getattr(e, "category", "unknown")
            if hasattr(cat, "value"):
                cat = cat.value
            categories[cat] += 1

        cat_summary = ", ".join(
            f"{cat}: {count}" for cat, count in categories.most_common(5)
        )
        print(f"      📊 Categories: {cat_summary}", flush=True)

    def process_pending_recordings(
        self, limit: Optional[int] = None
    ) -> tuple[int, int]:
        """Process all pending recordings using their transcripts.

        Args:
            limit: Maximum number to process

        Returns:
            Tuple of (success_count, failure_count)
        """
        pending = get_pending_chronos_recordings(self.db, limit=limit or 100)

        logger.info(f"Found {len(pending)} pending recordings")

        success_count = 0
        failure_count = 0

        for rec in pending:
            ok = self.process_recording_id(rec.recording_id)
            if ok:
                success_count += 1
            else:
                failure_count += 1

        logger.info(
            f"Processing complete: {success_count} success, {failure_count} failures"
        )
        return (success_count, failure_count)

    def process_recording_id(
        self,
        recording_id: str,
        *,
        delete_existing_events: bool = False,
    ) -> bool:
        """Process a single recording by ID.

        This is used by the UI to reprocess an individual recording on demand.
        """

        rec = get_chronos_recording(self.db, recording_id)
        if not rec:
            logger.error(f"Recording not found in Chronos DB: {recording_id}")
            return False

        try:
            from app_v2.services.xray import xray_log
            _proc_t0 = _time.perf_counter()

            # Mark as in-progress early so we can spot crashes mid-batch.
            mark_chronos_recording_status(
                self.db, rec.recording_id, "processing", error_message=None
            )
            xray_log("gemini", "start",
                     f"Handing this recording to Gemini AI")

            if delete_existing_events:
                deleted = delete_chronos_events_by_recording(self.db, recording_id)
                logger.info(f"Deleted {deleted} existing events for {recording_id}")

            # Fetch file details from Plaud API
            _api_t0 = _time.perf_counter()
            file_details = self.plaud.get_recording(rec.recording_id)
            _api_ms = (_time.perf_counter() - _api_t0) * 1000
            xray_log("ingest", "plaud-api",
                     f"Got the recording info from Plaud",
                     duration_ms=round(_api_ms, 1))

            # Best-effort: refresh the recording title from Plaud if present.
            try:
                plaud_title = file_details.get("title")
                if plaud_title and (not getattr(rec, "title", None)):
                    upsert_chronos_recording(
                        session=self.db,
                        recording_id=rec.recording_id,
                        title=plaud_title,
                        created_at=rec.created_at,
                        duration_seconds=rec.duration_seconds,
                        local_audio_path=rec.local_audio_path,
                        source=rec.source,
                        device_id=rec.device_id,
                        checksum=rec.checksum,
                    )
            except Exception:
                pass

            # Extract transcript
            transcript_text = self._extract_transcript(file_details)

            # Cache transcript for UI/library browsing.
            if transcript_text:
                try:
                    set_chronos_recording_transcript(
                        self.db, rec.recording_id, transcript_text
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to cache transcript for {rec.recording_id}: {e}"
                    )

            if not transcript_text:
                logger.warning(f"No transcript for {rec.recording_id}")
                xray_log("gemini", "skip",
                         f"This recording has no transcript — nothing to analyze",
                         level="warn")
                mark_chronos_recording_status(
                    self.db,
                    rec.recording_id,
                    "failed",
                    error_message="No transcript available in Plaud source_list",
                )
                return False

            # Process through Gemini — pass real recording date for temporal anchoring
            recording_date = ""
            if rec.created_at:
                try:
                    from datetime import datetime as dt_cls

                    if isinstance(rec.created_at, str):
                        recording_date = rec.created_at[:10]
                    else:
                        recording_date = rec.created_at.strftime("%Y-%m-%d")
                except Exception:
                    pass

            # Build Plaud AI context if available (summary + extracted data)
            plaud_context = None
            try:
                parts = []
                ai_summary = getattr(rec, "plaud_ai_summary", None)
                if ai_summary:
                    parts.append(f"AI Summary: {ai_summary}")
                extracted = getattr(rec, "plaud_extracted_data", None)
                if extracted and isinstance(extracted, dict):
                    import json as _json

                    parts.append(
                        f"Extracted Data: {_json.dumps(extracted, default=str)[:2000]}"
                    )
                if parts:
                    plaud_context = "\n\n".join(parts)
            except Exception:
                pass

            output = self.process_transcript_text(
                transcript_text,
                rec.recording_id,
                recording_date=recording_date,
                plaud_context=plaud_context,
            )

            if not output or not output.events:
                logger.warning(f"No events extracted for {rec.recording_id}")
                xray_log("gemini", "fail",
                         f"Gemini couldn't find anything meaningful in this recording",
                         level="warn")
                mark_chronos_recording_status(
                    self.db,
                    rec.recording_id,
                    "failed",
                    error_message="Gemini returned no events",
                )
                return False

            # Store events in database (convert Pydantic schema -> ORM model)
            # Generate real UUIDs instead of using Gemini's placeholder values
            db_events = [
                ChronosEventModel(
                    event_id=str(uuid.uuid4()),  # REAL UUID, not Gemini's placeholder
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
                        e.category.value
                        if hasattr(e.category, "value")
                        else str(e.category)
                    ),
                    sentiment=e.sentiment,
                    keywords=e.keywords,
                    speaker=(
                        e.speaker.value
                        if hasattr(e.speaker, "value")
                        else str(e.speaker)
                    ),
                    raw_transcript_snippet=e.raw_transcript_snippet,
                    gemini_reasoning=e.gemini_reasoning,
                )
                for e in output.events
            ]
            add_chronos_events(self.db, db_events)

            # Update status
            mark_chronos_recording_status(
                self.db, rec.recording_id, "completed", error_message=None
            )

            _proc_ms = (_time.perf_counter() - _proc_t0) * 1000
            xray_log("gemini", "done",
                     f"Done! Found {len(output.events)} moments in this recording",
                     duration_ms=round(_proc_ms, 1))

            logger.info(f"✓ Processed {rec.recording_id}: {len(output.events)} events")
            return True

        except Exception as e:
            logger.error(f"Failed to process {rec.recording_id}: {e}")
            xray_log("gemini", "error",
                     f"This recording crashed the processor: {str(e)[:60]}",
                     level="error")
            mark_chronos_recording_status(
                self.db,
                rec.recording_id,
                "failed",
                error_message=str(e),
            )
            return False
