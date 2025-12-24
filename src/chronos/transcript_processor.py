"""
Chronos Transcript Processor - Process Plaud transcripts without audio.

Since Plaud API doesn't provide audio downloads (presigned_url is null),
this module processes transcripts directly through Gemini for event extraction.
"""

import json
import logging
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
            return (resp.text or "").strip()
        except Exception as e:
            # If repair fails for any reason, bubble up the original JSON error
            # by returning the unmodified content (caller will re-attempt parse and fail).
            logger.error(f"JSON repair call failed (model={model_name}): {e}")
            return broken_json

    def process_transcript_text(
        self,
        transcript_text: str,
        recording_id: str,
        max_retries: int = 3,
    ) -> Optional[GeminiEventOutput]:
        """Process transcript text through Gemini (modified for text input).

        Args:
            transcript_text: Raw transcript from Plaud
            recording_id: Recording ID

        Returns:
            GeminiEventOutput with extracted events
        """
        # Build prompt (same as audio version)
        prompt = self.engine._build_prompt(recording_id)

        # Combine prompt with transcript
        full_prompt = f"""{prompt}

**RAW TRANSCRIPT:**

{transcript_text}

Extract events from this transcript following the schema exactly."""

        for attempt in range(max_retries):
            try:
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

                response = self.engine.client.models.generate_content(
                    model=self.engine.model_name,
                    contents=full_prompt,
                    config=config,
                )

                # Parse and validate
                # If Structured Outputs parsing is available, prefer it.
                parsed = getattr(response, "parsed", None)
                if parsed is not None:
                    validated = GeminiEventOutput(**parsed)
                    logger.info(
                        f"Extracted {validated.total_events} events from transcript"
                    )
                    return validated

                raw_text = (response.text or "").strip()

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

                logger.info(
                    f"Extracted {validated.total_events} events from transcript"
                )
                return validated

            except ValidationError as e:
                logger.error(f"Pydantic validation failed: {e}")
                return None

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                if attempt < max_retries - 1:
                    continue
                return None

            except Exception as e:
                logger.error(f"Failed to process transcript: {e}")
                if attempt < max_retries - 1:
                    continue
                return None

        return None

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
            # Mark as in-progress early so we can spot crashes mid-batch.
            mark_chronos_recording_status(
                self.db, rec.recording_id, "processing", error_message=None
            )

            if delete_existing_events:
                deleted = delete_chronos_events_by_recording(self.db, recording_id)
                logger.info(f"Deleted {deleted} existing events for {recording_id}")

            # Fetch file details from Plaud API
            file_details = self.plaud.get_recording(rec.recording_id)

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
                mark_chronos_recording_status(
                    self.db,
                    rec.recording_id,
                    "failed",
                    error_message="No transcript available in Plaud source_list",
                )
                return False

            # Process through Gemini
            output = self.process_transcript_text(transcript_text, rec.recording_id)

            if not output or not output.events:
                logger.warning(f"No events extracted for {rec.recording_id}")
                mark_chronos_recording_status(
                    self.db,
                    rec.recording_id,
                    "failed",
                    error_message="Gemini returned no events",
                )
                return False

            # Store events in database (convert Pydantic schema -> ORM model)
            db_events = [
                ChronosEventModel(
                    event_id=e.event_id,
                    recording_id=e.recording_id,
                    start_ts=e.start_ts,
                    end_ts=e.end_ts,
                    day_of_week=str(e.day_of_week),
                    hour_of_day=e.hour_of_day,
                    clean_text=e.clean_text,
                    category=str(e.category),
                    sentiment=e.sentiment,
                    keywords=e.keywords,
                    speaker=str(e.speaker),
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

            logger.info(f"✓ Processed {rec.recording_id}: {len(output.events)} events")
            return True

        except Exception as e:
            logger.error(f"Failed to process {rec.recording_id}: {e}")
            mark_chronos_recording_status(
                self.db,
                rec.recording_id,
                "failed",
                error_message=str(e),
            )
            return False
