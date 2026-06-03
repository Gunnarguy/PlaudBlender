"""
Chronos Transcript Processor - Process Plaud transcripts without audio.

Since Plaud API doesn't provide audio downloads (presigned_url is null),
this module processes transcripts directly through Gemini for event extraction.
"""

import json
import logging
import time as _time
import uuid
from typing import Any, Callable, Optional

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
from src.models.chronos_schemas import ChronosEvent, EventCategory
from src.chronos.engine import CHRONOS_CLEAN_PROMPT, ChronosEngine, GeminiEventOutput
from src.chronos.openai_service import OpenAIResponseService
from src.chronos.genai_helpers import (
    is_model_not_found,
    is_model_temporarily_unavailable,
    is_permission_denied,
    normalize_thinking_level,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


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
        self.engine = engine
        self.settings = get_settings()
        self._last_processing_error: Optional[str] = None

    def _get_engine(self) -> ChronosEngine:
        if self.engine is None:
            self.engine = ChronosEngine()
        return self.engine

    def _build_prompt(self, recording_id: str, recording_date: str = "") -> str:
        if recording_date:
            prompt_date = recording_date
        else:
            from datetime import datetime

            prompt_date = datetime.now().strftime("%Y-%m-%d")

        if self.engine is not None:
            return self.engine._build_prompt(recording_id, prompt_date)

        prompt = CHRONOS_CLEAN_PROMPT.replace("{{RECORDING_ID}}", recording_id)
        return prompt.replace("{{RECORDING_DATE}}", prompt_date)

    def _get_json_repair_model_name(self) -> str:
        """Choose the cheapest model that can safely retry malformed JSON."""
        analyst_model = (
            getattr(self.settings, "chronos_analyst_model", "") or ""
        ).strip()
        if (
            getattr(self.settings, "chronos_allow_paid_gemini_fallback", False)
            and analyst_model
        ):
            return analyst_model
        return self._get_engine().model_name

    def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        step: str,
        detail: str = "",
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(step, detail)
        except Exception:
            logger.debug("Progress callback failed", exc_info=True)

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

    def _repair_json_with_gemini(
        self,
        broken_json: str,
        recording_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
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

        model_name = self._get_json_repair_model_name()
        thinking_level = normalize_thinking_level(
            getattr(self.settings, "chronos_thinking_level", "")
        )
        self._emit_progress(
            progress_callback,
            "JSON repair",
            f"Repairing malformed Gemini output ({len(snippet):,} chars)",
        )
        from app_v2.services.xray import xray_log

        xray_log(
            "gemini",
            "json-repair",
            f"Gemini's answer was garbled — asking it to try again ({len(snippet):,} chars)",
        )
        _t0 = _time.perf_counter()
        try:
            engine = self._get_engine()
            config: Any = {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
            if thinking_level is not None:
                config["thinking_config"] = types.ThinkingConfig(
                    thinking_level=thinking_level
                )

            resp = engine.client.models.generate_content(
                model=model_name,
                contents=repair_prompt,
                config=config,
            )
            _ms = (_time.perf_counter() - _t0) * 1000
            repaired = (resp.text or "").strip()
            self._emit_progress(
                progress_callback,
                "JSON repaired",
                f"Recovered {len(repaired):,} chars of valid JSON",
            )
            xray_log(
                "gemini",
                "json-repair",
                f"Gemini cleaned up its own mess ({len(repaired):,} chars)",
                duration_ms=round(_ms, 1),
            )
            # Track cost for JSON repair call
            _repair_usage = getattr(resp, "usage_metadata", None)
            if _repair_usage:
                from src.chronos.cost_tracker import track_usage

                track_usage(
                    model_name,
                    "generate",
                    input_tokens=getattr(_repair_usage, "prompt_token_count", 0),
                    output_tokens=getattr(_repair_usage, "candidates_token_count", 0),
                )
            return repaired
        except Exception as e:
            _ms = (_time.perf_counter() - _t0) * 1000
            self._emit_progress(
                progress_callback,
                "JSON repair failed",
                str(e)[:80],
            )
            xray_log(
                "gemini",
                "json-repair",
                f"Couldn't salvage it: {str(e)[:60]}",
                duration_ms=round(_ms, 1),
                level="error",
            )
            logger.error(f"JSON repair call failed (model={model_name}): {e}")
            return broken_json

    def _get_processing_provider(self) -> str:
        provider = (
            (getattr(self.settings, "chronos_processing_provider", "gemini") or "gemini")
            .strip()
            .lower()
        )
        if provider == "auto":
            if getattr(self.settings, "gemini_api_key", None):
                return "gemini"
            if self._local_processing_available():
                return "local"
            if self._openai_processing_available():
                return "openai"
            return "local"
        if provider == "openai" and not self._openai_processing_available():
            if self._local_processing_available():
                return "local"
            if getattr(self.settings, "gemini_api_key", None):
                return "gemini"
        if provider not in {"gemini", "openai", "local"}:
            if getattr(self.settings, "gemini_api_key", None):
                return "gemini"
            if self._local_processing_available():
                return "local"
            return "gemini"
        return provider

    def _provider_label(self, provider: Optional[str] = None) -> str:
        resolved = (provider or self._get_processing_provider()).strip().lower()
        if resolved == "gemini":
            return "Gemini"
        if resolved == "local":
            return "Local"
        return "OpenAI"

    def _openai_processing_available(self) -> bool:
        if not bool(getattr(self.settings, "chronos_openai_enabled", False)):
            return False
        api_key = getattr(self.settings, "openai_api_key", None)
        return bool(str(api_key).strip()) if api_key is not None else False

    def _local_processing_available(self) -> bool:
        return bool(getattr(self.settings, "chronos_local_llm_enabled", False))

    def _parse_recording_start(self, recording_date: str):
        from datetime import datetime, timezone

        raw = (recording_date or "").strip()
        if raw:
            try:
                if len(raw) == 10:
                    return datetime.fromisoformat(raw).replace(hour=9, tzinfo=timezone.utc)
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)

    @staticmethod
    def _chunk_transcript_words(transcript_text: str, chunk_words: int = 420) -> list[str]:
        words = transcript_text.split()
        if not words:
            return []
        return [" ".join(words[index : index + chunk_words]) for index in range(0, len(words), chunk_words)]

    @staticmethod
    def _local_category_for_text(text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ("meeting", "call", "discussed", "talked with", "conversation")):
            return EventCategory.MEETING.value
        if any(term in lowered for term in ("bug", "code", "api", "repo", "deploy", "database", "project", "client")):
            return EventCategory.WORK.value
        if any(term in lowered for term in ("lunch", "break", "walk", "coffee", "sleep", "rest")):
            return EventCategory.BREAK.value
        if any(term in lowered for term in ("idea", "maybe", "could", "what if", "brainstorm")):
            return EventCategory.IDEA.value
        if any(term in lowered for term in ("feel", "felt", "thinking about", "realized", "reflect")):
            return EventCategory.REFLECTION.value
        return EventCategory.UNKNOWN.value

    @staticmethod
    def _local_keywords_for_text(text: str, limit: int = 8) -> list[str]:
        import re
        from collections import Counter

        stopwords = {
            "about", "after", "again", "also", "because", "before", "being", "could",
            "from", "have", "into", "just", "like", "more", "really", "that", "their",
            "then", "there", "these", "they", "this", "through", "with", "would",
        }
        words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
        counts = Counter(word for word in words if word not in stopwords)
        return [word for word, _count in counts.most_common(limit)]

    def process_transcript_text(
        self,
        transcript_text: str,
        recording_id: str,
        max_retries: int = 3,
        verbose: bool = True,
        recording_date: str = "",
        plaud_context: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[GeminiEventOutput]:
        """Process transcript text via the configured provider."""
        self._last_processing_error = None

        # Skip transcripts that are too short to extract meaningful events.
        if len(transcript_text.strip()) < 100:
            self._last_processing_error = (
                "Transcript too short to extract meaningful events"
            )
            return None

        provider = self._get_processing_provider()
        if provider == "gemini":
            return self._process_transcript_text_gemini(
                transcript_text,
                recording_id,
                max_retries=max_retries,
                verbose=verbose,
                recording_date=recording_date,
                plaud_context=plaud_context,
                progress_callback=progress_callback,
            )

        if provider == "local":
            return self._process_transcript_text_local(
                transcript_text,
                recording_id,
                verbose=verbose,
                recording_date=recording_date,
                plaud_context=plaud_context,
                progress_callback=progress_callback,
            )

        if provider == "openai":
            return self._process_transcript_text_openai(
                transcript_text,
                recording_id,
                verbose=verbose,
                recording_date=recording_date,
                plaud_context=plaud_context,
                progress_callback=progress_callback,
            )

        gemini_output = self._process_transcript_text_gemini(
            transcript_text,
            recording_id,
            max_retries=max_retries,
            verbose=verbose,
            recording_date=recording_date,
            plaud_context=plaud_context,
            progress_callback=progress_callback,
        )
        if gemini_output and gemini_output.events:
            return gemini_output

        gemini_error = self._last_processing_error
        from app_v2.services.xray import xray_log

        xray_log(
            "pipeline",
            "fallback",
            "Gemini stalled or failed — trying OpenAI instead",
            level="warn",
        )

        if not self._openai_processing_available():
            self._last_processing_error = gemini_error or (
                "Gemini failed and OpenAI fallback is unavailable because OPENAI_API_KEY is not configured"
            )
            return None

        self._last_processing_error = None
        openai_output = self._process_transcript_text_openai(
            transcript_text,
            recording_id,
            verbose=verbose,
            recording_date=recording_date,
            plaud_context=plaud_context,
            progress_callback=progress_callback,
        )
        if openai_output and openai_output.events:
            return openai_output

        openai_error = self._last_processing_error
        if gemini_error and openai_error:
            self._last_processing_error = (
                f"Gemini failed: {gemini_error}; OpenAI failed: {openai_error}"
            )
        else:
            self._last_processing_error = openai_error or gemini_error
        return None

    def _process_transcript_text_local(
        self,
        transcript_text: str,
        recording_id: str,
        *,
        verbose: bool = True,
        recording_date: str = "",
        plaud_context: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[GeminiEventOutput]:
        """Process transcript text without cloud LLMs using local chunk events."""
        from datetime import timedelta
        from app_v2.services.xray import xray_log

        self._last_processing_error = None
        chunks = self._chunk_transcript_words(transcript_text)
        if not chunks:
            self._last_processing_error = "Transcript too short"
            return None

        start_anchor = self._parse_recording_start(recording_date)
        event_minutes = 10
        events: list[ChronosEvent] = []
        dropped_short_chunks = 0

        self._emit_progress(
            progress_callback,
            "Local processing",
            f"Building {len(chunks)} local transcript chunks without cloud AI",
        )
        xray_log(
            "local",
            "extract",
            f"Building {len(chunks)} local transcript chunks without cloud AI",
            provider="ollama",
            model=getattr(self.settings, "chronos_local_llm_model", "local"),
            status="local",
        )

        for index, chunk in enumerate(chunks):
            event_start = start_anchor + timedelta(minutes=event_minutes * index)
            event_end = event_start + timedelta(minutes=event_minutes)
            clean_text = " ".join(chunk.split()).strip()
            if plaud_context and index == 0:
                clean_text = f"{clean_text}\n\nPlaud context: {plaud_context[:700].strip()}"

            if len(clean_text) < 10:
                dropped_short_chunks += 1
                continue

            events.append(
                ChronosEvent(
                    event_id=str(uuid.uuid4()),
                    recording_id=recording_id,
                    start_ts=event_start,
                    end_ts=event_end,
                    day_of_week=event_start.strftime("%A"),
                    hour_of_day=event_start.hour,
                    clean_text=clean_text,
                    category=self._local_category_for_text(clean_text),
                    category_confidence=0.45,
                    sentiment=0.0,
                    keywords=self._local_keywords_for_text(clean_text),
                    speaker="unknown",
                    raw_transcript_snippet=chunk[:500],
                    gemini_reasoning="local deterministic transcript chunk; no Gemini/OpenAI call",
                )
            )

        if not events:
            self._last_processing_error = "Transcript chunks were too short to extract meaningful events"
            return None

        if verbose:
            print(
                "      🧠 Local mode: extracted "
                f"{len(events)} chunk events without Gemini/OpenAI"
                + (
                    f" (skipped {dropped_short_chunks} short chunks)"
                    if dropped_short_chunks
                    else ""
                ),
                flush=True,
            )

        self._emit_progress(progress_callback, "Local events extracted", f"{len(events)} events")
        return GeminiEventOutput(
            events=events,
            total_events=len(events),
            processing_metadata={
                "provider": "local",
                "mode": "deterministic_chunker",
                "chunk_words": 420,
                "cloud_ai_used": False,
            },
        )

    def _process_transcript_text_openai(
        self,
        transcript_text: str,
        recording_id: str,
        *,
        verbose: bool = True,
        recording_date: str = "",
        plaud_context: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[GeminiEventOutput]:
        """Process transcript text through OpenAI structured outputs."""
        prompt = self._build_prompt(recording_id, recording_date)
        self._emit_progress(
            progress_callback,
            "Prompt built",
            f"{len(transcript_text.split()):,} words ready for OpenAI",
        )

        plaud_section = ""
        if plaud_context:
            plaud_section = (
                "\n\n**PLAUD AI CONTEXT** (use this to guide categorization, "
                "sentiment, and structure — but always extract events from the "
                "raw transcript below):\n\n"
                f"{plaud_context}\n"
            )

        full_prompt = f"""{prompt}{plaud_section}

**RAW TRANSCRIPT:**

{transcript_text}

Extract events from this transcript following the schema exactly."""

        if verbose:
            print(
                f"      📊 Transcript: {len(transcript_text.split()):,} words, {len(transcript_text):,} chars",
                flush=True,
            )
            print(
                f"      🤖 Model: {self.settings.openai_model} (OpenAI)",
                flush=True,
            )
            print("      📤 Sending to OpenAI API...", flush=True)

        self._emit_progress(
            progress_callback,
            "OpenAI request sent",
            f"{len(transcript_text.split()):,} words · {len(transcript_text):,} chars",
        )

        svc = OpenAIResponseService()
        result = svc.extract_events(
            full_prompt,
            recording_id=recording_id,
            model=(
                getattr(self.settings, "chronos_cleaning_model", None)
                or getattr(self.settings, "openai_model", None)
            ),
        )
        if "error" in result:
            self._last_processing_error = result["error"]
            self._emit_progress(
                progress_callback,
                "OpenAI failed",
                result["error"][:80],
            )
            if verbose:
                print(f"      ❌ Error: {result['error'][:80]}", flush=True)
            return None

        output = result.get("output")
        if not output or not output.events:
            self._last_processing_error = "OpenAI returned no events"
            return None

        self._emit_progress(
            progress_callback,
            "Events extracted",
            f"{output.total_events} events",
        )
        if verbose:
            print(f"      ✅ Extracted {output.total_events} events", flush=True)
            self._print_event_summary(output.events)

        self._last_processing_error = None
        return output

    def _process_transcript_text_gemini(
        self,
        transcript_text: str,
        recording_id: str,
        max_retries: int = 3,
        verbose: bool = True,
        recording_date: str = "",
        plaud_context: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
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
        self._last_processing_error = None

        try:
            engine = self._get_engine()
        except Exception as e:
            self._last_processing_error = str(e)
            return None

        # Skip transcripts that are too short to extract meaningful events
        MIN_TRANSCRIPT_CHARS = 100
        if len(transcript_text.strip()) < MIN_TRANSCRIPT_CHARS:
            self._last_processing_error = "Transcript too short"
            if verbose:
                print(
                    f"      ⚠️ Transcript too short ({len(transcript_text.strip())} chars < {MIN_TRANSCRIPT_CHARS}), skipping",
                    flush=True,
                )
            from app_v2.services.xray import xray_log

            xray_log(
                "gemini",
                "skip",
                f"Only {len(transcript_text.strip())} characters — too short for Gemini to work with",
            )
            logger.warning(
                f"Skipping {recording_id}: transcript too short ({len(transcript_text.strip())} chars)"
            )
            return None

        # Build prompt (same as audio version)
        prompt = self._build_prompt(recording_id, recording_date)
        self._emit_progress(
            progress_callback,
            "Prompt built",
            f"{len(transcript_text.split()):,} words ready for Gemini",
        )

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
            print(f"      🤖 Model: {engine.model_name}", flush=True)
            print(f"      📤 Sending to Gemini API...", flush=True)

        from app_v2.services.xray import xray_log

        _prompt_words = len(transcript_text.split())
        _prompt_chars = len(transcript_text)
        self._emit_progress(
            progress_callback,
            "Gemini request sent",
            f"{_prompt_words:,} words · {_prompt_chars:,} chars",
        )
        xray_log(
            "gemini",
            "prompt",
            f"Feeding {_prompt_words:,} words to Gemini — 'read this and tell me what happened'",
        )

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

                config: Any = {
                    "response_mime_type": "application/json",
                    "response_json_schema": GeminiEventOutput.model_json_schema(),
                    "temperature": 0.2,
                }
                if engine._thinking_level is not None:
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level=engine._thinking_level
                    )

                # Use streaming to show real-time progress
                STREAM_TIMEOUT = 900  # 15 min max for any single Gemini call
                if verbose:
                    response_text = ""
                    token_count = 0
                    events_found = 0
                    last_print_len = 0
                    start_time = __import__("time").time()
                    last_chunk_time = start_time
                    last_progress_update = start_time

                    stream = engine.client.models.generate_content_stream(
                        model=engine.model_name,
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

                        if now - last_progress_update >= 2:
                            self._emit_progress(
                                progress_callback,
                                "Streaming response",
                                f"{len(response_text):,} chars · {events_found} events · {now - start_time:.0f}s",
                            )
                            last_progress_update = now

                    # Final newline after streaming
                    elapsed = __import__("time").time() - start_time
                    print(
                        f"\r      📝 Streaming: {len(response_text):,} chars | {events_found} events | {elapsed:.0f}s ✓",
                        flush=True,
                    )
                    self._emit_progress(
                        progress_callback,
                        "Gemini response received",
                        f"{events_found} events spotted in {elapsed:.0f}s",
                    )

                    xray_log(
                        "gemini",
                        "stream",
                        f"Gemini read it all — wrote {len(response_text):,} chars and spotted {events_found} moments",
                        duration_ms=round(elapsed * 1000, 1),
                    )

                    # Get final usage from last chunk if available
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        _in_tok = getattr(usage, "prompt_token_count", 0)
                        _out_tok = getattr(usage, "candidates_token_count", 0)
                        print(
                            f"      📊 Tokens - Input: {_in_tok:,} | Output: {_out_tok:,}",
                            flush=True,
                        )
                        xray_log(
                            "gemini",
                            "tokens",
                            f"Gemini used {_in_tok:,} words reading + {_out_tok:,} words writing = {_in_tok + _out_tok:,} total",
                        )
                        from src.chronos.cost_tracker import track_usage

                        track_usage(
                            engine.model_name,
                            "generate",
                            input_tokens=_in_tok,
                            output_tokens=_out_tok,
                            recording_id=recording_id,
                        )

                    # Parse the accumulated response
                    raw_text = response_text.strip()
                else:
                    # Non-verbose: use regular call
                    response = engine.client.models.generate_content(
                        model=engine.model_name,
                        contents=full_prompt,
                        config=config,
                    )
                    raw_text = (response.text or "").strip()

                    # Track cost for non-verbose mode
                    _nv_usage = getattr(response, "usage_metadata", None)
                    if _nv_usage:
                        from src.chronos.cost_tracker import track_usage

                        track_usage(
                            engine.model_name,
                            "generate",
                            input_tokens=getattr(_nv_usage, "prompt_token_count", 0),
                            output_tokens=getattr(
                                _nv_usage, "candidates_token_count", 0
                            ),
                            recording_id=recording_id,
                        )

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
                self._emit_progress(
                    progress_callback,
                    "Parsing JSON",
                    f"{len(raw_text):,} chars",
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
                    repaired = self._repair_json_with_gemini(
                        raw_json,
                        recording_id,
                        progress_callback=progress_callback,
                    )

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
                self._emit_progress(
                    progress_callback,
                    "Events extracted",
                    f"{validated.total_events} events",
                )

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
                    _c = getattr(_ev, "category", "unknown")
                    _cats[str(getattr(_c, "value", _c))] += 1
                _cat_str = ", ".join(f"{c}:{n}" for c, n in _cats.most_common(5))
                xray_log(
                    "gemini",
                    "extract",
                    f"Pulled out {validated.total_events} moments — {_cat_str}",
                )

                logger.info(
                    f"Extracted {validated.total_events} events from transcript"
                )
                self._last_processing_error = None
                return validated

            except ValidationError as e:
                if verbose:
                    print(f"      ⚠️ Validation error: {str(e)[:100]}", flush=True)
                xray_log(
                    "gemini",
                    "retry",
                    f"Gemini's answer didn't make sense, trying again ({attempt + 1}/{max_retries})",
                    level="warn",
                )
                logger.error(f"Pydantic validation failed: {e}")
                if attempt < max_retries - 1:
                    import time as _time
                    _time.sleep(5)
                    continue
                self._last_processing_error = (
                    f"Gemini validation failed: {str(e)[:120]}"
                )
                return None

            except json.JSONDecodeError as e:
                if verbose:
                    print(f"      ⚠️ JSON parse error, retrying...", flush=True)
                xray_log(
                    "gemini",
                    "retry",
                    f"Gemini gave a weird answer, trying again ({attempt + 1}/{max_retries})",
                    level="warn",
                )
                logger.error(f"JSON parse error: {e}")
                if attempt < max_retries - 1:
                    import time as _time
                    _time.sleep(5)
                    continue
                self._last_processing_error = (
                    f"Gemini returned invalid JSON: {str(e)[:120]}"
                )
                return None

            except Exception as e:
                # 403 PERMISSION_DENIED = project banned; bail immediately
                if is_permission_denied(e):
                    if verbose:
                        print(
                            f"      ❌ Error: {str(e)[:80]}",
                            flush=True,
                        )
                    xray_log(
                        "gemini",
                        "error",
                        "Gemini project access denied (403) — no point retrying",
                        level="error",
                    )
                    logger.error("Gemini 403 PERMISSION_DENIED — aborting retries: %s", e)
                    self._last_processing_error = str(e)
                    return None

                failover_model = engine.pick_failover_model(e)
                if failover_model:
                    previous_model = engine.model_name
                    engine.model_name = failover_model
                    if verbose:
                        print(
                            f"      ↪ Switching model: {previous_model} -> {failover_model}",
                            flush=True,
                        )
                    self._emit_progress(
                        progress_callback,
                        "Switching Gemini model",
                        f"{previous_model} -> {failover_model}",
                    )
                    reason = (
                        "isn't available to this API key"
                        if is_model_not_found(e)
                        else "is under heavy load right now"
                    )
                    xray_log(
                        "gemini",
                        "fallback",
                        f"{previous_model} {reason} — switching to {failover_model}",
                        level="warn",
                    )
                    logger.warning(
                        "Gemini model '%s' %s; switching to '%s'",
                        previous_model,
                        reason,
                        failover_model,
                    )
                    continue

                transient_unavailable = is_model_temporarily_unavailable(e)
                if verbose:
                    print(f"      ❌ Error: {str(e)[:80]}", flush=True)
                if transient_unavailable and attempt < max_retries - 1:
                    xray_log(
                        "gemini",
                        "retry",
                        "Gemini is under heavy load — waiting and trying again",
                        level="warn",
                    )
                else:
                    xray_log(
                        "gemini",
                        "error",
                        f"Something went wrong with Gemini: {str(e)[:60]}",
                        level="error",
                    )
                logger.error(f"Failed to process transcript: {e}")
                if attempt < max_retries - 1:
                    # Backoff delay: 35s for 429/503 (rate limit/overload), 5s otherwise
                    import time as _time
                    delay = 35 if transient_unavailable else 5
                    if verbose:
                        print(f"      ⏳ Waiting {delay}s before retry...", flush=True)
                    _time.sleep(delay)
                    continue
                self._last_processing_error = str(e)
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
            category = getattr(category, "value", category)
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
            cat = getattr(cat, "value", cat)
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
            ok = self.process_recording_id(str(rec.recording_id))
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
        progress_callback: Optional[ProgressCallback] = None,
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
            provider = self._get_processing_provider()
            provider_label = self._provider_label(provider)
            record_id = str(rec.recording_id)
            record_title = getattr(rec, "title", None)
            created_at = getattr(rec, "created_at", None)
            duration_seconds = getattr(rec, "duration_seconds", 0)
            local_audio_path = getattr(rec, "local_audio_path", "")
            source = getattr(rec, "source", "plaud")
            device_id = getattr(rec, "device_id", None)
            checksum = getattr(rec, "checksum", None)
            cached_transcript = getattr(rec, "transcript", None)

            # Mark as in-progress early so we can spot crashes mid-batch.
            mark_chronos_recording_status(
                self.db, record_id, "processing", error_message=None
            )
            self._emit_progress(
                progress_callback, "Starting extraction", record_id[:20]
            )
            xray_log(
                "pipeline",
                "start",
                f"Handing this recording to {provider_label} extraction",
            )

            if delete_existing_events:
                deleted = delete_chronos_events_by_recording(self.db, recording_id)
                logger.info(f"Deleted {deleted} existing events for {recording_id}")

            use_cached_transcript = (
                str(source or "").strip().lower() == "notion"
                or record_id.startswith("notion:")
            )

            if use_cached_transcript:
                self._emit_progress(
                    progress_callback, "Loading cached transcript", record_id[:20]
                )
                transcript_text = str(cached_transcript or "").strip()
                if transcript_text:
                    xray_log(
                        "ingest",
                        "transcript-cache",
                        "Using the transcript already stored in Chronos",
                    )
            else:
                # Fetch file details from Plaud API
                self._emit_progress(
                    progress_callback, "Fetching transcript", record_id[:20]
                )
                _api_t0 = _time.perf_counter()
                file_details = self.plaud.get_recording(record_id)
                _api_ms = (_time.perf_counter() - _api_t0) * 1000
                xray_log(
                    "ingest",
                    "plaud-api",
                    f"Got the recording info from Plaud",
                    duration_ms=round(_api_ms, 1),
                )

                # Best-effort: refresh the recording title from Plaud if present.
                try:
                    plaud_title = file_details.get("title")
                    if plaud_title and (not record_title) and created_at is not None:
                        upsert_chronos_recording(
                            session=self.db,
                            recording_id=record_id,
                            title=plaud_title,
                            created_at=created_at,
                            duration_seconds=duration_seconds,
                            local_audio_path=local_audio_path,
                            source=source,
                            device_id=device_id,
                            checksum=checksum,
                        )
                except Exception:
                    pass

                # Extract transcript
                transcript_text = self._extract_transcript(file_details)

                # Cache transcript for UI/library browsing.
                if transcript_text:
                    self._emit_progress(
                        progress_callback,
                        "Transcript fetched",
                        f"{len(transcript_text.split()):,} words",
                    )
                    try:
                        set_chronos_recording_transcript(
                            self.db, record_id, transcript_text
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache transcript for {record_id}: {e}")

            if not transcript_text:
                logger.warning(f"No transcript for {record_id}")
                xray_log(
                    "pipeline",
                    "skip",
                    f"This recording has no transcript — nothing to analyze",
                    level="warn",
                )
                mark_chronos_recording_status(
                    self.db,
                    record_id,
                    "failed",
                    error_message=(
                        "No cached transcript available for Notion import"
                        if use_cached_transcript
                        else "No transcript available in Plaud source_list"
                    ),
                )
                return False

            # Process through the configured provider — pass real recording date
            # for temporal anchoring.
            recording_date = ""
            if created_at:
                try:
                    if isinstance(created_at, str):
                        recording_date = created_at[:10]
                    else:
                        recording_date = created_at.strftime("%Y-%m-%d")
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
                record_id,
                recording_date=recording_date,
                plaud_context=plaud_context,
                progress_callback=progress_callback,
            )

            # Auto-retry once only for single-provider Gemini mode. The auto mode
            # already chains providers and doesn't need a second outer retry.
            if (
                provider == "gemini"
                and (not output or not output.events)
                and len(transcript_text.strip()) >= 500
            ):
                logger.info(
                    f"Retrying {record_id} — {provider_label} returned no events on first attempt"
                )
                xray_log(
                    "pipeline",
                    "retry",
                    f"{provider_label} blanked on a real transcript — giving it one more shot",
                    level="warn",
                )
                self._emit_progress(
                    progress_callback,
                    f"Retrying {provider_label}",
                    "first attempt returned no events",
                )
                output = self.process_transcript_text(
                    transcript_text,
                    record_id,
                    recording_date=recording_date,
                    plaud_context=plaud_context,
                    progress_callback=progress_callback,
                )

            if not output or not output.events:
                logger.warning(f"No events extracted for {record_id}")
                failure_reason = self._last_processing_error
                if not failure_reason:
                    if provider == "auto":
                        failure_reason = "No AI provider returned any events"
                    else:
                        failure_reason = f"{provider_label} returned no events"
                xray_log(
                    "pipeline",
                    "fail",
                    failure_reason,
                    level="warn",
                )
                mark_chronos_recording_status(
                    self.db,
                    record_id,
                    "failed",
                    error_message=failure_reason,
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
                    day_of_week=str(getattr(e.day_of_week, "value", e.day_of_week)),
                    hour_of_day=e.hour_of_day,
                    clean_text=e.clean_text,
                    category=str(getattr(e.category, "value", e.category)),
                    sentiment=e.sentiment,
                    keywords=e.keywords,
                    speaker=str(getattr(e.speaker, "value", e.speaker)),
                    raw_transcript_snippet=e.raw_transcript_snippet,
                    gemini_reasoning=e.gemini_reasoning,
                )
                for e in output.events
            ]
            add_chronos_events(self.db, db_events)
            self._emit_progress(
                progress_callback,
                "Events stored",
                f"{len(db_events)} saved to Chronos",
            )

            # Update status
            mark_chronos_recording_status(
                self.db, record_id, "completed", error_message=None
            )
            self._emit_progress(
                progress_callback,
                "Recording complete",
                f"{len(output.events)} events extracted",
            )

            _proc_ms = (_time.perf_counter() - _proc_t0) * 1000
            xray_log(
                "pipeline",
                "done",
                f"Done with {provider_label}! Found {len(output.events)} moments in this recording",
                duration_ms=round(_proc_ms, 1),
            )

            logger.info(f"✓ Processed {record_id}: {len(output.events)} events")
            return True

        except Exception as e:
            logger.error(f"Failed to process {recording_id}: {e}")
            xray_log(
                "pipeline",
                "error",
                f"This recording crashed the processor: {str(e)[:60]}",
                level="error",
            )
            mark_chronos_recording_status(
                self.db,
                str(recording_id),
                "failed",
                error_message=str(e),
            )
            return False
