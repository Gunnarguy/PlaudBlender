"""Chronos cognitive engine powered by Gemini.

This module handles the "clean verbatim" reconstruction of audio recordings.
It transforms raw, erratic voice data into structured ChronosEvent objects.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from google.genai import types
from pydantic import ValidationError

from src.config import get_settings
from src.models.chronos_schemas import (
    ChronosEvent,
    GeminiEventOutput,
)

from src.chronos.genai_helpers import (
    get_genai_client,
    is_model_not_found,
    is_model_temporarily_unavailable,
    normalize_thinking_level,
    pick_first_available,
    pick_first_available_or_known,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# System Prompt for "Clean as F*ck" Reconstruction
# ═══════════════════════════════════════════════════════════════════

CHRONOS_CLEAN_PROMPT = """You are an expert cognitive editor processing a voice recording.

**INPUT:** A 5-7 hour audio recording of someone's work day. The audio is raw, jumbled, and erratic. It contains:
- Stream-of-consciousness thinking out loud
- Topic switches without transitions
- Filler words (um, uh, like, you know)
- False starts and abandoned sentences
- Long silences (breaks, deep work)

**YOUR TASK:** Transform this into a clean, structured timeline of events.

**CRITICAL RULES:**

1. **STRICT CONTEXT PRESERVATION**
   - Do NOT summarize. Retain every distinct thought, opinion, and technical detail.
   - If the user mentions "Project Alpha needs a new database schema", you must output that exact intent.
   - Preserve terminology exactly as spoken (project names, technical terms, proper nouns).

2. **AGGRESSIVE NOISE REMOVAL**
   - Remove ALL filler words: um, uh, like, you know, so, basically, actually
   - Remove stutters and false starts
   - If a sentence is aborted and restarted, keep ONLY the final coherent version
   - Example: "I was going to... no wait, the plan is to refactor the API" → "The plan is to refactor the API"

3. **TOPIC SEGMENTATION**
   - Break the continuous stream into discrete "Events"
   - An Event is defined by a shift in topic or activity
   - Examples of topic shifts:
     * Coding → Lunch break
     * Meeting discussion → Email review
     * Technical problem → Personal reflection
   - Each event should represent 2-15 minutes of cohesive thought

4. **TEMPORAL ACCURACY**
   - This recording was made on: {{RECORDING_DATE}}
   - Each event MUST use this date for start_ts and end_ts
   - Distribute event times across the recording duration on that date
   - If there's a 10+ minute silence, create a "break" event
   - Do NOT use dates from examples - use the RECORDING DATE above

5. **CATEGORIZATION**
   - Assign each event to ONE category:
     * work: professional tasks, coding, problem-solving
     * personal: life thoughts, family, health
     * meeting: discussions with others (if detected)
     * deep_work: focused technical work with minimal interruption
     * break: eating, resting, context switches
     * reflection: thinking about past decisions or future plans
     * idea: brainstorming, creative thinking
     * unknown: unclear or transitional moments
   - Also assign a category_confidence score from 0.0 to 1.0:
     * 1.0 = certain (clearly a meeting with multiple speakers)
     * 0.7+ = confident (fits the category well)
     * 0.4-0.7 = moderate (could be another category)
     * <0.4 = low confidence (ambiguous content)

6. **SENTIMENT (REQUIRED & NEUTRAL BANNED)**
   - Assign a sentiment score from -1.0 (very negative/frustrated) to 1.0 (very positive/excited)
   - BANNED: You MUST NOT output exactly 0.0. A neutral score of 0.0 is strictly forbidden.
   - Even mundane or professional tasks lean slightly positive (+0.1, productive) or negative (-0.1, tedious).
   - Base this on tone, word choice, and energy level. You must take a stance.

7. **KEYWORDS**
   - Extract 3-5 high-value, specific keywords that capture the concrete essence of this event.
   - Keywords must represent key subjects, proper nouns, projects, tools, or concrete concepts (e.g. "auth-refactor", "Stryker", "Stanford", "suture", "VNC").
   - STRICTLY avoid generic verbs (doing, going, make, think, thank), pronouns (him, us, they), generic nouns (stuff, thing, guys, dude), or generic question/conversational words (how, what, please).

**OUTPUT FORMAT:**

Return a JSON object with this EXACT structure:

```json
{
  "events": [
    {
      "event_id": "uuid-string",
      "recording_id": "{{RECORDING_ID}}",
      "start_ts": "{{RECORDING_DATE}}T09:15:32Z",
      "end_ts": "{{RECORDING_DATE}}T09:18:45Z",
      "day_of_week": "Monday",
      "hour_of_day": 9,
      "clean_text": "Reviewed the Sprint planning doc. The team agreed to prioritize the authentication refactor. I'm concerned about the timeline but optimistic about the new architecture.",
      "category": "work",
      "category_confidence": 0.85,
      "sentiment": 0.3,
      "keywords": ["sprint", "authentication", "refactor", "architecture"],
      "speaker": "self_talk"
    }
  ],
  "total_events": 42,
  "processing_metadata": {
    "audio_duration_seconds": 25200,
    "total_silence_seconds": 3600,
    "quality_notes": "High quality recording, minimal background noise"
  }
}
```

**CONSTRAINTS:**

- Minimum 10 events for a 1-hour recording
- Minimum 50 events for a 7-hour recording
- Each event's clean_text must be at least 20 characters
- Do NOT output events with empty or placeholder text
- Timestamps must be chronologically ordered
- All day_of_week values must be: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- All category values must be: work, personal, meeting, deep_work, break, reflection, idea, unknown
- All speaker values must be: self_talk, conversation, unknown

**REMEMBER:** You are reconstructing a narrative, not transcribing verbatim. Clean it, but preserve the meaning.
"""


class ChronosEngine:
    """Gemini-powered cognitive engine for audio reconstruction.

    Handles:
    - Audio file upload to Gemini File API
    - Clean verbatim prompt execution
    - JSON validation against Pydantic schemas
    - Retry logic for transient failures
    """

    def __init__(self):
        """Initialize Gemini client."""
        self.settings = get_settings()

        if not self.settings.gemini_api_key:
            raise ValueError(
                "CHRONOS_GEMINI_API_KEY not set. Set a dedicated Chronos key or "
                "opt into the shared GEMINI_API_KEY with CHRONOS_ALLOW_SHARED_GEMINI_KEY=1"
            )

        self.client = get_genai_client()

        # Select model based on config, but prefer a model that is actually
        # available to the configured API key.
        configured = (self.settings.chronos_cleaning_model or "").strip()
        fallback = pick_first_available(
            configured,
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-2.5-flash",
        )
        self.model_name = fallback or configured or "gemini-3-flash-preview"

        self._thinking_level = normalize_thinking_level(
            getattr(self.settings, "chronos_thinking_level", "")
        )

        logger.info(f"Initialized ChronosEngine with model: {self.model_name}")

    def _failover_candidates(
        self,
        current_model: str,
        *,
        transient_unavailable: bool,
    ) -> List[str]:
        """Return a small ordered list of fallback models for the current mode."""
        current = (current_model or "").strip()
        configured = (self.settings.chronos_cleaning_model or "").strip()

        if transient_unavailable:
            if "pro" in current:
                return ["gemini-2.5-pro", "gemini-2.5-flash"]
            if "flash" in current:
                return ["gemini-2.5-flash"]
            return ["gemini-2.5-flash"]

        return [
            configured,
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
        ]

    def pick_failover_model(
        self,
        err: Exception,
        current_model: Optional[str] = None,
    ) -> Optional[str]:
        """Pick the next model to try for not-found or transient availability errors."""
        current = (current_model or self.model_name or "").strip()
        if not current:
            return None

        if is_model_not_found(err):
            candidates = self._failover_candidates(current, transient_unavailable=False)
        elif is_model_temporarily_unavailable(err):
            candidates = self._failover_candidates(current, transient_unavailable=True)
        else:
            return None

        deduped: List[str] = []
        for candidate in candidates:
            candidate = (candidate or "").strip()
            if not candidate or candidate == current or candidate in deduped:
                continue
            deduped.append(candidate)

        return pick_first_available_or_known(*deduped)

    def _upload_audio_file(self, audio_path: str):
        """Upload audio file to Gemini Files API.

        Args:
            audio_path: Path to local audio file

        Returns:
            File handle for generation

        Raises:
            ValueError: If file upload fails
        """
        if not Path(audio_path).exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        logger.info(f"Uploading audio file: {audio_path}")
        file_handle = self.client.files.upload(file=audio_path)

        # Some media types may require processing. Best-effort polling.
        # (Files API is Gemini Developer API only.)
        try:
            while (
                getattr(getattr(file_handle, "state", None), "name", None)
                == "PROCESSING"
            ):
                logger.debug("Waiting for file processing...")
                time.sleep(5)
                file_handle = self.client.files.get(name=file_handle.name)

            if getattr(getattr(file_handle, "state", None), "name", None) == "FAILED":
                raise ValueError(
                    f"File upload failed: {getattr(file_handle, 'state', None)}"
                )
        except Exception:
            # If state polling isn't supported for this file type/account, just proceed.
            pass

        logger.info(f"File uploaded successfully: {file_handle.name}")
        return file_handle

    def _build_prompt(self, recording_id: str, recording_date: str = "") -> str:
        """Build the full prompt with recording context.

        Args:
            recording_id: Recording ID to inject into prompt
            recording_date: ISO date string (YYYY-MM-DD) for temporal anchoring

        Returns:
            str: Complete prompt
        """
        # Default to today if no date provided
        if not recording_date:
            from datetime import datetime

            recording_date = datetime.now().strftime("%Y-%m-%d")
        prompt = CHRONOS_CLEAN_PROMPT.replace("{{RECORDING_ID}}", recording_id)
        return prompt.replace("{{RECORDING_DATE}}", recording_date)

    def process_audio(
        self,
        audio_path: str,
        recording_id: str,
        max_retries: int = 3,
        recording_date: str = "",
    ) -> Optional[GeminiEventOutput]:
        """Process audio file and extract structured events.

        Args:
            audio_path: Path to local audio file
            recording_id: Recording ID for provenance
            max_retries: Number of retry attempts for transient failures

        Returns:
            GeminiEventOutput: Validated event structure, or None if failed
        """
        for attempt in range(max_retries):
            try:
                # Upload audio
                file_handle = self._upload_audio_file(audio_path)

                # Build prompt
                prompt = self._build_prompt(recording_id, recording_date)

                # Generate with strict JSON output
                logger.info(
                    f"Generating events (attempt {attempt + 1}/{max_retries})..."
                )
                config: dict = {
                    "response_mime_type": "application/json",
                    # Structured Outputs: ask the API to enforce schema adherence.
                    "response_json_schema": GeminiEventOutput.model_json_schema(),
                    "temperature": 0.2,
                }
                if self._thinking_level is not None:
                    config["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self._thinking_level
                    )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[file_handle, prompt],
                    config=config,
                )

                # Track cost
                _usage = getattr(response, "usage_metadata", None)
                if _usage:
                    from src.chronos.cost_tracker import track_usage as _track

                    _track(
                        self.model_name,
                        "generate",
                        input_tokens=getattr(_usage, "prompt_token_count", 0),
                        output_tokens=getattr(_usage, "candidates_token_count", 0),
                        recording_id=recording_id,
                    )

                # Parse JSON
                output_data = getattr(response, "parsed", None)
                if output_data is None:
                    raw_json = response.text or ""
                    logger.debug(f"Raw Gemini response: {raw_json[:500]}...")
                    output_data = json.loads(raw_json)

                # Validate with Pydantic
                validated = GeminiEventOutput(**output_data)
                logger.info(f"Successfully extracted {validated.total_events} events")

                # Clean up uploaded file
                try:
                    self.client.files.delete(name=file_handle.name)
                    logger.debug(f"Deleted uploaded file: {file_handle.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete file: {e}")

                return validated

            except ValidationError as e:
                logger.error(f"Pydantic validation failed: {e}")
                # Validation errors are not transient - fail immediately
                return None

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    return None

            except Exception as e:
                failover_model = self.pick_failover_model(e)
                if failover_model:
                    previous_model = self.model_name
                    reason = (
                        "not available to this API key"
                        if is_model_not_found(e)
                        else "temporarily unavailable"
                    )
                    logger.warning(
                        "Model '%s' is %s; switching to %s",
                        previous_model,
                        reason,
                        failover_model,
                    )
                    self.model_name = failover_model
                    continue

                logger.error(f"Processing error: {e}")
                if attempt < max_retries - 1:
                    delay = (
                        max(15, 2**attempt)
                        if is_model_temporarily_unavailable(e)
                        else 2**attempt
                    )
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    return None

        return None

    def process_audio_to_events(
        self,
        audio_path: str,
        recording_id: str,
        recording_date: str = "",
    ) -> Optional[List[ChronosEvent]]:
        """Convenience method that returns just the event list.

        Args:
            audio_path: Path to audio file
            recording_id: Recording ID
            recording_date: ISO date string (YYYY-MM-DD) for temporal anchoring

        Returns:
            List[ChronosEvent]: List of validated events, or None if failed
        """
        output = self.process_audio(
            audio_path, recording_id, recording_date=recording_date
        )
        return output.events if output else None


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════


def validate_event_quality(events: List[ChronosEvent], duration_seconds: int) -> bool:
    """Check if event extraction meets quality standards.

    Args:
        events: List of extracted events
        duration_seconds: Original recording duration

    Returns:
        bool: True if quality standards met
    """
    if not events:
        logger.warning("No events extracted")
        return False

    # Heuristic: expect at least 1 event per 10 minutes
    expected_min = duration_seconds // 600
    if len(events) < expected_min:
        logger.warning(f"Too few events: {len(events)} (expected >= {expected_min})")
        return False

    # Check for empty events
    empty_count = sum(1 for e in events if len(e.clean_text.strip()) < 20)
    if empty_count > len(events) * 0.1:  # More than 10% empty
        logger.warning(f"Too many empty events: {empty_count}/{len(events)}")
        return False

    logger.info(f"Event quality check passed: {len(events)} events")
    return True
