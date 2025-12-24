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
    DayOfWeek,
    EventCategory,
)

from src.chronos.genai_helpers import (
    get_genai_client,
    is_model_not_found,
    normalize_thinking_level,
    pick_first_available,
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
   - Each event MUST have accurate start_ts and end_ts based on the audio timestamps
   - If there's a 10+ minute silence, create a "break" event
   - Do NOT invent timestamps - use the actual audio timing

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

6. **SENTIMENT (OPTIONAL BUT ENCOURAGED)**
   - Assign a sentiment score from -1.0 (very negative/frustrated) to 1.0 (very positive/excited)
   - 0.0 = neutral
   - Base this on tone, word choice, and energy level

**OUTPUT FORMAT:**

Return a JSON object with this EXACT structure:

```json
{
  "events": [
    {
      "event_id": "uuid-string",
      "recording_id": "{{RECORDING_ID}}",
      "start_ts": "2025-10-27T09:15:32Z",
      "end_ts": "2025-10-27T09:18:45Z",
      "day_of_week": "Monday",
      "hour_of_day": 9,
      "clean_text": "Reviewed the Sprint planning doc. The team agreed to prioritize the authentication refactor. I'm concerned about the timeline but optimistic about the new architecture.",
      "category": "work",
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
            raise ValueError("GEMINI_API_KEY not set in environment")

        self.client = get_genai_client()

        # Select model based on config, but prefer a model that is actually
        # available to the configured API key.
        configured = (self.settings.chronos_cleaning_model or "").strip()
        fallback = pick_first_available(
            configured,
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
        )
        self.model_name = fallback or configured or "gemini-3-flash-preview"

        self._thinking_level = normalize_thinking_level(
            getattr(self.settings, "chronos_thinking_level", "")
        )

        logger.info(f"Initialized ChronosEngine with model: {self.model_name}")

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

    def _build_prompt(self, recording_id: str) -> str:
        """Build the full prompt with recording context.

        Args:
            recording_id: Recording ID to inject into prompt

        Returns:
            str: Complete prompt
        """
        return CHRONOS_CLEAN_PROMPT.replace("{{RECORDING_ID}}", recording_id)

    def process_audio(
        self,
        audio_path: str,
        recording_id: str,
        max_retries: int = 3,
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
                prompt = self._build_prompt(recording_id)

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
                # If the selected model doesn't exist for this API key, try a
                # sane fallback once so transcript-first pipelines don't hard fail.
                if (
                    is_model_not_found(e)
                    and self.model_name != "gemini-3-flash-preview"
                ):
                    logger.warning(
                        f"Model '{self.model_name}' not found/available; switching to gemini-3-flash-preview"
                    )
                    self.model_name = "gemini-3-flash-preview"
                    continue

                logger.error(f"Processing error: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    time.sleep(2**attempt)
                else:
                    return None

        return None

    def process_audio_to_events(
        self,
        audio_path: str,
        recording_id: str,
    ) -> Optional[List[ChronosEvent]]:
        """Convenience method that returns just the event list.

        Args:
            audio_path: Path to audio file
            recording_id: Recording ID

        Returns:
            List[ChronosEvent]: List of validated events, or None if failed
        """
        output = self.process_audio(audio_path, recording_id)
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
