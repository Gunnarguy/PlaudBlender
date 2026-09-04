"""OpenAI Responses API service for RAG and conversational queries.

Uses the OpenAI Responses API (not Chat Completions) for rich,
context-aware answers backed by Chronos event data.
"""

import hashlib
import logging
import os
import time
from typing import Any, Callable, List, Optional, TypeVar

from pydantic import BaseModel, Field

from src.config import get_settings, normalize_openai_model_name
from src.models.chronos_schemas import ChronosEvent, GeminiEventOutput

logger = logging.getLogger(__name__)

_ResponseT = TypeVar("_ResponseT")


class _OpenAIEventOutput(BaseModel):
    """OpenAI-compatible subset of GeminiEventOutput.

    Drops ``processing_metadata`` (Optional[dict]) which produces
    a bare ``object`` JSON-Schema type that OpenAI rejects because
    it requires ``additionalProperties: false`` on every object.
    """

    events: List[ChronosEvent] = Field(..., description="Array of reconstructed events")
    total_events: int = Field(0, ge=0, description="Total events extracted")


class OpenAIResponseService:
    """Wraps OpenAI Responses API for Chronos RAG queries."""

    _CONNECTION_CACHE: dict[str, tuple[float, tuple[bool, str]]] = {}
    _CONNECTION_CACHE_TTL_SECONDS = 120.0
    _REQUEST_TIMEOUT_SECONDS = 150.0
    _MAX_RETRY_ATTEMPTS = 3
    # Reasoning effort for every OpenAI call that does not name its own.
    # This was pinned to "low", which quietly capped the analyst and cleaning
    # models well under what they can do. "high" is the default because bulk
    # transcript chunking shares this service; Ask overrides it upward.
    # The API accepts none/low/medium/high/xhigh -- there is no "max".
    _VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
    _DEFAULT_REASONING_EFFORT = (
        os.getenv("CHRONOS_REASONING_EFFORT", "high").strip().lower()
        if os.getenv("CHRONOS_REASONING_EFFORT", "high").strip().lower()
        in {"none", "low", "medium", "high", "xhigh"}
        else "high"
    )
    _DEFAULT_MAX_OUTPUT_TOKENS = 1800
    _EXTRACTION_CACHE_KEY_VERSION = "v1"

    @staticmethod
    def _supports_temperature(model_name: Optional[str]) -> bool:
        if not model_name:
            return True
        normalized = normalize_openai_model_name(model_name).lower()
        return not normalized.startswith("gpt-5")

    def __init__(self):
        settings = get_settings()
        self._enabled = bool(getattr(settings, "chronos_openai_enabled", False))
        self._api_key = settings.openai_api_key
        self._model = normalize_openai_model_name(settings.openai_model)
        self._temperature = settings.openai_temperature
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._enabled and self._api_key)

    def _get_client(self):
        if not self.available:
            raise RuntimeError("OpenAI is disabled by CHRONOS_OPENAI_ENABLED=0")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                timeout=self._REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    @staticmethod
    def _retryable_error_types() -> tuple[type[BaseException], ...]:
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )

            return (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except Exception:
            return ()

    @staticmethod
    def _exception_name(exc: Exception) -> str:
        return exc.__class__.__name__

    @classmethod
    def _format_openai_error(cls, exc: Exception) -> str:
        name = cls._exception_name(exc)
        message = (str(exc) or name).strip()
        status_code = getattr(exc, "status_code", None)

        if name == "AuthenticationError":
            return "OpenAI authentication failed. Check OPENAI_API_KEY."
        if name == "RateLimitError":
            return f"OpenAI rate limited the request: {message}"
        if name == "APITimeoutError":
            return f"OpenAI request timed out: {message}"
        if name == "APIConnectionError":
            return f"OpenAI connection failed: {message}"
        if name == "BadRequestError":
            return f"OpenAI rejected the request: {message}"
        if status_code is not None:
            return f"OpenAI API error ({status_code}): {message}"
        return message

    @staticmethod
    def _sleep_before_retry(delay_seconds: float) -> None:
        time.sleep(delay_seconds)

    @classmethod
    def _call_with_retry(
        cls,
        operation_name: str,
        func: Callable[[], _ResponseT],
        *,
        xray_log=None,
        xray_source: Optional[str] = None,
    ) -> _ResponseT:
        retryable_errors = cls._retryable_error_types()

        for attempt in range(1, cls._MAX_RETRY_ATTEMPTS + 1):
            try:
                return func()
            except Exception as exc:
                is_retryable = bool(retryable_errors) and isinstance(
                    exc, retryable_errors
                )
                if not is_retryable or attempt >= cls._MAX_RETRY_ATTEMPTS:
                    raise

                delay_seconds = float(attempt)
                logger.warning(
                    "%s hit a transient OpenAI error on attempt %s/%s: %s",
                    operation_name,
                    attempt,
                    cls._MAX_RETRY_ATTEMPTS,
                    cls._format_openai_error(exc),
                )
                if xray_log and xray_source:
                    xray_log(
                        xray_source,
                        "openai",
                        f"{operation_name} hit a transient OpenAI error — retrying",
                        detail=f"attempt={attempt + 1}/{cls._MAX_RETRY_ATTEMPTS} wait={delay_seconds:.1f}s",
                        level="warn",
                    )
                cls._sleep_before_retry(delay_seconds)

        raise RuntimeError(f"{operation_name} exhausted retries")

    @staticmethod
    def _extract_output_text(response) -> str:
        direct = (getattr(response, "output_text", None) or "").strip()
        if direct:
            return direct

        pieces: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text = getattr(content, "text", "")
                    if text:
                        pieces.append(text)

        return "".join(pieces).strip()

    @staticmethod
    def _extract_reasoning_summary_text(response) -> str:
        summary_text = ""
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for summary_item in getattr(item, "summary", []) or []:
                if getattr(summary_item, "type", None) == "summary_text":
                    summary_text += getattr(summary_item, "text", "")
        return summary_text.strip()

    @staticmethod
    def _incomplete_reason(response) -> Optional[str]:
        details = getattr(response, "incomplete_details", None)
        if details is None:
            return None
        if isinstance(details, dict):
            return details.get("reason")
        return getattr(details, "reason", None)

    @classmethod
    def _format_incomplete_response_error(cls, response) -> str:
        reason = cls._incomplete_reason(response) or "unknown"
        if reason == "max_output_tokens":
            return "OpenAI ran out of output tokens before finishing the answer. Increase max_output_tokens or lower reasoning effort."
        return f"OpenAI returned an incomplete response ({reason})."

    @staticmethod
    def _cache_usage(usage) -> tuple[int, int]:
        """Return prompt-cache read and write token counts from Responses usage."""
        details = getattr(usage, "input_tokens_details", None)
        if details is None:
            return 0, 0
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0)
            written = details.get("cache_write_tokens", 0)
        else:
            cached = getattr(details, "cached_tokens", 0)
            written = getattr(details, "cache_write_tokens", 0)
        return int(cached or 0), int(written or 0)

    @classmethod
    def _extraction_cache_key(cls, model: str, instructions: str) -> str:
        """Route identical extraction instructions to the same prompt cache."""
        instructions_hash = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:12]
        return (
            "plaudblender:chronos-events:"
            f"{cls._EXTRACTION_CACHE_KEY_VERSION}:{model}:{instructions_hash}"
        )

    def extract_events(
        self,
        prompt: str,
        *,
        recording_id: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict:
        """Extract Chronos events from a transcript using structured OpenAI output."""
        if not self.available:
            return {"error": "OPENAI_API_KEY not configured"}

        instructions = system_prompt or (
            "You are Chronos, an event extraction engine. "
            "Read the transcript and return only structured Chronos events that satisfy the provided schema. "
            "Do not summarize away concrete details. Preserve exact meaning, names, and technical terminology."
        )

        try:
            from app_v2.services.xray import xray_log
        except ImportError:
            xray_log = None

        try:
            import time as _time

            _t0 = _time.perf_counter()

            if xray_log:
                xray_log(
                    "pipeline",
                    "openai",
                    f"Sending transcript to OpenAI ({len(prompt.split()):,} words)",
                )

            client = self._get_client()
            request_model = normalize_openai_model_name(model or self._model)
            kwargs: dict[str, Any] = {
                "model": request_model,
                "instructions": instructions,
                "input": prompt,
                "text_format": _OpenAIEventOutput,
                "max_output_tokens": 32768,
                "prompt_cache_key": self._extraction_cache_key(
                    request_model, instructions
                ),
            }

            if self._temperature is not None and self._supports_temperature(
                request_model
            ):
                kwargs["temperature"] = self._temperature

            response = self._call_with_retry(
                "OpenAI extraction",
                lambda: client.responses.parse(**kwargs),
                xray_log=xray_log,
                xray_source="pipeline",
            )
            _elapsed = (_time.perf_counter() - _t0) * 1000
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)
            cached_tokens, cache_write_tokens = self._cache_usage(usage)

            parsed = response.output_parsed
            if parsed is None:
                raw_text = self._extract_output_text(response)
                if not raw_text.strip():
                    incomplete_reason = self._incomplete_reason(response)
                    if incomplete_reason:
                        return {
                            "error": self._format_incomplete_response_error(response)
                        }
                    return {"error": "OpenAI returned no structured output"}
                try:
                    parsed = _OpenAIEventOutput.model_validate_json(raw_text)
                except Exception as exc:
                    return {
                        "error": "OpenAI structured output could not be parsed: "
                        + str(exc)
                    }

            # Convert to GeminiEventOutput for downstream compatibility
            result_output = GeminiEventOutput(
                events=parsed.events,
                processing_metadata=None,
                total_events=parsed.total_events,
            )

            from src.chronos.cost_tracker import track_usage

            track_usage(
                response.model,
                "generate",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                recording_id=recording_id,
            )

            if xray_log:
                xray_log(
                    "pipeline",
                    "openai",
                    f"OpenAI extracted {result_output.total_events} events",
                    duration_ms=round(_elapsed, 1),
                    detail=(
                        f"model={response.model} in={input_tokens} out={output_tokens} "
                        f"cache_read={cached_tokens} cache_write={cache_write_tokens}"
                    ),
                )

            return {
                "output": result_output,
                "model": response.model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "total_tokens": total_tokens,
                },
            }
        except Exception as e:
            message = self._format_openai_error(e)
            logger.exception("OpenAI transcript extraction failed")
            if xray_log:
                xray_log(
                    "pipeline",
                    "openai",
                    f"OpenAI extraction error: {message[:100]}",
                    level="error",
                )
            return {"error": message}

    def ask(
        self,
        question: str,
        context_events: List[dict],
        system_prompt: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        verbosity: Optional[str] = None,
        service_tier: Optional[str] = None,
    ) -> dict:
        """Answer a question using retrieved event context via the Responses API.

        Args:
            question: User's natural language question.
            context_events: List of dicts with date, time, category, text fields.
            system_prompt: Optional system instructions override.
            previous_response_id: For follow-up questions (stateful conversation).
            model: Override the configured OpenAI model for this request.
            reasoning: Reasoning effort for reasoning-capable GPT-5 models.
            reasoning_summary: Optional reasoning summary level.
            temperature: Optional sampling temperature.
            top_p: Optional nucleus sampling value.
            max_output_tokens: Optional cap including reasoning + visible output.
            verbosity: Optional text verbosity level.
            service_tier: Optional service tier override.

        Returns:
            dict with 'answer', 'model', 'response_id', 'usage' keys.
        """
        if not self.available:
            return {"error": "OPENAI_API_KEY not configured"}

        # Build evidence blocks from search hits plus expanded day summaries.
        context_lines = []
        for evt in context_events:
            kind = evt.get("kind", "search_hit")
            date = evt.get("date", "?")
            time = evt.get("time", "")
            category = evt.get("category", "unknown")
            text = evt.get("text", "")

            if kind == "expanded_day":
                header = f"[Expanded day {date}] ({category})"
            else:
                rank = evt.get("rank")
                score = evt.get("score")
                rank_part = f"Hit #{rank}" if rank else "Search hit"
                score_part = f" score={score}" if score is not None else ""
                header = f"[{rank_part}{score_part} | {date} {time}] ({category})"

            context_lines.append(f"{header}\n{text}".strip())
        context_block = "\n\n".join(context_lines)
        context_chars = len(context_block)
        context_dates = len(
            {
                evt.get("date")
                for evt in context_events
                if evt.get("date") and evt.get("date") != "?"
            }
        )

        default_system = (
            "You are Chronos, an AI assistant answering questions about the user's "
            "voice recordings and daily events. Use only the retrieved evidence blocks, "
            "which may include exact search hits and expanded day summaries for relevant dates. "
            "Lead with the direct answer, then synthesize concrete insights when the evidence "
            "supports them — recurring themes, changes over time, clusters of activity, useful "
            "date expansions, and notable absences. Use the expanded day evidence when a relevant "
            "date appears instead of staying narrowly on a single snippet. Cite specific dates, times, "
            "and moments whenever possible, and clearly separate direct evidence from inference. If the "
            "evidence is thin, say exactly what is missing instead of guessing."
        )

        instructions = system_prompt or default_system

        user_message = f"""Based on these events from my voice recordings:

{context_block}

Question: {question}"""

        try:
            from app_v2.services.xray import xray_log
        except ImportError:
            xray_log = None

        try:
            import time as _time

            _t0 = _time.perf_counter()

            if xray_log:
                _reasoning_label = reasoning or "default"
                request_model = normalize_openai_model_name(model or self._model)
                xray_log(
                    "search",
                    "openai",
                    f"Asking GPT: {question[:80]}…",
                    detail=(
                        f"model={request_model} reasoning={_reasoning_label} "
                        f"events={len(context_events)} chars={context_chars} dates={context_dates}"
                    ),
                )

            client = self._get_client()
            request_model = normalize_openai_model_name(model or self._model)

            kwargs: dict[str, Any] = {
                "model": request_model,
                "instructions": instructions,
                "input": [{"role": "user", "content": user_message}],
            }

            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            valid_reasoning = {"none", "low", "medium", "high", "xhigh"}
            effective_reasoning = (
                reasoning
                if reasoning in valid_reasoning
                else self._DEFAULT_REASONING_EFFORT
            )
            reasoning_config: dict[str, Any] = {}
            if effective_reasoning:
                reasoning_config["effort"] = effective_reasoning
            if reasoning_summary and reasoning_summary in {"auto"}:
                reasoning_config["summary"] = reasoning_summary
            if reasoning_config:
                kwargs["reasoning"] = reasoning_config

            if self._supports_temperature(request_model):
                if temperature is not None:
                    kwargs["temperature"] = float(temperature)
                elif self._temperature is not None:
                    kwargs["temperature"] = self._temperature

            if top_p is not None:
                kwargs["top_p"] = float(top_p)

            resolved_max_output_tokens = (
                int(max_output_tokens)
                if max_output_tokens is not None
                else self._DEFAULT_MAX_OUTPUT_TOKENS
            )
            kwargs["max_output_tokens"] = resolved_max_output_tokens

            if verbosity and verbosity in {"low", "medium", "high"}:
                kwargs["text"] = {"verbosity": verbosity}

            if service_tier and service_tier in {"auto", "default", "flex", "priority"}:
                kwargs["service_tier"] = service_tier

            response = self._call_with_retry(
                "OpenAI Ask Chronos request",
                lambda: client.responses.create(**kwargs),
                xray_log=xray_log,
                xray_source="search",
            )
            _elapsed = (_time.perf_counter() - _t0) * 1000

            incomplete_reason = self._incomplete_reason(response)
            answer = self._extract_output_text(response)
            reasoning_summary_text = self._extract_reasoning_summary_text(response)

            if incomplete_reason and not answer:
                message = self._format_incomplete_response_error(response)
                if xray_log:
                    xray_log("search", "openai", message, level="error")
                return {"error": message}

            if not answer:
                message = "OpenAI returned no visible answer text."
                if xray_log:
                    xray_log("search", "openai", message, level="error")
                return {"error": message}

            # Track cost
            usage = getattr(response, "usage", None)
            _inp_tok = getattr(usage, "input_tokens", 0)
            _out_tok = getattr(usage, "output_tokens", 0)
            cached_tokens, cache_write_tokens = self._cache_usage(usage)
            output_details = getattr(usage, "output_tokens_details", None)
            reasoning_tokens = (
                getattr(output_details, "reasoning_tokens", 0)
                if output_details is not None
                else 0
            )
            from src.chronos.cost_tracker import track_usage

            track_usage(
                response.model, "search", input_tokens=_inp_tok, output_tokens=_out_tok
            )

            if xray_log:
                xray_log(
                    "search",
                    "ai-answer",
                    f"GPT answered in {_elapsed / 1000:.1f}s — {_inp_tok}+{_out_tok} tokens",
                    duration_ms=round(_elapsed, 1),
                    detail=(
                        f"model={response.model} in={_inp_tok} out={_out_tok} "
                        f"reasoning={reasoning_tokens} cache_read={cached_tokens} "
                        f"cache_write={cache_write_tokens} "
                        f"total={getattr(usage, 'total_tokens', _inp_tok + _out_tok)}"
                    ),
                    level="perf",
                )

            config = {
                "provider": "openai",
                "model": request_model,
                "reasoning": effective_reasoning,
                "reasoning_summary": reasoning_summary,
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "max_output_tokens": resolved_max_output_tokens,
                "verbosity": verbosity,
                "service_tier": kwargs.get("service_tier"),
            }
            if incomplete_reason:
                config["incomplete_reason"] = incomplete_reason

            return {
                "answer": answer,
                "model": response.model,
                "response_id": response.id,
                "reasoning_summary": reasoning_summary_text or None,
                "config": config,
                "usage": {
                    "input_tokens": _inp_tok,
                    "output_tokens": _out_tok,
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": getattr(usage, "total_tokens", _inp_tok + _out_tok),
                },
            }
        except Exception as e:
            message = self._format_openai_error(e)
            logger.exception("OpenAI Responses API call failed")
            if xray_log:
                xray_log(
                    "search", "openai", f"OpenAI error: {message[:100]}", level="error"
                )
            return {"error": message}

    def check_connection(self, quick: bool = False) -> tuple:
        """Test OpenAI connectivity. Returns (ok: bool, detail: str)."""
        if not self._enabled:
            return False, "OpenAI disabled by CHRONOS_OPENAI_ENABLED=0"
        if not self.available:
            return False, "OPENAI_API_KEY not set"

        cache_key = f"{self._model}:{'quick' if quick else 'full'}"
        cached = self._CONNECTION_CACHE.get(cache_key)
        now = time.time()
        if cached and now - cached[0] < self._CONNECTION_CACHE_TTL_SECONDS:
            return cached[1]

        try:
            client = self._get_client()
            if quick:
                models = client.models.list()
                result = (
                    True,
                    f"Connected — {self._model} (found {len(models.data)} models)",
                )
            else:
                response = self._call_with_retry(
                    "OpenAI readiness check",
                    lambda: client.responses.create(
                        model=self._model,
                        instructions="Reply with exactly OK.",
                        input=[{"role": "user", "content": "Health check"}],
                        max_output_tokens=16,
                        store=False,
                    ),
                )
                if self._incomplete_reason(response):
                    result = (False, self._format_incomplete_response_error(response))
                else:
                    output = self._extract_output_text(response)
                    if output:
                        result = (True, f"Responses API ready — {self._model}")
                    else:
                        result = (
                            False,
                            f"Responses API reachable, but {self._model} returned no text",
                        )

            self._CONNECTION_CACHE[cache_key] = (now, result)
            return result
        except Exception as e:
            result = (False, self._format_openai_error(e)[:120])
            self._CONNECTION_CACHE[cache_key] = (now, result)
            return result
