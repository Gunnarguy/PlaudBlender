"""Shared Ask Chronos orchestration.

Keeps the OpenAI-first/Gemini-fallback behavior consistent across the Dash UI,
REST API, and MCP tool.
"""

import logging
import os
from typing import Any, Optional

from src.config import get_settings
from src.chronos.genai_helpers import get_genai_client, pick_first_available_or_known
from src.chronos.openai_service import OpenAIResponseService

logger = logging.getLogger(__name__)


class ChronosAskService:
    def __init__(self):
        self.settings = get_settings()
        self._openai = OpenAIResponseService()

    @property
    def available(self) -> bool:
        return bool(
            self._openai.available
            or getattr(self.settings, "gemini_api_key", None)
            or getattr(self.settings, "chronos_local_llm_enabled", False)
        )

    @staticmethod
    def _prefers_gemini(model_name: Optional[str]) -> bool:
        normalized = (model_name or "").strip().lower()
        if normalized.startswith("models/"):
            normalized = normalized.split("/", 1)[1]
        return normalized.startswith("gemini-")

    def _allows_openai_fallback(self, requested_model: Optional[str]) -> bool:
        provider = (self.settings.chronos_processing_provider or "").strip().lower()
        if provider in {"openai", "auto"}:
            return True
        if requested_model and not self._prefers_gemini(requested_model):
            return True
        return False

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are Chronos, an AI assistant answering questions about the user's "
            "voice recordings and daily events. Use only the retrieved evidence blocks, "
            "which may include exact search hits and expanded day summaries for relevant dates. "
            "Honor the time window implied by the question. If the question asks about lately, recently, "
            "this week, or the last few weeks, stay anchored to those dates instead of drifting to older evidence. "
            "Start with a direct answer, then support it with the strongest concrete evidence. "
            "Cite specific dates, times, and moments whenever possible. If the evidence is thin, "
            "say exactly what is missing instead of guessing."
        )

    @staticmethod
    def _format_context_event(event: dict[str, Any]) -> str:
        kind = event.get("kind", "search_hit")
        date = event.get("date", "?")
        time = event.get("time", "")
        category = event.get("category", "unknown")
        text = event.get("text", "")

        if kind == "expanded_day":
            header = f"[Expanded day {date}] ({category})"
        else:
            rank = event.get("rank")
            score = event.get("score")
            rank_part = f"Hit #{rank}" if rank else "Search hit"
            score_part = f" score={score}" if score is not None else ""
            header = f"[{rank_part}{score_part} | {date} {time}] ({category})"

        return f"{header}\n{text}".strip()

    def _select_gemini_model(self, model_override: Optional[str] = None) -> str:
        configured = (
            model_override or getattr(self.settings, "chronos_analyst_model", "") or ""
        ).strip()
        if configured.startswith("models/"):
            configured = configured.split("/", 1)[1]

        configured_candidate = configured if configured.startswith("gemini-") else ""
        fallback = pick_first_available_or_known(
            configured_candidate,
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
        )
        return fallback or "gemini-2.5-flash"

    def _ask_with_gemini(
        self,
        question: str,
        context_events: list[dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        if not getattr(self.settings, "gemini_api_key", None):
            return {
                "error": (
                    "CHRONOS_GEMINI_API_KEY not configured. Set a dedicated Chronos "
                    "key or opt into the shared GEMINI_API_KEY with "
                    "CHRONOS_ALLOW_SHARED_GEMINI_KEY=1"
                )
            }

        prompt = "\n\n".join(self._format_context_event(evt) for evt in context_events)
        instructions = system_prompt or self._default_system_prompt()
        model_name = self._select_gemini_model(model)

        full_prompt = f"""{instructions}

Evidence blocks from the user's recordings:

{prompt}

Question: {question}

Answer using only the evidence above. If the evidence is thin or missing, say so plainly."""

        try:
            client = get_genai_client()
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            answer = (getattr(response, "text", "") or "").strip()
            if not answer:
                return {"error": "Gemini returned no visible answer text"}

            usage = getattr(response, "usage_metadata", None)
            if usage:
                from src.chronos.cost_tracker import track_usage

                track_usage(
                    model_name,
                    "generate",
                    input_tokens=getattr(usage, "prompt_token_count", 0),
                    output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
                    + (getattr(usage, "thoughts_token_count", 0) or 0),
                )

            return {
                "answer": answer,
                "model": model_name,
                "response_id": None,
                "reasoning_summary": None,
                "config": {
                    "provider": "gemini",
                    "fallback_from": "openai",
                },
                "usage": {
                    "input_tokens": (
                        getattr(usage, "prompt_token_count", 0) if usage else 0
                    ),
                    "output_tokens": (
                        getattr(usage, "candidates_token_count", 0) if usage else 0
                    ),
                    "reasoning_tokens": 0,
                    "total_tokens": (
                        getattr(usage, "total_token_count", 0) if usage else 0
                    ),
                },
            }
        except Exception as exc:
            logger.exception("Gemini Ask Chronos fallback failed")
            return {"error": str(exc)}

    def _ask_with_local(
        self,
        question: str,
        context_events: list[dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            from src.chronos.local_llm_service import LocalLLMService

            local = LocalLLMService(settings=self.settings)
            budget_words = max(350, int(local.max_context / 1.5) - 450)
            blocks: list[str] = []
            used_words = 0
            for event in context_events:
                block = self._format_context_event(event)
                block_words = len(block.split())
                if blocks and used_words + block_words > budget_words:
                    break
                blocks.append(block)
                used_words += block_words

            evidence = "\n\n".join(blocks).strip()
            instructions = system_prompt or self._default_system_prompt()
            prompt = f"""{instructions}

You are running in local offline degraded mode on a small Raspberry Pi model. Use only the evidence below. If the evidence is missing or too thin, say that plainly. Keep the answer concise and do not invent facts.

Evidence blocks:

{evidence or "No evidence blocks were provided."}

Question: {question}

Answer:"""
            result = local.generate(prompt, task="ask", temperature=0.0, timeout=180.0)
            if result.get("skipped"):
                return {
                    "error": f"Local LLM skipped: {result.get('reason', 'unknown reason')}"
                }

            answer = (result.get("text") or "").strip()
            if not answer:
                return {"error": "Local LLM returned no visible answer text"}

            return {
                "answer": answer,
                "model": result.get("model")
                or getattr(self.settings, "chronos_local_llm_model", "local"),
                "response_id": None,
                "reasoning_summary": None,
                "config": {
                    "provider": result.get("provider")
                    or getattr(self.settings, "chronos_local_llm_provider", "local"),
                    "local_degraded": True,
                    "evidence_blocks_used": len(blocks),
                },
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                },
            }
        except Exception as exc:
            logger.exception("Local Ask Chronos fallback failed")
            return {"error": str(exc)}

    def ask(
        self,
        question: str,
        context_events: list[dict[str, Any]],
        *,
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
    ) -> dict[str, Any]:
        requested_model = (
            model
            or getattr(self.settings, "chronos_analyst_model", None)
            or getattr(self.settings, "openai_model", None)
        )

        if self._prefers_gemini(requested_model):
            gemini_result = self._ask_with_gemini(
                question,
                context_events,
                system_prompt=system_prompt,
                model=requested_model,
            )
            if "error" not in gemini_result:
                return gemini_result
            if not self._openai.available or not self._allows_openai_fallback(
                requested_model
            ):
                local_result = self._ask_with_local(
                    question,
                    context_events,
                    system_prompt=system_prompt,
                )
                return local_result if "error" not in local_result else gemini_result

            logger.warning("Gemini Ask Chronos failed: %s", gemini_result["error"])

        if not self._openai.available:
            if getattr(self.settings, "gemini_api_key", None):
                return self._ask_with_gemini(
                    question,
                    context_events,
                    system_prompt=system_prompt,
                    model=requested_model,
                )
            local_result = self._ask_with_local(
                question,
                context_events,
                system_prompt=system_prompt,
            )
            if "error" not in local_result:
                return local_result
            return {
                "error": "No AI provider configured (set CHRONOS_GEMINI_API_KEY, OPENAI_API_KEY, or enable CHRONOS_LOCAL_LLM_ENABLED)"
            }

        # Ask is a handful of calls a day against the whole corpus, so it
        # reasons harder than the bulk chunking that shares this service.
        if not reasoning:
            _ask_effort = os.getenv("CHRONOS_ASK_REASONING_EFFORT", "xhigh").strip().lower()
            reasoning = _ask_effort if _ask_effort in {
                "none", "low", "medium", "high", "xhigh"
            } else "xhigh"

        result = self._openai.ask(
            question=question,
            context_events=context_events,
            system_prompt=system_prompt,
            previous_response_id=previous_response_id,
            model=(
                requested_model
                if not self._prefers_gemini(requested_model)
                else getattr(self.settings, "openai_model", None)
            ),
            reasoning=reasoning,
            reasoning_summary=reasoning_summary,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            verbosity=verbosity,
            service_tier=service_tier,
        )
        if "error" in result:
            logger.warning("OpenAI Ask Chronos failed: %s", result["error"])
            local_result = self._ask_with_local(
                question,
                context_events,
                system_prompt=system_prompt,
            )
            return local_result if "error" not in local_result else result

        config = dict(result.get("config") or {})
        config.setdefault("provider", "openai")
        result["config"] = config
        return result
