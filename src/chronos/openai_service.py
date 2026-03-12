"""OpenAI Responses API service for RAG and conversational queries.

Uses the OpenAI Responses API (not Chat Completions) for rich,
context-aware answers backed by Chronos event data.
"""

import json
import logging
from typing import List, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIResponseService:
    """Wraps OpenAI Responses API for Chronos RAG queries."""

    def __init__(self):
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model
        self._temperature = settings.openai_temperature
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def ask(
        self,
        question: str,
        context_events: List[dict],
        system_prompt: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> dict:
        """Answer a question using retrieved event context via the Responses API.

        Args:
            question: User's natural language question.
            context_events: List of dicts with date, time, category, text fields.
            system_prompt: Optional system instructions override.
            previous_response_id: For follow-up questions (stateful conversation).
            reasoning: Reasoning effort for GPT-5.4: none, low, medium, high, xhigh.
                       None uses model default.

        Returns:
            dict with 'answer', 'model', 'response_id', 'usage' keys.
        """
        if not self.available:
            return {"error": "OPENAI_API_KEY not configured"}

        # Build event context block
        context_lines = []
        for evt in context_events:
            line = f"[{evt.get('date', '?')} {evt.get('time', '')}] ({evt.get('category', 'unknown')}) {evt.get('text', '')}"
            context_lines.append(line)
        context_block = "\n".join(context_lines)

        default_system = (
            "You are Chronos, an AI assistant that answers questions about the user's "
            "voice recordings and daily events. You have access to timestamped events "
            "extracted from Plaud voice recordings. Answer concisely and accurately, "
            "referencing specific dates and times when relevant. If the events don't "
            "contain enough information, say so honestly."
        )

        instructions = system_prompt or default_system

        user_message = f"""Based on these events from my voice recordings:

{context_block}

Question: {question}"""

        try:
            client = self._get_client()

            kwargs = {
                "model": self._model,
                "instructions": instructions,
                "input": user_message,
                "temperature": self._temperature,
            }

            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            # GPT-5.4 reasoning levels: none, low, medium, high, xhigh
            valid_reasoning = {"low", "medium", "high", "xhigh"}
            if reasoning and reasoning in valid_reasoning:
                kwargs["reasoning"] = {"effort": reasoning}

            response = client.responses.create(**kwargs)

            # Extract text from response
            answer = ""
            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            answer += content.text

            return {
                "answer": answer,
                "model": response.model,
                "response_id": response.id,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.exception("OpenAI Responses API call failed")
            return {"error": str(e)}

    def check_connection(self) -> tuple:
        """Test OpenAI connectivity. Returns (ok: bool, detail: str)."""
        if not self.available:
            return False, "OPENAI_API_KEY not set"
        try:
            client = self._get_client()
            models = client.models.list()
            model_ids = [m.id for m in models.data[:5]]
            return True, f"Connected — {self._model} (found {len(models.data)} models)"
        except Exception as e:
            return False, str(e)[:80]
