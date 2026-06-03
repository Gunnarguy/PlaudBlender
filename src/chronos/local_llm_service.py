"""Optional local LLM sidecar support for Chronos.

Designed for tiny helper tasks on constrained machines (Raspberry Pi) and for
observability. The service is disabled by default and every decision is visible
through X-Ray/trace metadata.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Optional

from src.config import Settings, get_settings
from src.chronos.trace_service import trace_span

logger = logging.getLogger(__name__)


class LocalLLMService:
    """Small HTTP client for Ollama or llama.cpp-compatible local servers."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.enabled = bool(getattr(self.settings, "chronos_local_llm_enabled", False))
        self.provider = (
            getattr(self.settings, "chronos_local_llm_provider", "ollama") or "ollama"
        ).strip().lower()
        self.base_url = (
            getattr(self.settings, "chronos_local_llm_base_url", "http://127.0.0.1:11434")
            or "http://127.0.0.1:11434"
        ).strip().rstrip("/")
        self.model = (getattr(self.settings, "chronos_local_llm_model", "") or "").strip()
        self.max_context = int(getattr(self.settings, "chronos_local_llm_max_context", 4096) or 4096)
        raw_tasks = getattr(
            self.settings,
            "chronos_local_llm_allowed_tasks",
            "json_repair,entity_extract,classify,ask",
        )
        self.allowed_tasks = {
            task.strip().lower() for task in str(raw_tasks).split(",") if task.strip()
        }

    # ------------------------------------------------------------------
    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec - local configurable endpoint
            body = response.read().decode("utf-8", errors="replace")
        if not body.strip():
            return {}
        return json.loads(body)

    def status(self) -> dict[str, Any]:
        """Return local runtime health without raising."""
        base = {
            "enabled": self.enabled,
            "ok": False,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "max_context": self.max_context,
            "allowed_tasks": sorted(self.allowed_tasks),
        }
        if not self.enabled:
            base["detail"] = "disabled by CHRONOS_LOCAL_LLM_ENABLED"
            return base

        try:
            if self.provider == "ollama":
                data = self._request_json("GET", "/api/tags", timeout=3.0)
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                base.update({"ok": True, "models": models, "model_available": self.model in models})
            else:
                # llama.cpp server usually exposes /health or OpenAI-compatible /v1/models.
                try:
                    data = self._request_json("GET", "/health", timeout=3.0)
                    base.update({"ok": True, "detail": data or "healthy"})
                except Exception:
                    data = self._request_json("GET", "/v1/models", timeout=3.0)
                    base.update({"ok": True, "models": data.get("data", [])})
        except Exception as exc:
            base.update({"ok": False, "error": str(exc)[:300]})
        return base

    def can_run_task(self, task: str, prompt: str) -> tuple[bool, str]:
        """Return whether the local model should be used for this task."""
        normalized = (task or "").strip().lower()
        if not self.enabled:
            return False, "local LLM disabled"
        if normalized not in self.allowed_tasks:
            return False, f"task '{normalized}' is not allowed locally"
        approx_tokens = max(1, int(len(prompt.split()) * 1.3))
        if approx_tokens > self.max_context:
            return False, f"prompt too large for local context ({approx_tokens}>{self.max_context})"
        status = self.status()
        if not status.get("ok"):
            return False, status.get("error") or status.get("detail") or "local runtime unavailable"
        return True, "ok"

    def generate(
        self,
        prompt: str,
        *,
        task: str = "classify",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Generate text through the local sidecar with trace visibility."""
        allowed, reason = self.can_run_task(task, prompt)
        if not allowed:
            try:
                from app_v2.services.xray import xray_log

                xray_log(
                    "local",
                    "skip",
                    f"Skipped local LLM: {reason}",
                    provider=self.provider,
                    model=self.model,
                    status="skipped",
                    metadata={"task": task},
                    level="warn",
                )
            except Exception:
                pass
            return {"skipped": True, "reason": reason}

        with trace_span(
            operation="local-generate",
            source="local",
            stage="local_llm",
            provider=self.provider,
            model=self.model,
            message=f"Local LLM {task}",
            input_value=prompt,
            metadata={"task": task},
        ):
            if self.provider == "ollama":
                data = self._request_json(
                    "POST",
                    "/api/generate",
                    payload={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                    timeout=timeout,
                )
                return {
                    "text": data.get("response", ""),
                    "model": data.get("model", self.model),
                    "provider": self.provider,
                    "raw": data,
                }

            # llama.cpp OpenAI-compatible completion path
            data = self._request_json(
                "POST",
                "/v1/chat/completions",
                payload={
                    "model": self.model or "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=timeout,
            )
            text = ""
            choices = data.get("choices") or []
            if choices:
                text = ((choices[0].get("message") or {}).get("content") or "")
            return {"text": text, "model": self.model, "provider": self.provider, "raw": data}

    def embed(
        self,
        inputs: str | list[str],
        *,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> list[list[float]]:
        """Create embeddings through an Ollama-compatible local sidecar."""
        if not self.enabled:
            raise RuntimeError("local LLM disabled")
        if self.provider != "ollama":
            raise RuntimeError("local embeddings currently require Ollama")

        requested_model = (model or self.model or "").strip()
        if not requested_model:
            raise RuntimeError("local embedding model is not configured")

        payload_input = inputs if isinstance(inputs, list) else inputs
        data = self._request_json(
            "POST",
            "/api/embed",
            payload={"model": requested_model, "input": payload_input},
            timeout=timeout,
        )
        embeddings = data.get("embeddings") or []
        if not embeddings and isinstance(inputs, str):
            legacy = self._request_json(
                "POST",
                "/api/embeddings",
                payload={"model": requested_model, "prompt": inputs},
                timeout=timeout,
            )
            embedding = legacy.get("embedding") or []
            embeddings = [embedding] if embedding else []
        if not embeddings:
            raise RuntimeError(f"local embedding model {requested_model} returned no vectors")
        return [list(vector) for vector in embeddings]
