"""AGY bridge client: Chronos transcript extraction on the owner's Google AI Ultra plan.

WHY. The owner's rule (2026-09-30): Gemini runs as ``gemini-3.8-flash-high`` through the
AGY (Antigravity) subscription, never a metered API. JobScoutOS already runs a bridge for
exactly this on the Pi host -- the user unit ``jobscout-agy-bridge`` on :8799, ``POST
/complete`` with a shared token -- so Chronos reuses it (CHRONOS_PROCESSING_PROVIDER=agy).

Measured on the Pi, 2026-09-30, with a 4-event sample transcript:

- ``agy --json-schema`` does not resolve ``$ref``/``$defs``. With Pydantic's schema the
  agent looped ~9 turns (190k tokens, 119 s) and still returned the wrong fields. With the
  schema flattened it returned the right fields in 92 s. So :func:`flatten_schema` first.
- The reply can hold the JSON object twice (pretty-printed, then compact). Parse every
  top-level object and keep the last one that validates.
- One call is one ``agy`` process of ~400 MB RSS for 1-3+ minutes; the bridge runs at most
  two at once (shared with JobScoutOS). Fine for ~5 recordings a day; not for per-event
  micro-calls, which is why graph entity extraction stays on its own provider.
- The bridge splits only its ``system`` field into conversation turns (each under ~110 KB),
  and Linux refuses one argument over 128 KiB. So the long material (instructions +
  transcript) goes in ``system`` and ``user`` stays a one-line task.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from src.config import get_settings
from src.models.chronos_schemas import GeminiEventOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Chronos, an event extraction engine. "
    "Read the transcript and return only structured Chronos events that satisfy the provided schema. "
    "Do not summarize away concrete details. Preserve exact meaning, names, and technical terminology."
)
TASK = (
    "Extract every event from the RAW TRANSCRIPT in your instructions, following the schema exactly. "
    "Return only the JSON object."
)


def flatten_schema(schema: dict) -> dict:
    """Inline ``$defs``/``$ref``: agy's ``--json-schema`` does not follow references."""
    defs = schema.get("$defs") or {}

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 20:
            raise ValueError("schema $ref nesting too deep (recursive model?)")
        if isinstance(node, dict):
            if "$ref" in node:
                target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
                target.update({k: v for k, v in node.items() if k != "$ref"})
                return walk(target, depth + 1)
            return {k: walk(v, depth) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(item, depth) for item in node]
        return node

    return walk(schema)


def parse_json_objects(text: str) -> list[dict]:
    """Every top-level JSON object in ``text``, in order (tolerates fences and prose)."""
    decoder = json.JSONDecoder()
    found: list[dict] = []
    index = 0
    while True:
        index = text.find("{", index)
        if index < 0:
            return found
        try:
            obj, end = decoder.raw_decode(text, index)
        except ValueError:
            index += 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        index = end


class AgyBridgeService:
    """Thin client for the host's AGY bridge (see module docstring)."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        # Comma-separated, tried in order: e.g. the GPD's bridge through the Pi's SSH tunnel
        # first (its RAM, not the Pi's), then the Pi's own bridge when the GPD is away.
        raw = getattr(self.settings, "chronos_agy_bridge_url", "http://127.0.0.1:8799") or ""
        self.urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()] or ["http://127.0.0.1:8799"]
        self.url = self.urls[0]
        self.model = getattr(self.settings, "chronos_agy_model", "gemini-3.8-flash-high")
        self.timeout_s = int(getattr(self.settings, "chronos_agy_timeout_seconds", 900))
        self.token_file = getattr(self.settings, "chronos_agy_token_file", "")

    def _token(self) -> str:
        try:
            with open(self.token_file, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    @property
    def available(self) -> bool:
        return bool(self._token())

    def complete(self, system: str, user: str, schema: Optional[dict] = None) -> dict:
        """POST /complete to each configured bridge in turn until one answers ok.

        Always returns a dict with ``ok``; never raises. ``bridge`` names the one that answered.
        """
        token = self._token()
        if not token:
            return {"ok": False, "error": f"AGY bridge token not readable at {self.token_file}"}
        errors = []
        for url in self.urls:
            result = self._complete_at(url, token, system, user, schema)
            if result.get("ok"):
                result["bridge"] = url
                return result
            errors.append(f"{url}: {result.get('error')}")
            logger.warning("AGY bridge %s failed: %s", url, str(result.get("error"))[:200])
        return {"ok": False, "error": " | ".join(errors)}

    def _complete_at(self, url: str, token: str, system: str, user: str, schema: Optional[dict]) -> dict:
        body = {
            "system": system,
            "user": user,
            "schema": schema,
            "model": self.model,
            "timeout_s": self.timeout_s,
        }
        request = urllib.request.Request(
            url + "/complete",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Bridge-Token": token},
        )
        # The bridge may queue up to 600 s for a free slot before it starts the call.
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s + 700) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": f"AGY bridge HTTP {exc.code}: {exc.read()[:200]!r}"}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return {"ok": False, "error": f"AGY bridge unreachable at {url}: {exc}"}

    def extract_events(self, instructions: str, *, recording_id: str) -> dict:
        """Same contract as ``OpenAIResponseService.extract_events``:
        ``{"output": GeminiEventOutput, "model": str, "usage": dict}`` or ``{"error": str}``."""
        from src.chronos.openai_service import _OpenAIEventOutput

        try:
            from app_v2.services.xray import xray_log
        except ImportError:
            xray_log = None

        if xray_log:
            xray_log("pipeline", "agy", f"Sending transcript to AGY ({self.model})")
        started = time.perf_counter()
        schema = flatten_schema(_OpenAIEventOutput.model_json_schema())
        result = self.complete(SYSTEM_PROMPT + "\n\n" + instructions, TASK, schema)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        if not result.get("ok"):
            error = str(result.get("error") or "AGY bridge returned ok=false")
            logger.error("AGY extraction failed for %s: %s", recording_id, error)
            if xray_log:
                xray_log("pipeline", "agy", f"AGY error: {error[:100]}", level="error")
            return {"error": error}

        parsed = None
        last_error: Optional[Exception] = None
        for candidate in reversed(parse_json_objects(result.get("text") or "")):
            try:
                parsed = _OpenAIEventOutput.model_validate(candidate)
                break
            except Exception as exc:  # pydantic ValidationError and friends
                last_error = exc
        if parsed is None:
            detail = f": {last_error}" if last_error else ""
            return {"error": f"AGY returned no valid event JSON{detail}"[:500]}

        usage = result.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        model_label = f"agy/{result.get('model') or self.model}"

        from src.chronos.cost_tracker import track_usage

        # Subscription calls: logged for volume, priced at $0 (see cost_tracker).
        track_usage(
            model_label,
            "generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            recording_id=recording_id,
        )
        output = GeminiEventOutput(
            events=parsed.events,
            processing_metadata=None,
            total_events=len(parsed.events),
        )
        if xray_log:
            xray_log(
                "pipeline",
                "agy",
                f"AGY extracted {output.total_events} events",
                duration_ms=elapsed_ms,
                detail=f"model={model_label} in={input_tokens} out={output_tokens} "
                f"turns={result.get('turns')} secs={result.get('secs')}",
            )
        return {
            "output": output,
            "model": model_label,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": int(usage.get("thinking_tokens") or 0),
                "total_tokens": input_tokens + output_tokens,
            },
        }
