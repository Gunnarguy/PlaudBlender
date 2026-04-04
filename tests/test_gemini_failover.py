import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from src.chronos.engine import ChronosEngine
from src.chronos.transcript_processor import TranscriptProcessor


def test_pick_failover_model_uses_stable_flash_for_preview_overload(monkeypatch):
    engine = ChronosEngine.__new__(ChronosEngine)
    engine.__dict__["settings"] = SimpleNamespace(
        chronos_cleaning_model="gemini-3-flash-preview"
    )
    engine.model_name = "gemini-3-flash-preview"

    monkeypatch.setattr(
        "src.chronos.engine.pick_first_available_or_known",
        lambda *candidates: candidates[0] if candidates else None,
    )

    chosen = engine.pick_failover_model(
        Exception(
            "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}"
        )
    )

    assert chosen == "gemini-2.5-flash"


def test_transcript_processor_switches_models_after_unavailable(monkeypatch):
    monkeypatch.setattr("app_v2.services.xray.xray_log", lambda *args, **kwargs: None)

    payload = {
        "events": [
            {
                "event_id": "evt-1",
                "recording_id": "rec-1",
                "start_ts": "2026-04-02T10:00:00Z",
                "end_ts": "2026-04-02T10:05:00Z",
                "day_of_week": "Thursday",
                "hour_of_day": 10,
                "clean_text": "Reviewed the Gemini fallback path and stabilized transcript processing.",
                "category": "work",
                "sentiment": 0.2,
                "keywords": ["gemini", "fallback"],
                "speaker": "self_talk",
            }
        ],
        "total_events": 1,
    }

    class DummyEngine:
        def __init__(self):
            self.model_name = "gemini-3-flash-preview"
            self._thinking_level = None
            self.calls = []
            self.client = SimpleNamespace(
                models=SimpleNamespace(generate_content=self.generate_content)
            )

        def _build_prompt(self, recording_id: str, recording_date: str = "") -> str:
            return "Prompt"

        def pick_failover_model(self, err: Exception, current_model=None):
            if (
                "503 UNAVAILABLE" in str(err)
                and self.model_name == "gemini-3-flash-preview"
            ):
                return "gemini-2.5-flash"
            return None

        def generate_content(self, *, model, contents, config):
            self.calls.append(model)
            if len(self.calls) == 1:
                raise Exception(
                    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}"
                )
            return SimpleNamespace(
                text=json.dumps(payload),
                parsed=None,
                usage_metadata=None,
            )

    engine = DummyEngine()
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=cast(Any, engine),
    )

    result = processor.process_transcript_text(
        transcript_text=(
            "This is a sufficiently long transcript about debugging Gemini failures. "
            * 8
        ),
        recording_id="rec-1",
        max_retries=2,
        verbose=False,
        recording_date="2026-04-02",
    )

    assert result is not None
    assert result.total_events == 1
    assert engine.calls == ["gemini-3-flash-preview", "gemini-2.5-flash"]
    assert engine.model_name == "gemini-2.5-flash"
