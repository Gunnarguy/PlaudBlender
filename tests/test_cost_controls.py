from types import SimpleNamespace
from unittest.mock import Mock

from src.chronos import cost_tracker
from src.chronos.transcript_processor import TranscriptProcessor
from src.models.chronos_schemas import GeminiEventOutput


def _sample_output() -> GeminiEventOutput:
    return GeminiEventOutput.model_validate(
        {
            "events": [
                {
                    "event_id": "evt-1",
                    "recording_id": "rec-1",
                    "start_ts": "2026-04-02T10:00:00Z",
                    "end_ts": "2026-04-02T10:05:00Z",
                    "day_of_week": "Wednesday",
                    "hour_of_day": 10,
                    "clean_text": "Reviewed the provider fallback path and extracted structured events.",
                    "category": "work",
                    "sentiment": 0.2,
                    "keywords": ["provider", "fallback"],
                    "speaker": "self_talk",
                }
            ],
            "total_events": 1,
        }
    )


def test_normalize_model_name_strips_google_prefix():
    assert (
        cost_tracker.normalize_model_name("models/gemini-3-flash-preview")
        == "gemini-3-flash-preview"
    )


def test_paid_gemini_flash_uses_paid_tier(monkeypatch):
    monkeypatch.setattr(
        cost_tracker,
        "get_settings",
        lambda: SimpleNamespace(gemini_billing_tier="paid"),
    )

    assert (
        cost_tracker.estimate_cost("gemini-3-flash-preview", 1_000_000, 1_000_000)
        == 3.5
    )


def test_legacy_openai_model_alias_uses_current_pricing():
    pricing = cost_tracker.get_pricing("gpt-5-mini")

    assert pricing["label"] == "GPT-5.4 Mini"
    assert pricing["input_per_mtok"] == 0.75
    assert pricing["output_per_mtok"] == 4.50


def test_json_repair_defaults_to_cleaning_model():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(model_name="gemini-3-flash-preview"),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_allow_paid_gemini_fallback=False,
        chronos_analyst_model="gemini-3.1-pro-preview",
    )

    assert processor._get_json_repair_model_name() == "gemini-3-flash-preview"


def test_json_repair_can_opt_into_paid_fallback():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(model_name="gemini-3-flash-preview"),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_allow_paid_gemini_fallback=True,
        chronos_analyst_model="gemini-3.1-pro-preview",
    )

    assert processor._get_json_repair_model_name() == "gemini-3.1-pro-preview"


def test_invalid_processing_provider_defaults_to_auto():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="totally-bogus"
    )

    assert processor._get_processing_provider() == "auto"


def test_openai_provider_does_not_require_gemini_engine(monkeypatch):
    output = _sample_output()

    class DummyOpenAIService:
        def extract_events(self, prompt, *, recording_id, system_prompt=None):
            assert "RAW TRANSCRIPT" in prompt
            assert recording_id == "rec-1"
            return {"output": output, "model": "gpt-5.4", "usage": {}}

    monkeypatch.setattr(
        "src.chronos.transcript_processor.OpenAIResponseService",
        lambda: DummyOpenAIService(),
    )

    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=None,
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="openai",
        openai_model="gpt-5.4",
    )

    result = processor.process_transcript_text(
        transcript_text=("This transcript is long enough to process. " * 20),
        recording_id="rec-1",
        verbose=False,
        recording_date="2026-04-02",
    )

    assert result is not None
    assert result.total_events == 1
    assert processor.engine is None


def test_auto_provider_falls_back_to_openai(monkeypatch):
    output = _sample_output()
    order = []

    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="auto",
        openai_api_key="test-key",
        openai_model="gpt-5.4",
    )

    def fake_gemini(*args, **kwargs):
        order.append("gemini")
        processor._last_processing_error = "403 from Gemini"
        return None

    def fake_openai(*args, **kwargs):
        order.append("openai")
        return output

    monkeypatch.setattr(processor, "_process_transcript_text_gemini", fake_gemini)
    monkeypatch.setattr(processor, "_process_transcript_text_openai", fake_openai)
    monkeypatch.setattr("app_v2.services.xray.xray_log", lambda *args, **kwargs: None)

    result = processor.process_transcript_text(
        transcript_text=("This transcript is long enough to process. " * 20),
        recording_id="rec-1",
        verbose=False,
        recording_date="2026-04-02",
    )

    assert result is not None
    assert result.total_events == 1
    assert order == ["gemini", "openai"]


def test_auto_provider_skips_openai_without_key(monkeypatch):
    order = []

    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="auto",
        openai_api_key=None,
        openai_model="gpt-5.4",
    )

    def fake_gemini(*args, **kwargs):
        order.append("gemini")
        processor._last_processing_error = "Gemini returned no events"
        return None

    def fake_openai(*args, **kwargs):
        order.append("openai")
        raise AssertionError("OpenAI fallback should not run without a key")

    monkeypatch.setattr(processor, "_process_transcript_text_gemini", fake_gemini)
    monkeypatch.setattr(processor, "_process_transcript_text_openai", fake_openai)
    monkeypatch.setattr("app_v2.services.xray.xray_log", lambda *args, **kwargs: None)

    result = processor.process_transcript_text(
        transcript_text=("This transcript is long enough to process. " * 20),
        recording_id="rec-1",
        verbose=False,
        recording_date="2026-04-02",
    )

    assert result is None
    assert order == ["gemini"]
    assert "Gemini returned no events" in str(processor._last_processing_error)
