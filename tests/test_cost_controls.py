from datetime import datetime
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


def test_invalid_processing_provider_defaults_to_gemini_when_no_paid_provider_enabled():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="totally-bogus"
    )

    assert processor._get_processing_provider() == "gemini"


def test_openai_provider_falls_back_to_local_when_openai_disabled():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="openai",
        chronos_openai_enabled=False,
        openai_api_key="test-key",
        chronos_local_llm_enabled=True,
        gemini_api_key=None,
    )

    assert processor._get_processing_provider() == "local"


def test_local_provider_extracts_chunk_events_without_cloud():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=None,
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="local",
        chronos_local_llm_enabled=True,
        chronos_local_llm_model="qwen2.5:0.5b",
    )

    result = processor.process_transcript_text(
        transcript_text=("Worked on local Chronos processing and removed OpenAI quota usage. " * 80),
        recording_id="rec-local",
        verbose=False,
        recording_date="2026-05-28",
    )

    assert result is not None
    assert result.total_events and result.total_events > 0
    assert result.processing_metadata["provider"] == "local"
    assert result.processing_metadata["cloud_ai_used"] is False


def test_local_provider_skips_short_trailing_chunk():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=None,
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="local",
        chronos_local_llm_enabled=True,
        chronos_local_llm_model="qwen2.5:0.5b",
    )

    transcript_text = (("planning " * 420) + "my dog.").strip()

    result = processor.process_transcript_text(
        transcript_text=transcript_text,
        recording_id="rec-local-short-tail",
        verbose=False,
        recording_date="2026-05-28",
    )

    assert result is not None
    assert result.total_events == 1
    assert result.events[0].clean_text.startswith("planning planning")


def test_openai_provider_does_not_require_gemini_engine(monkeypatch):
    output = _sample_output()

    class DummyOpenAIService:
        def extract_events(self, prompt, *, recording_id, system_prompt=None, model=None):
            assert "RAW TRANSCRIPT" in prompt
            assert recording_id == "rec-1"
            assert model == "gpt-5.5"
            return {"output": output, "model": "gpt-5.5", "usage": {}}

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
        chronos_openai_enabled=True,
        openai_model="gpt-5.4",
        openai_api_key="test-key",
        chronos_cleaning_model="gpt-5.5",
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


def test_auto_provider_aliases_to_openai(monkeypatch):
    output = _sample_output()
    order = []

    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="auto",
        chronos_openai_enabled=True,
        openai_api_key="test-key",
        openai_model="gpt-5.4",
        chronos_cleaning_model="gpt-5.5",
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
    assert order == ["openai"]


def test_explicit_gemini_provider_keeps_gemini_path(monkeypatch):
    order = []

    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="gemini",
        openai_api_key="test-key",
        openai_model="gpt-5.4",
        chronos_cleaning_model="gpt-5.5",
    )

    def fake_gemini(*args, **kwargs):
        order.append("gemini")
        processor._last_processing_error = "Gemini returned no events"
        return None

    def fake_openai(*args, **kwargs):
        order.append("openai")
        raise AssertionError("OpenAI should not run for explicit gemini provider")

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


def test_process_recording_id_uses_cached_transcript_for_notion(monkeypatch):
    recording = SimpleNamespace(
        recording_id="notion:page-1",
        title="Notion page",
        created_at=datetime(2026, 5, 31, 12, 0, 0),
        duration_seconds=300,
        local_audio_path="",
        source="notion",
        device_id="notion",
        checksum=None,
        transcript=("This transcript is long enough to process without refetching from Plaud. " * 10),
        plaud_ai_summary=None,
        plaud_extracted_data=None,
    )

    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )
    processor.__dict__["settings"] = SimpleNamespace(
        chronos_processing_provider="local",
        chronos_local_llm_enabled=True,
        chronos_local_llm_model="qwen2.5:0.5b",
    )

    events_output = _sample_output().model_copy(
        update={"events": [event.model_copy(update={"recording_id": "notion:page-1"}) for event in _sample_output().events]}
    )

    monkeypatch.setattr(
        "src.chronos.transcript_processor.get_chronos_recording",
        lambda db, recording_id: recording,
    )
    status_updates = []
    monkeypatch.setattr(
        "src.chronos.transcript_processor.mark_chronos_recording_status",
        lambda db, recording_id, status, error_message=None: status_updates.append(
            (recording_id, status, error_message)
        ),
    )
    stored_events = []
    monkeypatch.setattr(
        "src.chronos.transcript_processor.add_chronos_events",
        lambda db, events: stored_events.extend(events) or len(events),
    )
    monkeypatch.setattr(
        "src.chronos.transcript_processor.delete_chronos_events_by_recording",
        lambda db, recording_id: 0,
    )
    monkeypatch.setattr(
        "src.chronos.transcript_processor.set_chronos_recording_transcript",
        lambda db, recording_id, transcript_text: None,
    )
    monkeypatch.setattr(
        processor,
        "process_transcript_text",
        lambda transcript_text, record_id, **kwargs: events_output,
    )
    monkeypatch.setattr("app_v2.services.xray.xray_log", lambda *args, **kwargs: None)

    ok = processor.process_recording_id("notion:page-1")

    assert ok is True
    processor.plaud.get_recording.assert_not_called()
    assert stored_events
    assert status_updates[0][1] == "processing"
    assert status_updates[-1][1] == "completed"


def test_process_transcript_text_sets_reason_for_short_transcripts():
    processor = TranscriptProcessor(
        db_session=Mock(),
        plaud_client=Mock(),
        engine=Mock(),
    )

    result = processor.process_transcript_text(
        transcript_text="too short",
        recording_id="rec-short",
        verbose=False,
    )

    assert result is None
    assert processor._last_processing_error == (
        "Transcript too short to extract meaningful events"
    )
