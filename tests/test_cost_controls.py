from types import SimpleNamespace
from unittest.mock import Mock

from src.chronos import cost_tracker
from src.chronos.transcript_processor import TranscriptProcessor


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
