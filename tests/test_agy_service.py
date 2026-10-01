import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.chronos import cost_tracker
from src.chronos.agy_service import AgyBridgeService, flatten_schema, parse_json_objects
from src.chronos.openai_service import _OpenAIEventOutput
from src.chronos.transcript_processor import TranscriptProcessor
from src.models.chronos_schemas import GeminiEventOutput

EVENT = {
    "event_id": "rec-1-1",
    "recording_id": "rec-1",
    "start_ts": "2026-09-30T09:15:00",
    "end_ts": "2026-09-30T10:00:00",
    "day_of_week": "Wednesday",
    "hour_of_day": 9,
    "clean_text": "OR prep meeting with Dr. Patel about the sterilization checklist.",
    "category": "meeting",
    "keywords": ["OR prep"],
    "speaker": "self_talk",
}


def _settings(tmp_path, token="tok", **overrides):
    token_file = tmp_path / "token"
    if token is not None:
        token_file.write_text(token)
    base = dict(
        chronos_processing_provider="agy",
        chronos_agy_token_file=str(token_file),
        chronos_agy_bridge_url="http://127.0.0.1:1",
        chronos_agy_model="gemini-3.8-flash-high",
        chronos_agy_timeout_seconds=60,
        chronos_agy_fallback_openai=True,
        chronos_openai_enabled=True,
        openai_api_key="test-key",
        chronos_local_llm_enabled=False,
        gemini_api_key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _output() -> GeminiEventOutput:
    return GeminiEventOutput.model_validate({"events": [EVENT], "total_events": 1})


def test_flatten_schema_inlines_refs_for_agy():
    flat = flatten_schema(_OpenAIEventOutput.model_json_schema())
    text = json.dumps(flat)
    assert "$ref" not in text and "$defs" not in text
    event = flat["properties"]["events"]["items"]
    assert "clean_text" in event["properties"]
    assert "Wednesday" in json.dumps(event["properties"]["day_of_week"])


def test_parse_json_objects_handles_duplicate_reply_fences_and_prose():
    obj = {"events": [EVENT], "total_events": 1}
    text = "Here you go:\n```json\n" + json.dumps(obj, indent=2) + "\n```\n" + json.dumps(obj)
    found = parse_json_objects(text)
    assert len(found) == 2 and found[-1] == obj


def test_agy_pricing_is_zero():
    pricing = cost_tracker.get_pricing("agy/gemini-3.8-flash-high")
    assert pricing["input_per_mtok"] == 0 and pricing["output_per_mtok"] == 0
    assert cost_tracker.estimate_cost("agy/gemini-3.8-flash-high", 100_000, 50_000) == 0


def test_complete_without_token_fails_closed_without_network(tmp_path):
    svc = AgyBridgeService(_settings(tmp_path, token=None))
    assert svc.available is False
    result = svc.complete("system", "user")
    assert result["ok"] is False and "token" in result["error"]


def test_extract_events_takes_last_valid_object_and_logs_zero_cost_usage(tmp_path, monkeypatch):
    svc = AgyBridgeService(_settings(tmp_path))
    good = {"events": [EVENT], "total_events": 7}
    reply = json.dumps({"events": [{"bogus": True}]}) + "\n" + json.dumps(good)
    captured = {}
    monkeypatch.setattr(
        svc, "complete",
        lambda system, user, schema=None: captured.update(system=system, user=user, schema=schema)
        or {"ok": True, "text": reply, "usage": {"input_tokens": 10, "output_tokens": 5}, "model": "gemini-3.8-flash-high"},
    )
    tracked = []
    monkeypatch.setattr(cost_tracker, "track_usage", lambda *a, **k: tracked.append((a, k)))

    result = svc.extract_events("INSTRUCTIONS + RAW TRANSCRIPT", recording_id="rec-1")

    assert result["output"].total_events == 1  # normalized to the real count
    assert result["model"] == "agy/gemini-3.8-flash-high"
    assert "RAW TRANSCRIPT" in captured["system"] and len(captured["user"]) < 300
    assert "$ref" not in json.dumps(captured["schema"])
    assert tracked and tracked[0][0][:2] == ("agy/gemini-3.8-flash-high", "generate")


def test_extract_events_reports_bridge_errors(tmp_path, monkeypatch):
    svc = AgyBridgeService(_settings(tmp_path))
    monkeypatch.setattr(svc, "complete", lambda *a, **k: {"ok": False, "error": "bridge busy"})
    assert svc.extract_events("x", recording_id="rec-1") == {"error": "bridge busy"}

    monkeypatch.setattr(svc, "complete", lambda *a, **k: {"ok": True, "text": "no json here"})
    assert "no valid event JSON" in svc.extract_events("x", recording_id="rec-1")["error"]


def _processor(settings):
    processor = TranscriptProcessor(db_session=Mock(), plaud_client=Mock(), engine=Mock())
    processor.__dict__["settings"] = settings
    return processor


def test_agy_provider_resolution(tmp_path):
    assert _processor(_settings(tmp_path))._get_processing_provider() == "agy"
    # token missing: metered fallback only when the owner allows it
    (tmp_path / "empty").mkdir()
    no_token = _settings(tmp_path / "empty", token=None)
    assert _processor(no_token)._get_processing_provider() == "openai"
    no_token.chronos_agy_fallback_openai = False
    assert _processor(no_token)._get_processing_provider() == "agy"
    assert _processor(_settings(tmp_path))._provider_label() == "AGY"


def test_agy_success_never_touches_openai(tmp_path):
    processor = _processor(_settings(tmp_path))
    processor._process_transcript_text_agy = Mock(return_value=_output())
    processor._process_transcript_text_openai = Mock()
    out = processor.process_transcript_text("word " * 40, "rec-1", verbose=False)
    assert out.total_events == 1
    processor._process_transcript_text_openai.assert_not_called()


def test_agy_failure_falls_back_to_openai_when_allowed(tmp_path):
    processor = _processor(_settings(tmp_path))

    def agy_fails(*a, **k):
        processor._last_processing_error = "bridge busy"
        return None

    processor._process_transcript_text_agy = Mock(side_effect=agy_fails)
    processor._process_transcript_text_openai = Mock(return_value=_output())
    assert processor.process_transcript_text("word " * 40, "rec-1", verbose=False).total_events == 1

    processor.settings.chronos_agy_fallback_openai = False
    processor._process_transcript_text_openai.reset_mock()
    assert processor.process_transcript_text("word " * 40, "rec-1", verbose=False) is None
    processor._process_transcript_text_openai.assert_not_called()
    assert processor._last_processing_error == "bridge busy"



def test_bridges_are_tried_in_order_until_one_answers(tmp_path, monkeypatch):
    svc = AgyBridgeService(_settings(tmp_path, chronos_agy_bridge_url="http://127.0.0.1:8798, http://127.0.0.1:8799"))
    assert svc.urls == ["http://127.0.0.1:8798", "http://127.0.0.1:8799"]
    calls = []

    def fake(url, token, system, user, schema):
        calls.append(url)
        if url.endswith("8798"):
            return {"ok": False, "error": "authentication failed or timed out"}
        return {"ok": True, "text": "{}"}

    monkeypatch.setattr(svc, "_complete_at", fake)
    result = svc.complete("s", "u")
    assert result["ok"] and result["bridge"] == "http://127.0.0.1:8799" and calls == ["http://127.0.0.1:8798", "http://127.0.0.1:8799"]

    calls.clear()
    monkeypatch.setattr(svc, "_complete_at", lambda url, *a: calls.append(url) or {"ok": True, "text": "{}"})
    assert svc.complete("s", "u")["bridge"] == "http://127.0.0.1:8798" and calls == ["http://127.0.0.1:8798"]

    monkeypatch.setattr(svc, "_complete_at", lambda url, *a: {"ok": False, "error": f"down {url[-4:]}"})
    failed = svc.complete("s", "u")
    assert not failed["ok"] and "down 8798" in failed["error"] and "down 8799" in failed["error"]
