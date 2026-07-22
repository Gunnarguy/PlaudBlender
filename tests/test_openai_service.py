from types import SimpleNamespace

from src.chronos.openai_service import OpenAIResponseService


class _TransientOpenAIError(Exception):
    pass


def _make_settings():
    return SimpleNamespace(
        chronos_openai_enabled=True,
        openai_api_key="test-openai-key",
        openai_model="gpt-5.4",
        openai_temperature=0.7,
    )


def test_call_with_retry_retries_transient_errors(monkeypatch):
    attempts = []

    monkeypatch.setattr(
        OpenAIResponseService,
        "_retryable_error_types",
        staticmethod(lambda: (_TransientOpenAIError,)),
    )
    monkeypatch.setattr(
        OpenAIResponseService,
        "_sleep_before_retry",
        staticmethod(lambda _seconds: None),
    )

    def flaky_call():
        attempts.append("call")
        if len(attempts) < 3:
            raise _TransientOpenAIError("temporary outage")
        return "ok"

    result = OpenAIResponseService._call_with_retry("OpenAI ask", flaky_call)

    assert result == "ok"
    assert len(attempts) == 3


def test_ask_prefers_output_text(monkeypatch):
    monkeypatch.setattr("src.chronos.openai_service.get_settings", _make_settings)

    response = SimpleNamespace(
        output_text="Direct answer from output_text",
        output=[],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        model="gpt-5.4",
        id="resp_123",
        incomplete_details=None,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: response)
    )

    svc = OpenAIResponseService()
    monkeypatch.setattr(svc, "_get_client", lambda: client)
    monkeypatch.setattr(svc, "_call_with_retry", lambda _op, func, **_kwargs: func())
    monkeypatch.setattr(
        "src.chronos.cost_tracker.track_usage",
        lambda *args, **kwargs: None,
    )

    result = svc.ask("What happened?", [{"date": "2026-01-01", "time": "10:00 AM", "category": "work", "text": "Did the thing"}])

    assert result["answer"] == "Direct answer from output_text"
    assert result["config"]["provider"] == "openai"
    assert result["usage"]["total_tokens"] == 20


def test_ask_returns_error_for_incomplete_empty_response(monkeypatch):
    monkeypatch.setattr("src.chronos.openai_service.get_settings", _make_settings)

    response = SimpleNamespace(
        output_text="",
        output=[],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=0,
            total_tokens=12,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        model="gpt-5.4",
        id="resp_456",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: response)
    )

    svc = OpenAIResponseService()
    monkeypatch.setattr(svc, "_get_client", lambda: client)
    monkeypatch.setattr(svc, "_call_with_retry", lambda _op, func, **_kwargs: func())

    result = svc.ask("What happened?", [{"date": "2026-01-01", "time": "10:00 AM", "category": "work", "text": "Did the thing"}])

    assert "error" in result
    assert "output tokens" in result["error"]


def test_check_connection_uses_responses_api(monkeypatch):
    monkeypatch.setattr("src.chronos.openai_service.get_settings", _make_settings)

    response = SimpleNamespace(
        output_text="OK",
        output=[],
        incomplete_details=None,
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: response),
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[])),
    )

    svc = OpenAIResponseService()
    monkeypatch.setattr(svc, "_get_client", lambda: client)
    monkeypatch.setattr(svc, "_call_with_retry", lambda _op, func, **_kwargs: func())
    monkeypatch.setattr(OpenAIResponseService, "_CONNECTION_CACHE", {})

    ok, detail = svc.check_connection()

    assert ok is True
    assert "Responses API ready" in detail


def test_ask_uses_low_reasoning_and_default_output_cap(monkeypatch):
    monkeypatch.setattr("src.chronos.openai_service.get_settings", _make_settings)

    captured = {}
    response = SimpleNamespace(
        output_text="Condensed answer",
        output=[],
        usage=SimpleNamespace(
            input_tokens=20,
            output_tokens=12,
            total_tokens=32,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
        model="gpt-5.4",
        id="resp_defaults",
        incomplete_details=None,
    )

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(
        responses=SimpleNamespace(create=create)
    )

    svc = OpenAIResponseService()
    monkeypatch.setattr(svc, "_get_client", lambda: client)
    monkeypatch.setattr(svc, "_call_with_retry", lambda _op, func, **_kwargs: func())
    monkeypatch.setattr(
        "src.chronos.cost_tracker.track_usage",
        lambda *args, **kwargs: None,
    )

    result = svc.ask(
        "What happened?",
        [{"date": "2026-01-01", "time": "10:00 AM", "category": "work", "text": "Did the thing"}],
    )

    assert captured["reasoning"]["effort"] == "low"
    assert captured["max_output_tokens"] == OpenAIResponseService._DEFAULT_MAX_OUTPUT_TOKENS
    assert "temperature" not in captured
    assert "prompt_cache_key" not in captured
    assert result["config"]["reasoning"] == "low"


def test_extract_events_uses_stable_cache_key_and_reports_cache_usage(monkeypatch):
    monkeypatch.setattr("src.chronos.openai_service.get_settings", _make_settings)

    captured = {}
    response = SimpleNamespace(
        output_parsed=SimpleNamespace(
            events=[
                {
                    "event_id": "event-123",
                    "recording_id": "recording-123",
                    "start_ts": "2026-07-17T10:00:00Z",
                    "end_ts": "2026-07-17T10:05:00Z",
                    "day_of_week": "Friday",
                    "hour_of_day": 10,
                    "clean_text": "A concrete event extracted from the transcript.",
                }
            ],
            total_events=1,
        ),
        output_text="",
        output=[],
        usage=SimpleNamespace(
            input_tokens=2400,
            output_tokens=20,
            total_tokens=2420,
            input_tokens_details=SimpleNamespace(
                cached_tokens=1600,
                cache_write_tokens=0,
            ),
        ),
        model="gpt-5.4",
        incomplete_details=None,
    )

    def parse(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    svc = OpenAIResponseService()
    monkeypatch.setattr(svc, "_get_client", lambda: client)
    monkeypatch.setattr(svc, "_call_with_retry", lambda _op, func, **_kwargs: func())
    monkeypatch.setattr(
        "src.chronos.cost_tracker.track_usage",
        lambda *args, **kwargs: None,
    )

    result = svc.extract_events(
        "A sufficiently long transcript body",
        recording_id="recording-123",
    )

    assert captured["prompt_cache_key"].startswith(
        "plaudblender:chronos-events:v1:gpt-5.4:"
    )
    assert result["usage"]["cached_tokens"] == 1600
    assert result["usage"]["cache_write_tokens"] == 0


def test_extraction_cache_key_changes_with_instructions():
    default_key = OpenAIResponseService._extraction_cache_key(
        "gpt-5.4", "default instructions"
    )
    custom_key = OpenAIResponseService._extraction_cache_key(
        "gpt-5.4", "custom instructions"
    )

    assert default_key == OpenAIResponseService._extraction_cache_key(
        "gpt-5.4", "default instructions"
    )
    assert default_key != custom_key
