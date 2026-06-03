from types import SimpleNamespace

from src.chronos.ask_service import ChronosAskService


def test_ask_service_returns_openai_success(monkeypatch):
    class DummyOpenAIService:
        available = True

        def ask(self, *args, **kwargs):
            assert kwargs["model"] == "gpt-5.5"
            return {
                "answer": "OpenAI answer",
                "model": "gpt-5.5",
                "response_id": "resp_123",
                "reasoning_summary": None,
                "config": {"provider": "openai"},
                "usage": {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0, "total_tokens": 15},
            }

    monkeypatch.setattr(
        "src.chronos.ask_service.get_settings",
        lambda: SimpleNamespace(openai_model="gpt-5.4", chronos_analyst_model="gpt-5.5"),
    )
    monkeypatch.setattr(
        "src.chronos.ask_service.OpenAIResponseService",
        lambda: DummyOpenAIService(),
    )

    svc = ChronosAskService()
    result = svc.ask("What happened?", [{"text": "evidence"}])

    assert result["answer"] == "OpenAI answer"
    assert result["config"]["provider"] == "openai"


def test_ask_service_surfaces_openai_error_without_gemini_fallback(monkeypatch):
    class DummyOpenAIService:
        available = True

        def ask(self, *args, **kwargs):
            return {"error": "rate limited"}

    monkeypatch.setattr(
        "src.chronos.ask_service.get_settings",
        lambda: SimpleNamespace(openai_model="gpt-5.4", chronos_analyst_model="gpt-5.5"),
    )
    monkeypatch.setattr(
        "src.chronos.ask_service.OpenAIResponseService",
        lambda: DummyOpenAIService(),
    )

    svc = ChronosAskService()
    result = svc.ask("What happened?", [{"text": "evidence"}])

    assert result["error"] == "rate limited"


def test_ask_service_reports_missing_providers(monkeypatch):
    class DummyOpenAIService:
        available = False

        def ask(self, *args, **kwargs):
            raise AssertionError("OpenAI should not be called")

    monkeypatch.setattr(
        "src.chronos.ask_service.get_settings",
        lambda: SimpleNamespace(openai_model="gpt-5.5", chronos_analyst_model="gpt-5.5"),
    )
    monkeypatch.setattr(
        "src.chronos.ask_service.OpenAIResponseService",
        lambda: DummyOpenAIService(),
    )

    svc = ChronosAskService()
    result = svc.ask("What happened?", [{"text": "evidence"}])

    assert "error" in result
    assert "No AI provider configured" in result["error"]
    assert "OPENAI_API_KEY" in result["error"]
