import json
from datetime import datetime
from types import SimpleNamespace

from src.chronos import cost_tracker
from src.chronos import graph_rag
from src.chronos.graph_rag import EntityExtractor, Entity, EntityType
from src.chronos.graph_service import ChronosGraphExtractor


def _settings(tmp_path, token="tok"):
    token_file = tmp_path / "token"
    if token is not None:
        token_file.write_text(token)
    return SimpleNamespace(
        chronos_processing_provider="agy",
        chronos_agy_token_file=str(token_file),
        chronos_agy_bridge_url="http://127.0.0.1:1",
        chronos_agy_model="gemini-3.8-flash-high",
        chronos_agy_timeout_seconds=60,
        openai_api_key="must-not-be-used",
        gemini_api_key=None,
    )


def _extractor(tmp_path, monkeypatch, token="tok"):
    monkeypatch.setattr(graph_rag, "get_settings", lambda: _settings(tmp_path, token))
    return EntityExtractor()


def test_agy_provider_uses_bridge_not_openai(tmp_path, monkeypatch):
    ex = _extractor(tmp_path, monkeypatch)
    assert ex.supports_batch is True
    assert ex.llm.model == "agy/gemini-3.8-flash-high"
    assert not hasattr(ex, "_openai_client")


def test_agy_without_token_disables_extraction_instead_of_paying(tmp_path, monkeypatch):
    ex = _extractor(tmp_path, monkeypatch, token=None)
    assert ex.llm is None and ex.supports_batch is False


def test_batch_maps_answers_to_events_and_logs_zero_cost(tmp_path, monkeypatch):
    ex = _extractor(tmp_path, monkeypatch)
    answer = {"events": [
        {"event_id": "e1", "people": [{"name": "Dr. Patel", "role": "surgeon"}], "topics": ["Sterilization Checklist"]},
        {"event_id": "e2", "organizations": ["Tailscale"]},
        {"event_id": "not-asked", "people": [{"name": "Ghost"}]},
    ]}
    sent = {}

    def fake_complete(system, user, schema=None):
        sent.update(system=system, user=user, schema=schema)
        return {"ok": True, "text": json.dumps(answer) + "\n" + json.dumps(answer),
                "usage": {"input_tokens": 100, "output_tokens": 50}}

    monkeypatch.setattr(ex._agy, "complete", fake_complete)
    tracked = []
    monkeypatch.setattr(cost_tracker, "track_usage", lambda *a, **k: tracked.append((a, k)))

    out = ex.extract_entities_batch([("e1", "Met Dr. Patel about the checklist"), ("e2", "Tailscale keys"), ("e3", "lunch")])

    assert set(out) == {"e1", "e2"}  # e3 unanswered, "not-asked" ignored
    names = {e.name for e in out["e1"][0]}
    assert "Dr. Patel" in names
    assert "=== EVENT e1 ===" in sent["system"] and "=== EVENT e3 ===" in sent["system"]
    assert sent["schema"]["properties"]["events"]["type"] == "array"
    assert tracked[0][0][:2] == ("agy/gemini-3.8-flash-high", "entity")
    assert cost_tracker.estimate_cost("agy/gemini-3.8-flash-high", 100, 50) == 0


class _Event(SimpleNamespace):
    pass


def _events(n):
    return [
        _Event(event_id=f"e{i}", recording_id="r1", clean_text=f"text {i}",
               start_ts=datetime(2026, 9, 30, 9, i), category=SimpleNamespace(value="work"))
        for i in range(n)
    ]


class _FakeBatchExtractor:
    supports_batch = True

    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def extract_entities_batch(self, items):
        self.calls.append([doc_id for doc_id, _ in items])
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("bridge busy")
        # answer all but the last event of each batch
        return {
            doc_id: ([Entity(id=Entity.generate_id(f"P{doc_id}", EntityType.PERSON), name=f"P{doc_id}",
                             entity_type=EntityType.PERSON)], [])
            for doc_id, _ in items[:-1]
        }

    def extract_entities(self, *a, **k):  # must never be used on the batch path
        raise AssertionError("per-event path used")


def _graph_extractor(extractor):
    gx = ChronosGraphExtractor.__new__(ChronosGraphExtractor)
    gx.entity_extractor = extractor
    gx.community_detector = None
    return gx


def test_graph_batches_by_size_and_counts_skipped_events(monkeypatch):
    monkeypatch.setenv("CHRONOS_AGY_ENTITY_BATCH_SIZE", "4")
    fake = _FakeBatchExtractor(fail_first=True)
    gx = _graph_extractor(fake)
    seen = []
    entities, graph = gx.extract_from_events(_events(10), progress_callback=seen.append)
    # batches of 4,4,2; the first one failed once and was retried
    assert [len(c) for c in fake.calls] == [4, 4, 4, 2]
    assert len(seen) == 10
    assert len(entities) == 3 + 3 + 1
    assert gx.last_failed_events == 3


def test_non_batch_extractor_keeps_per_event_path():
    calls = []

    class _PerEvent:
        def extract_entities(self, text, doc_id):
            calls.append(doc_id)
            return [], []

    gx = _graph_extractor(_PerEvent())
    gx.extract_from_events(_events(3))
    assert calls == ["e0", "e1", "e2"] and gx.last_failed_events == 0
