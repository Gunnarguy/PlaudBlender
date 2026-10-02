import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from src.chronos.graph_rag import Entity, EntityType
from src.chronos.graph_service import ChronosGraphExtractor


def _events(texts):
    return [
        SimpleNamespace(event_id=f"e{i}", recording_id="r1", clean_text=t,
                        start_ts=datetime(2026, 9, 30, 9, 0) + timedelta(minutes=i), category=SimpleNamespace(value="work"))
        for i, t in enumerate(texts)
    ]


class _CountingBatch:
    supports_batch = True

    def __init__(self):
        self.seen = []

    def extract_entities_batch(self, items):
        self.seen.extend(doc_id for doc_id, _ in items)
        return {
            doc_id: ([Entity(id=Entity.generate_id(text, EntityType.TOPIC), name=text,
                             entity_type=EntityType.TOPIC)], [])
            for doc_id, text in items
        }


def _gx(extractor, cache_path):
    gx = ChronosGraphExtractor.__new__(ChronosGraphExtractor)
    gx.entity_extractor = extractor
    gx.community_detector = None
    gx.cache_path = cache_path
    gx._fresh = {}
    return gx


def test_second_build_reuses_cache_and_only_extracts_changed_events(tmp_path):
    cache = tmp_path / "entity_cache.json"
    first = _CountingBatch()
    entities, _ = _gx(first, cache).extract_from_events(_events(["Tailscale", "Qdrant", "Notion"]))
    assert sorted(first.seen) == ["e0", "e1", "e2"] and len(entities) == 3
    assert json.loads(cache.read_text())["version"] == ChronosGraphExtractor.CACHE_VERSION

    second = _CountingBatch()
    entities, _ = _gx(second, cache).extract_from_events(_events(["Tailscale", "Qdrant", "Notion"]))
    assert second.seen == [] and len(entities) == 3  # all three replayed from the cache

    third = _CountingBatch()
    entities, _ = _gx(third, cache).extract_from_events(_events(["Tailscale", "Qdrant v1.19", "Notion"]))
    assert third.seen == ["e1"]  # only the edited event goes back to the model
    assert {e["name"] for e in entities} == {"Tailscale", "Qdrant v1.19", "Notion"}


def test_cache_is_pruned_to_the_current_input_set(tmp_path):
    cache = tmp_path / "entity_cache.json"
    _gx(_CountingBatch(), cache).extract_from_events(_events(["a", "b", "c"]))
    _gx(_CountingBatch(), cache).extract_from_events(_events(["a"]))
    assert set(json.loads(cache.read_text())["events"]) == {"e0"}


def test_unreadable_cache_just_means_re_extraction(tmp_path):
    cache = tmp_path / "entity_cache.json"
    cache.write_text("{not json")
    ex = _CountingBatch()
    entities, _ = _gx(ex, cache).extract_from_events(_events(["x"]))
    assert ex.seen == ["e0"] and len(entities) == 1


def test_no_cache_path_keeps_old_behaviour(tmp_path):
    ex = _CountingBatch()
    _gx(ex, None).extract_from_events(_events(["x", "y"]))
    _gx(ex, None).extract_from_events(_events(["x", "y"]))
    assert ex.seen == ["e0", "e1", "e0", "e1"]
    assert list(tmp_path.iterdir()) == []


class _Killed(BaseException):
    """Like the SIGTERM a unit timeout delivers: not an Exception, so nothing catches it."""


class _DiesOnThirdBatch(_CountingBatch):
    def extract_entities_batch(self, items):
        if len(self.seen) >= 2 * 25:
            raise _Killed()
        return super().extract_entities_batch(items)


def test_progress_survives_a_killed_build(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONOS_AGY_ENTITY_BATCH_SIZE", "25")
    cache = tmp_path / "entity_cache.json"
    texts = [f"topic {i}" for i in range(80)]
    import pytest
    with pytest.raises(_Killed):
        _gx(_DiesOnThirdBatch(), cache).extract_from_events(_events(texts))
    saved = json.loads(cache.read_text())["events"]
    assert len(saved) == 50  # the two finished batches were kept

    resume = _CountingBatch()
    entities, _ = _gx(resume, cache).extract_from_events(_events(texts))
    assert len(resume.seen) == 30 and len(entities) == 80  # only the unfinished events go to the model


def test_budget_extracts_newest_first_and_resumes(tmp_path):
    """A run extracts at most max_new uncached moments; the next run picks up the rest."""
    cache = tmp_path / "entity_cache.json"
    texts = ["newest", "middle", "oldest"]  # callers pass events newest first
    first = _CountingBatch()
    gx = _gx(first, cache)
    entities, _ = gx.extract_from_events(_events(texts), max_new=1)
    assert first.seen == ["e0"] and gx.pending_events == 2
    assert {e["name"] for e in entities} == {"newest"}

    second = _CountingBatch()
    gx = _gx(second, cache)
    entities, _ = gx.extract_from_events(_events(texts), max_new=1)
    assert second.seen == ["e1"] and gx.pending_events == 1
    assert {e["name"] for e in entities} == {"newest", "middle"}  # cached + new

    third = _CountingBatch()
    gx = _gx(third, cache)
    gx.extract_from_events(_events(texts), max_new=5)
    assert third.seen == ["e2"] and gx.pending_events == 0


def test_cache_file_is_read_once_per_build(tmp_path, monkeypatch):
    cache = tmp_path / "entity_cache.json"
    gx = _gx(_CountingBatch(), cache)
    reads = []
    real = ChronosGraphExtractor._load_cache
    monkeypatch.setattr(ChronosGraphExtractor, "_load_cache", lambda self: reads.append(1) or real(self))
    monkeypatch.setenv("CHRONOS_AGY_ENTITY_BATCH_SIZE", "1")  # one save per batch -> 4 saves
    gx.extract_from_events(_events(["a", "b", "c", "d"]))
    assert len(reads) == 1
    assert set(json.loads(cache.read_text())["events"]) == {"e0", "e1", "e2", "e3"}
