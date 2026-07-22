from datetime import datetime, timedelta

from app_v2.services import data_service as data_service_module


def _event(event_id, recording_id, keywords, category="work"):
    start = datetime(2026, 7, 17, 9, 0) + timedelta(minutes=int(event_id[-1]))
    return data_service_module.Event(
        id=event_id,
        recording_id=recording_id,
        start_ts=start,
        end_ts=start + timedelta(minutes=5),
        clean_text="A concrete event description long enough for the graph fixture.",
        category=category,
        sentiment=0.2,
        keywords=keywords,
        speaker="self_talk",
        duration_seconds=300,
        day_of_week="Friday",
        hour_of_day=9,
    )


def _build_graph(monkeypatch, events):
    monkeypatch.setattr(
        data_service_module.ChronosDataService,
        "_init_services",
        lambda self: None,
    )
    service = data_service_module.ChronosDataService()
    monkeypatch.setattr(service, "_get_all_events", lambda: events)
    return service._build_graph_from_events()


def _topic_ids(graph):
    return {
        node["data"]["id"]
        for node in graph.nodes
        if node["data"].get("type") == "topic"
    }


def test_graph_rejects_transcript_debris_and_profanity(monkeypatch):
    graph = _build_graph(
        monkeypatch,
        [
            _event(
                "event1",
                "recording1",
                ["Was", "People", "Not", "Fucking", "Qdrant indexing", "Authentication"],
            ),
            _event(
                "event2",
                "recording2",
                ["Was", "People", "Not", "Fucking", "Qdrant indexing", "Authentication"],
            ),
        ],
    )

    assert _topic_ids(graph) == {"kw:qdrant indexing", "kw:authentication"}


def test_graph_does_not_backfill_with_singleton_keywords(monkeypatch):
    graph = _build_graph(
        monkeypatch,
        [
            _event("event1", "recording1", ["Qdrant indexing", "Authentication"]),
        ],
    )

    assert _topic_ids(graph) == set()


def test_graph_trims_low_value_words_around_specific_subject(monkeypatch):
    graph = _build_graph(
        monkeypatch,
        [
            _event("event1", "recording1", ["The Plaud API"]),
            _event("event2", "recording2", ["Plaud API"]),
        ],
    )

    assert _topic_ids(graph) == {"kw:plaud api"}


def test_topics_endpoint_uses_the_same_quality_gate(monkeypatch):
    events = [
        _event("event1", "recording1", ["People", "The Plaud API", "one-off detail"]),
        _event("event2", "recording2", ["People", "Plaud API"]),
    ]
    monkeypatch.setattr(
        data_service_module.ChronosDataService,
        "_init_services",
        lambda self: None,
    )
    service = data_service_module.ChronosDataService()
    monkeypatch.setattr(service, "_get_all_events", lambda: events)

    assert service.get_all_topics() == [("plaud api", 2)]
