"""The entity graph is read by Ask and the API (2026-10-01); before, nothing read it."""

import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.chronos import entity_graph
from src.chronos.ask_service import ChronosAskService
from src.chronos.graph_rag import Entity, EntityType

MIKE = Entity.generate_id("Mike", EntityType.PERSON)
PATEL = Entity.generate_id("Dr. Patel", EntityType.PERSON)
GUNNAR = Entity.generate_id("Gunnar", EntityType.PERSON)
TAHOE = Entity.generate_id("Tahoe", EntityType.LOCATION)
WORK = Entity.generate_id("work", EntityType.TOPIC)


def _payload():
    return {
        "version": 1,
        "events": 3,
        "nodes": [
            {"id": MIKE, "name": "Mike", "type": "person", "mentions": 9, "aliases": [],
             "first_seen": "2025-11-02T10:00:00", "last_seen": "2026-09-30T11:00:00", "events": ["e2", "e1"]},
            {"id": PATEL, "name": "Dr. Patel", "type": "person", "mentions": 4, "aliases": [],
             "first_seen": "2026-01-05T09:15:00", "last_seen": "2026-09-30T09:15:00", "events": ["e3"]},
            {"id": GUNNAR, "name": "Gunnar", "type": "person", "mentions": 40, "aliases": [],
             "first_seen": "2025-10-01T08:00:00", "last_seen": "2026-09-30T12:00:00", "events": []},
            {"id": TAHOE, "name": "Tahoe", "type": "location", "mentions": 3, "aliases": [],
             "first_seen": "2026-09-30T11:00:00", "last_seen": "2026-09-30T11:00:00", "events": ["e2"]},
            {"id": WORK, "name": "work", "type": "topic", "mentions": 50, "aliases": [], "events": []},
        ],
        "edges": [
            {"source": GUNNAR, "target": MIKE, "type": "knows", "weight": 3, "evidence": ["my buddy Mike"]},
            {"source": MIKE, "target": TAHOE, "type": "co_mentioned", "weight": 7, "evidence": []},
        ],
    }


@pytest.fixture()
def graph_file(tmp_path, monkeypatch):
    path = tmp_path / "entity_graph.json"
    path.write_text(json.dumps(_payload()))
    monkeypatch.setattr(entity_graph, "graph_path", lambda: path)
    entity_graph._cache.update(path=None, mtime=None, graph=None)
    return path


class _Svc:
    def _get_all_events(self):
        mk = lambda i, text: SimpleNamespace(id=i, start_ts=datetime(2026, 9, 30, 9, 0), clean_text=text)
        return [mk("e1", "Mike said the cabin is booked."), mk("e2", "Called Mike about the Tahoe trip."),
                mk("e3", "OR prep meeting with Dr. Patel."), mk("e9", "unrelated")]


def test_question_matches_people_places_and_honorifics_not_single_word_topics(graph_file):
    graph = entity_graph.load_entity_graph()
    names = [n["name"] for n in graph.match_question("What did Mike and Patel say about the Tahoe work?")]
    assert set(names) == {"Mike", "Dr. Patel", "Tahoe"}  # "work" is a single-word topic -> ignored
    assert graph.match_question("nothing relevant here") == []


def test_ask_graph_context_has_links_with_evidence_and_recent_moments(graph_file):
    entries = entity_graph.ask_graph_context(_Svc(), "When is the Tahoe trip with Mike?")
    mike = next(e for e in entries if e["name"] == "Mike")
    assert mike["kind"] == "graph_entity" and mike["category"] == "person"
    # the edge is Gunnar -knows-> Mike: Mike's block must not say "Mike knows Gunnar"
    assert 'Gunnar (person) knows Mike x3 "my buddy Mike"' in mike["text"]
    assert "mentioned with Tahoe (location) x7" in mike["text"]
    assert "Called Mike about the Tahoe trip." in mike["text"] and "unrelated" not in mike["text"]


def test_ask_graph_context_outgoing_edges_read_from_the_named_entity(graph_file):
    entries = entity_graph.ask_graph_context(_Svc(), "How does Gunnar know Mike?")
    gunnar = next(e for e in entries if e["name"] == "Gunnar")
    assert 'Gunnar knows Mike (person) x3 "my buddy Mike"' in gunnar["text"]


def test_missing_or_broken_graph_file_is_harmless(tmp_path, monkeypatch):
    monkeypatch.setattr(entity_graph, "graph_path", lambda: tmp_path / "nope.json")
    entity_graph._cache.update(path=None, mtime=None, graph=None)
    assert entity_graph.ask_graph_context(_Svc(), "Mike?") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{half written")
    assert entity_graph.load_entity_graph(bad) is None


def test_ask_formatter_labels_graph_entries():
    text = ChronosAskService._format_context_event(
        {"kind": "graph_entity", "name": "Mike", "category": "person", "date": "2026-09-30", "text": "Mike (person)"}
    )
    assert text.startswith("[Knowledge graph: Mike (person), last seen 2026-09-30]")


@pytest.fixture()
def client(graph_file):
    from api.dependencies import get_service

    get_service.cache_clear()
    with patch.dict(os.environ, {"CHRONOS_API_KEY": ""}, clear=False):
        from api.main import app

        app.dependency_overrides[get_service] = lambda: _Svc()
        yield TestClient(app, raise_server_exceptions=False)
        app.dependency_overrides.clear()


def test_api_entities_search_and_detail(client):
    top = client.get("/api/v1/graph/entities?limit=3").json()
    assert [n["name"] for n in top["nodes"]] == ["work", "Gunnar", "Mike"]
    assert any(e["type"] == "knows" for e in top["edges"])
    people = client.get("/api/v1/graph/entities?type=person").json()["nodes"]
    assert {n["type"] for n in people} == {"person"}

    found = client.get("/api/v1/graph/entities/search?q=pat").json()["results"]
    assert [n["name"] for n in found] == ["Dr. Patel"]

    detail = client.get(f"/api/v1/graph/entities/{MIKE}").json()
    assert detail["entity"]["name"] == "Mike"
    assert detail["neighbors"][0]["type"] == "knows" and detail["neighbors"][0]["evidence"] == ["my buddy Mike"]
    assert detail["neighbors"][0]["direction"] == "in"  # Gunnar knows Mike
    assert [m["id"] for m in detail["recent_events"]] == ["e2", "e1"]
    assert client.get("/api/v1/graph/entities/doesnotexist").status_code == 404


def test_api_without_a_graph_returns_empty(tmp_path, monkeypatch):
    from api.dependencies import get_service

    monkeypatch.setattr(entity_graph, "graph_path", lambda: tmp_path / "missing.json")
    entity_graph._cache.update(path=None, mtime=None, graph=None)
    get_service.cache_clear()
    with patch.dict(os.environ, {"CHRONOS_API_KEY": ""}, clear=False):
        from api.main import app

        app.dependency_overrides[get_service] = lambda: _Svc()
        try:
            client = TestClient(app, raise_server_exceptions=False)
            assert client.get("/api/v1/graph/entities").json() == {"nodes": [], "edges": [], "events": 0}
            assert client.get("/api/v1/graph/entities/search?q=x").json() == {"results": []}
        finally:
            app.dependency_overrides.clear()
