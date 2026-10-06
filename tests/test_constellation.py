"""Entity Constellation (2026-10-02): communities, a stable 2D map, weekly activity and the
recording/day index, precomputed by export_json; served by /api/v1/graph/constellation,
/api/v1/graph/recordings/{id}/entities and /api/v1/graph/days/{date}/entities.

All names are made up.
"""

import json
import math
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.chronos import constellation as C
from src.chronos.graph_rag import EntityExtractor
from src.chronos.graph_service import ChronosGraphExtractor

# --------------------------------------------------------------------------- fixtures


def _node(node_id, name, kind, mentions):
    return {"id": node_id, "name": name, "type": kind, "mentions": mentions}


def _edge(a, b, weight=1.0, kind="co_mentioned"):
    return {"source": a, "target": b, "type": kind, "weight": weight}


NODES = [
    # an office: people, an organization, a project, a busy topic
    _node("p_avery", "Avery", "person", 30),
    _node("p_blake", "Blake", "person", 12),
    _node("o_north", "Northwind Labs", "organization", 9),
    _node("j_harbor", "harbor project", "project", 5),
    _node("t_budget", "budget review", "topic", 40),
    # a trip: places, a person, topics
    _node("l_lake", "Lakeview", "location", 8),
    _node("p_corin", "Corin", "person", 6),
    _node("t_trail", "trail map", "topic", 4),
    _node("t_cabin", "cabin booking", "topic", 3),
    _node("l_pine", "Pinecrest", "location", 2),
    # linked to nothing
    _node("a_plants", "water the plants", "action", 1),
]
OFFICE = ["p_avery", "p_blake", "o_north", "j_harbor", "t_budget"]
TRIP = ["l_lake", "p_corin", "t_trail", "t_cabin", "l_pine"]


def _clique(ids, weight):
    return [_edge(a, b, weight) for i, a in enumerate(ids) for b in ids[i + 1:]]


EDGES = (
    _clique(OFFICE, 3.0)
    + _clique(TRIP, 3.0)
    + [_edge("p_avery", "p_blake", 2.0, "works_with"), _edge("p_avery", "l_lake", 1.0)]  # one bridge
)


def _as_previous(tmp_path, result, edges, **layout_overrides):
    """Write a build result the way export_json does and read it back as `previous`."""
    path = tmp_path / "entity_graph.json"
    payload = {
        "layout": {**result["layout"], **layout_overrides},
        "nodes": [{"id": i, "x": x, "y": y, "community": result["community"][i]}
                  for i, (x, y) in result["positions"].items()],
        "edges": edges,
    }
    path.write_text(json.dumps(payload))
    return C.read_previous_layout(path)


# --------------------------------------------------------------------------- communities + layout


def test_communities_are_found_sorted_and_labelled_by_people_before_topics():
    result = C.build(NODES, EDGES)
    groups = result["communities"]
    assert [c["id"] for c in groups] == list(range(len(groups)))
    assert [c["size"] for c in groups] == [5, 5, 1]  # largest first; tie -> more mentions first
    office, trip, alone = groups
    assert {result["community"][n] for n in OFFICE} == {0}
    assert {result["community"][n] for n in TRIP} == {1}
    # "budget review" has the most mentions but is a topic: people/orgs/projects/places first
    assert office["label"] == "Avery · Blake · Northwind Labs"
    assert office["top"] == ["p_avery", "p_blake", "o_north"]
    assert trip["label"] == "Lakeview · Corin · Pinecrest"
    assert alone["label"] == "water the plants"
    for c in groups:
        assert set(c) >= {"id", "label", "size", "x", "y", "radius"}


def test_layout_is_deterministic_inside_the_unit_disc_and_forms_regions():
    first = C.build(NODES, EDGES)
    again = C.build(list(reversed(NODES)), list(reversed(EDGES)))  # input order doesn't matter
    assert first["positions"] == again["positions"]
    assert first["community"] == again["community"]
    assert first["layout"]["mode"] == "full" and first["layout"]["version"] == C.LAYOUT_VERSION

    pos = first["positions"]
    assert all(math.hypot(x, y) <= 1.0 + 1e-9 for x, y in pos.values())
    office, trip = first["communities"][0], first["communities"][1]
    for members, own, other in ((OFFICE, office, trip), (TRIP, trip, office)):
        for n in members:
            mine = math.dist(pos[n], (own["x"], own["y"]))
            assert mine < math.dist(pos[n], (other["x"], other["y"]))
    # an entity with no link sits on the outer ring, beyond every linked one
    ring = math.hypot(*pos["a_plants"])
    assert all(math.hypot(*pos[n]) < ring for n in OFFICE + TRIP)


def test_one_new_entity_leaves_the_map_in_place_and_starts_next_to_its_strongest_link(tmp_path):
    first = C.build(NODES, EDGES)
    previous = _as_previous(tmp_path, first, EDGES)

    nodes = NODES + [_node("p_dana", "Dana", "person", 2)]
    edges = EDGES + [_edge("p_dana", "p_blake", 4.0), _edge("p_dana", "l_pine", 1.0)]
    second = C.build(nodes, edges, previous)

    assert second["layout"]["mode"] == "incremental"
    assert second["layout"]["kept"] == len(NODES) and second["layout"]["attached"] == 1
    for node_id, (x, y) in first["positions"].items():
        assert second["positions"][node_id] == pytest.approx((x, y), abs=1e-4)
    unit = second["layout"]["unit"]
    assert math.dist(second["positions"]["p_dana"], second["positions"]["p_blake"]) < 6 * unit
    assert second["community"]["p_dana"] == second["community"]["p_blake"]  # joins its strongest link's
    # same inputs (previous file included) -> same output
    assert C.build(nodes, edges, previous)["positions"] == second["positions"]


def test_unlinked_entity_that_gains_a_link_leaves_the_ring(tmp_path):
    first = C.build(NODES, EDGES)
    previous = _as_previous(tmp_path, first, EDGES)
    edges = EDGES + [_edge("a_plants", "t_cabin", 2.0)]
    second = C.build(NODES, edges, previous)
    assert second["layout"]["mode"] == "incremental" and second["layout"]["rejoined"] == 1
    assert math.dist(second["positions"]["a_plants"], second["positions"]["t_cabin"]) < 6 * second["layout"]["unit"]
    assert second["community"]["a_plants"] == second["community"]["t_cabin"]
    for node_id in OFFICE + TRIP:
        assert second["positions"][node_id] == pytest.approx(first["positions"][node_id], abs=1e-4)


def test_new_unlinked_group_is_packed_at_the_edge_and_the_map_still_fits(tmp_path):
    first = C.build(NODES, EDGES)
    previous = _as_previous(tmp_path, first, EDGES)
    island = ["t_kiln", "t_glaze", "t_wheel"]
    nodes = NODES + [_node(i, i[2:], "topic", 2) for i in island]
    second = C.build(nodes, EDGES + _clique(island, 2.0), previous)
    assert second["layout"]["mode"] == "incremental" and second["layout"]["packed"] == 3
    assert len({second["community"][i] for i in island}) == 1
    assert all(math.hypot(x, y) <= 1.0 + 1e-9 for x, y in second["positions"].values())


def test_full_layout_again_when_the_map_doubled_turned_onto_the_old_one(tmp_path):
    first = C.build(NODES, EDGES)
    # the same map, turned a quarter, recorded as laid out when it had only 4 entities
    turned = {**first, "positions": {i: (-y, x) for i, (x, y) in first["positions"].items()}}
    previous = _as_previous(tmp_path, turned, EDGES, base_nodes=4)
    again = C.build(NODES, EDGES, previous)
    assert again["layout"]["mode"] == "relayout" and again["layout"]["base_nodes"] == len(NODES)
    for node_id, xy in turned["positions"].items():
        assert again["positions"][node_id] == pytest.approx(xy, abs=1e-6)


def _spread(positions):
    return max(math.hypot(x, y) for x, y in positions.values())


def test_relayout_keeps_its_own_scale_when_the_old_map_was_squeezed(tmp_path):
    """2026-10-06: a relayout copied the old map's scale, so a squeezed map stayed
    squeezed (the 800 most-mentioned entities ended up inside a 0.005-wide patch)."""
    first = C.build(NODES, EDGES)
    squeezed = {**first, "positions": {i: (x * 0.01, y * 0.01) for i, (x, y) in first["positions"].items()}}
    previous = _as_previous(tmp_path, squeezed, EDGES, base_nodes=4)
    again = C.build(NODES, EDGES, previous)
    assert again["layout"]["mode"] == "relayout"
    # not squeezed (it was 0.01x before the fix); re-centring moves it a little
    assert 0.6 * _spread(first["positions"]) < _spread(again["positions"]) <= 1.0


def test_a_map_squeezed_by_incremental_builds_is_laid_out_again(tmp_path):
    first = C.build(NODES, EDGES)
    unit = first["layout"]["unit"]
    shrunk = _as_previous(tmp_path, first, EDGES, base_nodes=len(NODES), unit=unit * 0.3, full_unit=unit)
    assert C.build(NODES, EDGES, shrunk)["layout"]["mode"] != "incremental"
    healthy = _as_previous(tmp_path, first, EDGES, base_nodes=len(NODES), unit=unit * 0.8, full_unit=unit)
    again = C.build(NODES, EDGES, healthy)
    assert again["layout"]["mode"] == "incremental" and again["layout"]["full_unit"] == pytest.approx(unit, rel=1e-3)


def test_a_previous_map_from_another_layout_version_is_not_reused(tmp_path):
    first = C.build(NODES, EDGES)
    previous = _as_previous(tmp_path, first, EDGES, version=C.LAYOUT_VERSION + 99)
    assert C.build(NODES, EDGES, previous)["layout"]["mode"] == "full"
    assert C.read_previous_layout(tmp_path / "missing.json") is None


def test_empty_graph():
    result = C.build([], [])
    assert result["positions"] == {} and result["communities"] == []


# --------------------------------------------------------------------------- weeks + index


def test_iso_weeks_and_local_days():
    from zoneinfo import ZoneInfo

    assert C.iso_week("2026-10-02") == "2026-W40"
    assert C.iso_week("2026-09-27") == "2026-W39"  # a Sunday closes its ISO week
    assert C.iso_week("2027-01-01") == "2026-W53"  # ISO year, not calendar year
    from src.chronos.entity_graph import _week_key  # the API's copy agrees

    assert [_week_key(d) for d in ("2026-09-27", "2027-01-01")] == ["2026-W39", "2026-W53"]
    # stored timestamps are naive UTC; the timeline shows them in the local zone
    assert C.local_day(datetime(2026, 10, 2, 3, 0), ZoneInfo("America/Los_Angeles")) == "2026-10-01"


def test_moment_days_follow_the_recordings_timeline_day():
    moments = {
        "e1": ("r1", "2026-09-27T23:50:00"),
        "e2": ("r1", "2026-09-28T00:20:00"),  # after midnight, same recording
        "e3": ("r2", "2026-09-30T10:00:00"),
        "e4": ("", "2026-10-01T08:00:00"),  # no recording: its own date
    }
    days = C.moment_days(moments, {"r2": "2026-09-29"})
    assert days == {"e1": "2026-09-27", "e2": "2026-09-27", "e3": "2026-09-29", "e4": "2026-10-01"}


def test_weeks_and_index_count_moments():
    moments = {
        "e1": ("r1", "2026-09-27T23:50:00"),
        "e2": ("r1", "2026-09-28T00:20:00"),
        "e3": ("r2", "2026-09-30T10:00:00"),
    }
    entity_events = {
        "p_avery": [("e1", "2026-09-27T23:50:00"), ("e2", "2026-09-28T00:20:00"), ("e3", "2026-09-30T10:00:00")],
        "l_lake": [("e3", "2026-09-30T10:00:00")],
        "not_kept": [("e1", "2026-09-27T23:50:00")],
    }
    weeks, index = C.build_activity({"p_avery", "l_lake"}, entity_events, moments)
    assert weeks == {"p_avery": {"2026-W39": 2, "2026-W40": 1}, "l_lake": {"2026-W40": 1}}
    assert index == {
        "recordings": {"r1": {"p_avery": 2}, "r2": {"l_lake": 1, "p_avery": 1}},
        "days": {"2026-09-27": {"p_avery": 2}, "2026-09-30": {"l_lake": 1, "p_avery": 1}},
    }


def test_recording_days_from_db_use_the_timelines_local_date(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.database.models import Base, ChronosRecording

    monkeypatch.setenv("TZ", "America/Los_Angeles")
    engine = create_engine(f"sqlite:///{tmp_path / 'days.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    for rid, created in (("r_evening", datetime(2026, 10, 2, 3, 30)), ("r_noon", datetime(2026, 10, 1, 19, 0))):
        session.add(ChronosRecording(recording_id=rid, title="x", created_at=created, duration_seconds=60,
                                     local_audio_path="x.m4a", processing_status="completed"))
    session.commit()
    days = C.recording_days_from_db(session, ["r_evening", "r_noon", "r_missing", None])
    assert days == {"r_evening": "2026-10-01", "r_noon": "2026-10-01"}
    session.close()


# --------------------------------------------------------------------------- export_json


class _Batch:
    supports_batch = True

    def __init__(self, answers):
        self.answers = answers

    def extract_entities_batch(self, items):
        extractor = EntityExtractor.__new__(EntityExtractor)
        return {doc_id: extractor._parse_entities_from_response(self.answers[doc_id], doc_id, 100)
                for doc_id, _ in items if doc_id in self.answers}


def _extractor(answers):
    gx = ChronosGraphExtractor.__new__(ChronosGraphExtractor)
    gx.entity_extractor = _Batch(answers)
    gx.community_detector = None
    gx.cache_path = None
    gx._fresh = {}
    gx._entity_stats = {}
    return gx


ANSWERS = {
    "e0": {"people": [{"name": "Avery"}, {"name": "Blake"}], "locations": ["Lakeview"],
           "topics": ["trail map"]},
    "e1": {"people": [{"name": "Avery"}], "locations": ["Lakeview"], "topics": ["trail map"]},
    "e2": {"people": [{"name": "Blake"}], "organizations": ["Northwind Labs"]},
    "e3": {"people": [{"name": "Avery"}, {"name": "Blake"}]},
}


def _moments():
    def event(event_id, recording_id, when):
        return SimpleNamespace(event_id=event_id, recording_id=recording_id, clean_text=f"text {event_id}",
                               start_ts=when, category=SimpleNamespace(value="work"))

    return [
        event("e0", "rec_a", datetime(2026, 9, 27, 23, 50)),  # Sunday, week 39
        event("e1", "rec_a", datetime(2026, 9, 28, 0, 20)),  # after midnight: still rec_a's day
        event("e2", "rec_b", datetime(2026, 9, 30, 10, 0)),
        event("e3", "rec_c", datetime(2026, 10, 1, 9, 0)),
    ]


def test_export_writes_the_constellation_and_the_index(tmp_path):
    from src.chronos.graph_rag import Entity, EntityType

    gx = _extractor(ANSWERS)
    gx.extract_from_events(_moments())
    out = tmp_path / "entity_graph.json"
    counts = gx.export_json(out, recording_days={"rec_b": "2026-09-29"})
    data = json.loads(out.read_text())
    assert counts == {"nodes": len(data["nodes"]), "edges": len(data["edges"])}
    assert data["version"] == 2 and data["generated_at"] and data["layout"]["mode"] == "full"
    assert data["communities"] and data["dropped_topics"] == 0

    avery = Entity.generate_id("Avery", EntityType.PERSON)
    lake = Entity.generate_id("Lakeview", EntityType.LOCATION)
    by_id = {n["id"]: n for n in data["nodes"]}
    for node in data["nodes"]:
        assert {"x", "y", "community", "weeks"} <= set(node)
        assert math.hypot(node["x"], node["y"]) <= 1.0
    # e0, e1 -> rec_a's day 2026-09-27 (week 39); e3 -> 2026-10-01 (week 40)
    assert by_id[avery]["weeks"] == {"2026-W39": 2, "2026-W40": 1}
    assert by_id[lake]["weeks"] == {"2026-W39": 2}

    index = json.loads((tmp_path / "entity_index.json").read_text())
    blake = Entity.generate_id("Blake", EntityType.PERSON)
    north = Entity.generate_id("Northwind Labs", EntityType.ORGANIZATION)
    assert index["recordings"]["rec_a"][avery] == 2 and index["recordings"]["rec_a"][lake] == 2
    assert index["recordings"]["rec_b"] == {blake: 1, north: 1}
    assert set(index["days"]) == {"2026-09-27", "2026-09-29", "2026-10-01"}  # rec_b from recording_days
    assert index["days"]["2026-10-01"] == {avery: 1, blake: 1}

    # the next build keeps every position (a rebuild of the same moments)
    gx.extract_from_events(_moments())
    gx.export_json(out)
    rebuilt = {n["id"]: (n["x"], n["y"]) for n in json.loads(out.read_text())["nodes"]}
    assert rebuilt == {n["id"]: (n["x"], n["y"]) for n in data["nodes"]}
    assert json.loads(out.read_text())["layout"]["mode"] == "incremental"


def test_export_still_writes_the_graph_when_the_constellation_fails(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("layout exploded")

    monkeypatch.setattr(C, "build", boom)
    gx = _extractor(ANSWERS)
    gx.extract_from_events(_moments())
    out = tmp_path / "entity_graph.json"
    gx.export_json(out)
    data = json.loads(out.read_text())
    assert data["nodes"] and "x" not in data["nodes"][0] and "communities" not in data


def test_assembly_version_bumped_so_the_pipeline_rebuilds():
    assert ChronosGraphExtractor.ASSEMBLY_VERSION == 3


# --------------------------------------------------------------------------- API

import os  # noqa: E402
from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.chronos import entity_graph  # noqa: E402


def _api_files(tmp_path):
    """entity_graph.json + entity_index.json as export_json writes them, for NODES/EDGES."""
    result = C.build(NODES, EDGES)
    weeks = {
        "p_avery": {"2026-W30": 3, "2026-W39": 2}, "t_budget": {"2026-W39": 5}, "p_blake": {"2026-W40": 1},
        "o_north": {"2026-W20": 1}, "j_harbor": {"2026-W39": 1}, "l_lake": {"2026-W39": 2},
        "p_corin": {"2026-W39": 1}, "t_trail": {"2026-W38": 2}, "t_cabin": {"2026-W38": 2},
        "l_pine": {"2026-W10": 1}, "a_plants": {"2026-W40": 1},
    }
    nodes = []
    for node in NODES:
        x, y = result["positions"][node["id"]]
        nodes.append({**node, "aliases": [], "first_seen": "2026-03-02T09:00:00", "last_seen": "2026-09-30T18:00:00",
                      "events": ["ev1", "ev2"] if node["id"] == "p_avery" else [],
                      "x": round(x, 4), "y": round(y, 4), "community": result["community"][node["id"]],
                      "weeks": weeks[node["id"]]})
    edges = [{**e, "evidence": []} for e in EDGES]
    payload = {"version": 2, "generated_at": "2026-10-02T14:00:00+00:00", "events": 42, "dropped_topics": 7,
               "layout": result["layout"], "communities": result["communities"], "nodes": nodes, "edges": edges}
    (tmp_path / "entity_graph.json").write_text(json.dumps(payload))
    (tmp_path / "entity_index.json").write_text(json.dumps({
        "version": 1,
        "recordings": {"rec_a": {"p_avery": 2, "l_lake": 3, "gone_entity": 9}, "rec_b": {"p_blake": 1}},
        "days": {"2026-09-27": {"p_avery": 2, "l_lake": 3}, "2026-09-29": {"p_blake": 1}},
    }))
    return result


def _reset_caches():
    entity_graph._cache.update(path=None, mtime=None, graph=None)
    entity_graph._index_cache.update(path=None, mtime=None, index=None)


class _Svc:
    def _get_all_events(self):
        return [
            SimpleNamespace(id="ev1", recording_id="rec_a", category="work", start_ts=datetime(2026, 9, 27, 23, 50),
                            clean_text="Avery booked the Lakeview cabin."),
            SimpleNamespace(id="ev2", recording_id="rec_a", category="social", start_ts=datetime(2026, 9, 28, 0, 20),
                            clean_text="Avery and Blake talked budgets."),
        ]


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from api.dependencies import get_service

    result = _api_files(tmp_path)
    monkeypatch.setattr(entity_graph, "graph_path", lambda: tmp_path / "entity_graph.json")
    _reset_caches()
    get_service.cache_clear()
    with patch.dict(os.environ, {"CHRONOS_API_KEY": ""}, clear=False):
        from api.main import app

        app.dependency_overrides[get_service] = lambda: _Svc()
        yield SimpleNamespace(client=TestClient(app, raise_server_exceptions=False), result=result)
        app.dependency_overrides.clear()
    _reset_caches()


def test_constellation_returns_the_map_with_meta(api):
    body = api.client.get("/api/v1/graph/constellation").json()
    assert set(body) == {"meta", "communities", "nodes", "edges"}
    names = [n["name"] for n in body["nodes"]]
    assert names[:3] == ["budget review", "Avery", "Blake"]  # by mentions
    node = body["nodes"][0]
    assert set(node) == {"id", "name", "type", "mentions", "community", "x", "y", "first_seen", "last_seen", "weeks"}
    assert node["weeks"] == {"2026-W39": 5}
    meta = body["meta"]
    assert meta["nodes_total"] == meta["nodes_matching"] == meta["nodes_shown"] == len(NODES)
    assert meta["edges_total"] == len(EDGES) and meta["edges_shown"] == len(EDGES)
    assert meta["events_covered"] == 42 and meta["generated_at"] == "2026-10-02T14:00:00+00:00"
    assert (meta["first_week"], meta["last_week"]) == ("2026-W10", "2026-W40")
    assert meta["layout_version"] == C.LAYOUT_VERSION and meta["dropped_single_moment_topics"] == 7
    assert meta["truncated"] is False and meta["truncation"] == []
    # stated relationships come before co-mentions
    assert body["edges"][0]["type"] == "works_with"
    assert set(body["edges"][0]) == {"source", "target", "type", "weight"}
    assert {c["id"] for c in body["communities"]} == {0, 1, 2} and all(c["shown"] >= 1 for c in body["communities"])


def test_constellation_limit_and_filters_say_what_was_cut(api):
    body = api.client.get("/api/v1/graph/constellation?limit=2").json()
    assert [n["id"] for n in body["nodes"]] == ["t_budget", "p_avery"]
    assert {(e["source"], e["target"]) for e in body["edges"]} <= {("p_avery", "t_budget"), ("t_budget", "p_avery")}
    meta = body["meta"]
    assert meta["truncated"] is True and meta["nodes_shown"] == 2 and meta["nodes_matching"] == len(NODES)
    assert meta["truncation"][0]["what"] == "nodes" and meta["truncation"][0]["of"] == len(NODES)
    assert [c["id"] for c in body["communities"]] == [0] and body["communities"][0]["shown"] == 2

    people_places = api.client.get("/api/v1/graph/constellation?types=person,location").json()
    assert {n["type"] for n in people_places["nodes"]} == {"person", "location"}
    assert people_places["meta"]["filters"]["types"] == ["location", "person"]

    # active in the window by ISO week: 2026-09-21..2026-09-27 is week 39
    window = api.client.get("/api/v1/graph/constellation?since=2026-09-21&until=2026-09-27").json()
    assert {n["id"] for n in window["nodes"]} == {"p_avery", "t_budget", "j_harbor", "l_lake", "p_corin"}
    since_only = api.client.get("/api/v1/graph/constellation?since=2026-09-28").json()
    assert {n["id"] for n in since_only["nodes"]} == {"p_blake", "a_plants"}
    assert api.client.get("/api/v1/graph/constellation?since=yesterday").status_code == 422


def test_constellation_caps_edges_and_reports_it():
    graph = entity_graph.EntityGraph.from_payload({
        "nodes": [{"id": n, "name": n, "type": "topic", "mentions": 1} for n in ("a", "b", "c")],
        "edges": [_edge("a", "b", 5.0), _edge("b", "c", 1.0), _edge("a", "c", 1.0, "related_to")],
    })
    body = entity_graph.constellation(graph, limit=10, edges_per_node=1)
    assert [(e["source"], e["target"]) for e in body["edges"]] == [("a", "c"), ("a", "b"), ("b", "c")]
    body = entity_graph.constellation(graph, limit=1, edges_per_node=1)
    assert body["edges"] == [] and body["meta"]["edges_matching"] == 0
    graph.edges.extend([_edge("a", "b", 2.0, "knows")])
    body = entity_graph.constellation(graph, limit=2, edges_per_node=1)
    assert [e["type"] for e in body["edges"]] == ["knows", "co_mentioned"]
    tight = entity_graph.constellation(graph, limit=3, edges_per_node=1)
    assert len(tight["edges"]) == 3 and tight["meta"]["edges_matching"] == 4
    assert tight["meta"]["truncation"] == [{"what": "edges", "shown": 3, "of": 4, "rule": tight["meta"]["truncation"][0]["rule"]}]


def test_recording_and_day_entities(api):
    rec = api.client.get("/api/v1/graph/recordings/rec_a/entities").json()
    assert rec["recording_id"] == "rec_a" and rec["total"] == 2
    assert [(e["id"], e["count"]) for e in rec["entities"]] == [("l_lake", 3), ("p_avery", 2)]  # gone_entity skipped
    lake = rec["entities"][0]
    assert set(lake) == {"id", "name", "type", "mentions", "community", "x", "y", "count"}
    assert (lake["x"], lake["y"]) == tuple(round(v, 4) for v in api.result["positions"]["l_lake"])
    assert api.client.get("/api/v1/graph/recordings/unknown/entities").json() == {
        "recording_id": "unknown", "entities": [], "total": 0}

    day = api.client.get("/api/v1/graph/days/2026-09-29/entities").json()
    assert day == {"date": "2026-09-29", "total": 1, "entities": [day["entities"][0]]}
    assert day["entities"][0]["id"] == "p_blake" and day["entities"][0]["count"] == 1
    assert api.client.get("/api/v1/graph/days/2026-01-01/entities").json()["entities"] == []
    assert api.client.get("/api/v1/graph/days/not-a-date/entities").status_code == 422


def test_entity_detail_has_map_position_weeks_and_moment_context(api):
    detail = api.client.get("/api/v1/graph/entities/p_avery").json()
    entity = detail["entity"]
    assert entity["weeks"] == {"2026-W30": 3, "2026-W39": 2}
    assert entity["community"] == api.result["community"]["p_avery"]
    assert (entity["x"], entity["y"]) == tuple(round(v, 4) for v in api.result["positions"]["p_avery"])
    assert entity["name"] == "Avery" and entity["first_seen"] == "2026-03-02T09:00:00"  # unchanged fields
    first = detail["recent_events"][0]
    assert first["id"] == "ev1" and first["recording_id"] == "rec_a" and first["category"] == "work"
    assert first["start_ts"].startswith("2026-09-27T23:50") and first["text"]


def test_constellation_endpoints_are_empty_without_files(tmp_path, monkeypatch):
    from api.dependencies import get_service

    monkeypatch.setattr(entity_graph, "graph_path", lambda: tmp_path / "missing" / "entity_graph.json")
    _reset_caches()
    get_service.cache_clear()
    with patch.dict(os.environ, {"CHRONOS_API_KEY": ""}, clear=False):
        from api.main import app

        app.dependency_overrides[get_service] = lambda: _Svc()
        try:
            client = TestClient(app, raise_server_exceptions=False)
            body = client.get("/api/v1/graph/constellation").json()
            assert body["nodes"] == [] and body["edges"] == [] and body["communities"] == []
            assert body["meta"]["nodes_total"] == 0 and body["meta"]["truncated"] is False
            assert client.get("/api/v1/graph/recordings/rec_a/entities").json()["entities"] == []
            assert client.get("/api/v1/graph/days/2026-09-27/entities").json()["entities"] == []
        finally:
            app.dependency_overrides.clear()
            _reset_caches()


def test_old_export_without_layout_still_serves(tmp_path, monkeypatch):
    """Before the first pipeline run with the constellation: nodes without x/y/weeks."""
    graph = entity_graph.EntityGraph.from_payload({
        "version": 1, "events": 3,
        "nodes": [{"id": "p_avery", "name": "Avery", "type": "person", "mentions": 3}],
        "edges": [],
    })
    body = entity_graph.constellation(graph)
    assert body["nodes"][0]["x"] is None and body["nodes"][0]["weeks"] == {}
    assert body["meta"]["layout_version"] is None and body["communities"] == []
    assert entity_graph.constellation(graph, since="2026-09-01")["nodes"] == []  # no weeks, not active
