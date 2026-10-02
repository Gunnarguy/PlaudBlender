"""The entity graph has real entity-to-entity edges and a usable JSON export (2026-10-01).

Before: relationships pointed from an event id to an entity, events are not graph nodes, so
the assembled graph had 1,733 nodes and 14 edges -- and nothing read it.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from src.chronos.graph_rag import Entity, EntityExtractor, EntityType, RelationType
from src.chronos.graph_service import ChronosGraphExtractor


def _parse(data, doc_id="e1"):
    extractor = EntityExtractor.__new__(EntityExtractor)  # parsing needs no LLM
    return extractor._parse_entities_from_response(data, doc_id, 500)


def test_honorifics_and_case_merge_people_but_not_other_types():
    assert Entity.generate_id("Dr. Patel", EntityType.PERSON) == Entity.generate_id("patel", EntityType.PERSON)
    assert Entity.generate_id("  Mike  ", EntityType.PERSON) == Entity.generate_id("mike", EntityType.PERSON)
    assert Entity.generate_id("Dr. Pepper", EntityType.ORGANIZATION) != Entity.generate_id("Pepper", EntityType.ORGANIZATION)


def test_locations_and_typed_relationships_with_evidence():
    entities, rels = _parse({
        "people": [{"name": "Gunnar"}, {"name": "Dr. Patel", "role": "surgeon"}],
        "locations": ["St. Mary's"],
        "relationships": [
            {"source": "Gunnar", "source_type": "person", "relation": "works_with",
             "target": "Patel", "target_type": "person", "evidence": "OR prep meeting with Dr. Patel"},
            # endpoint only named inside the relationship -> entity is created
            {"source": "Mike", "source_type": "person", "relation": "knows",
             "target": "Gunnar", "target_type": "person", "evidence": "called Mike about Tahoe"},
            # rejected: unknown relation, unknown type, self-loop
            {"source": "Gunnar", "source_type": "person", "relation": "hates", "target": "Mike", "target_type": "person"},
            {"source": "Gunnar", "source_type": "pet", "relation": "knows", "target": "Mike", "target_type": "person"},
            {"source": "Gunnar", "source_type": "person", "relation": "knows", "target": "gunnar", "target_type": "person"},
        ],
    })
    by_name = {e.name: e for e in entities}
    assert by_name["St. Mary's"].entity_type == EntityType.LOCATION
    assert "Mike" in by_name
    typed = [r for r in rels if r.relation_type in (RelationType.WORKS_WITH, RelationType.KNOWS)]
    assert len(typed) == 2
    works = next(r for r in typed if r.relation_type == RelationType.WORKS_WITH)
    assert works.target_id == Entity.generate_id("Dr. Patel", EntityType.PERSON)
    assert works.metadata["evidence"] == ["OR prep meeting with Dr. Patel"]


class _Batch:
    supports_batch = True

    def __init__(self, answers):
        self.answers = answers

    def extract_entities_batch(self, items):
        extractor = EntityExtractor.__new__(EntityExtractor)
        return {doc_id: extractor._parse_entities_from_response(self.answers[doc_id], doc_id, 100)
                for doc_id, _ in items if doc_id in self.answers}


def _events(n):
    base = datetime(2026, 9, 30, 9, 0)
    return [SimpleNamespace(event_id=f"e{i}", recording_id="r1", clean_text=f"text {i}",
                            start_ts=base + timedelta(hours=i), category=SimpleNamespace(value="work"))
            for i in range(n)]


def _gx(answers):
    gx = ChronosGraphExtractor.__new__(ChronosGraphExtractor)
    gx.entity_extractor = _Batch(answers)
    gx.community_detector = None
    gx.cache_path = None
    gx._fresh = {}
    gx._entity_stats = {}
    return gx


def test_graph_has_entity_edges_comentions_and_a_json_export(tmp_path):
    answers = {
        "e0": {"people": [{"name": "Gunnar"}, {"name": "Mike"}], "locations": ["Tahoe"],
               "relationships": [{"source": "Gunnar", "source_type": "person", "relation": "knows",
                                  "target": "Mike", "target_type": "person", "evidence": "my buddy Mike"}]},
        "e1": {"people": [{"name": "Mike"}], "locations": ["Tahoe"], "actions": [{"task": "book the cabin"}]},
        "e2": {"people": [{"name": "Dr. Patel"}], "organizations": ["St. Mary's"]},
    }
    gx = _gx(answers)
    _, graph = gx.extract_from_events(_events(3))
    mike = Entity.generate_id("Mike", EntityType.PERSON)
    tahoe = Entity.generate_id("Tahoe", EntityType.LOCATION)
    gunnar = Entity.generate_id("Gunnar", EntityType.PERSON)
    assert graph.has_edge(mike, tahoe) and graph.has_edge(gunnar, mike)
    # Mike + Tahoe co-mentioned in two moments -> one edge, weight 2
    co = [r for r in gx._knowledge_graph.relationships
          if r.relation_type == RelationType.CO_MENTIONED and {r.source_id, r.target_id} == {mike, tahoe}]
    assert len(co) == 1 and co[0].weight == 2
    assert graph.number_of_edges() >= 4

    out = tmp_path / "entity_graph.json"
    counts = gx.export_json(out)
    data = json.loads(out.read_text())
    assert counts == {"nodes": len(data["nodes"]), "edges": len(data["edges"])}
    node = next(n for n in data["nodes"] if n["id"] == mike)
    assert node["events"] == ["e1", "e0"]  # latest first
    assert node["first_seen"] < node["last_seen"]
    knows = next(e for e in data["edges"] if e["type"] == "knows")
    assert knows["evidence"] == ["my buddy Mike"]
    # actions never get co-mention links
    action_ids = {n["id"] for n in data["nodes"] if n["type"] == "action"}
    assert not any(e["type"] == "co_mentioned" and (e["source"] in action_ids or e["target"] in action_ids)
                   for e in data["edges"])


def test_cache_version_bumped_for_the_new_extraction_format():
    assert ChronosGraphExtractor.CACHE_VERSION == 2


def test_speaker_labels_are_not_people():
    answers = {
        "e0": {"people": [{"name": "Speaker 10"}, {"name": "Jeff"}, {"name": "speaker_4"}],
               "relationships": [{"source": "Speaker 10", "source_type": "person", "relation": "works_with",
                                  "target": "Jeff", "target_type": "person", "evidence": "x"}]},
    }
    gx = _gx(answers)
    _, graph = gx.extract_from_events(_events(1))
    names = {e.name for e in gx._knowledge_graph.entities.values()}
    assert names == {"Jeff"}
    assert graph.number_of_edges() == 0
