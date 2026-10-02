#!/usr/bin/env python3
"""Time and memory of the entity-graph export with the constellation, on made-up data.

    python scripts/benchmark_constellation.py [--nodes 5000] [--edges 30000] [--moments 17650]

Builds a synthetic graph shaped like the real one (heavy-tailed communities, hub entities,
~10% stated relationships, some isolated entities), then runs ChronosGraphExtractor.export_json
three times into a temporary directory: a full layout, a rebuild with one new entity, and a
rebuild with 10% new entities. Prints seconds and the growth of the process's peak RSS; with
--tracemalloc, the peak of Python + numpy allocations instead (tracemalloc slows the run ~5x,
so its seconds are not the real ones). Nothing outside the temporary directory is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
import tracemalloc
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chronos.graph_rag import Entity, EntityType, KnowledgeGraph, Relationship, RelationType  # noqa: E402
from src.chronos.graph_service import ChronosGraphExtractor  # noqa: E402

TYPES = ["topic", "person", "location", "organization", "action", "project"]
STATED = ["works_with", "works_on", "member_of", "part_of", "uses", "located_in", "knows", "related_to"]


def synthetic(n_nodes: int, n_edges: int, n_moments: int, n_recordings: int = 800, seed: int = 7):
    """(extractor ready to export, entity ids) for a made-up graph."""
    rng = np.random.default_rng(seed)
    kinds = rng.choice(TYPES, p=[0.55, 0.15, 0.10, 0.08, 0.08, 0.04], size=n_nodes)
    sizes, left = [], n_nodes
    while left > 0:
        size = int(min(left, max(1, rng.lognormal(2.3, 1.0))))
        sizes.append(size)
        left -= size
    home = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    popularity = rng.pareto(1.5, n_nodes) + 1.0
    ids = [f"syn{i:06d}" for i in range(n_nodes)]

    def pick(members, count):
        p = popularity[members] / popularity[members].sum()
        return rng.choice(members, size=count, p=p)

    kg = KnowledgeGraph()
    for i, node_id in enumerate(ids):
        kg.add_entity(Entity(id=node_id, name=f"Entity {i}", entity_type=EntityType(kinds[i]),
                             mention_count=int(popularity[i] * 3)))
    weights: dict[tuple[int, int, str], float] = {}
    size_p = np.array(sizes, dtype=float) ** 2
    size_p /= size_p.sum()
    everyone = np.arange(n_nodes)
    while len(weights) < n_edges:
        if rng.random() < 0.85:
            c = rng.choice(len(sizes), p=size_p)
            if sizes[c] < 2:
                continue
            a, b = pick(np.arange(starts[c], starts[c + 1]), 2)
        else:
            a, b = pick(everyone, 2)
        if a == b:
            continue
        kind = STATED[rng.integers(len(STATED))] if rng.random() < 0.1 else "co_mentioned"
        key = (min(a, b), max(a, b), kind)
        weights[key] = weights.get(key, 0.0) + 1.0
    for (a, b, kind), w in weights.items():
        kg.relationships.append(Relationship(source_id=ids[a], target_id=ids[b],
                                             relation_type=RelationType(kind), weight=w))

    gx = ChronosGraphExtractor.__new__(ChronosGraphExtractor)
    gx._knowledge_graph = kg
    gx._entity_stats = {}
    gx._moments = {}
    first_day = date(2026, 2, 1)
    recording_day = {f"rec{r:04d}": first_day + timedelta(days=int(rng.integers(240))) for r in range(n_recordings)}
    recordings = sorted(recording_day)
    for m in range(n_moments):
        event_id = f"ev{m:06d}"
        recording_id = recordings[int(rng.integers(n_recordings))]
        when = datetime.combine(recording_day[recording_id], datetime.min.time()) + timedelta(minutes=int(rng.integers(600)))
        iso = when.isoformat()
        gx._moments[event_id] = (recording_id, iso)
        c = int(rng.integers(len(sizes)))
        local = pick(np.arange(starts[c], starts[c + 1]), int(rng.integers(1, 8)))
        named = set(local.tolist()) | set(pick(everyone, int(rng.integers(0, 4))).tolist())
        kg.document_entities[event_id] = {ids[i] for i in named}
        for i in named:
            stat = gx._entity_stats.setdefault(ids[i], {"first": iso, "last": iso, "events": []})
            stat["first"], stat["last"] = min(stat["first"], iso), max(stat["last"], iso)
            stat["events"].append((event_id, iso))
    for i, node_id in enumerate(ids):  # every topic needs 2 moments to be exported
        stat = gx._entity_stats.setdefault(node_id, {"first": "", "last": "", "events": []})
        while len(stat["events"]) < 2:
            event_id = f"ev{int(rng.integers(n_moments)):06d}"
            stat["events"].append((event_id, gx._moments[event_id][1]))
    return gx, ids


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure(label: str, gx: ChronosGraphExtractor, path: Path, traced: bool) -> None:
    rss_before = _peak_rss_mb()
    if traced:
        tracemalloc.start()
    started = time.perf_counter()
    counts = gx.export_json(path)
    seconds = time.perf_counter() - started
    if traced:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory = f"tracemalloc peak {peak / 2**20:.0f} MB"
    else:
        memory = f"peak RSS +{_peak_rss_mb() - rss_before:.0f} MB (now {_peak_rss_mb():.0f} MB)"
    data = json.loads(path.read_text())
    layout = data.get("layout", {})
    print(
        f"{label}: {counts['nodes']} nodes, {counts['edges']} edges, {len(data.get('communities', []))} communities; "
        f"export {seconds:.1f} s (layout {layout.get('seconds')} s, mode {layout.get('mode')}); {memory}; "
        f"graph file {path.stat().st_size / 2**20:.1f} MB, index {path.with_name('entity_index.json').stat().st_size / 2**20:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--edges", type=int, default=30000)
    parser.add_argument("--moments", type=int, default=17650)
    parser.add_argument("--tracemalloc", action="store_true", help="measure allocations (slow)")
    args = parser.parse_args()

    print(f"process peak RSS after imports: {_peak_rss_mb():.0f} MB")
    gx, ids = synthetic(args.nodes, args.edges, args.moments)
    print(f"synthetic graph built; process peak RSS {_peak_rss_mb():.0f} MB")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "entity_graph.json"
        measure("full layout", gx, path, args.tracemalloc)

        kg = gx._knowledge_graph
        newcomer = Entity(id="syn-new-1", name="New entity", entity_type=EntityType.PERSON, mention_count=2)
        kg.add_entity(newcomer)
        kg.relationships.append(Relationship(source_id=ids[0], target_id=newcomer.id,
                                             relation_type=RelationType.CO_MENTIONED, weight=2.0))
        gx._entity_stats[newcomer.id] = {"first": "", "last": "", "events": [("ev000000", gx._moments["ev000000"][1])]}
        before = {n["id"]: (n["x"], n["y"]) for n in json.loads(path.read_text())["nodes"]}
        measure("one new entity", gx, path, args.tracemalloc)
        after = {n["id"]: (n["x"], n["y"]) for n in json.loads(path.read_text())["nodes"]}
        moved = max(abs(after[i][0] - before[i][0]) + abs(after[i][1] - before[i][1]) for i in before)
        print(f"  largest move of an existing entity: {moved:.4f}")

        rng = np.random.default_rng(11)
        extra = max(1, args.nodes // 10)
        for j in range(extra):
            node_id = f"syn-grow-{j}"
            kg.add_entity(Entity(id=node_id, name=f"Grown {j}", entity_type=EntityType.TOPIC, mention_count=2))
            anchor = ids[int(rng.integers(len(ids)))] if j % 5 else f"syn-grow-{max(0, j - 1)}"
            if anchor != node_id:
                kg.relationships.append(Relationship(source_id=anchor, target_id=node_id,
                                                     relation_type=RelationType.CO_MENTIONED, weight=1.0))
            gx._entity_stats[node_id] = {"first": "", "last": "", "events": [("ev000001", ""), ("ev000002", "")]}
        measure(f"{extra} new entities", gx, path, args.tracemalloc)


if __name__ == "__main__":
    main()
