"""Read side of the entity knowledge graph (data/cache/graphs/entity_graph.json).

The pipeline writes the file (ChronosGraphExtractor.export_json) and entity_index.json next
to it. This module loads them (cached by mtime), finds the people, places, organizations and
projects a question names, and turns them into Ask evidence and API payloads, including the
constellation map. Nothing here calls a model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Types a question can name directly; topics only count when they are multi-word phrases
# ("knee surgery"), since single words ("work") match far too much.
MATCH_TYPES = ("person", "location", "organization", "project")
_NON_WORD = re.compile(r"[^\w']+", re.UNICODE)
_cache: dict[str, Any] = {"path": None, "mtime": None, "graph": None}
_index_cache: dict[str, Any] = {"path": None, "mtime": None, "index": None}
INDEX_NAME = "entity_index.json"


def graph_path() -> Path:
    from src.config import get_settings

    return Path(get_settings().chronos_graph_cache_dir) / "entity_graph.json"


def index_path() -> Path:
    return graph_path().with_name(INDEX_NAME)


def _key(text: str, entity_type: str = "") -> str:
    from src.chronos.graph_rag import Entity, EntityType

    try:
        kind = EntityType(entity_type)
    except ValueError:
        kind = EntityType.TOPIC
    return " ".join(_NON_WORD.sub(" ", Entity.canonical_name(text, kind)).split())


@dataclass
class EntityGraph:
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    events: int = 0
    adjacency: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    by_name: dict[str, list[str]] = field(default_factory=dict)
    communities: list[dict[str, Any]] = field(default_factory=list)
    layout: dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[str] = None
    dropped_topics: Optional[int] = None
    first_week: Optional[str] = None
    last_week: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EntityGraph":
        nodes = {n["id"]: n for n in payload.get("nodes", []) if n.get("id")}
        edges = [e for e in payload.get("edges", []) if e.get("source") in nodes and e.get("target") in nodes]
        graph = cls(nodes=nodes, edges=edges, events=int(payload.get("events") or 0),
                    communities=list(payload.get("communities") or []), layout=dict(payload.get("layout") or {}),
                    generated_at=payload.get("generated_at"), dropped_topics=payload.get("dropped_topics"))
        for edge in edges:
            graph.adjacency.setdefault(edge["source"], []).append(edge)
            graph.adjacency.setdefault(edge["target"], []).append(edge)
        weeks = [week for node in nodes.values() for week in (node.get("weeks") or {})]
        if weeks:
            graph.first_week, graph.last_week = min(weeks), max(weeks)
        for node_id, node in nodes.items():
            for name in [node.get("name", "")] + list(node.get("aliases") or []):
                key = _key(name, node.get("type", ""))
                if len(key) >= 3:
                    graph.by_name.setdefault(key, []).append(node_id)
        return graph

    def neighbors(self, node_id: str, limit: int = 8) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """(edge, other node) pairs: stated relationships first, then co-mentions; heavier first."""
        pairs = []
        for edge in self.adjacency.get(node_id, []):
            other = edge["target"] if edge["source"] == node_id else edge["source"]
            if other in self.nodes:
                pairs.append((edge, self.nodes[other]))
        pairs.sort(key=lambda p: (p[0].get("type") == "co_mentioned", -float(p[0].get("weight") or 0)))
        return pairs[:limit]

    def search(self, query: str, limit: int = 20, entity_type: Optional[str] = None) -> list[dict[str, Any]]:
        needle = query.strip().casefold()
        if not needle:
            return []
        hits = [
            node for node in self.nodes.values()
            if (not entity_type or node.get("type") == entity_type)
            and (needle in str(node.get("name", "")).casefold()
                 or any(needle in str(a).casefold() for a in node.get("aliases") or []))
        ]
        hits.sort(key=lambda n: -int(n.get("mentions") or 0))
        return hits[:limit]

    def match_question(self, question: str, limit: int = 4) -> list[dict[str, Any]]:
        """Entities the question names, most specific (longest name) and most mentioned first."""
        text = f" {_key(question)} "
        found: dict[str, dict[str, Any]] = {}
        for key, node_ids in self.by_name.items():
            if f" {key} " not in text:
                continue
            for node_id in node_ids:
                node = self.nodes[node_id]
                kind = node.get("type")
                if kind in MATCH_TYPES or (kind == "topic" and " " in key):
                    found.setdefault(node_id, {**node, "_matched": key})
        ranked = sorted(found.values(), key=lambda n: (-len(n["_matched"]), -int(n.get("mentions") or 0)))
        return ranked[:limit]


def load_entity_graph(path: Optional[Path] = None) -> Optional[EntityGraph]:
    """The exported graph, re-read only when the file changes. None if it doesn't exist yet."""
    target = Path(path) if path else graph_path()
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return None
    if _cache["path"] == str(target) and _cache["mtime"] == mtime:
        return _cache["graph"]
    try:
        graph = EntityGraph.from_payload(json.loads(target.read_text()))
    except Exception as exc:  # a half-written or old file must never break Ask
        logger.warning("Entity graph unreadable at %s: %s", target, exc)
        return None
    _cache.update(path=str(target), mtime=mtime, graph=graph)
    return graph


def load_entity_index(path: Optional[Path] = None) -> dict[str, dict[str, dict[str, int]]]:
    """entity_index.json: {"recordings": {id: {entity: n}}, "days": {date: {entity: n}}},
    re-read only when the file changes; empty maps when it is missing or unreadable."""
    target = Path(path) if path else index_path()
    empty: dict[str, dict[str, dict[str, int]]] = {"recordings": {}, "days": {}}
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return empty
    if _index_cache["path"] == str(target) and _index_cache["mtime"] == mtime:
        return _index_cache["index"]
    try:
        payload = json.loads(target.read_text())
        index = {"recordings": dict(payload.get("recordings") or {}), "days": dict(payload.get("days") or {})}
    except Exception as exc:  # a half-written or old file must never break the API
        logger.warning("Entity index unreadable at %s: %s", target, exc)
        return empty
    _index_cache.update(path=str(target), mtime=mtime, index=index)
    return index


def node_payload(node: dict[str, Any]) -> dict[str, Any]:
    return {k: node.get(k) for k in ("id", "name", "type", "mentions", "aliases", "first_seen", "last_seen")}


# ---------------------------------------------------------------------------- constellation

CONSTELLATION_NODE_KEYS = ("id", "name", "type", "mentions", "community", "x", "y", "first_seen", "last_seen")


def constellation_node(node: dict[str, Any]) -> dict[str, Any]:
    return {**{k: node.get(k) for k in CONSTELLATION_NODE_KEYS}, "weeks": dict(node.get("weeks") or {})}


def _week_key(day: Optional[str]) -> Optional[str]:
    """'2026-10-02' -> '2026-W40', the keys of a node's weeks (constellation.iso_week)."""
    if not day:
        return None
    year, week, _ = date.fromisoformat(day[:10]).isocalendar()
    return f"{year}-W{week:02d}"


def constellation(
    graph: Optional[EntityGraph],
    *,
    limit: int = 800,
    types: Optional[set[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    edges_per_node: int = 4,
) -> dict[str, Any]:
    """The map payload: the ``limit`` most-mentioned entities of the wanted types that were
    active between ``since`` and ``until`` (YYYY-MM-DD, by ISO week: an entity counts when
    any of its weeks overlaps the window), the communities they belong to, and the links
    among them (stated relationships first, then co-mentions by weight; at most
    ``edges_per_node`` per entity shown). Everything cut is counted in meta["truncation"].
    """
    filters = {"types": sorted(types) if types else None, "since": since, "until": until, "limit": limit}
    if graph is None:
        return {"meta": {"generated_at": None, "events_covered": 0, "nodes_total": 0, "nodes_matching": 0,
                         "nodes_shown": 0, "edges_total": 0, "edges_matching": 0, "edges_shown": 0,
                         "communities_total": 0, "communities_shown": 0, "first_week": None, "last_week": None,
                         "layout_version": None, "layout_mode": None, "dropped_single_moment_topics": None,
                         "filters": filters, "truncated": False, "truncation": []},
                "communities": [], "nodes": [], "edges": []}
    lo, hi = _week_key(since), _week_key(until)

    def active(node: dict[str, Any]) -> bool:
        if lo is None and hi is None:
            return True
        return any((lo is None or week >= lo) and (hi is None or week <= hi) for week in (node.get("weeks") or {}))

    matching = [
        node for node in graph.nodes.values()
        if (not types or node.get("type") in types) and active(node)
    ]
    matching.sort(key=lambda n: (-int(n.get("mentions") or 0), str(n.get("name") or ""), n["id"]))
    shown = matching[:limit]
    ids = {node["id"] for node in shown}
    among = [e for e in graph.edges if e["source"] in ids and e["target"] in ids]
    among.sort(key=lambda e: (e.get("type") == "co_mentioned", -float(e.get("weight") or 0), e["source"], e["target"]))
    cap = edges_per_node * len(shown)
    edges = [{k: e.get(k) for k in ("source", "target", "type", "weight")} for e in among[:cap]]

    in_view: dict[Any, int] = {}
    for node in shown:
        if node.get("community") is not None:
            in_view[node["community"]] = in_view.get(node["community"], 0) + 1
    communities = [{**c, "shown": in_view[c["id"]]} for c in graph.communities if c.get("id") in in_view]

    truncation = []
    if len(matching) > len(shown):
        truncation.append({"what": "nodes", "shown": len(shown), "of": len(matching),
                           "rule": f"the {limit} most-mentioned entities that match the filters"})
    if len(among) > len(edges):
        truncation.append({"what": "edges", "shown": len(edges), "of": len(among),
                           "rule": f"links among the entities shown: stated relationships first, then "
                                   f"co-mentions by weight, at most {edges_per_node} per entity shown"})
    meta = {
        "generated_at": graph.generated_at,
        "events_covered": graph.events,
        "nodes_total": len(graph.nodes),
        "nodes_matching": len(matching),
        "nodes_shown": len(shown),
        "edges_total": len(graph.edges),
        "edges_matching": len(among),
        "edges_shown": len(edges),
        "communities_total": len(graph.communities),
        "communities_shown": len(communities),
        "first_week": graph.first_week,
        "last_week": graph.last_week,
        "layout_version": graph.layout.get("version"),
        "layout_mode": graph.layout.get("mode"),
        "dropped_single_moment_topics": graph.dropped_topics,
        "filters": filters,
        "truncated": bool(truncation),
        "truncation": truncation,
    }
    return {"meta": meta, "communities": communities, "nodes": [constellation_node(n) for n in shown], "edges": edges}


def entities_with_counts(graph: Optional[EntityGraph], counts: dict[str, int]) -> list[dict[str, Any]]:
    """Entities of one recording or day from the index, with where they sit on the map;
    most moments first. Ids the graph no longer has are skipped."""
    if graph is None:
        return []
    out = []
    for entity_id, count in (counts or {}).items():
        node = graph.nodes.get(entity_id)
        if node is None:
            continue
        out.append({**{k: node.get(k) for k in ("id", "name", "type", "mentions", "community", "x", "y")},
                    "count": int(count)})
    out.sort(key=lambda e: (-e["count"], -int(e.get("mentions") or 0), str(e.get("name") or ""), e["id"]))
    return out


def ask_graph_context(
    svc,
    question: str,
    *,
    max_entities: int = 4,
    events_per_entity: int = 3,
    neighbor_limit: int = 6,
    graph: Optional[EntityGraph] = None,
) -> list[dict[str, Any]]:
    """Ask evidence blocks for the entities a question names (kind "graph_entity")."""
    graph = graph or load_entity_graph()
    if graph is None:
        return []
    matches = graph.match_question(question, limit=max_entities)
    if not matches:
        return []

    wanted = {eid for node in matches for eid in (node.get("events") or [])[:events_per_entity]}
    moments: dict[str, Any] = {}
    if wanted:
        try:
            for event in svc._get_all_events():  # one pass instead of a scan per id
                if getattr(event, "id", None) in wanted:
                    moments[event.id] = event
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity graph: could not load moments: %s", exc)

    entries = []
    for node in matches:
        lines = [
            f"{node.get('name')} ({node.get('type')}): {node.get('mentions', 0)} mentions; "
            f"first seen {str(node.get('first_seen') or '?')[:10]}, last seen {str(node.get('last_seen') or '?')[:10]}."
        ]
        links = []
        for edge, other in graph.neighbors(node["id"], limit=neighbor_limit):
            relation = str(edge.get("type", "related_to")).replace("_", " ")
            evidence = (edge.get("evidence") or [""])[0]
            other_label = f"{other.get('name')} ({other.get('type')})"
            times = f"x{float(edge.get('weight') or 1):.0f}"
            # Whole statements keep the stated direction: "Jack member of Stryker", never
            # "Stryker: member of Jack" for an incoming edge.
            if edge.get("type") == "co_mentioned":
                link = f"mentioned with {other_label} {times}"
            elif edge.get("source") == node["id"]:
                link = f"{node.get('name')} {relation} {other_label} {times}"
            else:
                link = f"{other_label} {relation} {node.get('name')} {times}"
            links.append(link + (f' "{evidence}"' if evidence else ""))
        if links:
            lines.append("Linked: " + "; ".join(links))
        recent = [moments[eid] for eid in (node.get("events") or [])[:events_per_entity] if eid in moments]
        if recent:
            lines.append("Recent moments:")
            for event in recent:
                when = getattr(event, "start_ts", None)
                stamp = when.strftime("%Y-%m-%d %H:%M") if hasattr(when, "strftime") else "?"
                text = re.sub(r"\s+", " ", str(getattr(event, "clean_text", "")))[:220]
                lines.append(f"- {stamp}: {text}")
        entries.append({
            "kind": "graph_entity",
            "name": node.get("name"),
            "date": str(node.get("last_seen") or "")[:10],
            "time": "",
            "category": node.get("type", "entity"),
            "text": "\n".join(lines),
        })
    return entries
