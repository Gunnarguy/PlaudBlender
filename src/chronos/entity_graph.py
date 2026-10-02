"""Read side of the entity knowledge graph (data/cache/graphs/entity_graph.json).

The pipeline writes the file (ChronosGraphExtractor.export_json). This module loads it
(cached by mtime), finds the people, places, organizations and projects a question names,
and turns them into Ask evidence and API payloads. Nothing here calls a model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Types a question can name directly; topics only count when they are multi-word phrases
# ("knee surgery"), since single words ("work") match far too much.
MATCH_TYPES = ("person", "location", "organization", "project")
_NON_WORD = re.compile(r"[^\w']+", re.UNICODE)
_cache: dict[str, Any] = {"path": None, "mtime": None, "graph": None}


def graph_path() -> Path:
    from src.config import get_settings

    return Path(get_settings().chronos_graph_cache_dir) / "entity_graph.json"


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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EntityGraph":
        nodes = {n["id"]: n for n in payload.get("nodes", []) if n.get("id")}
        edges = [e for e in payload.get("edges", []) if e.get("source") in nodes and e.get("target") in nodes]
        graph = cls(nodes=nodes, edges=edges, events=int(payload.get("events") or 0))
        for edge in edges:
            graph.adjacency.setdefault(edge["source"], []).append(edge)
            graph.adjacency.setdefault(edge["target"], []).append(edge)
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


def node_payload(node: dict[str, Any]) -> dict[str, Any]:
    return {k: node.get(k) for k in ("id", "name", "type", "mentions", "aliases", "first_seen", "last_seen")}


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
            links.append(
                f"{relation} {other.get('name')} ({other.get('type')}, x{float(edge.get('weight') or 1):.0f})"
                + (f' "{evidence}"' if evidence else "")
            )
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
