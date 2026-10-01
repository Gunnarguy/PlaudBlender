"""Graph extraction integration for Chronos.

Bridges the ChronosEvent schema with the existing graph_rag.py module
to extract entities and relationships from cleaned narrative events.
"""

import logging
import os
import time as _time
from typing import List, Dict, Any, Tuple

import networkx as nx

from src.chronos.graph_rag import (
    CommunityDetector,
    Entity,
    EntityExtractor,
    EntityType,
    KnowledgeGraph,
    Relationship,
    RelationType,
)
from src.models.chronos_schemas import ChronosEvent

logger = logging.getLogger(__name__)


class ChronosGraphExtractor:
    """Extract entities and build knowledge graph from Chronos events."""

    CACHE_VERSION = 1

    def __init__(self, cache_path=None):
        """Initialize graph extraction components.

        Args:
            cache_path: optional JSON file of per-event extractions. With it, a rebuild
                only calls the model for events that are new or whose text changed.
        """
        self.cache_path = cache_path
        self._fresh: Dict[str, Dict[str, Any]] = {}
        self.entity_extractor = EntityExtractor()
        self.community_detector = CommunityDetector()

        # We keep the last KnowledgeGraph around because CommunityDetector
        # operates on KnowledgeGraph (it builds a NetworkX graph internally).
        self._knowledge_graph: KnowledgeGraph = KnowledgeGraph()

        logger.info("Initialized ChronosGraphExtractor")

    def extract_from_events(
        self,
        events: List[ChronosEvent],
        progress_callback=None,
    ) -> Tuple[List[Dict[str, Any]], nx.Graph]:
        """Extract entities and relationships from cleaned events.

        Args:
            events: List of ChronosEvent objects

        Returns:
            Tuple of (entities_list, networkx_graph)
        """
        logger.info(f"Extracting entities from {len(events)} events")

        from app_v2.services.xray import xray_log
        xray_log("graph", "extract",
                 f"Reading through {len(events)} moments to find people, places, and ideas")
        _ext_t0 = _time.perf_counter()

        # Reset for each extraction run.
        self._knowledge_graph = KnowledgeGraph()
        all_entities: List[Dict[str, Any]] = []
        self.last_failed_events = 0

        todo = self._apply_cache(events, all_entities, progress_callback)
        if todo:
            if getattr(self.entity_extractor, "supports_batch", False) is True:
                self._extract_batched(todo, all_entities, progress_callback)
            else:
                self._extract_one_by_one(todo, all_entities, progress_callback)
        self._save_cache(events)

        _ext_ms = (_time.perf_counter() - _ext_t0) * 1000
        xray_log("graph", "extract",
                 f"Found {len(all_entities)} people, places, and ideas across everything you've recorded",
                 duration_ms=round(_ext_ms, 1))
        logger.info(f"Extracted {len(all_entities)} total entities")

        # Build a NetworkX view (useful for quick stats + downstream visualization)
        _graph_t0 = _time.perf_counter()
        graph = nx.Graph()

        for entity_id, entity in self._knowledge_graph.entities.items():
            graph.add_node(
                entity_id,
                name=entity.name,
                type=getattr(entity.entity_type, "value", str(entity.entity_type)),
                mention_count=getattr(entity, "mention_count", 1),
            )

        for rel in self._knowledge_graph.relationships:
            if rel.source_id in graph.nodes and rel.target_id in graph.nodes:
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    weight=getattr(rel, "weight", 1.0),
                    type=getattr(rel.relation_type, "value", str(rel.relation_type)),
                )

        logger.info(
            f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )
        _graph_ms = (_time.perf_counter() - _graph_t0) * 1000
        xray_log("graph", "build",
                 f"Connected everything into a map: {graph.number_of_nodes()} ideas linked by {graph.number_of_edges()} connections",
                 duration_ms=round(_graph_ms, 1))

        return all_entities, graph

    @staticmethod
    def _text_hash(event) -> str:
        import hashlib

        return hashlib.sha1((event.clean_text or "").encode()).hexdigest()

    def _load_cache(self) -> Dict[str, Any]:
        import json
        from pathlib import Path

        if not self.cache_path or not Path(self.cache_path).exists():
            return {}
        try:
            data = json.loads(Path(self.cache_path).read_text())
            if data.get("version") == self.CACHE_VERSION:
                return data.get("events") or {}
        except Exception as e:  # a bad cache only costs a re-extraction
            logger.warning(f"Ignoring unreadable entity cache {self.cache_path}: {e}")
        return {}

    def _apply_cache(self, events, all_entities, progress_callback) -> list:
        """Replay cached extractions; return the events that still need the model."""
        self._fresh = {}
        cache = self._load_cache()
        todo = []
        for event in events:
            hit = cache.get(event.event_id)
            if not hit or hit.get("h") != self._text_hash(event):
                todo.append(event)
                continue
            try:
                entities = [
                    Entity(id=d["id"], name=d["name"], entity_type=EntityType(d["type"]),
                           aliases=d.get("aliases") or [], metadata=d.get("metadata") or {},
                           mention_count=d.get("mention_count", 1))
                    for d in hit.get("entities") or []
                ]
                relationships = [
                    Relationship(source_id=d["source"], target_id=d["target"],
                                 relation_type=RelationType(d["type"]),
                                 weight=d.get("weight", 1.0), metadata=d.get("metadata") or {})
                    for d in hit.get("relationships") or []
                ]
            except Exception:
                todo.append(event)
                continue
            self._add_event_result(event, entities, relationships, all_entities)
            if progress_callback:
                progress_callback(event.event_id)
        if self.cache_path:
            logger.info(f"Entity cache: {len(events) - len(todo)} reused, {len(todo)} to extract")
        return todo

    def _save_cache(self, events) -> None:
        import json
        from pathlib import Path

        if not self.cache_path:
            return
        wanted = {e.event_id for e in events}
        merged = {k: v for k, v in self._load_cache().items() if k in wanted}
        merged.update(self._fresh)
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": self.CACHE_VERSION, "events": merged}))
        os.replace(tmp, path)

    def _add_event_result(self, event, entities, relationships, all_entities) -> None:
        # Snapshot for the cache before KnowledgeGraph merges (and mutates) entities.
        self._fresh[event.event_id] = {
            "h": self._text_hash(event),
            "entities": [e.to_dict() for e in entities if hasattr(e, "to_dict")],
            "relationships": [r.to_dict() for r in relationships if hasattr(r, "to_dict")],
        }
        for ent in entities:
            self._knowledge_graph.add_entity(ent)
            # Preserve provenance in the exported dicts (helpful for debugging/UI).
            ent_dict = (
                ent.to_dict()
                if hasattr(ent, "to_dict")
                else {
                    "id": getattr(ent, "id", None),
                    "name": getattr(ent, "name", None),
                }
            )
            ent_dict["source_event_id"] = event.event_id
            ent_dict["source_recording_id"] = event.recording_id
            ent_dict["timestamp"] = event.start_ts.isoformat()
            ent_dict["category"] = event.category.value
            all_entities.append(ent_dict)

        for rel in relationships:
            self._knowledge_graph.add_relationship(rel)

        self._knowledge_graph.link_document(
            event.event_id,
            [e.id for e in entities],
        )

    def _extract_one_by_one(self, events, all_entities, progress_callback) -> None:
        from app_v2.services.xray import xray_log

        for event in events:
            try:
                # graph_rag.EntityExtractor expects a doc_id and returns strongly-typed
                # Entity and Relationship objects.
                entities, relationships = self.entity_extractor.extract_entities(
                    event.clean_text,
                    doc_id=event.event_id,
                )
                self._add_event_result(event, entities, relationships, all_entities)
            except Exception as e:
                logger.error(f"Failed to extract from event {event.event_id}: {e}")
                self.last_failed_events += 1
                xray_log("graph", "extract-error",
                         f"Skipped one — couldn't understand it",
                         detail=str(e)[:60], level="warn")
            finally:
                if progress_callback:
                    progress_callback(event.event_id)

    def _extract_batched(self, events, all_entities, progress_callback) -> None:
        """AGY path: CHRONOS_AGY_ENTITY_BATCH_SIZE events per model call (default 25).

        One agy call costs a 250-400 MB process for 12 s-2 min on the Pi, so ~650
        per-event calls a day were never an option; ~4 batched calls per build are.
        """
        from app_v2.services.xray import xray_log

        size = max(1, int(os.getenv("CHRONOS_AGY_ENTITY_BATCH_SIZE", "25")))
        for start in range(0, len(events), size):
            chunk = events[start : start + size]
            results = {}
            for attempt in (1, 2):
                try:
                    results = self.entity_extractor.extract_entities_batch(
                        [(e.event_id, e.clean_text) for e in chunk]
                    )
                    break
                except Exception as e:
                    logger.error(
                        f"AGY entity batch {start // size + 1} attempt {attempt} failed: {e}"
                    )
            for event in chunk:
                if event.event_id in results:
                    entities, relationships = results[event.event_id]
                    self._add_event_result(event, entities, relationships, all_entities)
                else:
                    self.last_failed_events += 1
                if progress_callback:
                    progress_callback(event.event_id)
            if len(results) < len(chunk):
                xray_log("graph", "extract-error",
                         f"Skipped {len(chunk) - len(results)} of {len(chunk)} moments in one batch",
                         level="warn")

    def detect_communities(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Detect communities in the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping community_id to list of node names
        """
        from app_v2.services.xray import xray_log
        _t0 = _time.perf_counter()
        # CommunityDetector operates on KnowledgeGraph, not the NetworkX graph.
        # (It will construct a NetworkX graph internally when needed.)
        communities = self.community_detector.detect_communities(self._knowledge_graph)
        _ms = (_time.perf_counter() - _t0) * 1000

        xray_log("graph", "communities",
                 f"Grouped everything into {len(communities)} clusters of related topics",
                 duration_ms=round(_ms, 1))
        logger.info(f"Detected {len(communities)} communities")

        # Return dicts for easy pickling / UI use.
        return [c.to_dict() for c in communities]

    def query_expansion(
        self,
        query_entities: List[str],
        graph: nx.Graph,
        max_hops: int = 2,
    ) -> List[str]:
        """Expand query using graph neighbors.

        Args:
            query_entities: Initial entity names from query
            graph: Knowledge graph
            max_hops: Maximum graph distance for expansion

        Returns:
            List of expanded entity names
        """
        expanded = set(query_entities)

        for entity in query_entities:
            if entity not in graph:
                continue

            # Get neighbors within max_hops
            neighbors = nx.single_source_shortest_path_length(
                graph,
                entity,
                cutoff=max_hops,
            )

            expanded.update(neighbors.keys())

        logger.info(f"Expanded {len(query_entities)} entities to {len(expanded)}")

        return list(expanded)
