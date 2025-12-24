"""Graph extraction integration for Chronos.

Bridges the ChronosEvent schema with the existing graph_rag.py module
to extract entities and relationships from cleaned narrative events.
"""

import logging
from typing import List, Dict, Any, Tuple

import networkx as nx

from src.ai.graph_rag import EntityExtractor, KnowledgeGraph, CommunityDetector
from src.models.chronos_schemas import ChronosEvent

logger = logging.getLogger(__name__)


class ChronosGraphExtractor:
    """Extract entities and build knowledge graph from Chronos events."""

    def __init__(self):
        """Initialize graph extraction components."""
        self.entity_extractor = EntityExtractor()
        self.community_detector = CommunityDetector()

        # We keep the last KnowledgeGraph around because CommunityDetector
        # operates on KnowledgeGraph (it builds a NetworkX graph internally).
        self._knowledge_graph: KnowledgeGraph = KnowledgeGraph()

        logger.info("Initialized ChronosGraphExtractor")

    def extract_from_events(
        self,
        events: List[ChronosEvent],
    ) -> Tuple[List[Dict[str, Any]], nx.Graph]:
        """Extract entities and relationships from cleaned events.

        Args:
            events: List of ChronosEvent objects

        Returns:
            Tuple of (entities_list, networkx_graph)
        """
        logger.info(f"Extracting entities from {len(events)} events")

        # Reset for each extraction run.
        self._knowledge_graph = KnowledgeGraph()
        all_entities: List[Dict[str, Any]] = []

        # Extract entities from each event
        for event in events:
            try:
                # graph_rag.EntityExtractor expects a doc_id and returns strongly-typed
                # Entity and Relationship objects.
                entities, relationships = self.entity_extractor.extract_entities(
                    event.clean_text,
                    doc_id=event.event_id,
                )

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

            except Exception as e:
                logger.error(f"Failed to extract from event {event.event_id}: {e}")
                continue

        logger.info(f"Extracted {len(all_entities)} total entities")

        # Build a NetworkX view (useful for quick stats + downstream visualization)
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

        return all_entities, graph

    def detect_communities(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Detect communities in the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping community_id to list of node names
        """
        # CommunityDetector operates on KnowledgeGraph, not the NetworkX graph.
        # (It will construct a NetworkX graph internally when needed.)
        communities = self.community_detector.detect_communities(self._knowledge_graph)

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
