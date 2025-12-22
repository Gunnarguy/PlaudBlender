"""Graph extraction integration for Chronos.

Bridges the ChronosEvent schema with the existing graph_rag.py module
to extract entities and relationships from cleaned narrative events.
"""

import logging
from typing import List, Dict, Any, Tuple
import networkx as nx

from src.processing.graph_rag import EntityExtractor, GraphBuilder, CommunityDetector
from src.models.chronos_schemas import ChronosEvent

logger = logging.getLogger(__name__)


class ChronosGraphExtractor:
    """Extract entities and build knowledge graph from Chronos events."""

    def __init__(self):
        """Initialize graph extraction components."""
        self.entity_extractor = EntityExtractor()
        self.graph_builder = GraphBuilder()
        self.community_detector = CommunityDetector()

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

        all_entities = []

        # Extract entities from each event
        for event in events:
            try:
                # Use the clean text for extraction
                entities = self.entity_extractor.extract_entities(event.clean_text)

                # Enrich with event metadata
                for entity in entities:
                    entity["source_event_id"] = event.event_id
                    entity["source_recording_id"] = event.recording_id
                    entity["timestamp"] = event.start_ts.isoformat()
                    entity["category"] = event.category.value

                all_entities.extend(entities)

            except Exception as e:
                logger.error(f"Failed to extract from event {event.event_id}: {e}")
                continue

        logger.info(f"Extracted {len(all_entities)} total entities")

        # Build graph
        graph = self.graph_builder.build_graph(all_entities)

        logger.info(
            f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )

        return all_entities, graph

    def detect_communities(self, graph: nx.Graph) -> Dict[str, List[str]]:
        """Detect communities in the graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dict mapping community_id to list of node names
        """
        communities = self.community_detector.detect_communities(graph)

        logger.info(f"Detected {len(communities)} communities")

        return communities

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
