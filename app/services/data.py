"""Data service bridging Dash UI to Chronos backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TimelineGroup:
    """A group of events for timeline display."""

    label: str
    date_key: str
    count: int
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GraphData:
    """Graph data for Cytoscape visualization."""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)


class ChronosDataService:
    """Bridge between Dash UI and Chronos backend services."""

    def __init__(self):
        """Initialize connections to backend services."""
        self._qdrant = None
        self._embedder = None
        self._graph = None
        self._db_session = None
        self._init_services()

    def _init_services(self):
        """Lazy-load backend services."""
        try:
            from src.chronos.qdrant_client import ChronosQdrantClient
            from src.chronos.embedding_service import ChronosEmbeddingService
            from src.database.engine import SessionLocal
            from src.config import get_settings

            settings = get_settings()

            if settings.gemini_api_key:
                try:
                    self._embedder = ChronosEmbeddingService()
                except Exception as e:
                    logger.warning(f"Could not init embedder: {e}")

            try:
                self._qdrant = ChronosQdrantClient()
            except Exception as e:
                logger.warning(f"Could not init Qdrant: {e}")

            try:
                self._db_session = SessionLocal()
            except Exception as e:
                logger.warning(f"Could not init DB: {e}")

            # Load graph if exists
            self._load_graph()

        except Exception as e:
            logger.error(f"Service init error: {e}")

    def _load_graph(self):
        """Load knowledge graph from cache."""
        try:
            from pathlib import Path
            import pickle

            graph_path = Path("data/cache/graphs/knowledge_graph.pkl")
            if graph_path.exists():
                with open(graph_path, "rb") as f:
                    data = pickle.load(f)

                # Handle dict wrapper format
                if isinstance(data, dict) and "graph" in data:
                    self._graph = data["graph"]
                    logger.info(
                        f"Loaded graph with {len(self._graph.nodes())} nodes from dict wrapper"
                    )
                elif hasattr(data, "nodes"):
                    self._graph = data
                    logger.info(f"Loaded graph with {len(self._graph.nodes())} nodes")
                else:
                    logger.warning(f"Unknown graph format: {type(data)}")
        except Exception as e:
            logger.debug(f"No cached graph: {e}")

    def get_all_events(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all events from Qdrant."""
        if not self._qdrant:
            return []

        try:
            from qdrant_client.models import ScrollRequest

            results = []
            offset = None

            while len(results) < limit:
                response = self._qdrant.client.scroll(
                    collection_name=self._qdrant.collection_name,
                    limit=min(100, limit - len(results)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, offset = response
                if not points:
                    break

                for point in points:
                    event = dict(point.payload) if point.payload else {}
                    event["id"] = str(point.id)
                    results.append(event)

                if offset is None:
                    break

            return results
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def get_timeline_groups(self) -> List[TimelineGroup]:
        """Get events grouped by date for timeline display."""
        events = self.get_all_events()

        # Group by date
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for event in events:
            # Try to get timestamp
            ts = event.get("event_timestamp") or event.get("timestamp")
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    elif isinstance(ts, (int, float)):
                        dt = datetime.fromtimestamp(ts)
                    else:
                        dt = datetime.now()
                    date_key = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_key = "unknown"
            else:
                date_key = "unknown"

            groups[date_key].append(event)

        # Sort and create TimelineGroup objects
        result = []
        today = datetime.now().date()

        for date_key in sorted(groups.keys(), reverse=True):
            events_list = groups[date_key]

            # Create nice label
            if date_key == "unknown":
                label = "📅 Unknown Date"
            else:
                try:
                    dt = datetime.strptime(date_key, "%Y-%m-%d").date()
                    if dt == today:
                        label = "📅 Today"
                    elif dt == today - timedelta(days=1):
                        label = "📅 Yesterday"
                    else:
                        label = f"📅 {dt.strftime('%b %d, %Y')}"
                except Exception:
                    label = f"📅 {date_key}"

            result.append(
                TimelineGroup(
                    label=label,
                    date_key=date_key,
                    count=len(events_list),
                    events=events_list,
                )
            )

        return result

    def get_graph_data(self) -> GraphData:
        """Get knowledge graph data for Cytoscape visualization - ALL nodes."""
        if not self._graph:
            return self._build_graph_from_events()

        nodes = []
        edges = []

        try:
            import networkx as nx

            # Calculate node importance (degree centrality)
            centrality = nx.degree_centrality(self._graph)

            total_nodes = len(self._graph.nodes())
            logger.info(f"Building graph with ALL {total_nodes} nodes")

            for node_id, data in self._graph.nodes(data=True):
                entity_type = data.get("type", "unknown")
                name = data.get("name", str(node_id))
                # Truncate long labels for display
                label = name[:15] + "..." if len(name) > 15 else name

                nodes.append(
                    {
                        "data": {
                            "id": str(node_id),
                            "label": label,
                            "full_label": name,
                            "type": entity_type,
                            "description": data.get("description", ""),
                            "size": 15 + (centrality.get(node_id, 0) * 80),
                        },
                        "classes": entity_type.lower(),
                    }
                )

            # ALL edges
            for source, target, data in self._graph.edges(data=True):
                edges.append(
                    {
                        "data": {
                            "id": f"{source}-{target}",
                            "source": str(source),
                            "target": str(target),
                            "label": data.get("relationship", ""),
                            "weight": data.get("weight", 1),
                        }
                    }
                )

            return GraphData(nodes=nodes, edges=edges)

        except Exception as e:
            logger.error(f"Error building graph data: {e}")
            return GraphData()

    def _build_graph_from_events(self) -> GraphData:
        """Build a simple graph from events if no cached graph exists."""
        events = self.get_all_events(limit=200)

        if not events:
            return GraphData()

        # Extract entities mentioned in events
        entity_counts: Dict[str, int] = defaultdict(int)
        event_entities: Dict[str, List[str]] = {}

        for event in events:
            event_id = event.get("id", "")
            text = event.get("narrative", "") or event.get("text", "")
            category = event.get("category", "general")

            # Simple entity extraction from narrative
            entities = []

            # Add category as entity
            if category:
                entities.append(f"cat:{category}")
                entity_counts[f"cat:{category}"] += 1

            # Add recording as entity
            recording_id = event.get("recording_id", "")
            if recording_id:
                short_id = recording_id[:8]
                entities.append(f"rec:{short_id}")
                entity_counts[f"rec:{short_id}"] += 1

            event_entities[event_id] = entities

        # Build nodes
        nodes = []
        for entity, count in entity_counts.items():
            entity_type = "category" if entity.startswith("cat:") else "recording"
            label = entity.split(":", 1)[1] if ":" in entity else entity

            nodes.append(
                {
                    "data": {
                        "id": entity,
                        "label": label,
                        "type": entity_type,
                        "count": count,
                        "size": 20 + (count * 3),
                    },
                    "classes": entity_type,
                }
            )

        # Build edges (entities that co-occur in same event)
        edges = []
        edge_set = set()

        for event_id, entities in event_entities.items():
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    edge_key = tuple(sorted([e1, e2]))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append(
                            {
                                "data": {
                                    "id": f"{e1}-{e2}",
                                    "source": e1,
                                    "target": e2,
                                }
                            }
                        )

        return GraphData(nodes=nodes, edges=edges)

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Semantic search for events."""
        if not self._qdrant or not self._embedder or not query.strip():
            return []

        try:
            # Embed query
            query_vector = self._embedder.embed_text(query, task_type="RETRIEVAL_QUERY")

            # Search using query_points (not deprecated search)
            results = self._qdrant.client.query_points(
                collection_name=self._qdrant.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

            events = []
            for hit in results.points:
                event = dict(hit.payload) if hit.payload else {}
                event["id"] = str(hit.id)
                event["score"] = hit.score
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get full details for a specific event."""
        if not self._qdrant:
            return None

        try:
            results = self._qdrant.client.retrieve(
                collection_name=self._qdrant.collection_name,
                ids=[event_id],
                with_payload=True,
            )

            if results:
                event = dict(results[0].payload) if results[0].payload else {}
                event["id"] = str(results[0].id)
                return event

            return None
        except Exception as e:
            logger.error(f"Error getting event {event_id}: {e}")
            return None

    def get_related_events(self, event_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get events related to the given event."""
        if not self._qdrant:
            return []

        try:
            # Get the event's vector
            results = self._qdrant.client.retrieve(
                collection_name=self._qdrant.collection_name,
                ids=[event_id],
                with_vectors=True,
            )

            if not results or not results[0].vector:
                return []

            # Search for similar using query method
            from qdrant_client.models import SearchRequest

            similar = self._qdrant.client.query_points(
                collection_name=self._qdrant.collection_name,
                query=results[0].vector,
                limit=limit + 1,  # +1 to exclude self
                with_payload=True,
            )

            events = []
            for hit in similar.points:
                if str(hit.id) != event_id:
                    event = dict(hit.payload) if hit.payload else {}
                    event["id"] = str(hit.id)
                    event["score"] = hit.score
                    events.append(event)

            return events[:limit]

        except Exception as e:
            logger.error(f"Error getting related events: {e}")
            return []

    def find_events_by_entity(
        self, entity_name: str, entity_type: str = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find events that mention a specific entity.

        This searches through events to find those that reference the entity
        in their narrative, actors, or extracted entities.
        """
        if not self._qdrant or not self._embedder:
            # Fallback to text search
            return self._text_search_events(entity_name, limit)

        try:
            # Use semantic search with the entity name as query
            query_vector = self._embedder.embed_single(entity_name)

            results = self._qdrant.client.query_points(
                collection_name=self._qdrant.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

            events = []
            for hit in results.points:
                event = dict(hit.payload) if hit.payload else {}
                event["id"] = str(hit.id)
                event["score"] = hit.score
                events.append(event)

            return events

        except Exception as e:
            logger.error(f"Error finding events by entity: {e}")
            return self._text_search_events(entity_name, limit)

    def _text_search_events(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback text search through events."""
        events = self.get_all_events(limit=500)
        query_lower = query.lower()

        matches = []
        for event in events:
            narrative = (event.get("narrative") or "").lower()
            text = (event.get("text") or "").lower()
            actors = (
                " ".join(event.get("actors", [])).lower() if event.get("actors") else ""
            )

            # Check if query appears in content
            if query_lower in narrative or query_lower in text or query_lower in actors:
                event["score"] = 1.0  # Exact match
                matches.append(event)
            elif any(word in narrative for word in query_lower.split()):
                event["score"] = 0.5  # Partial match
                matches.append(event)

        # Sort by score
        matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        return matches[:limit]

    def sync_from_plaud(self, days_back: int = 7) -> Dict[str, int]:
        """Sync recordings from Plaud and process them."""
        try:
            from src.chronos.ingest_service import ChronosIngestService
            from src.chronos.transcript_processor import ChronosTranscriptProcessor

            ingest = ChronosIngestService()
            result = ingest.fetch_and_store(days_back=days_back, fetch_all_pages=True)

            return {
                "fetched": result.get("success", 0),
                "failed": result.get("failed", 0),
            }
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "total_events": 0,
            "total_recordings": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "qdrant_connected": self._qdrant is not None,
            "embedder_ready": self._embedder is not None,
        }

        try:
            if self._qdrant:
                info = self._qdrant.client.get_collection(self._qdrant.collection_name)
                stats["total_events"] = info.points_count or 0

            if self._graph and hasattr(self._graph, "nodes"):
                stats["graph_nodes"] = len(self._graph.nodes())
                stats["graph_edges"] = len(self._graph.edges())

        except Exception as e:
            logger.error(f"Stats error: {e}")

        return stats


# Singleton instance
_data_service: Optional[ChronosDataService] = None


def get_data_service() -> ChronosDataService:
    """Get or create the singleton data service."""
    global _data_service
    if _data_service is None:
        _data_service = ChronosDataService()
    return _data_service
