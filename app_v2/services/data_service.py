"""Chronos Data Service - Recording-centric data access layer.

This service aggregates Qdrant events back to their source recordings
and provides day-level summaries for the UI.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Event:
    """A single event extracted from a recording."""

    id: str
    recording_id: str
    start_ts: datetime
    end_ts: datetime
    clean_text: str
    category: str
    sentiment: float
    keywords: List[str]
    speaker: str
    duration_seconds: float
    day_of_week: str
    hour_of_day: int

    @classmethod
    def from_qdrant(cls, point_id: str, payload: Dict[str, Any]) -> "Event":
        """Create Event from Qdrant point."""
        start_ts = payload.get("start_ts") or payload.get("timestamp")
        end_ts = payload.get("end_ts") or start_ts

        # Parse timestamps
        if isinstance(start_ts, str):
            start_dt = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        else:
            start_dt = datetime.now()

        if isinstance(end_ts, str):
            end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        else:
            end_dt = start_dt

        return cls(
            id=str(point_id),
            recording_id=payload.get("recording_id", "unknown"),
            start_ts=start_dt,
            end_ts=end_dt,
            clean_text=payload.get("clean_text", ""),
            category=payload.get("category", "unknown"),
            sentiment=payload.get("sentiment", 0.0),
            keywords=payload.get("keywords", []),
            speaker=payload.get("speaker", "unknown"),
            duration_seconds=payload.get("duration_seconds", 0),
            day_of_week=payload.get("day_of_week", ""),
            hour_of_day=payload.get("hour_of_day", 0),
        )


@dataclass
class RecordingSummary:
    """Summary of a recording with aggregated stats."""

    recording_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    event_count: int
    categories: Dict[str, int] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    avg_sentiment: float = 0.0

    @property
    def duration_formatted(self) -> str:
        """Format duration as HH:MM:SS or MM:SS."""
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        seconds = int(self.duration_seconds % 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    @property
    def time_range_formatted(self) -> str:
        """Format time range as 'HH:MM AM - HH:MM PM'."""
        start = self.start_time.strftime("%I:%M %p")
        end = self.end_time.strftime("%I:%M %p")
        return f"{start} - {end}"

    @property
    def top_category(self) -> str:
        """Get the most common category."""
        if not self.categories:
            return "unknown"
        return max(self.categories, key=self.categories.get)


@dataclass
class DaySummary:
    """Summary of all recordings for a day."""

    date: str  # YYYY-MM-DD
    date_display: str  # "Wednesday, Oct 29"
    total_duration_seconds: float
    recording_count: int
    event_count: int
    recordings: List[RecordingSummary] = field(default_factory=list)
    categories: Dict[str, int] = field(default_factory=dict)
    top_keywords: List[str] = field(default_factory=list)

    @property
    def duration_formatted(self) -> str:
        """Format duration as X.X hours."""
        hours = self.total_duration_seconds / 3600
        return f"{hours:.1f} hours"


@dataclass
class RecordingDetail:
    """Full recording with all events."""

    summary: RecordingSummary
    events: List[Event] = field(default_factory=list)

    @property
    def category_percentages(self) -> Dict[str, float]:
        """Get category distribution as percentages."""
        total = sum(self.summary.categories.values())
        if total == 0:
            return {}
        return {k: (v / total) * 100 for k, v in self.summary.categories.items()}


@dataclass
class TopicOccurrence:
    """A single occurrence of a topic in a recording."""

    event_id: str
    recording_id: str
    timestamp: datetime
    text_snippet: str
    category: str


@dataclass
class TopicTimeline:
    """Timeline of a topic across all recordings."""

    topic: str
    total_occurrences: int
    recording_count: int
    day_count: int
    occurrences: List[TopicOccurrence] = field(default_factory=list)


@dataclass
class SearchResult:
    """A search result with context."""

    event: Event
    score: float
    context_before: Optional[str] = None
    context_after: Optional[str] = None


@dataclass
class GraphData:
    """Graph data for Cytoscape visualization."""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Stats:
    """Overall statistics."""

    total_recordings: int
    total_events: int
    total_days: int
    total_duration_hours: float
    categories: Dict[str, int] = field(default_factory=dict)
    top_keywords: List[Tuple[str, int]] = field(default_factory=list)
    events_by_day_of_week: Dict[str, int] = field(default_factory=dict)
    events_by_hour: Dict[int, int] = field(default_factory=dict)
    # Enhanced analytics
    avg_sentiment: float = 0.0
    sentiment_distribution: Dict[str, int] = field(
        default_factory=dict
    )  # positive/neutral/negative counts
    avg_events_per_recording: float = 0.0
    avg_recording_duration_min: float = 0.0
    most_productive_day: str = ""
    most_productive_hour: int = 0
    longest_recording_min: float = 0.0
    pipeline_completion_rate: float = 0.0  # % of recordings fully processed


# ═══════════════════════════════════════════════════════════════════════════════
# DATA SERVICE
# ═══════════════════════════════════════════════════════════════════════════════


class ChronosDataService:
    """Main data service for Chronos UI.

    Provides recording-centric views by aggregating Qdrant events.
    """

    def __init__(self):
        """Initialize the data service."""
        self._qdrant = None
        self._embedder = None
        self._events_cache: List[Event] = []
        self._last_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Refresh cache every minute

        self._init_services()

    def _init_services(self):
        """Initialize backend services."""
        try:
            from src.chronos.qdrant_client import ChronosQdrantClient
            from src.chronos.embedding_service import ChronosEmbeddingService
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

        except Exception as e:
            logger.error(f"Service init error: {e}")

    def _get_all_events(self, force_refresh: bool = False) -> List[Event]:
        """Get all events from Qdrant with caching."""
        now = datetime.now()

        # Check cache validity
        if (
            not force_refresh
            and self._events_cache
            and self._last_cache_time
            and (now - self._last_cache_time).seconds < self._cache_ttl_seconds
        ):
            return self._events_cache

        if not self._qdrant:
            return []

        try:
            events = []
            offset = None

            while True:
                response = self._qdrant.client.scroll(
                    collection_name=self._qdrant.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points, offset = response
                if not points:
                    break

                for point in points:
                    event = Event.from_qdrant(point.id, point.payload or {})
                    events.append(event)

                if offset is None:
                    break

            # Sort by timestamp
            events.sort(key=lambda e: e.start_ts)

            # Update cache
            self._events_cache = events
            self._last_cache_time = now

            logger.info(f"Loaded {len(events)} events from Qdrant")
            return events

        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return self._events_cache or []

    def _aggregate_by_recording(
        self, events: List[Event]
    ) -> Dict[str, RecordingSummary]:
        """Aggregate events into recording summaries."""
        recordings: Dict[str, List[Event]] = defaultdict(list)

        for event in events:
            recordings[event.recording_id].append(event)

        summaries = {}
        for recording_id, rec_events in recordings.items():
            # Sort events by time
            rec_events.sort(key=lambda e: e.start_ts)

            # Calculate aggregates
            categories: Dict[str, int] = defaultdict(int)
            all_keywords: List[str] = []
            total_sentiment = 0.0

            for event in rec_events:
                categories[event.category] += 1
                all_keywords.extend(event.keywords)
                total_sentiment += event.sentiment

            # Get top keywords (deduplicated, by frequency)
            keyword_counts: Dict[str, int] = defaultdict(int)
            for kw in all_keywords:
                keyword_counts[kw.lower()] += 1
            top_keywords = [
                kw for kw, _ in sorted(keyword_counts.items(), key=lambda x: -x[1])[:10]
            ]

            # Calculate duration from first to last event
            start_time = rec_events[0].start_ts
            end_time = rec_events[-1].end_ts
            duration = (end_time - start_time).total_seconds()

            # Also sum individual event durations as backup
            event_duration = sum(e.duration_seconds for e in rec_events)
            duration = max(duration, event_duration)

            summaries[recording_id] = RecordingSummary(
                recording_id=recording_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                event_count=len(rec_events),
                categories=dict(categories),
                keywords=top_keywords,
                avg_sentiment=total_sentiment / len(rec_events) if rec_events else 0,
            )

        return summaries

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API - Day Views
    # ═══════════════════════════════════════════════════════════════════════════

    def get_days(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[DaySummary]:
        """Get all days with recording summaries.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            List of DaySummary sorted by date descending (newest first)
        """
        events = self._get_all_events()
        if not events:
            return []

        # First aggregate by recording
        recording_summaries = self._aggregate_by_recording(events)

        # Then group recordings by day
        days: Dict[str, List[RecordingSummary]] = defaultdict(list)

        for rec_summary in recording_summaries.values():
            day_key = rec_summary.start_time.strftime("%Y-%m-%d")
            days[day_key].append(rec_summary)

        # Build day summaries
        result = []
        for day_key, day_recordings in days.items():
            # Apply date filters
            if start_date and day_key < start_date:
                continue
            if end_date and day_key > end_date:
                continue

            # Sort recordings by start time
            day_recordings.sort(key=lambda r: r.start_time)

            # Aggregate day stats
            total_duration = sum(r.duration_seconds for r in day_recordings)
            total_events = sum(r.event_count for r in day_recordings)

            categories: Dict[str, int] = defaultdict(int)
            all_keywords: List[str] = []

            for rec in day_recordings:
                for cat, count in rec.categories.items():
                    categories[cat] += count
                all_keywords.extend(rec.keywords)

            # Deduplicate keywords
            keyword_counts: Dict[str, int] = defaultdict(int)
            for kw in all_keywords:
                keyword_counts[kw.lower()] += 1
            top_keywords = [
                kw for kw, _ in sorted(keyword_counts.items(), key=lambda x: -x[1])[:8]
            ]

            # Format date display
            try:
                dt = datetime.strptime(day_key, "%Y-%m-%d")
                date_display = dt.strftime("%A, %b %d")  # "Wednesday, Oct 29"
            except:
                date_display = day_key

            result.append(
                DaySummary(
                    date=day_key,
                    date_display=date_display,
                    total_duration_seconds=total_duration,
                    recording_count=len(day_recordings),
                    event_count=total_events,
                    recordings=day_recordings,
                    categories=dict(categories),
                    top_keywords=top_keywords,
                )
            )

        # Sort by date descending
        result.sort(key=lambda d: d.date, reverse=True)
        return result

    def get_day_detail(self, date: str) -> Optional[DaySummary]:
        """Get detailed view of a specific day.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            DaySummary with full recording list, or None if not found
        """
        days = self.get_days(start_date=date, end_date=date)
        return days[0] if days else None

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API - Recording Views
    # ═══════════════════════════════════════════════════════════════════════════

    def get_recording_detail(self, recording_id: str) -> Optional[RecordingDetail]:
        """Get full recording with all events.

        Args:
            recording_id: The recording ID

        Returns:
            RecordingDetail with all events, or None if not found
        """
        events = self._get_all_events()

        # Filter events for this recording
        rec_events = [e for e in events if e.recording_id == recording_id]
        if not rec_events:
            return None

        # Sort by time
        rec_events.sort(key=lambda e: e.start_ts)

        # Build summary
        summaries = self._aggregate_by_recording(rec_events)
        summary = summaries.get(recording_id)

        if not summary:
            return None

        return RecordingDetail(
            summary=summary,
            events=rec_events,
        )

    def get_events_for_recording(self, recording_id: str) -> List[Event]:
        """Get all events for a recording.

        Args:
            recording_id: The recording ID

        Returns:
            List of events sorted by timestamp
        """
        events = self._get_all_events()
        rec_events = [e for e in events if e.recording_id == recording_id]
        rec_events.sort(key=lambda e: e.start_ts)
        return rec_events

    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID.

        Args:
            event_id: The event ID

        Returns:
            Event or None if not found
        """
        events = self._get_all_events()
        for event in events:
            if event.id == event_id:
                return event
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API - Topics
    # ═══════════════════════════════════════════════════════════════════════════

    def get_all_topics(self) -> List[Tuple[str, int]]:
        """Get all unique topics/keywords with their counts.

        Returns:
            List of (keyword, count) tuples sorted by count descending
        """
        events = self._get_all_events()

        keyword_counts: Dict[str, int] = defaultdict(int)
        for event in events:
            for kw in event.keywords:
                keyword_counts[kw.lower()] += 1

        return sorted(keyword_counts.items(), key=lambda x: -x[1])

    def get_topic_timeline(self, topic: str) -> TopicTimeline:
        """Get timeline of a topic across all recordings.

        Args:
            topic: The topic/keyword to search for

        Returns:
            TopicTimeline with all occurrences
        """
        events = self._get_all_events()
        topic_lower = topic.lower()

        occurrences = []
        recording_ids = set()
        days = set()

        for event in events:
            # Check if topic appears in keywords or text
            in_keywords = any(topic_lower in kw.lower() for kw in event.keywords)
            in_text = topic_lower in event.clean_text.lower()

            if in_keywords or in_text:
                # Extract snippet around topic
                text = event.clean_text
                idx = text.lower().find(topic_lower)
                if idx >= 0:
                    start = max(0, idx - 50)
                    end = min(len(text), idx + len(topic) + 50)
                    snippet = "..." + text[start:end] + "..."
                else:
                    snippet = text[:150] + "..." if len(text) > 150 else text

                occurrences.append(
                    TopicOccurrence(
                        event_id=event.id,
                        recording_id=event.recording_id,
                        timestamp=event.start_ts,
                        text_snippet=snippet,
                        category=event.category,
                    )
                )

                recording_ids.add(event.recording_id)
                days.add(event.start_ts.strftime("%Y-%m-%d"))

        # Sort by timestamp
        occurrences.sort(key=lambda o: o.timestamp, reverse=True)

        return TopicTimeline(
            topic=topic,
            total_occurrences=len(occurrences),
            recording_count=len(recording_ids),
            day_count=len(days),
            occurrences=occurrences,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API - Knowledge Graph
    # ═══════════════════════════════════════════════════════════════════════════

    def get_graph_data(self) -> GraphData:
        """Get knowledge graph data for Cytoscape visualization.

        Tries to load from cached NetworkX pickle first,
        falls back to building from events.
        """
        try:
            from pathlib import Path
            import pickle

            graph_path = Path("data/cache/graphs/knowledge_graph.pkl")
            if graph_path.exists():
                return self._load_graph_from_pickle(graph_path)
        except Exception as e:
            logger.debug(f"No cached graph: {e}")

        # Fallback: build simple graph from events
        return self._build_graph_from_events()

    def _load_graph_from_pickle(self, graph_path) -> GraphData:
        """Load NetworkX graph from pickle and convert to Cytoscape format."""
        import pickle
        import networkx as nx

        with open(graph_path, "rb") as f:
            data = pickle.load(f)

        # Handle dict wrapper or raw NetworkX graph
        if isinstance(data, dict) and "graph" in data:
            graph = data["graph"]
        elif hasattr(data, "nodes"):
            graph = data
        else:
            logger.warning(f"Unknown graph format: {type(data)}")
            return self._build_graph_from_events()

        # Calculate centrality for node sizing
        centrality = nx.degree_centrality(graph)

        nodes = []
        for node_id, node_data in graph.nodes(data=True):
            entity_type = node_data.get("type", "unknown")
            name = node_data.get("name", str(node_id))
            label = name[:18] + "…" if len(name) > 18 else name

            nodes.append(
                {
                    "data": {
                        "id": str(node_id),
                        "label": label,
                        "full_label": name,
                        "type": entity_type,
                        "description": node_data.get("description", ""),
                        "mention_count": node_data.get("mention_count", 0),
                        "size": 20 + (centrality.get(node_id, 0) * 80),
                    },
                    "classes": entity_type.lower(),
                }
            )

        edges = []
        for source, target, edge_data in graph.edges(data=True):
            edges.append(
                {
                    "data": {
                        "id": f"{source}-{target}",
                        "source": str(source),
                        "target": str(target),
                        "label": edge_data.get(
                            "relationship", edge_data.get("type", "")
                        ),
                        "weight": edge_data.get("weight", 1),
                    }
                }
            )

        logger.info(f"Loaded graph: {len(nodes)} nodes, {len(edges)} edges")
        return GraphData(nodes=nodes, edges=edges)

    def _build_graph_from_events(self) -> GraphData:
        """Build a simple entity graph from Qdrant events."""
        events = self._get_all_events()
        if not events:
            return GraphData()

        # Build category nodes and recording nodes
        category_counts: Dict[str, int] = defaultdict(int)
        recording_categories: Dict[str, List[str]] = defaultdict(list)

        for event in events:
            category_counts[event.category] += 1
            if event.category not in recording_categories[event.recording_id]:
                recording_categories[event.recording_id].append(event.category)

        nodes = []
        # Category nodes
        for cat, count in category_counts.items():
            nodes.append(
                {
                    "data": {
                        "id": f"cat:{cat}",
                        "label": cat.capitalize(),
                        "full_label": cat,
                        "type": "category",
                        "count": count,
                        "size": 25 + (count * 2),
                    },
                    "classes": "category",
                }
            )

        # Recording nodes (just short IDs)
        for rec_id, cats in recording_categories.items():
            short_id = rec_id[:8]
            nodes.append(
                {
                    "data": {
                        "id": f"rec:{short_id}",
                        "label": short_id,
                        "full_label": rec_id,
                        "type": "recording",
                        "count": len(cats),
                        "size": 20,
                    },
                    "classes": "recording",
                }
            )

        # Edges: recording → category
        edges = []
        edge_set = set()
        for rec_id, cats in recording_categories.items():
            short_id = rec_id[:8]
            for cat in cats:
                edge_key = (f"rec:{short_id}", f"cat:{cat}")
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append(
                        {
                            "data": {
                                "id": f"{edge_key[0]}-{edge_key[1]}",
                                "source": edge_key[0],
                                "target": edge_key[1],
                            }
                        }
                    )

        logger.info(f"Built graph from events: {len(nodes)} nodes, {len(edges)} edges")
        return GraphData(nodes=nodes, edges=edges)

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API - Search
    # ═══════════════════════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        limit: int = 20,
        categories: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """Semantic search for events with optional filters.

        Args:
            query: Search query
            limit: Maximum results
            categories: Filter by event categories
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)

        Returns:
            List of SearchResult with scores
        """
        if not self._qdrant or not self._embedder or not query.strip():
            return self._text_search(query, limit)

        try:
            from src.models.chronos_schemas import TemporalFilter
            from datetime import datetime as dt_cls

            # Build temporal filter if date params provided
            temporal_filter = None
            if start_date or end_date:
                temporal_filter = TemporalFilter(
                    start_date=(
                        dt_cls.strptime(start_date, "%Y-%m-%d") if start_date else None
                    ),
                    end_date=(
                        dt_cls.strptime(end_date, "%Y-%m-%d") if end_date else None
                    ),
                    hours_of_day=None,
                )

            # Embed query
            query_vector = self._embedder.embed_text(query, task_type="RETRIEVAL_QUERY")

            # Use hybrid search if filters are present
            if temporal_filter or categories:
                results = self._qdrant.search_hybrid(
                    query_vector=query_vector,
                    temporal_filter=temporal_filter,
                    categories=categories,
                    limit=limit,
                )
                search_results = []
                for hit in results:
                    event = Event.from_qdrant(hit["event_id"], hit.get("payload", {}))
                    search_results.append(
                        SearchResult(event=event, score=hit.get("score", 0.0) or 0.0)
                    )
                return search_results
            else:
                # Simple semantic search
                results = self._qdrant.client.query_points(
                    collection_name=self._qdrant.collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                search_results = []
                for hit in results.points:
                    event = Event.from_qdrant(hit.id, hit.payload or {})
                    search_results.append(SearchResult(event=event, score=hit.score))
                return search_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            return self._text_search(query, limit)

    def _text_search(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Fallback text search."""
        events = self._get_all_events()
        query_lower = query.lower()

        results = []
        for event in events:
            text = event.clean_text.lower()
            keywords = " ".join(event.keywords).lower()

            if query_lower in text or query_lower in keywords:
                score = 1.0 if query_lower in text else 0.5
                results.append(SearchResult(event=event, score=score))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API - Stats
    # ═══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Stats:
        """Get overall statistics with enhanced analytics."""
        events = self._get_all_events()

        if not events:
            return Stats(
                total_recordings=0,
                total_events=0,
                total_days=0,
                total_duration_hours=0,
            )

        recording_ids = set()
        days = set()
        categories: Dict[str, int] = defaultdict(int)
        keywords: Dict[str, int] = defaultdict(int)
        by_day_of_week: Dict[str, int] = defaultdict(int)
        by_hour: Dict[int, int] = defaultdict(int)
        total_duration = 0.0
        total_sentiment = 0.0
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        recording_durations: Dict[str, float] = defaultdict(float)
        recording_event_counts: Dict[str, int] = defaultdict(int)

        for event in events:
            recording_ids.add(event.recording_id)
            days.add(event.start_ts.strftime("%Y-%m-%d"))
            categories[event.category] += 1
            for kw in event.keywords:
                keywords[kw.lower()] += 1
            by_day_of_week[event.day_of_week] += 1
            by_hour[event.hour_of_day] += 1
            total_duration += event.duration_seconds
            total_sentiment += event.sentiment
            recording_durations[event.recording_id] += event.duration_seconds
            recording_event_counts[event.recording_id] += 1

            # Sentiment bucketing
            if event.sentiment > 0.15:
                sentiment_counts["positive"] += 1
            elif event.sentiment < -0.15:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1

        top_keywords = sorted(keywords.items(), key=lambda x: -x[1])[:20]

        # Enhanced metrics
        num_recordings = len(recording_ids)
        avg_sentiment = total_sentiment / len(events) if events else 0.0
        avg_events_per_rec = len(events) / num_recordings if num_recordings else 0.0
        avg_duration_min = (
            (sum(recording_durations.values()) / num_recordings / 60)
            if num_recordings
            else 0.0
        )
        longest_rec_min = max(recording_durations.values(), default=0) / 60

        most_productive_day = (
            max(by_day_of_week, key=by_day_of_week.get) if by_day_of_week else ""
        )
        most_productive_hour = max(by_hour, key=by_hour.get) if by_hour else 0

        # Pipeline completion rate
        pipeline_rate = 0.0
        try:
            db_stats = self.get_recording_db_stats()
            total_db = sum(db_stats.values())
            completed = db_stats.get("completed", 0)
            pipeline_rate = (completed / total_db * 100) if total_db else 0.0
        except Exception:
            pass

        return Stats(
            total_recordings=num_recordings,
            total_events=len(events),
            total_days=len(days),
            total_duration_hours=total_duration / 3600,
            categories=dict(categories),
            top_keywords=top_keywords,
            events_by_day_of_week=dict(by_day_of_week),
            events_by_hour=dict(by_hour),
            avg_sentiment=avg_sentiment,
            sentiment_distribution=sentiment_counts,
            avg_events_per_recording=avg_events_per_rec,
            avg_recording_duration_min=avg_duration_min,
            most_productive_day=most_productive_day,
            most_productive_hour=most_productive_hour,
            longest_recording_min=longest_rec_min,
            pipeline_completion_rate=pipeline_rate,
        )

    def refresh_cache(self):
        """Force refresh of the events cache."""
        self._get_all_events(force_refresh=True)

    def get_transcript(self, recording_id: str) -> Optional[str]:
        """Get the cached transcript for a recording from SQLite."""
        try:
            from src.database.engine import SessionLocal
            from src.database.chronos_repository import get_chronos_recording

            db = SessionLocal()
            try:
                rec = get_chronos_recording(db, recording_id)
                if rec and rec.transcript:
                    return rec.transcript
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
        return None

    def get_recording_db_stats(self) -> Dict[str, int]:
        """Get recording status counts from SQLite."""
        try:
            from src.database.engine import SessionLocal
            import sqlalchemy as sa

            db = SessionLocal()
            try:
                result = db.execute(
                    sa.text(
                        "SELECT processing_status, COUNT(*) FROM chronos_recordings GROUP BY processing_status"
                    )
                )
                return {row[0]: row[1] for row in result}
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching DB stats: {e}")
            return {}

    def reset_stuck_recordings(self) -> int:
        """Reset recordings stuck in 'processing' back to 'pending'."""
        try:
            from src.database.engine import SessionLocal
            import sqlalchemy as sa

            db = SessionLocal()
            try:
                result = db.execute(
                    sa.text(
                        "UPDATE chronos_recordings SET processing_status = 'pending', "
                        "error_message = NULL WHERE processing_status = 'processing'"
                    )
                )
                db.commit()
                return result.rowcount
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error resetting recordings: {e}")
            return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service: Optional[ChronosDataService] = None


def get_data_service() -> ChronosDataService:
    """Get or create the singleton data service."""
    global _service
    if _service is None:
        _service = ChronosDataService()
    return _service
