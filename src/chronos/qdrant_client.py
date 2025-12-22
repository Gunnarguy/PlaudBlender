"""Native Qdrant client for Chronos temporal-vector indexing.

This module provides a clean, Qdrant-native interface WITHOUT the
Pinecone compatibility shim. It exposes Qdrant's full payload filtering
and temporal search capabilities required by the Chronos architecture.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    PayloadSchemaType,
)

from src.config import get_settings
from src.models.chronos_schemas import ChronosEvent, TemporalFilter

logger = logging.getLogger(__name__)


class ChronosQdrantClient:
    """Native Qdrant client optimized for Chronos temporal-vector search.

    Key Features:
    - Payload indexes for day_of_week, hour_of_day, timestamp
    - Hybrid search: semantic + temporal filters
    - Scroll API for bulk analytics
    - No Pinecone compatibility cruft
    """

    def __init__(self, collection_name: Optional[str] = None):
        """Initialize Qdrant client.

        Args:
            collection_name: Override default collection name from config
        """
        self.settings = get_settings()
        self.collection_name = collection_name or self.settings.qdrant_collection_name

        # Initialize client
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )

        logger.info(
            f"Initialized ChronosQdrantClient for collection: {self.collection_name}"
        )

    def create_collection(
        self, vector_size: int = 768, force_recreate: bool = False
    ) -> None:
        """Create Chronos collection with temporal payload indexes.

        Args:
            vector_size: Embedding dimension (768 for Gemini text-embedding-004)
            force_recreate: Delete existing collection if present
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and force_recreate:
            logger.warning(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

        # Create payload indexes for temporal filtering
        logger.info("Creating payload indexes...")

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="day_of_week",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="hour_of_day",
            field_schema=PayloadSchemaType.INTEGER,
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="timestamp",
            field_schema=PayloadSchemaType.DATETIME,
        )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        logger.info("Collection and indexes created successfully")

    def upsert_event(
        self,
        event: ChronosEvent,
        embedding: List[float],
    ) -> str:
        """Upsert a single event with its embedding.

        Args:
            event: ChronosEvent with metadata
            embedding: Vector embedding (768-dim for Gemini)

        Returns:
            str: Point ID (event_id)
        """
        point = PointStruct(
            id=event.event_id,
            vector=embedding,
            payload={
                "recording_id": event.recording_id,
                "start_ts": event.start_ts.isoformat(),
                "end_ts": event.end_ts.isoformat(),
                "timestamp": event.start_ts.isoformat(),  # Primary temporal index
                "day_of_week": event.day_of_week.value,
                "hour_of_day": event.hour_of_day,
                "clean_text": event.clean_text,
                "category": event.category.value,
                "sentiment": event.sentiment,
                "keywords": event.keywords,
                "speaker": event.speaker.value,
                "duration_seconds": event.duration_seconds,
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

        return event.event_id

    def upsert_events_batch(
        self,
        events: List[ChronosEvent],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """Batch upsert events with embeddings.

        Args:
            events: List of ChronosEvent objects
            embeddings: Corresponding embeddings
            batch_size: Batch size for upserts

        Returns:
            int: Number of events upserted
        """
        if len(events) != len(embeddings):
            raise ValueError("Events and embeddings must have same length")

        total = 0

        for i in range(0, len(events), batch_size):
            batch_events = events[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            points = [
                PointStruct(
                    id=event.event_id,
                    vector=embedding,
                    payload={
                        "recording_id": event.recording_id,
                        "start_ts": event.start_ts.isoformat(),
                        "end_ts": event.end_ts.isoformat(),
                        "timestamp": event.start_ts.isoformat(),
                        "day_of_week": event.day_of_week.value,
                        "hour_of_day": event.hour_of_day,
                        "clean_text": event.clean_text,
                        "category": event.category.value,
                        "sentiment": event.sentiment,
                        "keywords": event.keywords,
                        "speaker": event.speaker.value,
                        "duration_seconds": event.duration_seconds,
                    },
                )
                for event, embedding in zip(batch_events, batch_embeddings)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            total += len(points)
            logger.info(f"Upserted batch: {total}/{len(events)} events")

        return total

    def search_hybrid(
        self,
        query_vector: Optional[List[float]] = None,
        temporal_filter: Optional[TemporalFilter] = None,
        categories: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: semantic similarity + temporal filters.

        Args:
            query_vector: Embedding for semantic search (optional)
            temporal_filter: Temporal constraints (optional)
            categories: Filter by event categories (optional)
            limit: Max results

        Returns:
            List of dicts with event data and scores
        """
        # Build filter
        must_conditions = []

        if temporal_filter:
            if temporal_filter.start_date:
                must_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(
                            gte=temporal_filter.start_date.isoformat(),
                            lte=(
                                temporal_filter.end_date.isoformat()
                                if temporal_filter.end_date
                                else None
                            ),
                        ),
                    )
                )

            if temporal_filter.days_of_week:
                must_conditions.append(
                    FieldCondition(
                        key="day_of_week",
                        match=MatchAny(
                            any=[d.value for d in temporal_filter.days_of_week]
                        ),
                    )
                )

            if temporal_filter.hours_of_day:
                must_conditions.append(
                    FieldCondition(
                        key="hour_of_day",
                        match=MatchAny(any=temporal_filter.hours_of_day),
                    )
                )

        if categories:
            must_conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchAny(any=categories),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Execute search
        if query_vector:
            # Semantic + filter
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
        else:
            # Filter-only (scroll)
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
            )[
                0
            ]  # scroll returns (points, next_offset)

        # Format results
        formatted = []
        for hit in results:
            formatted.append(
                {
                    "event_id": hit.id,
                    "score": getattr(hit, "score", None),
                    "payload": hit.payload,
                }
            )

        return formatted

    def get_events_by_day(
        self,
        day_of_week: str,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Retrieve all events for a specific day of week.

        Args:
            day_of_week: "Monday", "Tuesday", etc.
            limit: Max results

        Returns:
            List of event payloads
        """
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="day_of_week",
                        match=MatchValue(value=day_of_week),
                    )
                ]
            ),
            limit=limit,
        )

        return [{"event_id": p.id, "payload": p.payload} for p in points]

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict with collection info
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }
