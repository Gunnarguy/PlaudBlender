"""Chronos Data Service - Recording-centric data access layer.

This service aggregates Qdrant events back to their source recordings
and provides day-level summaries for the UI.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.database import SessionLocal
from src.database.models import (
    ChronosRecording as _ChronosRecordingModel,
    Recording as _RecordingModel,
)

# Detect local timezone once at import time so we convert UTC→local correctly
_LOCAL_TZ = datetime.now().astimezone().tzinfo

logger = logging.getLogger(__name__)

_PLAUD_WORKFLOW_ACTIVE_STATUSES = {"PENDING", "PROCESSING"}
_PLAUD_WORKFLOW_TERMINAL_STATUSES = {"SUCCESS", "FAILED", "CANCELLED"}


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
    category_confidence: Optional[float] = None

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

        # Cap individual event duration to 4 hours — Gemini sometimes
        # hallucinates end_ts months into the future.
        MAX_EVENT_DURATION = 4 * 3600  # 4 hours
        if (end_dt - start_dt).total_seconds() > MAX_EVENT_DURATION:
            end_dt = start_dt + timedelta(seconds=MAX_EVENT_DURATION)

        capped_duration = min(
            max((end_dt - start_dt).total_seconds(), 0),
            MAX_EVENT_DURATION,
        )

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
            duration_seconds=capped_duration,
            day_of_week=payload.get("day_of_week", ""),
            hour_of_day=payload.get("hour_of_day", 0),
            category_confidence=payload.get("category_confidence"),
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
    source: str = "plaud_cloud"  # plaud_cloud | usb_import | local
    has_plaud_ai: bool = False  # True if Plaud cloud AI summary exists

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
        return max(self.categories, key=lambda k: self.categories[k])


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
    ai_summary: Optional[str] = None  # One-line day summary from AI

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
    # Plaud cloud stats (fetched from API)
    plaud_cloud_stats: Optional[Dict[str, Any]] = None
    # Hour × category matrix: {hour: {category: count}}
    categories_by_hour: Dict[int, Dict[str, int]] = field(default_factory=dict)


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
        self._cache_lock = threading.Lock()

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
        from app_v2.services.xray import xray_log
        import time as _time
        now = datetime.now()

        # Check cache validity (read outside lock for fast path)
        if (
            not force_refresh
            and self._events_cache
            and self._last_cache_time
            and (now - self._last_cache_time).seconds < self._cache_ttl_seconds
        ):
            _age = (now - self._last_cache_time).seconds
            xray_log("data", "cache-hit",
                     f"Using cached data — {len(self._events_cache)} events, {_age}s old")
            return self._events_cache

        with self._cache_lock:
            # Double-check after acquiring lock
            if (
                not force_refresh
                and self._events_cache
                and self._last_cache_time
                and (datetime.now() - self._last_cache_time).seconds
                < self._cache_ttl_seconds
            ):
                return self._events_cache

            if not self._qdrant:
                return []

            try:
                events = []
                offset = None
                _scroll_t0 = _time.perf_counter()
                _scroll_pages = 0
                xray_log("data", "cache-miss",
                         f"Refreshing data from database…")

                while True:
                    response = self._qdrant.client.scroll(
                        collection_name=self._qdrant.collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )

                    points, offset = response
                    if not points:
                        break
                    _scroll_pages += 1

                    for point in points:
                        event = Event.from_qdrant(str(point.id), point.payload or {})
                        events.append(event)

                    if offset is None:
                        break

                # Sort by timestamp
                events.sort(key=lambda e: e.start_ts)

                # Update cache
                self._events_cache = events
                self._last_cache_time = datetime.now()
                _scroll_ms = (_time.perf_counter() - _scroll_t0) * 1000

                xray_log("data", "loaded",
                         f"Loaded {len(events)} events ({_scroll_pages} pages)",
                         duration_ms=round(_scroll_ms, 1), level="perf")
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

        # ── Bulk-load ground-truth timestamps from SQLite (one query) ──────
        _db = SessionLocal()
        try:
            _db_records: Dict[str, Any] = {
                str(r.recording_id): r
                for r in _db.query(_ChronosRecordingModel)
                .filter(
                    _ChronosRecordingModel.recording_id.in_(list(recordings.keys()))
                )
                .all()
            }
        except Exception as _e:
            logger.warning(f"SQLite lookup failed, falling back to Gemini times: {_e}")
            _db_records = {}
        finally:
            _db.close()

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

            # ── Ground-truth timestamps from Plaud DB ──────────────────────
            # SQLite stores created_at as naive UTC.  We attach tzinfo=UTC,
            # convert to the local timezone detected at startup, then strip
            # tzinfo so the rest of the code stays timezone-naive.
            # end_time = start + hardware duration (never trust Gemini end_ts).
            db_rec = _db_records.get(recording_id)
            if (
                db_rec is not None
                and db_rec.created_at is not None  # type: ignore[truthy-bool]
                and db_rec.duration_seconds is not None  # type: ignore[truthy-bool]
            ):
                utc_start = db_rec.created_at.replace(tzinfo=timezone.utc)  # type: ignore[union-attr]
                local_start = utc_start.astimezone(_LOCAL_TZ)
                start_time = local_start.replace(tzinfo=None)
                duration = float(db_rec.duration_seconds)  # type: ignore[arg-type]
                end_time = start_time + timedelta(seconds=duration)
            else:
                # Fallback: Gemini-derived times (less accurate)
                start_time = rec_events[0].start_ts
                duration = sum(e.duration_seconds for e in rec_events)
                end_time = start_time + timedelta(seconds=duration)

            summaries[recording_id] = RecordingSummary(
                recording_id=recording_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                event_count=len(rec_events),
                categories=dict(categories),
                keywords=top_keywords,
                avg_sentiment=total_sentiment / len(rec_events) if rec_events else 0,
                source=(
                    str(getattr(db_rec, "source", "plaud_cloud") or "plaud_cloud")
                    if db_rec
                    else "plaud_cloud"
                ),
                has_plaud_ai=(
                    bool(getattr(db_rec, "plaud_ai_summary", None)) if db_rec else False
                ),
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

            # Aggregate day stats — cap at 24h (wall-clock reality)
            MAX_DAY_SECONDS = 24 * 3600
            total_duration = min(
                sum(r.duration_seconds for r in day_recordings),
                MAX_DAY_SECONDS,
            )
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

            # Build one-line AI summary from recording-level summaries
            day_ai_summary = None
            try:
                from src.database.models import ChronosRecording

                db = SessionLocal()
                try:
                    snippets = []
                    for rec in day_recordings:
                        db_rec = (
                            db.query(ChronosRecording)
                            .filter(
                                ChronosRecording.recording_id == rec.recording_id,
                            )
                            .first()
                        )
                        if db_rec and getattr(db_rec, "plaud_ai_summary", None):
                            text = str(db_rec.plaud_ai_summary).strip()
                            first_sentence = text.split(".")[0].strip()
                            if first_sentence:
                                snippets.append(first_sentence[:120])
                    if snippets:
                        day_ai_summary = ". ".join(snippets[:3])
                        if not day_ai_summary.endswith("."):
                            day_ai_summary += "."
                finally:
                    db.close()
            except Exception:
                pass

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
                    ai_summary=day_ai_summary,
                )
            )

        # Sort by date descending
        result.sort(key=lambda d: d.date, reverse=True)
        return result

    def get_days_filled(self, last_n_days: Optional[int] = None) -> List[DaySummary]:
        """Get days with empty-day fill for a continuous timeline.

        Unlike get_days(), this fills in gaps so the timeline is
        continuous with zero-event placeholder days.

        Args:
            last_n_days: Show the last N calendar days (7, 14, 30, etc).
                         If None, span from earliest to latest data day.

        Returns:
            List of DaySummary sorted by date descending, with gaps filled.
        """
        data_days = self.get_days()
        if not data_days:
            return []

        # Build lookup by date string
        day_lookup = {d.date: d for d in data_days}

        # Determine range
        from datetime import date as date_cls

        if last_n_days:
            end_dt = date_cls.today()
            start_dt = end_dt - timedelta(days=last_n_days - 1)
        else:
            all_dates = sorted(day_lookup.keys())
            start_dt = datetime.strptime(all_dates[0], "%Y-%m-%d").date()
            end_dt = datetime.strptime(all_dates[-1], "%Y-%m-%d").date()

        # Walk every calendar day in range
        result: List[DaySummary] = []
        cursor = end_dt
        while cursor >= start_dt:
            key = cursor.strftime("%Y-%m-%d")
            if key in day_lookup:
                result.append(day_lookup[key])
            else:
                # Empty placeholder
                try:
                    display = cursor.strftime("%A, %b %d")
                except Exception:
                    display = key
                result.append(
                    DaySummary(
                        date=key,
                        date_display=display,
                        total_duration_seconds=0,
                        recording_count=0,
                        event_count=0,
                    )
                )
            cursor -= timedelta(days=1)

        return result  # already newest-first

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

        # Apply user category overrides from SQLite
        overrides = self.get_category_overrides(recording_id)
        for evt in rec_events:
            override = overrides.get(evt.id)
            if override:
                evt.category = override

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

        Builds graph from actual event data — categories, keywords, temporal
        patterns and co-occurrence relationships.
        """
        return self._build_graph_from_events()

    def _build_graph_from_events(self) -> GraphData:
        """Build a meaningful knowledge graph from event data.

        Creates three kinds of nodes:
        - Category hubs (work, meeting, personal, etc.)
        - Keyword/entity nodes (extracted from event keywords)
        - Date nodes (days with recordings)

        Edges represent:
        - Keyword → Category (keyword appears in events of that category)
        - Keyword → Keyword (co-occur in same recording)
        - Date → Category (activity on that day)
        """
        events = self._get_all_events()
        if not events:
            return GraphData()

        # ── Collect data ──────────────────────────────────────────────
        category_counts: Dict[str, int] = defaultdict(int)
        keyword_counts: Dict[str, int] = defaultdict(int)
        keyword_categories: Dict[str, set] = defaultdict(set)
        recording_keywords: Dict[str, set] = defaultdict(set)
        date_categories: Dict[str, set] = defaultdict(set)
        date_event_count: Dict[str, int] = defaultdict(int)
        keyword_sentiments: Dict[str, list] = defaultdict(list)

        # Normalize keywords
        def normalize(kw: str) -> str:
            return kw.strip().lower()

        # Skip low-value keywords
        stop_keywords = {
            "unknown",
            "none",
            "other",
            "general",
            "n/a",
            "na",
            "misc",
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "about",
        }

        for event in events:
            cat = event.category
            category_counts[cat] += 1
            date_key = event.start_ts.strftime("%Y-%m-%d")
            date_categories[date_key].add(cat)
            date_event_count[date_key] += 1

            for kw in event.keywords:
                nkw = normalize(kw)
                if len(nkw) < 2 or nkw in stop_keywords:
                    continue
                keyword_counts[nkw] += 1
                keyword_categories[nkw].add(cat)
                recording_keywords[event.recording_id].add(nkw)
                keyword_sentiments[nkw].append(event.sentiment)

        # ── Filter to meaningful keywords (appear 2+ times) ──────────
        significant_keywords = {
            kw for kw, count in keyword_counts.items() if count >= 2
        }

        # If too few pass the threshold, lower it
        if len(significant_keywords) < 10:
            significant_keywords = {
                kw for kw, count in keyword_counts.items() if count >= 1
            }

        # Cap at top 80 keywords by frequency to keep graph readable
        if len(significant_keywords) > 80:
            sorted_kws = sorted(
                significant_keywords, key=lambda k: keyword_counts[k], reverse=True
            )
            significant_keywords = set(sorted_kws[:80])

        # ── Build co-occurrence edges between keywords ────────────────
        keyword_cooccurrence: Dict[tuple, int] = defaultdict(int)
        for rec_id, kws in recording_keywords.items():
            relevant = kws & significant_keywords
            kw_list = sorted(relevant)
            for i in range(len(kw_list)):
                for j in range(i + 1, len(kw_list)):
                    keyword_cooccurrence[(kw_list[i], kw_list[j])] += 1

        # ── Build nodes ──────────────────────────────────────────────
        nodes = []

        # Category hub nodes (large)
        cat_colors = {
            "work": "#3b82f6",
            "meeting": "#8b5cf6",
            "personal": "#ec4899",
            "health": "#10b981",
            "deep_work": "#f59e0b",
            "errand": "#f97316",
            "social": "#14b8a6",
            "reflection": "#6366f1",
            "planning": "#0ea5e9",
        }
        for cat, count in category_counts.items():
            nodes.append(
                {
                    "data": {
                        "id": f"cat:{cat}",
                        "label": cat.replace("_", " ").title(),
                        "full_label": f"{cat} ({count} events)",
                        "type": "category",
                        "count": count,
                        "size": max(35, min(70, 25 + count)),
                        "color": cat_colors.get(cat, "#64748b"),
                    },
                    "classes": "category",
                }
            )

        # Keyword nodes
        max_kw_count = (
            max(keyword_counts[k] for k in significant_keywords)
            if significant_keywords
            else 1
        )
        for kw in significant_keywords:
            count = keyword_counts[kw]
            cats = keyword_categories[kw]
            avg_sent = (
                sum(keyword_sentiments[kw]) / len(keyword_sentiments[kw])
                if keyword_sentiments[kw]
                else 0
            )

            # Determine primary type based on simple heuristics
            entity_type = "topic"
            display = kw.title()
            if any(c.isupper() for c in kw) or len(kw.split()) <= 2:
                # Could be a person or project name
                entity_type = "topic"

            size = 15 + int((count / max_kw_count) * 30)

            nodes.append(
                {
                    "data": {
                        "id": f"kw:{kw}",
                        "label": display[:20] + "…" if len(display) > 20 else display,
                        "full_label": display,
                        "type": entity_type,
                        "count": count,
                        "mention_count": count,
                        "size": size,
                        "categories": ", ".join(sorted(cats)),
                        "sentiment": round(avg_sent, 2),
                    },
                    "classes": entity_type,
                }
            )

        # Date nodes (only if < 20 unique dates, otherwise too cluttered)
        unique_dates = sorted(date_categories.keys())
        if len(unique_dates) <= 20:
            for date_str in unique_dates:
                cats = date_categories[date_str]
                evt_count = date_event_count[date_str]
                from datetime import datetime as dt_cls

                try:
                    day_name = dt_cls.strptime(date_str, "%Y-%m-%d").strftime(
                        "%a %m/%d"
                    )
                except Exception:
                    day_name = date_str

                nodes.append(
                    {
                        "data": {
                            "id": f"date:{date_str}",
                            "label": day_name,
                            "full_label": f"{date_str} ({evt_count} events)",
                            "type": "date",
                            "count": evt_count,
                            "size": 18 + min(evt_count * 2, 20),
                        },
                        "classes": "date",
                    }
                )

        # ── Build edges ──────────────────────────────────────────────
        edges = []
        edge_id = 0

        # Keyword → Category edges (keyword appears in events of that category)
        for kw in significant_keywords:
            cats = keyword_categories[kw]
            for cat in cats:
                edge_id += 1
                edges.append(
                    {
                        "data": {
                            "id": f"e{edge_id}",
                            "source": f"kw:{kw}",
                            "target": f"cat:{cat}",
                            "weight": keyword_counts[kw],
                        }
                    }
                )

        # Keyword ↔ Keyword co-occurrence edges (same recording)
        for (kw1, kw2), cocount in keyword_cooccurrence.items():
            if cocount >= 2:  # Only strong co-occurrences
                edge_id += 1
                edges.append(
                    {
                        "data": {
                            "id": f"e{edge_id}",
                            "source": f"kw:{kw1}",
                            "target": f"kw:{kw2}",
                            "label": f"{cocount}x",
                            "weight": cocount,
                        }
                    }
                )

        # Date → Category edges
        if len(unique_dates) <= 20:
            for date_str, cats in date_categories.items():
                for cat in cats:
                    edge_id += 1
                    edges.append(
                        {
                            "data": {
                                "id": f"e{edge_id}",
                                "source": f"date:{date_str}",
                                "target": f"cat:{cat}",
                                "weight": 1,
                            }
                        }
                    )

        logger.info(f"Built knowledge graph: {len(nodes)} nodes, {len(edges)} edges")
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
        task_type: str = "RETRIEVAL_QUERY",
    ) -> List[SearchResult]:
        """Semantic search for events with optional filters.

        Args:
            query: Search query
            limit: Maximum results
            categories: Filter by event categories
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
            task_type: Gemini embedding task type — RETRIEVAL_QUERY for general
                search, QUESTION_ANSWERING for RAG Q&A.

        Returns:
            List of SearchResult with scores
        """
        if not self._qdrant or not self._embedder or not query.strip():
            return self._text_search(query, limit)

        from app_v2.services.xray import xray_log
        import time as _time
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
            _embed_t0 = _time.perf_counter()
            query_vector = self._embedder.embed_text(query, task_type=task_type)
            _embed_ms = (_time.perf_counter() - _embed_t0) * 1000
            xray_log("search", "embed",
                     f"Search text converted to numbers",
                     duration_ms=round(_embed_ms, 1),
                     detail=f"{len(query.split())} words")

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
                    event = Event.from_qdrant(str(hit.id), hit.payload or {})
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
        cats_by_hour: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
            cats_by_hour[event.hour_of_day][event.category] += 1
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

        # Use RECORDING durations from SQLite (not event durations which overlap
        # and double-count time).  Fall back to event-based sums only if the DB
        # query fails.
        real_total_duration_sec = 0.0
        real_avg_duration_min = 0.0
        real_longest_rec_min = 0.0
        real_total_recordings = num_recordings
        try:
            from src.database.engine import SessionLocal as _SL
            from src.database.models import ChronosRecording as _CR
            _db = _SL()
            try:
                db_recs = _db.query(_CR).all()
                real_total_recordings = len(db_recs)
                rec_durations = [float(r.duration_seconds or 0) for r in db_recs]  # type: ignore[arg-type]
                real_total_duration_sec = sum(rec_durations)
                real_avg_duration_min = (
                    (real_total_duration_sec / len(db_recs) / 60) if db_recs else 0.0
                )
                real_longest_rec_min = max(rec_durations, default=0) / 60
            finally:
                _db.close()
        except Exception:
            # Fallback to event-based (inaccurate but better than nothing)
            real_total_duration_sec = total_duration
            real_avg_duration_min = (
                (sum(recording_durations.values()) / num_recordings / 60)
                if num_recordings
                else 0.0
            )
            real_longest_rec_min = max(recording_durations.values(), default=0) / 60

        most_productive_day = (
            max(by_day_of_week, key=lambda k: by_day_of_week[k])
            if by_day_of_week
            else ""
        )
        most_productive_hour = max(by_hour, key=lambda k: by_hour[k]) if by_hour else 0

        # Pipeline completion rate
        pipeline_rate = 0.0
        try:
            db_stats = self.get_recording_db_stats()
            total_db = sum(db_stats.values())
            completed = db_stats.get("completed", 0)
            pipeline_rate = (completed / total_db * 100) if total_db else 0.0
        except Exception:
            pass

        # Plaud cloud stats (non-blocking)
        plaud_stats = None
        try:
            from src.plaud_client import PlaudClient

            plaud = PlaudClient()
            if plaud.oauth.is_authenticated:
                plaud_stats = plaud.get_recording_stats()
        except Exception as e:
            logger.debug(f"Could not fetch Plaud cloud stats: {e}")

        return Stats(
            total_recordings=real_total_recordings,
            total_events=len(events),
            total_days=len(days),
            total_duration_hours=real_total_duration_sec / 3600,
            categories=dict(categories),
            top_keywords=top_keywords,
            events_by_day_of_week=dict(by_day_of_week),
            events_by_hour=dict(by_hour),
            avg_sentiment=avg_sentiment,
            sentiment_distribution=sentiment_counts,
            avg_events_per_recording=avg_events_per_rec,
            avg_recording_duration_min=real_avg_duration_min,
            most_productive_day=most_productive_day,
            most_productive_hour=most_productive_hour,
            longest_recording_min=real_longest_rec_min,
            pipeline_completion_rate=pipeline_rate,
            plaud_cloud_stats=plaud_stats,
            categories_by_hour={h: dict(c) for h, c in cats_by_hour.items()},
        )

    def refresh_cache(self):
        """Force refresh of the events cache."""
        self._get_all_events(force_refresh=True)

    def get_transcript(self, recording_id: str) -> Optional[str]:
        """Get the cached transcript for a recording from SQLite."""
        try:
            from src.database.chronos_repository import get_chronos_recording

            db = SessionLocal()
            try:
                rec = get_chronos_recording(db, recording_id)
                if rec is not None and rec.transcript is not None:
                    return str(rec.transcript)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
        return None

    def get_ai_summary(self, recording_id: str) -> Optional[str]:
        """Get the Plaud AI summary for a recording from SQLite."""
        try:
            from src.database.chronos_repository import get_chronos_recording

            db = SessionLocal()
            try:
                rec = get_chronos_recording(db, recording_id)
                if rec is not None and rec.plaud_ai_summary is not None:
                    return str(rec.plaud_ai_summary)

                legacy = db.query(_RecordingModel).filter_by(id=recording_id).first()
                extra = self._coerce_recording_extra(legacy)
                plaud_summary = extra.get("plaud_summary")
                if isinstance(plaud_summary, str) and plaud_summary.strip():
                    return plaud_summary.strip()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching AI summary: {e}")
        return None

    def get_extracted_data(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get the Plaud AI_ETL extracted data for a recording."""
        try:
            db = SessionLocal()
            try:
                rec = (
                    db.query(_ChronosRecordingModel)
                    .filter_by(recording_id=recording_id)
                    .first()
                )
                if rec and rec.plaud_extracted_data:
                    import json as _json

                    data = rec.plaud_extracted_data
                    if isinstance(data, str):
                        data = _json.loads(data)
                    return data if isinstance(data, dict) else None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching extracted data: {e}")
        return None

    def get_plaud_workflow_transcript(self, recording_id: str) -> Optional[str]:
        """Get the Plaud cloud workflow transcript (if different from local cached)."""
        try:
            db = SessionLocal()
            try:
                legacy = (
                    db.query(_RecordingModel).filter_by(id=str(recording_id)).first()
                )
                if legacy and legacy.extra:
                    extra = legacy.extra if isinstance(legacy.extra, dict) else {}
                    return extra.get("plaud_workflow_transcript")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching workflow transcript: {e}")
        return None

    def get_workflow_status_for_recording(
        self, recording_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the current Plaud workflow status for a recording."""
        try:
            db = SessionLocal()
            try:
                rec = (
                    db.query(_ChronosRecordingModel)
                    .filter_by(recording_id=recording_id)
                    .first()
                )
                if rec and rec.plaud_workflow_id:
                    return {
                        "workflow_id": rec.plaud_workflow_id,
                        "status": rec.plaud_workflow_status,
                        "submitted_at": (
                            str(rec.plaud_workflow_submitted_at)
                            if rec.plaud_workflow_submitted_at
                            else None
                        ),
                        "completed_at": (
                            str(rec.plaud_workflow_completed_at)
                            if rec.plaud_workflow_completed_at
                            else None
                        ),
                        "template_id": rec.plaud_workflow_template_id,
                        "error": rec.plaud_workflow_error,
                        "has_summary": bool(rec.plaud_ai_summary),
                        "has_extracted_data": bool(rec.plaud_extracted_data),
                    }
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching workflow status: {e}")
        return None

    @staticmethod
    def _coerce_recording_extra(record: Optional[_RecordingModel]) -> Dict[str, Any]:
        """Return a safe mutable copy of Recording.extra."""
        if record is None:
            return {}
        extra = getattr(record, "extra", None)
        return dict(extra) if isinstance(extra, dict) else {}

    @staticmethod
    def _get_workflow_metadata(record: Optional[_RecordingModel]) -> Dict[str, Any]:
        """Extract Plaud workflow metadata from a legacy recording row."""
        extra = ChronosDataService._coerce_recording_extra(record)
        workflow = extra.get("plaud_workflow")
        return dict(workflow) if isinstance(workflow, dict) else {}

    def _persist_plaud_workflow_artifacts(
        self,
        db,
        recording_id: str,
        workflow_metadata: Dict[str, Any],
        summary: Optional[str] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
        transcript: Optional[str] = None,
    ) -> None:
        """Persist Plaud workflow metadata and outputs into SQLite.

        Writes to both the new dedicated columns on ChronosRecording AND
        the legacy Recording.extra JSON for backward compatibility.
        """
        chronos_record = (
            db.query(_ChronosRecordingModel)
            .filter_by(recording_id=recording_id)
            .first()
        )
        legacy_record = db.query(_RecordingModel).filter_by(id=recording_id).first()

        # Write to new dedicated columns on ChronosRecording
        if chronos_record:
            if summary:
                chronos_record.plaud_ai_summary = summary.strip()
            wf_id = workflow_metadata.get("workflow_id")
            if wf_id:
                chronos_record.plaud_workflow_id = str(wf_id)
            wf_status = workflow_metadata.get("status")
            if wf_status:
                chronos_record.plaud_workflow_status = str(wf_status).upper()
            submitted_at = workflow_metadata.get("submitted_at")
            if submitted_at and isinstance(submitted_at, str):
                try:
                    chronos_record.plaud_workflow_submitted_at = datetime.fromisoformat(
                        submitted_at
                    )
                except Exception:
                    pass
            completed_at = workflow_metadata.get("completed_at")
            if completed_at and isinstance(completed_at, str):
                try:
                    chronos_record.plaud_workflow_completed_at = datetime.fromisoformat(
                        completed_at
                    )
                except Exception:
                    pass
            template_id = workflow_metadata.get("template_id")
            if template_id:
                chronos_record.plaud_workflow_template_id = str(template_id)
            error = workflow_metadata.get("error")
            if error:
                chronos_record.plaud_workflow_error = str(error)[:500]
            elif wf_status and str(wf_status).upper() == "SUCCESS":
                chronos_record.plaud_workflow_error = None
            if extracted_data is not None:
                chronos_record.plaud_extracted_data = extracted_data

        # Also write to legacy Recording.extra for backward compat
        if legacy_record:
            extra = self._coerce_recording_extra(legacy_record)
            extra["plaud_workflow"] = workflow_metadata
            if summary and summary.strip():
                extra["plaud_summary"] = summary.strip()
            if extracted_data is not None:
                extra["plaud_extracted_data"] = extracted_data
            if transcript and transcript.strip():
                extra["plaud_workflow_transcript"] = transcript.strip()
            legacy_record.extra = extra

        db.commit()

    def get_plaud_workflow_stats(self, days_back: int = 30) -> Dict[str, Any]:
        """Summarize Plaud workflow coverage and current workflow state."""
        stats: Dict[str, Any] = {
            "recent_recordings": 0,
            "with_ai_summary": 0,
            "missing_ai_summary": 0,
            "ready_for_enrichment": 0,
            "workflow_pending": 0,
            "workflow_failed": 0,
            "workflow_success": 0,
            "last_submitted_at": None,
            "active_workflows": [],  # NEW: list of in-flight workflows
        }

        try:
            days_back = max(int(days_back), 1)
            cutoff = datetime.utcnow() - timedelta(days=days_back)

            db = SessionLocal()
            try:
                recent = (
                    db.query(_ChronosRecordingModel)
                    .filter(_ChronosRecordingModel.created_at >= cutoff)
                    .order_by(_ChronosRecordingModel.created_at.desc())
                    .all()
                )
                stats["recent_recordings"] = len(recent)

                for rec in recent:
                    has_summary = bool(rec.plaud_ai_summary)

                    # Fallback to legacy extra for summary check
                    if not has_summary:
                        legacy = (
                            db.query(_RecordingModel)
                            .filter_by(id=str(rec.recording_id))
                            .first()
                        )
                        extra = self._coerce_recording_extra(legacy)
                        has_summary = bool(
                            isinstance(extra.get("plaud_summary"), str)
                            and extra.get("plaud_summary", "").strip()
                        )

                    if has_summary:
                        stats["with_ai_summary"] += 1
                    else:
                        stats["missing_ai_summary"] += 1

                    # Use new dedicated columns first, fallback to legacy
                    workflow_status = ""
                    if rec.plaud_workflow_status:
                        workflow_status = str(rec.plaud_workflow_status).upper()
                    else:
                        legacy = (
                            db.query(_RecordingModel)
                            .filter_by(id=str(rec.recording_id))
                            .first()
                        )
                        workflow = self._get_workflow_metadata(legacy)
                        workflow_status = str(workflow.get("status") or "").upper()

                    if workflow_status in _PLAUD_WORKFLOW_ACTIVE_STATUSES:
                        stats["workflow_pending"] += 1
                        stats["active_workflows"].append(
                            {
                                "recording_id": str(rec.recording_id),
                                "workflow_id": rec.plaud_workflow_id,
                                "status": workflow_status,
                                "template_id": rec.plaud_workflow_template_id,
                                "title": rec.title or str(rec.recording_id)[:16],
                            }
                        )
                    elif workflow_status == "FAILED":
                        stats["workflow_failed"] += 1
                    elif workflow_status == "SUCCESS":
                        stats["workflow_success"] += 1

                    if (
                        not has_summary
                    ) and workflow_status not in _PLAUD_WORKFLOW_ACTIVE_STATUSES:
                        stats["ready_for_enrichment"] += 1

                    submitted_at = (
                        str(rec.plaud_workflow_submitted_at)
                        if rec.plaud_workflow_submitted_at
                        else None
                    )
                    if submitted_at:
                        last_submitted = stats["last_submitted_at"]
                        if not last_submitted or submitted_at > last_submitted:
                            stats["last_submitted_at"] = submitted_at
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching Plaud workflow stats: {e}")

        return stats

    def submit_plaud_workflows(
        self,
        days_back: int = 7,
        limit: int = 3,
        template_id: Optional[str] = None,
        model: str = "gemini",
    ) -> Dict[str, Any]:
        """Submit Plaud cloud workflows for recent recordings missing AI summaries."""
        try:
            from src.plaud_workflow import PlaudWorkflowClient

            days_back = max(int(days_back), 1)
            limit = max(int(limit), 1)
            template_id = (template_id or "").strip() or None
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            workflow_client = PlaudWorkflowClient()

            result: Dict[str, Any] = {
                "submitted": [],
                "skipped": [],
                "errors": [],
                "template_id": template_id,
                "model": model,
            }

            db = SessionLocal()
            try:
                recent = (
                    db.query(_ChronosRecordingModel)
                    .filter(_ChronosRecordingModel.created_at >= cutoff)
                    .order_by(_ChronosRecordingModel.created_at.desc())
                    .all()
                )

                candidates: List[_ChronosRecordingModel] = []
                for rec in recent:
                    # Check new columns first, then fall back to legacy
                    workflow_status = str(rec.plaud_workflow_status or "").upper()
                    if not workflow_status:
                        legacy = (
                            db.query(_RecordingModel)
                            .filter_by(id=str(rec.recording_id))
                            .first()
                        )
                        workflow = self._get_workflow_metadata(legacy)
                        workflow_status = str(workflow.get("status") or "").upper()

                    if rec.plaud_ai_summary is not None:
                        result["skipped"].append(
                            {
                                "recording_id": str(rec.recording_id),
                                "reason": "already_has_summary",
                            }
                        )
                        continue
                    if workflow_status in _PLAUD_WORKFLOW_ACTIVE_STATUSES:
                        result["skipped"].append(
                            {
                                "recording_id": str(rec.recording_id),
                                "reason": "workflow_in_progress",
                            }
                        )
                        continue
                    candidates.append(rec)
                    if len(candidates) >= limit:
                        break

                for rec in candidates:
                    recording_id = str(rec.recording_id)
                    try:
                        workflow_id = workflow_client.submit_workflow(
                            file_id=recording_id,
                            template_id=template_id,
                            include_summary=True,
                            workflow_name=f"chronos_sync_{recording_id[:8]}",
                            model=model,
                        )
                        workflow_metadata = {
                            "workflow_id": workflow_id,
                            "status": "PENDING",
                            "submitted_at": datetime.utcnow().isoformat(),
                            "source": "sync_view",
                            "template_id": template_id,
                            "completed_tasks": 0,
                            "total_tasks": 3 if template_id else 2,
                        }
                        self._persist_plaud_workflow_artifacts(
                            db,
                            recording_id=recording_id,
                            workflow_metadata=workflow_metadata,
                        )
                        result["submitted"].append(
                            {
                                "recording_id": recording_id,
                                "workflow_id": workflow_id,
                                "title": rec.title or recording_id[:16],
                            }
                        )
                    except Exception as exc:
                        logger.error(
                            f"Failed to submit Plaud workflow for {recording_id}: {exc}"
                        )
                        self._persist_plaud_workflow_artifacts(
                            db,
                            recording_id=recording_id,
                            workflow_metadata={
                                "workflow_id": None,
                                "status": "FAILED",
                                "submitted_at": datetime.utcnow().isoformat(),
                                "source": "sync_view",
                                "template_id": template_id,
                                "error": str(exc),
                            },
                        )
                        result["errors"].append(
                            {"recording_id": recording_id, "error": str(exc)}
                        )
            finally:
                db.close()

            return result
        except Exception as e:
            logger.error(f"Error submitting Plaud workflows: {e}")
            return {
                "submitted": [],
                "skipped": [],
                "errors": [{"recording_id": None, "error": str(e)}],
                "template_id": template_id,
            }

    def refresh_plaud_workflow_statuses(
        self,
        days_back: int = 30,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Refresh pending Plaud workflow statuses and persist completed outputs."""
        try:
            from src.plaud_workflow import PlaudWorkflowClient

            days_back = max(int(days_back), 1)
            limit = max(int(limit), 1)
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            workflow_client = PlaudWorkflowClient()

            result: Dict[str, Any] = {"pending": [], "completed": [], "failed": []}

            db = SessionLocal()
            try:
                recent = (
                    db.query(_ChronosRecordingModel)
                    .filter(_ChronosRecordingModel.created_at >= cutoff)
                    .order_by(_ChronosRecordingModel.created_at.desc())
                    .all()
                )
                record_ids = [str(rec.recording_id) for rec in recent]
                legacy_records = {}
                if record_ids:
                    legacy_records = {
                        str(rec.id): rec
                        for rec in db.query(_RecordingModel)
                        .filter(_RecordingModel.id.in_(record_ids))
                        .all()
                    }

                targets: List[Tuple[str, Dict[str, Any]]] = []
                for rec in recent:
                    workflow = self._get_workflow_metadata(
                        legacy_records.get(str(rec.recording_id))
                    )
                    if not workflow.get("workflow_id"):
                        continue
                    status = str(workflow.get("status") or "").upper()
                    if status in _PLAUD_WORKFLOW_ACTIVE_STATUSES:
                        targets.append((str(rec.recording_id), workflow))
                    if len(targets) >= limit:
                        break

                for recording_id, workflow in targets:
                    workflow_id = str(workflow.get("workflow_id"))
                    try:
                        status_info = workflow_client.get_workflow_status(workflow_id)
                        workflow_metadata = dict(workflow)
                        workflow_metadata.update(
                            {
                                "status": str(
                                    status_info.get("status")
                                    or workflow.get("status")
                                    or "PENDING"
                                ).upper(),
                                "completed_tasks": status_info.get(
                                    "completed_tasks",
                                    workflow.get("completed_tasks", 0),
                                ),
                                "total_tasks": status_info.get(
                                    "total_tasks",
                                    workflow.get("total_tasks", 0),
                                ),
                                "current_task": status_info.get("current_task"),
                                "last_checked_at": datetime.utcnow().isoformat(),
                            }
                        )

                        status = workflow_metadata["status"]
                        if status == "SUCCESS":
                            workflow_result = workflow_client.get_workflow_results(
                                workflow_id
                            )
                            workflow_metadata["completed_at"] = (
                                datetime.utcnow().isoformat()
                            )
                            self._persist_plaud_workflow_artifacts(
                                db,
                                recording_id=recording_id,
                                workflow_metadata=workflow_metadata,
                                summary=workflow_result.summary,
                                extracted_data=workflow_result.extracted_data,
                                transcript=workflow_result.transcript,
                            )
                            result["completed"].append(
                                {
                                    "recording_id": recording_id,
                                    "workflow_id": workflow_id,
                                }
                            )
                        elif status in _PLAUD_WORKFLOW_TERMINAL_STATUSES:
                            workflow_metadata["error"] = status_info.get("error")
                            self._persist_plaud_workflow_artifacts(
                                db,
                                recording_id=recording_id,
                                workflow_metadata=workflow_metadata,
                            )
                            result["failed"].append(
                                {
                                    "recording_id": recording_id,
                                    "workflow_id": workflow_id,
                                    "error": status_info.get("error"),
                                }
                            )
                        else:
                            self._persist_plaud_workflow_artifacts(
                                db,
                                recording_id=recording_id,
                                workflow_metadata=workflow_metadata,
                            )
                            result["pending"].append(
                                {
                                    "recording_id": recording_id,
                                    "workflow_id": workflow_id,
                                    "current_task": status_info.get("current_task"),
                                }
                            )
                    except Exception as exc:
                        logger.error(
                            f"Failed to refresh Plaud workflow for {recording_id}: {exc}"
                        )
                        failed_metadata = dict(workflow)
                        failed_metadata.update(
                            {
                                "status": "FAILED",
                                "error": str(exc),
                                "last_checked_at": datetime.utcnow().isoformat(),
                            }
                        )
                        self._persist_plaud_workflow_artifacts(
                            db,
                            recording_id=recording_id,
                            workflow_metadata=failed_metadata,
                        )
                        result["failed"].append(
                            {
                                "recording_id": recording_id,
                                "workflow_id": workflow_id,
                                "error": str(exc),
                            }
                        )
            finally:
                db.close()

            return result
        except Exception as e:
            logger.error(f"Error refreshing Plaud workflow statuses: {e}")
            return {
                "pending": [],
                "completed": [],
                "failed": [{"recording_id": None, "error": str(e)}],
            }

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
                return int(getattr(result, "rowcount", 0) or 0)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error resetting recordings: {e}")
            return 0

    def save_category_override(self, event_qdrant_id: str, new_category: str) -> bool:
        """Save a user category override for an event.

        Finds the ChronosEvent by its qdrant_point_id and sets user_category_override.
        """
        try:
            from src.database.engine import SessionLocal
            from src.database.models import ChronosEvent

            db = SessionLocal()
            try:
                evt = (
                    db.query(ChronosEvent)
                    .filter_by(qdrant_point_id=event_qdrant_id)
                    .first()
                )
                if evt:
                    setattr(evt, "user_category_override", new_category)
                    db.commit()
                    # Invalidate cache so next load reflects the change
                    self._get_all_events(force_refresh=True)
                    return True
                logger.warning(f"No event found with qdrant_point_id={event_qdrant_id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error saving category override: {e}")
        return False

    def get_upload_candidates(self) -> List[Dict[str, Any]]:
        """Return list of local audio files in data/raw/usb_import/ that can be uploaded."""
        try:
            from src.plaud_client import PlaudClient

            client = PlaudClient()
            return client.get_upload_candidates()
        except Exception as e:
            logger.error(f"Error getting upload candidates: {e}")
            return []

    def upload_and_process_files(
        self,
        file_paths: List[str],
        template_id: Optional[str] = None,
        model: str = "gemini",
    ) -> Dict[str, Any]:
        """Upload local audio files to Plaud cloud and submit full workflow pipeline.

        Returns dict with 'uploaded', 'errors' lists.
        """
        from src.plaud_client import PlaudClient
        from src.plaud_workflow import PlaudWorkflowManager, get_template_by_id

        result: Dict[str, Any] = {"uploaded": [], "errors": []}

        try:
            client = PlaudClient()
            manager = PlaudWorkflowManager()
            template = get_template_by_id(template_id) if template_id else None

            for path in file_paths:
                try:
                    upload_result = manager.upload_and_process(
                        plaud_client=client,
                        file_path=path,
                        template=template,
                        model=model,
                    )
                    result["uploaded"].append(
                        {
                            "path": path,
                            "file_id": upload_result.get("file_id"),
                            "workflow_id": upload_result.get("workflow_id"),
                        }
                    )
                except Exception as exc:
                    logger.error(f"Upload failed for {path}: {exc}")
                    result["errors"].append({"path": path, "error": str(exc)})
        except Exception as e:
            result["errors"].append({"path": "init", "error": str(e)})

        return result

    def submit_single_recording_workflow(
        self,
        recording_id: str,
        template_id: Optional[str] = None,
        model: str = "gemini",
    ) -> Dict[str, Any]:
        """Submit a Plaud cloud workflow for a single specific recording.

        Returns dict with workflow_id, status, or error.
        """
        try:
            from src.plaud_workflow import PlaudWorkflowClient
            from src.database.engine import SessionLocal

            db = SessionLocal()
            try:
                rec = (
                    db.query(_ChronosRecordingModel)
                    .filter_by(recording_id=recording_id)
                    .first()
                )
                if not rec:
                    return {"error": f"Recording {recording_id} not found"}

                workflow_client = PlaudWorkflowClient()
                effective_template = (template_id or "").strip() or None
                workflow_id = workflow_client.submit_workflow(
                    file_id=recording_id,
                    template_id=effective_template,
                    include_summary=True,
                    workflow_name=f"single_{recording_id[:8]}",
                    model=model,
                )
                workflow_metadata = {
                    "workflow_id": workflow_id,
                    "status": "PENDING",
                    "submitted_at": datetime.utcnow().isoformat(),
                    "source": "recording_detail",
                    "template_id": effective_template,
                    "completed_tasks": 0,
                    "total_tasks": 3 if effective_template else 2,
                }
                self._persist_plaud_workflow_artifacts(
                    db,
                    recording_id=recording_id,
                    workflow_metadata=workflow_metadata,
                )
                return {
                    "workflow_id": workflow_id,
                    "status": "PENDING",
                    "recording_id": recording_id,
                }
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error submitting workflow for {recording_id}: {e}")
            return {"error": str(e), "recording_id": recording_id}

    def get_category_overrides(self, recording_id: str) -> Dict[str, str]:
        """Get all user category overrides for events in a recording.

        Returns:
            Dict mapping qdrant_point_id → user_category_override
        """
        try:
            from src.database.engine import SessionLocal
            from src.database.models import ChronosEvent

            db = SessionLocal()
            try:
                events = (
                    db.query(
                        ChronosEvent.qdrant_point_id,
                        ChronosEvent.user_category_override,
                    )
                    .filter(
                        ChronosEvent.user_category_override.isnot(None),
                    )
                    .all()
                )
                return {str(e[0]): str(e[1]) for e in events if e[0]}
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error fetching category overrides: {e}")
        return {}


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
