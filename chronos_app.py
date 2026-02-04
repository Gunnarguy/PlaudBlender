"""Chronos Streamlit UI — Simplified Edition.

A clean, usable interface for transforming Plaud voice recordings into
a searchable knowledge timeline.

Design principles:
- Progressive disclosure: simple first, power options in "Advanced"
- One place for each thing: no duplicate paths
- Show state clearly: connection status, counts, what's pending
- Command visibility: preview what will run before running it
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Increase Python's integer string conversion limit for Gemini JSON responses
if sys.version_info >= (3, 11):
    sys.set_int_max_str_digits(0)

import streamlit as st

from src.config import get_settings
from src.chronos.qdrant_client import ChronosQdrantClient
from src.chronos.embedding_service import ChronosEmbeddingService
from src.models.chronos_schemas import DayOfWeek, EventCategory, TemporalFilter
from src.database import SessionLocal, init_db
from src.database.models import (
    ChronosRecording as ChronosRecordingDB,
    ChronosEvent as ChronosEventDB,
)
from src.database.chronos_repository import set_chronos_recording_transcript
from src.plaud_client import PlaudClient

# Page configuration
st.set_page_config(
    page_title="Chronos",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Clean up Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        opacity: 0.8;
        margin-bottom: 1rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
        border: 1px solid rgba(100, 150, 255, 0.3);
        background: rgba(100, 150, 255, 0.1);
    }
    .status-ok { border-color: rgba(100, 200, 100, 0.5); background: rgba(100, 200, 100, 0.15); }
    .status-warn { border-color: rgba(255, 180, 100, 0.5); background: rgba(255, 180, 100, 0.15); }
    .status-error { border-color: rgba(255, 100, 100, 0.5); background: rgba(255, 100, 100, 0.15); }

    .event-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(49, 50, 68, 0.60);
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-radius: 10px;
    }
    .muted { opacity: 0.7; font-size: 0.85rem; }

    /* Status bar at bottom */
    .statusbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(30, 30, 46, 0.95);
        border-top: 1px solid rgba(255,255,255,0.1);
        padding: 0.3rem 1rem;
        z-index: 1000;
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------------------------


@st.cache_resource
def get_qdrant_client() -> ChronosQdrantClient:
    """Initialize Qdrant client (cached)."""
    return ChronosQdrantClient()


@st.cache_resource
def get_embedder() -> ChronosEmbeddingService:
    """Initialize Gemini embedder (cached). Raises if GEMINI_API_KEY not set."""
    return ChronosEmbeddingService()


def _set_latency(ms: float):
    st.session_state.last_latency_ms = float(ms)


def _log_error(msg: str):
    logs = st.session_state.get("error_logs", [])
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    st.session_state.error_logs = logs[-100:]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def get_system_status(settings) -> Dict[str, Any]:
    """Check system health and return status dict."""
    status = {
        "gemini": bool(settings.gemini_api_key),
        "plaud": bool(settings.plaud_client_id and settings.plaud_client_secret),
        "qdrant": False,
        "qdrant_url": settings.qdrant_url,
        "points_count": 0,
        "recordings_count": 0,
        "pending_count": 0,
    }

    # Check Qdrant
    try:
        qdrant = get_qdrant_client()
        stats = qdrant.get_stats()
        status["qdrant"] = True
        status["points_count"] = stats.get("points_count", 0)
    except Exception:
        pass

    # Check database
    try:
        init_db()
        session = SessionLocal()
        status["recordings_count"] = session.query(ChronosRecordingDB).count()
        status["pending_count"] = (
            session.query(ChronosRecordingDB)
            .filter_by(processing_status="pending")
            .count()
        )
        session.close()
    except Exception:
        pass

    return status


def run_pipeline_command(args: List[str], header: str) -> int:
    """Run pipeline subprocess and stream output to UI with progress bars."""
    import re

    st.subheader(header)

    # Create UI elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    output_area = st.empty()
    lines: List[str] = []

    # Regex to parse progress lines like: "⏳ Gemini: [████░░░░] 3/10 (30%) → processing..."
    progress_pattern = re.compile(r"(\d+)/(\d+)\s*\((\d+)%\)")

    proc = subprocess.Popen(
        [sys.executable, "scripts/chronos_pipeline.py", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        clean_line = line.rstrip("\n")
        lines.append(clean_line)
        lines = lines[-300:]

        # Parse progress from line
        match = progress_pattern.search(clean_line)
        if match:
            current, total, pct = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            progress_bar.progress(pct / 100)
            status_text.markdown(f"**{current}/{total}** — {clean_line[:100]}")
        elif "📝 Streaming:" in clean_line:
            # Real-time streaming progress - this is the key update!
            status_text.markdown(f"🔄 **{clean_line.strip()}**")
        elif "⏳ Still processing" in clean_line:
            # Heartbeat - show elapsed time prominently
            status_text.markdown(f"🧠 **{clean_line.strip()}**")
        elif "📊 Transcript:" in clean_line:
            # Show transcript stats
            status_text.markdown(f"📊 **{clean_line.strip()}**")
        elif "📊 Tokens" in clean_line:
            # Token usage stats
            status_text.markdown(f"📊 **{clean_line.strip()}**")
        elif "🤖 Model:" in clean_line:
            status_text.markdown(f"🤖 **{clean_line.strip()}**")
        elif "📤 Sending to Gemini" in clean_line:
            status_text.markdown(f"📤 **Sending to Gemini API (streaming)...**")
        elif "🔍 Parsing" in clean_line:
            status_text.markdown(f"🔍 **{clean_line.strip()}**")
        elif "📋 Event preview" in clean_line:
            status_text.markdown(f"📋 **Parsing extracted events...**")
        elif "📊 Categories:" in clean_line:
            status_text.markdown(f"📊 **{clean_line.strip()}**")
        elif "🔄 Fetching transcript" in clean_line:
            status_text.markdown(f"📥 **Fetching transcript from Plaud...**")
        elif "📄 Recording" in clean_line:
            # Extract recording info
            status_text.markdown(f"**{clean_line}**")
        elif "✅" in clean_line:
            status_text.markdown(f"✅ {clean_line}")
        elif "❌" in clean_line:
            status_text.markdown(f"❌ {clean_line}")
        elif "════" in clean_line:
            # Phase header - show prominently
            status_text.markdown(f"**{clean_line.replace('═', '').strip()}**")

        output_area.code("\n".join(lines), language="text")

    result = proc.wait()
    progress_bar.progress(1.0)
    return result


def format_event_card(event: Dict[str, Any]) -> str:
    """Format a Qdrant event as HTML card."""
    payload = event.get("payload", {})
    category = payload.get("category", "unknown")
    score = event.get("score")
    score_txt = f" · score: {score:.3f}" if score else ""

    return f"""
    <div class="event-card">
        <div><span class="status-pill">{category}</span>
             <span class="muted">{payload.get('recording_id', '')[:20]}</span></div>
        <div style="margin-top:0.3rem; font-size:0.85rem; opacity:0.8;">
            {payload.get('start_ts', '')[:16]} → {payload.get('end_ts', '')[:16]}{score_txt}
        </div>
        <div style="margin-top:0.4rem;">{payload.get('clean_text', '')[:500]}</div>
        <div class="muted" style="margin-top:0.4rem;">
            Sentiment: {payload.get('sentiment', 0):.2f} ·
            Duration: {payload.get('duration_seconds', 0):.0f}s
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# DYNAMIC DATA UTILITIES — Auto-adapt to YOUR data
# ---------------------------------------------------------------------------


def string_to_color(s: str, saturation: float = 0.65, lightness: float = 0.55) -> str:
    """Generate a consistent HSL color from any string.

    Same string always produces same color. Great for dynamic categories.
    """
    import hashlib

    # Hash the string to get a consistent number
    hash_bytes = hashlib.md5(s.encode()).digest()
    hue = int.from_bytes(hash_bytes[:2], "big") % 360

    # Convert HSL to hex
    import colorsys

    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def get_dynamic_category_colors(categories: List[str]) -> Dict[str, str]:
    """Generate consistent colors for any list of categories."""
    return {cat: string_to_color(cat) for cat in categories}


@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_unique_values_from_qdrant(field: str) -> List[str]:
    """Get all unique values for a payload field from Qdrant using faceting.

    This dynamically discovers what categories, days, speakers, etc. exist.
    """
    try:
        qdrant = get_qdrant_client()

        # Use faceting API for efficient unique value retrieval
        from qdrant_client.models import FacetRequest

        result = qdrant.client.facet(
            collection_name=qdrant.collection_name,
            key=field,
            limit=1000,  # Get up to 1000 unique values
        )

        return sorted([hit.value for hit in result.hits if hit.value])
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_all_unique_categories() -> List[str]:
    """Get all unique categories from Qdrant."""
    cats = get_unique_values_from_qdrant("category")
    return cats if cats else ["unknown"]


@st.cache_data(ttl=60)
def get_all_unique_days() -> List[str]:
    """Get all unique days of week from Qdrant."""
    days = get_unique_values_from_qdrant("day_of_week")
    # Sort by weekday order
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return (
        sorted(days, key=lambda d: day_order.index(d) if d in day_order else 99)
        if days
        else day_order
    )


@st.cache_data(ttl=60)
def get_all_unique_speakers() -> List[str]:
    """Get all unique speakers from Qdrant."""
    speakers = get_unique_values_from_qdrant("speaker")
    return speakers if speakers else ["self_talk"]


@st.cache_data(ttl=60)
def get_all_unique_hours() -> tuple:
    """Get min/max hours that have data."""
    try:
        qdrant = get_qdrant_client()
        from qdrant_client.models import FacetRequest

        result = qdrant.client.facet(
            collection_name=qdrant.collection_name,
            key="hour_of_day",
            limit=24,
        )

        hours = [hit.value for hit in result.hits if hit.value is not None]
        if hours:
            return (min(hours), max(hours))
    except Exception:
        pass
    return (0, 23)


@st.cache_data(ttl=30)
def get_collection_field_stats() -> Dict[str, Any]:
    """Get comprehensive stats about all payload fields in the collection.

    Returns counts for each unique value in each indexed field.
    """
    stats = {
        "categories": {},
        "days": {},
        "hours": {},
        "speakers": {},
        "total_points": 0,
    }

    try:
        qdrant = get_qdrant_client()
        collection_info = qdrant.get_stats()
        stats["total_points"] = collection_info.get("points_count", 0)

        # Get category distribution
        from qdrant_client.models import FacetRequest

        for field, key in [
            ("categories", "category"),
            ("days", "day_of_week"),
            ("hours", "hour_of_day"),
            ("speakers", "speaker"),
        ]:
            try:
                result = qdrant.client.facet(
                    collection_name=qdrant.collection_name,
                    key=key,
                    limit=1000,
                )
                stats[field] = {hit.value: hit.count for hit in result.hits}
            except Exception:
                pass

    except Exception:
        pass

    return stats


# ---------------------------------------------------------------------------
# SESSION DETECTION — Group split recordings into logical sessions
# ---------------------------------------------------------------------------


@dataclass
class RecordingSession:
    """A logical session grouping multiple recordings.

    When Plaud splits long recordings (e.g., 5hr + 1.5hr chunks),
    we detect and group them here.
    """

    session_id: str
    recordings: List[Any]  # List of ChronosRecordingDB objects
    device_id: Optional[str]
    start_time: datetime
    end_time: datetime
    total_duration_seconds: int
    recording_count: int

    @property
    def date_str(self) -> str:
        return self.start_time.strftime("%Y-%m-%d")

    @property
    def time_range_str(self) -> str:
        start = self.start_time.strftime("%H:%M")
        end = self.end_time.strftime("%H:%M")
        return f"{start} → {end}"

    @property
    def duration_str(self) -> str:
        h = self.total_duration_seconds // 3600
        m = (self.total_duration_seconds % 3600) // 60
        return f"{h}h {m}m"


def detect_sessions(
    recordings: List[Any],
    gap_threshold_minutes: int = 15,
) -> List[RecordingSession]:
    """Detect recording sessions by grouping recordings with small time gaps.

    Algorithm:
    1. Group by device_id (or "unknown" if None)
    2. Sort each group by created_at
    3. If gap between end of rec A and start of rec B < threshold, same session
    4. Handle Plaud's 5-hour auto-split (typical gap is 0-5 minutes)

    Args:
        recordings: List of ChronosRecordingDB objects
        gap_threshold_minutes: Max gap to consider same session (default 15 min)

    Returns:
        List of RecordingSession objects, sorted by start_time desc
    """
    if not recordings:
        return []

    # Group recordings by device
    by_device: Dict[str, List[Any]] = defaultdict(list)
    for rec in recordings:
        device_key = rec.device_id or "unknown"
        by_device[device_key].append(rec)

    sessions: List[RecordingSession] = []

    for device_id, device_recs in by_device.items():
        # Sort by created_at (recording start time)
        sorted_recs = sorted(device_recs, key=lambda r: r.created_at or datetime.min)

        current_session: List[Any] = []

        for rec in sorted_recs:
            if not current_session:
                # Start new session
                current_session = [rec]
            else:
                # Check gap from previous recording's END to this one's START
                prev_rec = current_session[-1]
                prev_end = prev_rec.created_at + timedelta(
                    seconds=prev_rec.duration_seconds or 0
                )
                curr_start = rec.created_at

                gap_minutes = (curr_start - prev_end).total_seconds() / 60

                if gap_minutes <= gap_threshold_minutes and gap_minutes >= -5:
                    # Same session (allow small overlap due to timestamp precision)
                    current_session.append(rec)
                else:
                    # New session - save current one first
                    sessions.append(_build_session(current_session, device_id))
                    current_session = [rec]

        # Don't forget the last session
        if current_session:
            sessions.append(_build_session(current_session, device_id))

    # Sort sessions by start time descending (most recent first)
    sessions.sort(key=lambda s: s.start_time, reverse=True)

    return sessions


def _build_session(recordings: List[Any], device_id: str) -> RecordingSession:
    """Build a RecordingSession from a list of grouped recordings."""
    sorted_recs = sorted(recordings, key=lambda r: r.created_at or datetime.min)

    start_time = sorted_recs[0].created_at
    last_rec = sorted_recs[-1]
    end_time = last_rec.created_at + timedelta(seconds=last_rec.duration_seconds or 0)

    total_duration = sum(r.duration_seconds or 0 for r in recordings)

    # Generate session ID from first recording ID + date
    session_id = (
        f"session_{sorted_recs[0].recording_id[:8]}_{start_time.strftime('%Y%m%d')}"
    )

    return RecordingSession(
        session_id=session_id,
        recordings=sorted_recs,
        device_id=device_id if device_id != "unknown" else None,
        start_time=start_time,
        end_time=end_time,
        total_duration_seconds=total_duration,
        recording_count=len(recordings),
    )


@st.cache_data(ttl=30)
def get_session_events(recording_ids: tuple) -> List[Dict[str, Any]]:
    """Get all events from Qdrant for a list of recording IDs.

    Uses tuple for recording_ids because lists aren't hashable (caching).
    """
    try:
        qdrant = get_qdrant_client()
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        points, _ = qdrant.client.scroll(
            collection_name=qdrant.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="recording_id",
                        match=MatchAny(any=list(recording_ids)),
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
        )

        # Convert to dicts and sort by start_ts
        events = []
        for p in points:
            events.append({"id": str(p.id), "payload": p.payload})

        events.sort(
            key=lambda e: e["payload"].get("start_ts", "") if e["payload"] else ""
        )
        return events

    except Exception:
        return []


def get_all_unique_sessions() -> List[str]:
    """Get unique session IDs from existing recordings."""
    try:
        init_db()
        session = SessionLocal()
        recordings = session.query(ChronosRecordingDB).all()
        sessions = detect_sessions(recordings)
        session.close()
        return [s.session_id for s in sessions]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------------------


def page_home(settings, status: Dict[str, Any]):
    """Home page with quick status and one-click actions."""
    st.markdown('<div class="main-title">🕰️ Chronos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Your voice recordings, searchable and organized</div>',
        unsafe_allow_html=True,
    )

    # Status badges
    badges = []
    if status["qdrant"]:
        badges.append('<span class="status-pill status-ok">Qdrant ✓</span>')
    else:
        badges.append('<span class="status-pill status-error">Qdrant ✗</span>')
    if status["gemini"]:
        badges.append('<span class="status-pill status-ok">Gemini ✓</span>')
    else:
        badges.append(
            '<span class="status-pill status-warn">Gemini not configured</span>'
        )
    if status["plaud"]:
        badges.append('<span class="status-pill status-ok">Plaud ✓</span>')
    else:
        badges.append(
            '<span class="status-pill status-warn">Plaud not configured</span>'
        )

    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Recordings", status["recordings_count"])
    with col2:
        st.metric("⏳ Pending", status["pending_count"])
    with col3:
        st.metric("🔍 Indexed Events", status["points_count"])
    with col4:
        ready = status["qdrant"] and status["gemini"]
        st.metric("Status", "Ready" if ready else "Setup needed")

    st.markdown("---")

    # Quick actions
    st.subheader("⚡ Quick Actions")

    action_cols = st.columns(4)
    with action_cols[0]:
        if st.button(
            "🔄 Fetch New Recordings",
            width="stretch",
            disabled=not status["plaud"],
        ):
            code = run_pipeline_command(["--ingest", "--limit", "25"], "Fetching...")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with action_cols[1]:
        if st.button(
            "🧠 Process Pending",
            width="stretch",
            disabled=not status["gemini"],
        ):
            code = run_pipeline_command(["--process", "--limit", "10"], "Processing...")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with action_cols[2]:
        if st.button(
            "📤 Index to Qdrant",
            width="stretch",
            disabled=not (status["qdrant"] and status["gemini"]),
        ):
            code = run_pipeline_command(["--index", "--limit", "50"], "Indexing...")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with action_cols[3]:
        if st.button(
            "🚀 Run Full Pipeline",
            width="stretch",
            disabled=not (status["plaud"] and status["gemini"] and status["qdrant"]),
        ):
            code = run_pipeline_command(["--full", "--limit", "10"], "Full Pipeline")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    # Getting started guide (if not fully configured)
    if not (status["plaud"] and status["gemini"] and status["qdrant"]):
        st.markdown("---")
        st.subheader("🚀 Getting Started")

        steps = []
        if not status["plaud"]:
            steps.append(
                "1. **Configure Plaud OAuth** — Run `python plaud_setup.py` and add credentials to `.env`"
            )
        if not status["gemini"]:
            steps.append(
                "2. **Add Gemini API Key** — Get a key from [ai.google.dev](https://ai.google.dev) and add `GEMINI_API_KEY` to `.env`"
            )
        if not status["qdrant"]:
            steps.append(
                "3. **Start Qdrant** — Run `docker run -p 6333:6333 qdrant/qdrant`"
            )

        for step in steps:
            st.markdown(step)


# ---------------------------------------------------------------------------
# PAGE: SEARCH
# ---------------------------------------------------------------------------


def page_search(settings, status: Dict[str, Any]):
    """Semantic + temporal search."""
    st.header("🔍 Search")

    if not status["qdrant"]:
        st.error(
            "Qdrant is not connected. Start it with: `docker run -p 6333:6333 qdrant/qdrant`"
        )
        return

    col_query, col_results = st.columns([1, 2])

    with col_query:
        st.subheader("Query")

        query_text = st.text_area(
            "What are you looking for?",
            placeholder="e.g., What do I think about anxiety? / meetings about Project Alpha",
            height=80,
            disabled=not status["gemini"],
            help="Semantic search requires GEMINI_API_KEY",
        )

        # Temporal filters
        st.subheader("Filters")
        filter_type = st.selectbox(
            "Time filter",
            ["None", "Date Range", "Day of Week", "Hour of Day"],
            help="Filter by when events occurred",
        )

        temporal_filter: Optional[TemporalFilter] = None

        if filter_type == "Date Range":
            start = st.date_input("From", value=datetime.now() - timedelta(days=30))
            end = st.date_input("To", value=datetime.now())
            temporal_filter = TemporalFilter(
                start_date=datetime.combine(start, datetime.min.time()),
                end_date=datetime.combine(end, datetime.max.time()),
            )
        elif filter_type == "Day of Week":
            days = st.multiselect(
                "Days",
                options=[d.value for d in DayOfWeek],
                help="Find patterns by day",
            )
            if days:
                temporal_filter = TemporalFilter(
                    days_of_week=[DayOfWeek(d) for d in days]
                )
        elif filter_type == "Hour of Day":
            hours = st.slider("Hours", 0, 23, (9, 17))
            temporal_filter = TemporalFilter(
                hours_of_day=list(range(hours[0], hours[1] + 1))
            )

        categories = st.multiselect(
            "Categories",
            options=[c.value for c in EventCategory],
        )

        limit = st.slider("Max results", 10, 200, 50)

        search_btn = st.button("🔍 Search", type="primary", width="stretch")

        with st.expander("Advanced"):
            debug = st.checkbox("Show raw payloads")

    with col_results:
        if not search_btn:
            st.info("Configure your search on the left and click Search.")
            return

        qdrant = get_qdrant_client()
        query_vector = None

        if query_text and query_text.strip() and status["gemini"]:
            with st.spinner("Embedding query..."):
                t0 = time.perf_counter()
                embedder = get_embedder()
                query_vector = embedder.embed_text(
                    query_text.strip(), task_type="RETRIEVAL_QUERY"
                )
                _set_latency((time.perf_counter() - t0) * 1000)

        with st.spinner("Searching..."):
            try:
                t0 = time.perf_counter()
                results = qdrant.search_hybrid(
                    query_vector=query_vector,
                    temporal_filter=temporal_filter,
                    categories=categories or None,
                    limit=limit,
                )
                _set_latency((time.perf_counter() - t0) * 1000)
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        st.subheader(f"Results ({len(results)})")

        if not results:
            st.info("No events found. Try adjusting your filters.")
            return

        for i, event in enumerate(results, 1):
            with st.expander(
                f"{i}. {event.get('payload', {}).get('category', 'event')} — "
                f"{event.get('payload', {}).get('start_ts', '')[:10]}",
                expanded=(i <= 3),
            ):
                st.markdown(format_event_card(event), unsafe_allow_html=True)
                if debug:
                    st.json(event)


# ---------------------------------------------------------------------------
# PAGE: LIBRARY
# ---------------------------------------------------------------------------


def page_library(settings, status: Dict[str, Any]):
    """
    📚 ULTIMATE Recording Library — EVERYTHING exposed.

    Features:
    - Full transcript viewer with word count, search highlighting
    - All SQLite metadata fields visible
    - Extracted events with full Qdrant payloads
    - Vector similarity exploration
    - Processing actions with real-time feedback
    - SESSION GROUPING: Auto-detect split recordings as unified sessions
    """
    st.header("📚 Recording Library")
    st.markdown(
        '<p class="subtitle">Complete access to your recordings, transcripts, and extracted events</p>',
        unsafe_allow_html=True,
    )

    init_db()
    session = SessionLocal()

    # Initialize Qdrant client for vector operations
    qdrant_client = None
    if status["qdrant"]:
        try:
            qdrant_client = ChronosQdrantClient()
        except Exception:
            pass

    try:
        # ═══════════════════════════════════════════════════════════════
        # VIEW MODE TOGGLE — Recordings vs Sessions
        # ═══════════════════════════════════════════════════════════════
        view_col1, view_col2, view_col3 = st.columns([2, 1, 2])
        with view_col2:
            view_mode = st.radio(
                "View Mode",
                ["📁 Recordings", "🗂️ Sessions"],
                horizontal=True,
                help="Sessions group split recordings (5hr chunks) from the same device",
            )

        if view_mode == "🗂️ Sessions":
            _render_session_view(session, qdrant_client, status)
        else:
            _render_recording_view(session, qdrant_client, status)

    finally:
        session.close()


def _render_session_view(db_session, qdrant_client, status: Dict[str, Any]):
    """Render the session-based view of recordings."""

    # ═══════════════════════════════════════════════════════════════
    # SESSION FILTERS
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🗂️ Session View")
    st.info(
        "**Sessions** group recordings from the same device with small time gaps. "
        "Perfect for full-day recordings that Plaud splits into 5-hour chunks."
    )

    filter_cols = st.columns([1, 1, 1])
    with filter_cols[0]:
        days_back = st.number_input("Days back", 1, 365, 90, key="session_days_back")
    with filter_cols[1]:
        gap_threshold = st.slider(
            "Gap threshold (minutes)",
            1,
            60,
            15,
            help="Max gap between recordings to consider same session",
        )
    with filter_cols[2]:
        min_recordings = st.number_input(
            "Min recordings per session",
            1,
            10,
            1,
            help="Filter to sessions with at least this many recordings",
        )

    # Get all recordings
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    recordings = (
        db_session.query(ChronosRecordingDB)
        .filter(ChronosRecordingDB.created_at >= cutoff.replace(tzinfo=None))
        .all()
    )

    if not recordings:
        st.info("No recordings found in this time range.")
        return

    # Detect sessions
    sessions = detect_sessions(recordings, gap_threshold_minutes=gap_threshold)

    # Filter by min recordings
    sessions = [s for s in sessions if s.recording_count >= min_recordings]

    if not sessions:
        st.info("No sessions found with current filters. Try adjusting the threshold.")
        return

    # ═══════════════════════════════════════════════════════════════
    # SESSION SUMMARY STATS
    # ═══════════════════════════════════════════════════════════════
    stat_cols = st.columns(6)
    total_recs = sum(s.recording_count for s in sessions)
    total_duration = sum(s.total_duration_seconds for s in sessions)
    multi_chunk = sum(1 for s in sessions if s.recording_count > 1)
    avg_duration = total_duration // len(sessions) if sessions else 0

    stat_cols[0].metric("🗂️ Sessions", len(sessions))
    stat_cols[1].metric("📁 Recordings", total_recs)
    stat_cols[2].metric(
        "🔗 Multi-chunk", multi_chunk, help="Sessions with >1 recording"
    )
    stat_cols[3].metric(
        "⏱️ Total Duration",
        f"{total_duration // 3600}h {(total_duration % 3600) // 60}m",
    )
    stat_cols[4].metric(
        "📊 Avg Duration", f"{avg_duration // 3600}h {(avg_duration % 3600) // 60}m"
    )
    stat_cols[5].metric(
        "📅 Date Range",
        (
            f"{sessions[-1].date_str} → {sessions[0].date_str}"
            if len(sessions) > 1
            else sessions[0].date_str
        ),
    )

    # ═══════════════════════════════════════════════════════════════
    # SESSION LIST
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Sessions")

    # Sessions table
    rows = []
    for s in sessions:
        recording_statuses = Counter(r.processing_status for r in s.recordings)
        status_str = ", ".join(f"{v}×{k}" for k, v in recording_statuses.items())

        rows.append(
            {
                "Date": s.date_str,
                "Time": s.time_range_str,
                "Duration": s.duration_str,
                "Recordings": s.recording_count,
                "Device": s.device_id[:8] if s.device_id else "—",
                "Status": status_str,
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True, height=250)

    # ═══════════════════════════════════════════════════════════════
    # SESSION DETAIL
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 Session Detail")

    options = [
        f"{s.date_str} | {s.time_range_str} | {s.duration_str} ({s.recording_count} recs)"
        for s in sessions
    ]
    selected_idx = st.selectbox(
        "Select session to explore",
        range(len(options)),
        format_func=lambda i: options[i],
        key="session_selector",
    )

    if selected_idx is not None:
        selected_session = sessions[selected_idx]

        # Session tabs
        tab_overview, tab_recordings, tab_transcript, tab_events, tab_timeline = (
            st.tabs(
                [
                    "📊 Overview",
                    "📁 Recordings",
                    "📝 Combined Transcript",
                    "🎯 Events",
                    "⏱️ Timeline",
                ]
            )
        )

        # ─────────────────────────────────────────────────────────────
        # TAB: Overview
        # ─────────────────────────────────────────────────────────────
        with tab_overview:
            st.markdown("#### Session Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Session Info**")
                st.code(f"session_id: {selected_session.session_id}")
                st.code(f"device_id: {selected_session.device_id or 'Unknown'}")
                st.code(f"date: {selected_session.date_str}")
                st.code(f"time_range: {selected_session.time_range_str}")
                st.code(f"total_duration: {selected_session.duration_str}")
                st.code(f"recording_count: {selected_session.recording_count}")

            with col2:
                st.markdown("**Recording Breakdown**")
                for i, rec in enumerate(selected_session.recordings, 1):
                    dur = rec.duration_seconds or 0
                    st.markdown(
                        f"**{i}.** `{rec.recording_id[:16]}...` — "
                        f"{dur // 3600}h {(dur % 3600) // 60}m — "
                        f"*{rec.processing_status}*"
                    )

            # Processing status
            statuses = Counter(r.processing_status for r in selected_session.recordings)
            st.markdown("**Processing Status**")
            for status_name, count in statuses.items():
                pct = count / len(selected_session.recordings) * 100
                color = (
                    "status-ok"
                    if status_name == "completed"
                    else "status-warn" if status_name == "pending" else "status-error"
                )
                st.markdown(
                    f'<span class="status-pill {color}">{status_name}</span> {count} ({pct:.0f}%)',
                    unsafe_allow_html=True,
                )

        # ─────────────────────────────────────────────────────────────
        # TAB: Recordings
        # ─────────────────────────────────────────────────────────────
        with tab_recordings:
            st.markdown("#### Recordings in This Session")

            for i, rec in enumerate(selected_session.recordings, 1):
                with st.expander(
                    f"📁 Recording {i}: {rec.title or rec.recording_id[:24]}",
                    expanded=(i == 1),
                ):
                    r_cols = st.columns([2, 1])

                    with r_cols[0]:
                        st.code(f"recording_id: {rec.recording_id}")
                        st.code(f"title: {rec.title or 'Untitled'}")
                        st.code(f"created_at: {rec.created_at}")
                        st.code(
                            f"duration: {rec.duration_seconds // 60}m {rec.duration_seconds % 60}s"
                        )
                        if rec.transcript:
                            words = len(rec.transcript.split())
                            st.code(f"transcript: {words:,} words")

                    with r_cols[1]:
                        st.code(f"status: {rec.processing_status}")
                        st.code(
                            f"device: {rec.device_id[:8] if rec.device_id else 'Unknown'}"
                        )
                        st.code(f"source: {rec.source}")
                        if rec.processed_at:
                            st.code(
                                f"processed: {rec.processed_at.strftime('%Y-%m-%d %H:%M')}"
                            )

        # ─────────────────────────────────────────────────────────────
        # TAB: Combined Transcript
        # ─────────────────────────────────────────────────────────────
        with tab_transcript:
            st.markdown("#### Combined Session Transcript")

            all_transcripts = []
            total_words = 0

            for rec in selected_session.recordings:
                if rec.transcript:
                    all_transcripts.append(
                        f"\n{'='*60}\n"
                        f"📁 RECORDING: {rec.title or rec.recording_id[:24]}\n"
                        f"⏱️ {rec.duration_seconds // 60}m {rec.duration_seconds % 60}s | "
                        f"🕐 {rec.created_at.strftime('%H:%M') if rec.created_at else 'Unknown'}\n"
                        f"{'='*60}\n\n"
                        f"{rec.transcript}"
                    )
                    total_words += len(rec.transcript.split())

            if all_transcripts:
                combined = "\n\n".join(all_transcripts)

                # Stats
                stat_cols = st.columns(4)
                stat_cols[0].metric("📝 Total Words", f"{total_words:,}")
                stat_cols[1].metric("📁 Recordings", len(all_transcripts))
                stat_cols[2].metric("📄 Characters", f"{len(combined):,}")
                stat_cols[3].metric("⏱️ Est. Speaking", f"{total_words // 150}m")

                # Search
                search_q = st.text_input(
                    "🔍 Search in combined transcript",
                    placeholder="Find text across all recordings...",
                    key="session_transcript_search",
                )

                if search_q:
                    import re

                    pattern = re.compile(re.escape(search_q), re.IGNORECASE)
                    matches = list(pattern.finditer(combined))
                    st.info(f"Found **{len(matches)}** matches")

                # Display
                st.text_area(
                    "Combined transcript",
                    combined,
                    height=400,
                    label_visibility="collapsed",
                )

                st.download_button(
                    "📥 Download Combined Transcript",
                    combined,
                    file_name=f"{selected_session.session_id}_transcript.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No transcripts available for this session.")

        # ─────────────────────────────────────────────────────────────
        # TAB: Events
        # ─────────────────────────────────────────────────────────────
        with tab_events:
            st.markdown("#### All Events in Session")

            recording_ids = tuple(r.recording_id for r in selected_session.recordings)
            events = get_session_events(recording_ids)

            if events:
                # Event stats
                categories = Counter(
                    e["payload"].get("category", "unknown") for e in events
                )
                sentiments = [
                    e["payload"].get("sentiment", 0)
                    for e in events
                    if e["payload"].get("sentiment") is not None
                ]

                stat_cols = st.columns(5)
                stat_cols[0].metric("🎯 Total Events", len(events))
                stat_cols[1].metric("📁 Categories", len(categories))
                stat_cols[2].metric(
                    "💭 Avg Sentiment",
                    f"{sum(sentiments)/len(sentiments):.2f}" if sentiments else "—",
                )
                stat_cols[3].metric("📁 Recordings", len(recording_ids))
                stat_cols[4].metric("⏱️ Session Duration", selected_session.duration_str)

                # Category breakdown
                with st.expander("📊 Category Distribution", expanded=True):
                    cat_colors = get_dynamic_category_colors(list(categories.keys()))
                    for cat, count in categories.most_common():
                        pct = count / len(events) * 100
                        st.progress(pct / 100, text=f"{cat}: {count} ({pct:.1f}%)")

                # Event list
                for i, e in enumerate(events[:50]):  # Limit to 50 for performance
                    payload = e.get("payload", {})
                    with st.expander(
                        f"🎯 {i+1}. {payload.get('category', 'event')} — {payload.get('clean_text', '')[:50]}...",
                        expanded=(i < 3),
                    ):
                        st.markdown(format_event_card(e), unsafe_allow_html=True)
                        st.code(f"recording_id: {payload.get('recording_id', '')}")

                if len(events) > 50:
                    st.info(
                        f"Showing 50 of {len(events)} events. Use search to find specific content."
                    )
            else:
                st.info("No events found. Process recordings first.")

        # ─────────────────────────────────────────────────────────────
        # TAB: Timeline
        # ─────────────────────────────────────────────────────────────
        with tab_timeline:
            st.markdown("#### Session Timeline")

            recording_ids = tuple(r.recording_id for r in selected_session.recordings)
            events = get_session_events(recording_ids)

            if events:
                # Build timeline data
                import plotly.express as px
                import pandas as pd

                timeline_data = []
                for e in events:
                    payload = e.get("payload", {})
                    start_ts = payload.get("start_ts", "")
                    category = payload.get("category", "unknown")

                    if start_ts:
                        try:
                            ts = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                            timeline_data.append(
                                {
                                    "time": ts,
                                    "category": category,
                                    "text": payload.get("clean_text", "")[:100],
                                    "hour": ts.hour,
                                    "minute": ts.minute,
                                }
                            )
                        except Exception:
                            pass

                if timeline_data:
                    df = pd.DataFrame(timeline_data)

                    # Events over time
                    st.markdown("##### Events by Hour")
                    hour_counts = df.groupby("hour").size().reset_index(name="count")
                    fig = px.bar(
                        hour_counts,
                        x="hour",
                        y="count",
                        title=f"Event Distribution Across {selected_session.duration_str}",
                        labels={"hour": "Hour of Day", "count": "Events"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Category breakdown by hour
                    st.markdown("##### Categories by Hour")
                    cat_hour = (
                        df.groupby(["hour", "category"])
                        .size()
                        .reset_index(name="count")
                    )
                    fig2 = px.bar(
                        cat_hour,
                        x="hour",
                        y="count",
                        color="category",
                        title="Category Distribution by Hour",
                        barmode="stack",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No timeline data available.")
            else:
                st.info("No events to display on timeline.")


def _render_recording_view(db_session, qdrant_client, status: Dict[str, Any]):
    """Render the traditional recording-by-recording view."""

    # ═══════════════════════════════════════════════════════════════
    # FILTERS — All the controls
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🎛️ Filters")
    filter_cols = st.columns([1, 1, 1, 2])

    with filter_cols[0]:
        status_filter = st.multiselect(
            "Processing Status",
            ["pending", "processing", "completed", "failed"],
            default=["pending", "completed", "failed"],
        )
    with filter_cols[1]:
        days_back = st.number_input("Days back", 1, 365, 90, key="rec_days_back")
    with filter_cols[2]:
        sort_by = st.selectbox(
            "Sort by",
            ["created_at", "duration_seconds", "title"],
            index=0,
        )
    with filter_cols[3]:
        search_q = st.text_input(
            "🔍 Search",
            placeholder="Search title, ID, or transcript content...",
        )

    # Query
    q = db_session.query(ChronosRecordingDB)
    if status_filter:
        q = q.filter(ChronosRecordingDB.processing_status.in_(status_filter))
    if days_back:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        q = q.filter(ChronosRecordingDB.created_at >= cutoff.replace(tzinfo=None))
    if search_q:
        like = f"%{search_q}%"
        q = q.filter(
            (ChronosRecordingDB.recording_id.ilike(like))
            | (ChronosRecordingDB.title.ilike(like))
            | (ChronosRecordingDB.transcript.ilike(like))
        )

    # Apply sorting
    if sort_by == "created_at":
        q = q.order_by(ChronosRecordingDB.created_at.desc())
    elif sort_by == "duration_seconds":
        q = q.order_by(ChronosRecordingDB.duration_seconds.desc())
    else:
        q = q.order_by(ChronosRecordingDB.title.asc())

    recs = q.limit(200).all()

    if not recs:
        st.info("No recordings found. Go to **Pipeline** to fetch from Plaud.")
        return

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY STATS
    # ═══════════════════════════════════════════════════════════════
    stat_cols = st.columns(6)
    pending = sum(1 for r in recs if r.processing_status == "pending")
    completed = sum(1 for r in recs if r.processing_status == "completed")
    failed = sum(1 for r in recs if r.processing_status == "failed")
    with_transcript = sum(1 for r in recs if r.transcript)
    total_duration = sum(r.duration_seconds or 0 for r in recs)
    total_events = sum(
        db_session.query(ChronosEventDB).filter_by(recording_id=r.recording_id).count()
        for r in recs
    )

    stat_cols[0].metric("📊 Total", len(recs))
    stat_cols[1].metric("⏳ Pending", pending)
    stat_cols[2].metric("✅ Completed", completed)
    stat_cols[3].metric("❌ Failed", failed)
    stat_cols[4].metric("📝 With Transcript", with_transcript)
    stat_cols[5].metric(
        "⏱️ Total Duration",
        f"{total_duration // 3600}h {(total_duration % 3600) // 60}m",
    )

    # ═══════════════════════════════════════════════════════════════
    # RECORDINGS TABLE — Full data
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📋 Recordings")

    rows = []
    for r in recs:
        event_count = (
            db_session.query(ChronosEventDB)
            .filter_by(recording_id=r.recording_id)
            .count()
        )
        transcript_len = len(r.transcript) if r.transcript else 0
        word_count = len(r.transcript.split()) if r.transcript else 0

        rows.append(
            {
                "ID": r.recording_id,
                "Title": r.title or "—",
                "Created": (
                    r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—"
                ),
                "Duration": (
                    f"{r.duration_seconds // 60}m {r.duration_seconds % 60}s"
                    if r.duration_seconds
                    else "—"
                ),
                "Status": r.processing_status,
                "Events": event_count,
                "Transcript": f"{word_count:,} words" if transcript_len else "—",
                "Device": r.device_id[:8] if r.device_id else "—",
                "Processed": (
                    r.processed_at.strftime("%m-%d %H:%M") if r.processed_at else "—"
                ),
            }
        )

    st.dataframe(rows, width="stretch", hide_index=True, height=300)

    # ═══════════════════════════════════════════════════════════════
    # RECORDING DETAIL — Deep dive
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 Recording Detail")

    options = [
        f"{r.title or r.recording_id[:24]}... ({r.processing_status})" for r in recs
    ]
    selected_idx = st.selectbox(
        "Select recording to explore",
        range(len(options)),
        format_func=lambda i: options[i],
        key="rec_selector",
    )

    if selected_idx is not None:
        rec = recs[selected_idx]

        # TABS for organization
        tab_meta, tab_transcript, tab_events, tab_qdrant, tab_actions = st.tabs(
            ["📋 Metadata", "📝 Transcript", "🎯 Events", "🔮 Qdrant", "⚡ Actions"]
        )

        # ─────────────────────────────────────────────────────────────
        # TAB: Metadata — ALL fields
        # ─────────────────────────────────────────────────────────────
        with tab_meta:
            st.markdown("#### Complete Recording Metadata")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Identifiers**")
                st.code(f"recording_id: {rec.recording_id}")
                st.code(f"device_id: {rec.device_id or 'None'}")
                st.code(f"checksum: {rec.checksum or 'None'}")

            with col2:
                st.markdown("**Timestamps**")
                st.code(f"created_at: {rec.created_at}")
                st.code(f"ingested_at: {rec.ingested_at}")
                st.code(f"processed_at: {rec.processed_at or 'Not processed'}")
                st.code(f"transcript_cached_at: {rec.transcript_cached_at or 'None'}")

            with col3:
                st.markdown("**Processing**")
                st.code(f"status: {rec.processing_status}")
                st.code(f"source: {rec.source}")
                st.code(f"duration_seconds: {rec.duration_seconds}")
                if rec.error_message:
                    st.error(f"Error: {rec.error_message}")

            st.markdown("**File Paths**")
            st.code(f"local_audio_path: {rec.local_audio_path}")

            # Raw JSON dump
            with st.expander("📦 Raw SQLite Row (JSON)", expanded=False):
                raw_data = {
                    "recording_id": rec.recording_id,
                    "title": rec.title,
                    "created_at": str(rec.created_at),
                    "duration_seconds": rec.duration_seconds,
                    "local_audio_path": rec.local_audio_path,
                    "source": rec.source,
                    "device_id": rec.device_id,
                    "checksum": rec.checksum,
                    "processing_status": rec.processing_status,
                    "error_message": rec.error_message,
                    "processed_at": (
                        str(rec.processed_at) if rec.processed_at else None
                    ),
                    "ingested_at": str(rec.ingested_at),
                    "transcript_length": (len(rec.transcript) if rec.transcript else 0),
                }
                st.json(raw_data)

        # ─────────────────────────────────────────────────────────────
        # TAB: Transcript — Full text with stats
        # ─────────────────────────────────────────────────────────────
        with tab_transcript:
            if rec.transcript:
                transcript = rec.transcript
                words = transcript.split()
                chars = len(transcript)

                # Stats row
                ts_cols = st.columns(5)
                ts_cols[0].metric("📝 Characters", f"{chars:,}")
                ts_cols[1].metric("📖 Words", f"{len(words):,}")
                ts_cols[2].metric("📄 Paragraphs", transcript.count("\n\n") + 1)
                ts_cols[3].metric("⏱️ Est. Speaking Time", f"{len(words) // 150}m")
                ts_cols[4].metric(
                    "💾 Cached",
                    (
                        rec.transcript_cached_at.strftime("%Y-%m-%d")
                        if rec.transcript_cached_at
                        else "—"
                    ),
                )

                # Search within transcript
                search_in_transcript = st.text_input(
                    "🔍 Search in transcript",
                    placeholder="Find text...",
                    key="transcript_search",
                )

                if search_in_transcript:
                    # Highlight matches
                    import re

                    pattern = re.compile(re.escape(search_in_transcript), re.IGNORECASE)
                    matches = list(pattern.finditer(transcript))
                    st.info(
                        f"Found **{len(matches)}** matches for '{search_in_transcript}'"
                    )

                    # Show context around matches
                    for i, match in enumerate(matches[:10]):
                        start = max(0, match.start() - 100)
                        end = min(len(transcript), match.end() + 100)
                        context = transcript[start:end]
                        # Highlight the match
                        highlighted = pattern.sub(
                            f"**🔸{search_in_transcript}🔸**",
                            context,
                        )
                        st.markdown(f"**Match {i+1}:** ...{highlighted}...")

                # Full transcript display
                st.markdown("#### Full Transcript")
                st.text_area(
                    "Transcript content",
                    transcript,
                    height=400,
                    label_visibility="collapsed",
                )

                # Download button
                st.download_button(
                    "📥 Download Transcript (.txt)",
                    transcript,
                    file_name=f"{rec.recording_id[:16]}_transcript.txt",
                    mime="text/plain",
                )
            else:
                st.warning("No transcript cached for this recording.")
                st.markdown(
                    "Click **Fetch Transcript** in the Actions tab to retrieve it from Plaud."
                )

        # ─────────────────────────────────────────────────────────────
        # TAB: Events — All extracted events with full details
        # ─────────────────────────────────────────────────────────────
        with tab_events:
            events = (
                db_session.query(ChronosEventDB)
                .filter_by(recording_id=rec.recording_id)
                .order_by(ChronosEventDB.start_ts.asc())
                .all()
            )

            if events:
                st.markdown(f"#### {len(events)} Extracted Events")

                # Event summary stats
                ev_cols = st.columns(5)
                categories = Counter(e.category for e in events)
                sentiments = [e.sentiment for e in events if e.sentiment is not None]
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

                ev_cols[0].metric("🎯 Total Events", len(events))
                ev_cols[1].metric("📁 Categories", len(categories))
                ev_cols[2].metric("💭 Avg Sentiment", f"{avg_sentiment:.2f}")
                ev_cols[3].metric(
                    "🔑 Keywords", sum(len(e.keywords or []) for e in events)
                )
                ev_cols[4].metric(
                    "⏱️ Total Duration",
                    f"{sum((e.end_ts - e.start_ts).total_seconds() for e in events):.0f}s",
                )

                # Category breakdown
                with st.expander("📊 Category Distribution", expanded=True):
                    for cat, count in categories.most_common():
                        pct = count / len(events) * 100
                        st.progress(pct / 100, text=f"{cat}: {count} ({pct:.1f}%)")

                # Event list with full details
                st.markdown("#### Event Details")

                for i, e in enumerate(events):
                    with st.expander(
                        f"🎯 Event {i+1}: {e.clean_text[:60]}...",
                        expanded=i == 0,
                    ):
                        detail_cols = st.columns([2, 1])

                        with detail_cols[0]:
                            st.markdown(f"**Clean Text:**\n{e.clean_text}")
                            if e.raw_transcript_snippet:
                                st.markdown("**Raw Snippet:**")
                                st.code(e.raw_transcript_snippet[:500])
                            if e.gemini_reasoning:
                                st.markdown("**Gemini Reasoning:**")
                                st.info(e.gemini_reasoning)

                        with detail_cols[1]:
                            st.markdown("**Metadata:**")
                            st.code(f"event_id: {e.event_id}")
                            st.code(f"category: {e.category}")
                            st.code(f"speaker: {e.speaker}")
                            st.code(f"sentiment: {e.sentiment}")
                            st.code(f"day_of_week: {e.day_of_week}")
                            st.code(f"hour_of_day: {e.hour_of_day}")
                            st.code(f"start_ts: {e.start_ts}")
                            st.code(f"end_ts: {e.end_ts}")
                            if e.keywords:
                                st.markdown(f"**Keywords:** {', '.join(e.keywords)}")
                            if e.qdrant_point_id:
                                st.code(f"qdrant_point_id: {e.qdrant_point_id}")
            else:
                st.info("No events extracted yet. Process this recording first.")

        # ─────────────────────────────────────────────────────────────
        # TAB: Qdrant — Vector store exploration
        # ─────────────────────────────────────────────────────────────
        with tab_qdrant:
            if not qdrant_client:
                st.warning("Qdrant not connected. Start Qdrant to explore vectors.")
            else:
                st.markdown("#### Qdrant Vector Storage")

                # Get events for this recording from Qdrant
                try:
                    from qdrant_client.models import (
                        Filter,
                        FieldCondition,
                        MatchValue,
                    )

                    points, _ = qdrant_client.client.scroll(
                        collection_name=qdrant_client.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="recording_id",
                                    match=MatchValue(value=rec.recording_id),
                                )
                            ]
                        ),
                        limit=100,
                        with_vectors=True,
                        with_payload=True,
                    )

                    if points:
                        st.success(
                            f"Found **{len(points)}** vectors in Qdrant for this recording"
                        )

                        # Vector stats
                        vec_cols = st.columns(4)
                        vec_cols[0].metric("🔢 Vectors", len(points))
                        vec_cols[1].metric(
                            "📐 Dimensions",
                            len(points[0].vector) if points[0].vector else 0,
                        )
                        vec_cols[2].metric(
                            "📊 Avg Vector Norm",
                            f"{sum(sum(v**2 for v in p.vector)**0.5 for p in points if p.vector) / len(points):.2f}",
                        )
                        vec_cols[3].metric(
                            "🏷️ Payload Fields",
                            (len(points[0].payload.keys()) if points[0].payload else 0),
                        )

                        # Payload schema exploration
                        st.markdown("#### Payload Schema")
                        if points[0].payload:
                            schema_data = {}
                            for key, value in points[0].payload.items():
                                schema_data[key] = {
                                    "type": type(value).__name__,
                                    "example": (str(value)[:100] if value else "None"),
                                }
                            st.json(schema_data)

                        # Full point inspection
                        st.markdown("#### Point Inspector")
                        point_options = [
                            f"{p.id[:16]}... ({p.payload.get('category', '?')})"
                            for p in points
                        ]
                        selected_point_idx = st.selectbox(
                            "Select point to inspect",
                            range(len(point_options)),
                            format_func=lambda i: point_options[i],
                            key="point_selector",
                        )

                        if selected_point_idx is not None:
                            point = points[selected_point_idx]

                            pt_cols = st.columns([2, 1])

                            with pt_cols[0]:
                                st.markdown("**Full Payload:**")
                                st.json(point.payload)

                            with pt_cols[1]:
                                st.markdown("**Vector Preview (first 20 dims):**")
                                if point.vector:
                                    vec_preview = point.vector[:20]
                                    st.code(
                                        "\n".join(
                                            f"[{i}]: {v:.6f}"
                                            for i, v in enumerate(vec_preview)
                                        )
                                    )
                                    st.caption(
                                        f"... and {len(point.vector) - 20} more dimensions"
                                    )

                            # Find similar vectors
                            st.markdown("#### 🔗 Find Similar Events")
                            if st.button(
                                "Search Similar Vectors", key="similar_search"
                            ):
                                similar = qdrant_client.client.query_points(
                                    collection_name=qdrant_client.collection_name,
                                    query=point.vector,
                                    limit=5,
                                ).points

                                st.markdown("**Top 5 Similar Events:**")
                                for sim in similar:
                                    if sim.id != point.id:
                                        st.markdown(
                                            f"- **Score: {sim.score:.4f}** | "
                                            f"{sim.payload.get('category', '?')} | "
                                            f"{sim.payload.get('clean_text', '')[:80]}..."
                                        )
                    else:
                        st.info(
                            "No vectors found in Qdrant for this recording. Index it first."
                        )
                except Exception as ex:
                    st.error(f"Qdrant query error: {ex}")

        # ─────────────────────────────────────────────────────────────
        # TAB: Actions — Processing controls
        # ─────────────────────────────────────────────────────────────
        with tab_actions:
            st.markdown("#### Processing Actions")

            action_cols = st.columns(4)
            with action_cols[0]:
                if st.button(
                    "🧠 Process with Gemini",
                    width="stretch",
                    disabled=not status["gemini"],
                ):
                    code = run_pipeline_command(
                        [
                            "--process",
                            "--recording-id",
                            rec.recording_id,
                            "--limit",
                            "1",
                        ],
                        f"Processing {rec.recording_id[:16]}...",
                    )
                    if code == 0:
                        st.success("Done!")
                        st.rerun()

            with action_cols[1]:
                if st.button(
                    "📤 Index to Qdrant",
                    width="stretch",
                    disabled=not status["gemini"],
                ):
                    code = run_pipeline_command(
                        [
                            "--index",
                            "--recording-id",
                            rec.recording_id,
                            "--limit",
                            "50",
                        ],
                        f"Indexing {rec.recording_id[:16]}...",
                    )
                    if code == 0:
                        st.success("Done!")
                        st.rerun()

            with action_cols[2]:
                if st.button(
                    "🔄 Force Reprocess",
                    width="stretch",
                    disabled=not status["gemini"],
                ):
                    code = run_pipeline_command(
                        [
                            "--process",
                            "--recording-id",
                            rec.recording_id,
                            "--force",
                            "--limit",
                            "1",
                        ],
                        f"Force reprocessing...",
                    )
                    if code == 0:
                        st.success("Done!")
                        st.rerun()

            with action_cols[3]:
                if st.button(
                    "📝 Fetch Transcript",
                    width="stretch",
                    disabled=not status["plaud"],
                ):
                    try:
                        client = PlaudClient()
                        file_details = client.get_recording(rec.recording_id)
                        for source in file_details.get("source_list", []):
                            if source.get("data_type") == "transaction":
                                segments = json.loads(source.get("data_content", "[]"))
                                transcript = " ".join(
                                    s.get("content", "") for s in segments
                                )
                                if transcript.strip():
                                    set_chronos_recording_transcript(
                                        db_session, rec.recording_id, transcript
                                    )
                                    st.success(f"Cached {len(transcript):,} chars")
                                    st.rerun()
                                break
                        else:
                            st.warning("No transcript found in Plaud")
                    except Exception as e:
                        st.error(f"Failed: {e}")


# ---------------------------------------------------------------------------
# PAGE: PIPELINE
# ---------------------------------------------------------------------------


def page_pipeline(settings, status: Dict[str, Any]):
    """Simple 3-step pipeline with advanced options."""
    st.header("⚡ Pipeline")
    st.markdown("Fetch → Process → Index — in three simple steps")

    # Simple mode
    st.subheader("Quick Pipeline")

    cols = st.columns(3)

    with cols[0]:
        st.markdown("### Step 1: Fetch")
        st.caption("Pull recordings from Plaud API")
        if st.button(
            "🔄 Fetch from Plaud",
            width="stretch",
            disabled=not status["plaud"],
            type="primary",
        ):
            code = run_pipeline_command(
                ["--ingest", "--limit", "25"], "Fetching recordings..."
            )
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with cols[1]:
        st.markdown("### Step 2: Process")
        st.caption("Clean transcripts with Gemini AI")
        if st.button(
            "🧠 Process with AI",
            width="stretch",
            disabled=not status["gemini"],
            type="primary",
        ):
            code = run_pipeline_command(
                ["--process", "--limit", "10"], "Processing transcripts..."
            )
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with cols[2]:
        st.markdown("### Step 3: Index")
        st.caption("Make searchable in Qdrant")
        if st.button(
            "📤 Index to Qdrant",
            width="stretch",
            disabled=not (status["gemini"] and status["qdrant"]),
            type="primary",
        ):
            code = run_pipeline_command(
                ["--index", "--limit", "50"], "Indexing events..."
            )
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    st.markdown("---")

    # Full pipeline
    st.subheader("Full Pipeline")
    if st.button(
        "🚀 Run All Steps",
        width="stretch",
        disabled=not (status["plaud"] and status["gemini"] and status["qdrant"]),
    ):
        code = run_pipeline_command(
            ["--full", "--limit", "10"], "Running full pipeline..."
        )
        st.success("Complete!" if code == 0 else f"Failed (code {code})")

    # Advanced options (collapsed)
    with st.expander("⚙️ Advanced Options"):
        st.markdown("### Custom Limits")
        adv_cols = st.columns(4)
        with adv_cols[0]:
            ingest_limit = st.number_input("Ingest limit", 1, 500, 25)
        with adv_cols[1]:
            process_limit = st.number_input("Process limit", 1, 500, 10)
        with adv_cols[2]:
            index_limit = st.number_input("Index limit", 1, 500, 50)
        with adv_cols[3]:
            graph_limit = st.number_input("Graph limit", 1, 500, 25)

        st.markdown("### Single Recording Override")
        recording_id = st.text_input(
            "Recording ID (optional)", placeholder="plaud_recording_xxx"
        )
        force = st.checkbox("Force reprocess (delete existing events)")
        fetch_all = st.checkbox("Fetch ALL recordings (slow, first-time only)")

        st.markdown("### Custom Commands")

        cmd_cols = st.columns(4)
        with cmd_cols[0]:
            if st.button("Custom Ingest", width="stretch"):
                args = ["--ingest", "--limit", str(ingest_limit)]
                if fetch_all:
                    args.append("--fetch-all")
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Ingest")

        with cmd_cols[1]:
            if st.button("Custom Process", width="stretch"):
                args = ["--process", "--limit", str(process_limit)]
                if recording_id:
                    args += ["--recording-id", recording_id]
                if force:
                    args.append("--force")
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Process")

        with cmd_cols[2]:
            if st.button("Custom Index", width="stretch"):
                args = ["--index", "--limit", str(index_limit)]
                if recording_id:
                    args += ["--recording-id", recording_id]
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Index")

        with cmd_cols[3]:
            if st.button("Custom Graph", width="stretch"):
                args = ["--graph", "--limit", str(graph_limit)]
                if recording_id:
                    args += ["--recording-id", recording_id]
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Graph")

        st.markdown("### Diagnostics")
        diag_cols = st.columns(2)
        with diag_cols[0]:
            if st.button("Preflight Check", width="stretch"):
                code = run_pipeline_command(["--preflight"], "Preflight")
        with diag_cols[1]:
            if st.button("Smoke Test (with API call)", width="stretch"):
                code = run_pipeline_command(["--preflight-smoke"], "Smoke Test")


# ---------------------------------------------------------------------------
# PAGE: PLAUD
# ---------------------------------------------------------------------------


def page_plaud(settings, status: Dict[str, Any]):
    """Consolidated Plaud integration page with deep device integration."""
    st.header("📱 Plaud Integration")

    # Check if we have any Plaud config (OAuth or API token)
    if not status["plaud"]:
        st.error("Plaud not configured.")

        st.markdown(
            """
        ### 🔐 Setup Required

        You need to configure Plaud OAuth to fetch recordings.

        **Option 1: Click the button below** (opens browser for OAuth)
        """
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🔐 Authenticate with Plaud", type="primary", width="stretch"):
                try:
                    from src.plaud_oauth import PlaudOAuthClient

                    oauth = PlaudOAuthClient()
                    st.info("🌐 Opening browser for authentication...")
                    st.warning(
                        "After authenticating in the browser, come back here and refresh the page."
                    )

                    # Run in a subprocess so it doesn't block Streamlit
                    import threading

                    def run_auth():
                        try:
                            oauth.authenticate_interactive()
                        except Exception as e:
                            print(f"Auth error: {e}")

                    thread = threading.Thread(target=run_auth, daemon=True)
                    thread.start()

                except Exception as e:
                    st.error(f"Failed to start OAuth: {e}")
                    st.code(str(e))

        st.markdown(
            """
        **Option 2: Run from terminal**
        ```bash
        python plaud_setup.py
        ```

        After setup, you'll be able to:
        - Auto-detect devices when plugged in via USB
        - Monitor device battery, storage, and sync status
        - Automatically sync recordings to Chronos
        """
        )
        return

    # Connection check - also handle token refresh
    try:
        client = PlaudClient()
        user = client.get_user_info()
        st.success(
            f"✅ Connected as **{user.get('email', user.get('id', 'Unknown'))}**"
        )
    except Exception as e:
        st.warning(f"API connection issue: {e}")

        # Offer re-authentication
        if st.button("🔄 Re-authenticate with Plaud"):
            try:
                from src.plaud_oauth import PlaudOAuthClient
                import threading

                oauth = PlaudOAuthClient()
                st.info("🌐 Opening browser for re-authentication...")

                def run_auth():
                    try:
                        oauth.authenticate_interactive()
                    except Exception as ex:
                        print(f"Auth error: {ex}")

                thread = threading.Thread(target=run_auth, daemon=True)
                thread.start()
                st.warning("After authenticating, refresh this page.")
            except Exception as ex:
                st.error(f"Failed: {ex}")

        st.info("USB detection and local features will still work.")
        user = None

    # Main tabs - now with Device Integration as the first tab
    tab_devices, tab_overview, tab_workflows, tab_webhooks = st.tabs(
        ["📱 Devices", "📊 Overview", "🔄 Workflows", "🔔 Webhooks"]
    )

    with tab_devices:
        # Use the new deep device integration panel
        from gui.components.device_integration import render_device_integration

        render_device_integration()

    with tab_overview:
        st.subheader("Quick Stats")
        stats_cols = st.columns(4)

        try:
            from src.plaud_admin import PlaudAdminClient

            admin = PlaudAdminClient(plaud_client=client)

            devices = admin.list_devices()
            webhooks = admin.list_webhooks()

            with stats_cols[0]:
                st.metric("API Devices", len(devices))
            with stats_cols[1]:
                st.metric("Webhooks", len(webhooks))
            with stats_cols[2]:
                st.metric("Recordings (local)", status["recordings_count"])
            with stats_cols[3]:
                st.metric("Pending", status["pending_count"])
        except Exception as e:
            st.warning(f"Could not load stats: {e}")

        st.markdown("---")
        st.subheader("Quick Actions")
        quick_cols = st.columns(3)
        with quick_cols[0]:
            if st.button("🔄 Fetch Latest Recordings", width="stretch"):
                code = run_pipeline_command(
                    ["--ingest", "--limit", "25"], "Fetching..."
                )
                st.rerun() if code == 0 else None
        with quick_cols[1]:
            if st.button("📜 Get Recording Stats", width="stretch"):
                try:
                    stats = client.get_recording_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"Failed: {e}")
        with quick_cols[2]:
            if st.button("👤 User Info", width="stretch"):
                if user:
                    st.json(user)
                else:
                    st.warning("User info not available")

    with tab_workflows:
        from gui.components.workflow_panel import render_workflow_panel

        render_workflow_panel()

    with tab_webhooks:
        from gui.components.webhook_panel import render_webhook_panel

        render_webhook_panel()


# ---------------------------------------------------------------------------
# PAGE: TIMELINE (ULTIMATE Qdrant-Powered Timeline)
# ---------------------------------------------------------------------------


def page_timeline(settings, status: Dict[str, Any]):
    """
    🔮 ULTIMATE Timeline — Everything from Qdrant exposed.

    Features:
    - Direct Qdrant collection stats and schema
    - Full payload field exploration
    - Advanced filtering with all Qdrant operators
    - Scroll API for bulk data access
    - Faceting by any indexed field
    - Similarity search from timeline
    - Raw query builder
    """
    st.header("📅 Timeline & Qdrant Explorer")
    st.markdown(
        '<p class="subtitle">Your complete knowledge timeline with full Qdrant access</p>',
        unsafe_allow_html=True,
    )

    # Initialize connections
    init_db()
    session = SessionLocal()

    qdrant_client = None
    qdrant_stats = None
    if status["qdrant"]:
        try:
            qdrant_client = ChronosQdrantClient()
            qdrant_stats = qdrant_client.get_stats()
        except Exception as ex:
            st.error(f"Qdrant connection error: {ex}")

    try:
        # ═══════════════════════════════════════════════════════════════
        # QDRANT COLLECTION OVERVIEW
        # ═══════════════════════════════════════════════════════════════
        if qdrant_stats:
            st.markdown("### 🗄️ Qdrant Collection Overview")

            stat_cols = st.columns(5)
            stat_cols[0].metric("📊 Collection", qdrant_stats["collection_name"])
            stat_cols[1].metric("🔢 Total Points", f"{qdrant_stats['points_count']:,}")
            stat_cols[2].metric("📐 Vectors", f"{qdrant_stats['vectors_count']:,}")
            stat_cols[3].metric(
                "🔍 Indexed", f"{qdrant_stats['indexed_vectors_count']:,}"
            )
            stat_cols[4].metric("🟢 Status", qdrant_stats["status"])

            # Get collection info for schema
            try:
                collection_info = qdrant_client.client.get_collection(
                    qdrant_client.collection_name
                )

                with st.expander("🔧 Collection Configuration", expanded=False):
                    config_cols = st.columns(2)

                    with config_cols[0]:
                        st.markdown("**Vector Config:**")
                        if hasattr(collection_info.config, "params"):
                            params = collection_info.config.params
                            st.code(f"size: {getattr(params, 'vectors', {})}")
                        st.code(
                            f"optimizer_status: {getattr(collection_info, 'optimizer_status', 'N/A')}"
                        )

                    with config_cols[1]:
                        st.markdown("**Payload Schema (Indexed Fields):**")
                        if (
                            hasattr(collection_info, "payload_schema")
                            and collection_info.payload_schema
                        ):
                            for (
                                field_name,
                                field_info,
                            ) in collection_info.payload_schema.items():
                                data_type = getattr(field_info, "data_type", "unknown")
                                st.code(f"{field_name}: {data_type}")
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════
        # MAIN TABS
        # ═══════════════════════════════════════════════════════════════
        tab_explore, tab_heatmap, tab_query, tab_facets, tab_raw = st.tabs(
            [
                "🔍 Explore",
                "🔥 Heatmap",
                "🎯 Query Builder",
                "📊 Facets",
                "🔧 Raw Access",
            ]
        )

        # Load data from SQLite for filtering UI
        events_db = (
            session.query(ChronosEventDB).order_by(ChronosEventDB.start_ts.desc()).all()
        )
        recordings_db = session.query(ChronosRecordingDB).all()
        rec_lookup = {r.recording_id: r for r in recordings_db}

        # Detect sessions for filtering
        detected_sessions = detect_sessions(recordings_db, gap_threshold_minutes=15)
        session_lookup = {}  # recording_id -> session_id
        for sess in detected_sessions:
            for rec in sess.recordings:
                session_lookup[rec.recording_id] = sess

        # DYNAMIC CATEGORY COLORS — Auto-generated for ANY category
        all_cats_in_data = sorted(set(e.category or "unknown" for e in events_db))
        CATEGORY_COLORS = get_dynamic_category_colors(all_cats_in_data)

        # ─────────────────────────────────────────────────────────────────
        # TAB: EXPLORE — Timeline with full filters
        # ─────────────────────────────────────────────────────────────────
        with tab_explore:
            if not events_db:
                st.info("No events yet. Process recordings first.")
            else:
                st.markdown("### 🎛️ Filters")

                # Convert events to dicts with session info
                events_data = []
                for e in events_db:
                    sess = session_lookup.get(e.recording_id)
                    events_data.append(
                        {
                            "event_id": e.event_id,
                            "recording_id": e.recording_id,
                            "start_ts": e.start_ts,
                            "end_ts": e.end_ts,
                            "day_of_week": e.day_of_week,
                            "hour_of_day": e.hour_of_day,
                            "clean_text": e.clean_text,
                            "category": e.category or "unknown",
                            "sentiment": e.sentiment or 0.0,
                            "keywords": e.keywords or [],
                            "speaker": e.speaker or "self_talk",
                            "qdrant_point_id": e.qdrant_point_id,
                            "session_id": sess.session_id if sess else None,
                            "session_date": sess.date_str if sess else None,
                        }
                    )

                # Session filter toggle
                session_filter_enabled = st.checkbox(
                    "🗂️ Filter by Session",
                    value=False,
                    help="Group recordings by session (for split recordings)",
                )

                if session_filter_enabled and detected_sessions:
                    session_options = [
                        f"{s.date_str} | {s.time_range_str} | {s.duration_str} ({s.recording_count} recs)"
                        for s in detected_sessions
                    ]
                    selected_session_idx = st.selectbox(
                        "Select Session",
                        range(len(session_options)),
                        format_func=lambda i: session_options[i],
                        key="timeline_session_selector",
                    )
                    selected_session = detected_sessions[selected_session_idx]
                    session_recording_ids = {
                        r.recording_id for r in selected_session.recordings
                    }

                    # Filter events to selected session
                    events_data = [
                        e
                        for e in events_data
                        if e["recording_id"] in session_recording_ids
                    ]

                    st.success(
                        f"📌 Viewing session: **{selected_session.date_str}** | "
                        f"{selected_session.duration_str} | "
                        f"{selected_session.recording_count} recordings | "
                        f"{len(events_data)} events"
                    )

                if not events_data:
                    st.info("No events match the current filters.")
                    return

                # Filter row 1 - Main filters
                filter_cols = st.columns([2, 2, 2, 2])

                with filter_cols[0]:
                    min_date = min(e["start_ts"].date() for e in events_data)
                    max_date = max(e["start_ts"].date() for e in events_data)
                    date_range = st.date_input(
                        "📅 Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                    )
                    start_date, end_date = (
                        date_range
                        if isinstance(date_range, tuple) and len(date_range) == 2
                        else (min_date, max_date)
                    )

                with filter_cols[1]:
                    all_categories = sorted(set(e["category"] for e in events_data))
                    selected_categories = st.multiselect(
                        f"📁 Categories ({len(all_categories)})",
                        all_categories,
                        default=all_categories,
                    )

                with filter_cols[2]:
                    # Dynamic days - only show days that have data
                    all_days_in_data = sorted(
                        set(e["day_of_week"] for e in events_data if e["day_of_week"]),
                        key=lambda d: (
                            [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ].index(d)
                            if d
                            in [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ]
                            else 99
                        ),
                    )
                    selected_days = st.multiselect(
                        f"📆 Days ({len(all_days_in_data)})",
                        all_days_in_data,
                        default=all_days_in_data,
                    )

                with filter_cols[3]:
                    # Dynamic speakers
                    all_speakers_in_data = sorted(
                        set(e["speaker"] for e in events_data if e.get("speaker"))
                    )
                    if all_speakers_in_data:
                        selected_speakers = st.multiselect(
                            f"🎤 Speakers ({len(all_speakers_in_data)})",
                            all_speakers_in_data,
                            default=all_speakers_in_data,
                        )
                    else:
                        selected_speakers = []

                # Filter row 2 - Ranges
                range_cols = st.columns([1, 1, 1])

                with range_cols[0]:
                    # Dynamic hour range based on data
                    min_hour = min(e["hour_of_day"] for e in events_data)
                    max_hour = max(e["hour_of_day"] for e in events_data)
                    hour_range = st.slider("⏰ Hours", 0, 23, (min_hour, max_hour))

                with range_cols[1]:
                    # Dynamic sentiment range
                    sentiments_in_data = [
                        e["sentiment"]
                        for e in events_data
                        if e["sentiment"] is not None
                    ]
                    if sentiments_in_data:
                        min_sent = min(sentiments_in_data)
                        max_sent = max(sentiments_in_data)
                    else:
                        min_sent, max_sent = -1.0, 1.0
                    sentiment_range = st.slider(
                        "💭 Sentiment", -1.0, 1.0, (min_sent, max_sent)
                    )

                with range_cols[2]:
                    # Quick stats about filter state
                    st.markdown(f"**Data range:** {min_date} to {max_date}")
                    st.markdown(f"**Hours with data:** {min_hour}:00 - {max_hour}:00")

                # Apply filters (including speaker if available)
                filtered_events = [
                    e
                    for e in events_data
                    if (
                        e["start_ts"].date() >= start_date
                        and e["start_ts"].date() <= end_date
                    )
                    and e["category"] in selected_categories
                    and e["day_of_week"] in selected_days
                    and (not selected_speakers or e.get("speaker") in selected_speakers)
                    and e["hour_of_day"] >= hour_range[0]
                    and e["hour_of_day"] <= hour_range[1]
                    and (e["sentiment"] or 0) >= sentiment_range[0]
                    and (e["sentiment"] or 0) <= sentiment_range[1]
                ]

                st.markdown(
                    f"**Showing {len(filtered_events):,} of {len(events_data):,} events** "
                    f"({len(all_categories)} categories, {len(all_days_in_data)} days, {len(all_speakers_in_data)} speakers)"
                )

                # Summary stats
                if filtered_events:
                    stat_cols = st.columns(6)
                    categories = Counter(e["category"] for e in filtered_events)
                    sentiments = [
                        e["sentiment"] for e in filtered_events if e["sentiment"]
                    ]
                    avg_sentiment = (
                        sum(sentiments) / len(sentiments) if sentiments else 0
                    )
                    total_duration = sum(
                        (e["end_ts"] - e["start_ts"]).total_seconds()
                        for e in filtered_events
                    )

                    stat_cols[0].metric("🎯 Events", len(filtered_events))
                    stat_cols[1].metric("📁 Categories", len(categories))
                    stat_cols[2].metric("💭 Avg Sentiment", f"{avg_sentiment:.2f}")
                    stat_cols[3].metric(
                        "🔑 Keywords", sum(len(e["keywords"]) for e in filtered_events)
                    )
                    stat_cols[4].metric("⏱️ Duration", f"{total_duration / 3600:.1f}h")
                    stat_cols[5].metric(
                        "📊 In Qdrant",
                        sum(1 for e in filtered_events if e["qdrant_point_id"]),
                    )

                # Timeline visualization
                st.markdown("### 📊 Daily Event Distribution")

                import pandas as pd
                from collections import defaultdict

                daily_counts = defaultdict(lambda: defaultdict(int))
                for e in filtered_events:
                    date_str = e["start_ts"].strftime("%Y-%m-%d")
                    daily_counts[date_str][e["category"]] += 1

                if daily_counts:
                    dates = sorted(daily_counts.keys())
                    chart_data = []
                    for date in dates:
                        row = {"date": date}
                        for cat in selected_categories:
                            row[cat] = daily_counts[date].get(cat, 0)
                        chart_data.append(row)

                    df = pd.DataFrame(chart_data)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")

                    st.bar_chart(df, height=350)

                # Event list with full payload
                st.markdown("### 📋 Event Details")

                # Search within
                search_text = st.text_input(
                    "🔍 Search within events", placeholder="keyword..."
                )

                display_events = filtered_events
                if search_text:
                    display_events = [
                        e
                        for e in filtered_events
                        if search_text.lower() in e["clean_text"].lower()
                        or search_text.lower() in " ".join(e["keywords"]).lower()
                    ]
                    st.info(
                        f"Found {len(display_events)} events matching '{search_text}'"
                    )

                # Pagination
                events_per_page = 20
                total_pages = max(
                    1, (len(display_events) + events_per_page - 1) // events_per_page
                )
                page_num = st.number_input("Page", 1, total_pages, 1)

                start_idx = (page_num - 1) * events_per_page
                page_events = display_events[start_idx : start_idx + events_per_page]

                for i, e in enumerate(page_events):
                    cat_color = CATEGORY_COLORS.get(e["category"], "#7F8C8D")
                    time_str = e["start_ts"].strftime("%Y-%m-%d %H:%M")
                    duration = (e["end_ts"] - e["start_ts"]).total_seconds()
                    rec = rec_lookup.get(e["recording_id"])
                    rec_title = (
                        rec.title if rec and rec.title else e["recording_id"][:16]
                    )

                    with st.expander(
                        f"{time_str} | {e['category']} | {e['clean_text'][:60]}...",
                        expanded=False,
                    ):
                        ev_cols = st.columns([3, 1])

                        with ev_cols[0]:
                            st.markdown(f"**Full Text:**\n{e['clean_text']}")
                            if e["keywords"]:
                                st.markdown(f"**Keywords:** {', '.join(e['keywords'])}")

                        with ev_cols[1]:
                            st.markdown("**Metadata:**")
                            st.code(f"event_id: {e['event_id'][:16]}...")
                            st.code(f"category: {e['category']}")
                            st.code(f"speaker: {e['speaker']}")
                            st.code(f"sentiment: {e['sentiment']:.3f}")
                            st.code(f"day_of_week: {e['day_of_week']}")
                            st.code(f"hour_of_day: {e['hour_of_day']}")
                            st.code(f"duration: {duration:.0f}s")
                            st.code(f"recording: {rec_title}")
                            if e["qdrant_point_id"]:
                                st.code(f"qdrant_id: {e['qdrant_point_id'][:16]}...")

                st.caption(
                    f"Page {page_num} of {total_pages} ({len(display_events)} events)"
                )

        # ─────────────────────────────────────────────────────────────────
        # TAB: HEATMAP — Temporal patterns
        # ─────────────────────────────────────────────────────────────────
        with tab_heatmap:
            if not events_db:
                st.info("No events yet.")
            else:
                st.markdown("### 🔥 Activity Heatmap")
                st.markdown("*When do you record? Discover your temporal patterns.*")

                import pandas as pd
                import numpy as np

                # Use filtered_events from Explore tab if available
                events_for_heatmap = (
                    events_data if not "filtered_events" in dir() else events_data
                )

                # Dynamic day order based on data
                day_order = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                days_with_data = set(
                    e["day_of_week"] for e in events_for_heatmap if e["day_of_week"]
                )

                heatmap_data = np.zeros((24, 7))

                for e in events_for_heatmap:
                    day_idx = (
                        day_order.index(e["day_of_week"])
                        if e["day_of_week"] in day_order
                        else 0
                    )
                    hour_idx = e["hour_of_day"]
                    heatmap_data[hour_idx, day_idx] += 1

                # Plotly heatmap
                try:
                    import plotly.express as px

                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(
                            x="Day of Week", y="Hour of Day", color="Event Count"
                        ),
                        x=day_order,
                        y=[f"{h:02d}:00" for h in range(24)],
                        color_continuous_scale="Viridis",
                        aspect="auto",
                    )
                    fig.update_layout(
                        height=600,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cdd6f4"),
                    )
                    st.plotly_chart(fig, width="stretch")
                except ImportError:
                    heatmap_df = pd.DataFrame(
                        heatmap_data,
                        index=[f"{h:02d}:00" for h in range(24)],
                        columns=day_order,
                    )
                    st.dataframe(
                        heatmap_df.style.background_gradient(cmap="viridis", axis=None)
                    )

                # Summary stats
                stat_cols = st.columns(4)
                busiest_hour = int(heatmap_data.sum(axis=1).argmax())
                busiest_day_idx = int(heatmap_data.sum(axis=0).argmax())
                total_events = len(events_for_heatmap)
                total_days = len(set(e["start_ts"].date() for e in events_for_heatmap))

                stat_cols[0].metric("🔥 Busiest Hour", f"{busiest_hour:02d}:00")
                stat_cols[1].metric("📅 Busiest Day", day_order[busiest_day_idx])
                stat_cols[2].metric(
                    "📊 Avg Events/Day", f"{total_events / max(total_days, 1):.1f}"
                )
                stat_cols[3].metric("⚡ Peak Hour Count", int(heatmap_data.max()))

        # ─────────────────────────────────────────────────────────────────
        # TAB: QUERY BUILDER — Advanced Qdrant filtering
        # ─────────────────────────────────────────────────────────────────
        with tab_query:
            st.markdown("### 🎯 Qdrant Query Builder")
            st.markdown("*Build complex queries using Qdrant's full filter syntax*")

            if not qdrant_client:
                st.warning("Qdrant not connected.")
            else:
                # Get dynamic values from Qdrant
                dynamic_categories = get_all_unique_categories()
                dynamic_days = get_all_unique_days()
                dynamic_speakers = get_all_unique_speakers()

                # Show what's available
                with st.expander(
                    "📊 Available Filter Values (from your data)", expanded=False
                ):
                    info_cols = st.columns(4)
                    info_cols[0].metric("Categories", len(dynamic_categories))
                    info_cols[1].metric("Days", len(dynamic_days))
                    info_cols[2].metric("Speakers", len(dynamic_speakers))
                    info_cols[3].metric(
                        "Hours",
                        f"{get_all_unique_hours()[0]}-{get_all_unique_hours()[1]}",
                    )

                query_cols = st.columns([1, 1])

                with query_cols[0]:
                    st.markdown("**Semantic Search (optional)**")
                    semantic_query = st.text_input(
                        "Query text", placeholder="Find events about..."
                    )

                    st.markdown("**Category Filter**")
                    query_categories = st.multiselect(
                        "Categories",
                        dynamic_categories,
                        help=f"Found {len(dynamic_categories)} unique categories in your data",
                    )

                    st.markdown("**Day of Week Filter**")
                    query_days = st.multiselect(
                        "Days",
                        dynamic_days,
                        help=f"Found {len(dynamic_days)} unique days in your data",
                    )

                    st.markdown("**Speaker Filter**")
                    query_speakers = st.multiselect(
                        "Speakers",
                        dynamic_speakers,
                        help=f"Found {len(dynamic_speakers)} unique speakers",
                    )

                with query_cols[1]:
                    st.markdown("**Hour Range**")
                    min_hour, max_hour = get_all_unique_hours()
                    query_hour_range = st.slider(
                        "Hour range", 0, 23, (min_hour, max_hour), key="query_hours"
                    )

                    st.markdown("**Sentiment Range**")
                    query_sentiment = st.slider(
                        "Sentiment", -1.0, 1.0, (-1.0, 1.0), key="query_sentiment"
                    )

                    st.markdown("**Result Limit**")
                    query_limit = st.number_input("Max results", 1, 1000, 50)

                if st.button("🔍 Execute Query", type="primary"):
                    from qdrant_client.models import (
                        Filter,
                        FieldCondition,
                        MatchAny,
                        MatchValue,
                        Range,
                    )

                    # Build filter
                    must_conditions = []

                    if query_categories:
                        must_conditions.append(
                            FieldCondition(
                                key="category", match=MatchAny(any=query_categories)
                            )
                        )

                    if query_days:
                        must_conditions.append(
                            FieldCondition(
                                key="day_of_week", match=MatchAny(any=query_days)
                            )
                        )

                    if query_speakers:
                        must_conditions.append(
                            FieldCondition(
                                key="speaker", match=MatchAny(any=query_speakers)
                            )
                        )

                    if query_hour_range != (min_hour, max_hour):
                        must_conditions.append(
                            FieldCondition(
                                key="hour_of_day",
                                range=Range(
                                    gte=query_hour_range[0], lte=query_hour_range[1]
                                ),
                            )
                        )

                    if query_sentiment != (-1.0, 1.0):
                        must_conditions.append(
                            FieldCondition(
                                key="sentiment",
                                range=Range(
                                    gte=query_sentiment[0], lte=query_sentiment[1]
                                ),
                            )
                        )

                    query_filter = (
                        Filter(must=must_conditions) if must_conditions else None
                    )

                    try:
                        if semantic_query:
                            # Semantic search with filter
                            embedding_service = ChronosEmbeddingService()
                            query_vector = embedding_service.embed_text(semantic_query)

                            results = qdrant_client.client.query_points(
                                collection_name=qdrant_client.collection_name,
                                query=query_vector,
                                query_filter=query_filter,
                                limit=query_limit,
                                with_payload=True,
                            ).points
                        else:
                            # Filter-only scroll
                            results, _ = qdrant_client.client.scroll(
                                collection_name=qdrant_client.collection_name,
                                scroll_filter=query_filter,
                                limit=query_limit,
                                with_payload=True,
                            )

                        st.success(f"Found {len(results)} results")

                        # Display results
                        for r in results:
                            score = getattr(r, "score", None)
                            score_txt = f" | Score: {score:.4f}" if score else ""

                            with st.expander(
                                f"{r.payload.get('category', '?')} | {r.payload.get('clean_text', '')[:60]}...{score_txt}"
                            ):
                                st.json(r.payload)

                    except Exception as ex:
                        st.error(f"Query error: {ex}")

        # ─────────────────────────────────────────────────────────────────
        # TAB: FACETS — Qdrant faceting (DYNAMIC)
        # ─────────────────────────────────────────────────────────────────
        with tab_facets:
            st.markdown("### 📊 Field Facets (Live from Qdrant)")
            st.markdown(
                "*Real-time value distributions across ALL indexed payload fields*"
            )

            if not qdrant_client:
                st.warning("Qdrant not connected.")
            else:
                # Use the cached stats function for efficiency
                field_stats = get_collection_field_stats()
                total_points = field_stats["total_points"]

                st.metric("📊 Total Indexed Points", f"{total_points:,}")
                st.markdown("---")

                facet_col1, facet_col2 = st.columns(2)

                with facet_col1:
                    st.markdown("#### 📁 Category Distribution")
                    category_counts = field_stats.get("categories", {})
                    if category_counts:
                        cat_colors = get_dynamic_category_colors(
                            list(category_counts.keys())
                        )
                        for cat, count in sorted(
                            category_counts.items(), key=lambda x: -x[1]
                        ):
                            pct = count / total_points * 100 if total_points else 0
                            color = cat_colors.get(cat, "#7F8C8D")
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                                f'<div style="width:12px;height:12px;background:{color};border-radius:2px;"></div>'
                                f'<div style="flex:1;">{cat}</div>'
                                f'<div style="opacity:0.7;">{count:,} ({pct:.1f}%)</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No category data yet")

                    st.markdown("---")
                    st.markdown("#### 🎤 Speaker Distribution")
                    speaker_counts = field_stats.get("speakers", {})
                    if speaker_counts:
                        for speaker, count in sorted(
                            speaker_counts.items(), key=lambda x: -x[1]
                        ):
                            pct = count / total_points * 100 if total_points else 0
                            st.markdown(f"**{speaker}**: {count:,} ({pct:.1f}%)")
                    else:
                        st.info("No speaker data yet")

                with facet_col2:
                    st.markdown("#### 📆 Day Distribution")
                    day_counts = field_stats.get("days", {})
                    if day_counts:
                        day_order = [
                            "Monday",
                            "Tuesday",
                            "Wednesday",
                            "Thursday",
                            "Friday",
                            "Saturday",
                            "Sunday",
                        ]
                        # Sort by weekday order
                        sorted_days = sorted(
                            day_counts.items(),
                            key=lambda x: (
                                day_order.index(x[0]) if x[0] in day_order else 99
                            ),
                        )
                        for day, count in sorted_days:
                            pct = count / total_points * 100 if total_points else 0
                            st.progress(
                                pct / 100, text=f"{day}: {count:,} ({pct:.1f}%)"
                            )
                    else:
                        st.info("No day data yet")

                    st.markdown("---")
                    st.markdown("#### ⏰ Hour Distribution")
                    hour_counts = field_stats.get("hours", {})
                    if hour_counts:
                        import pandas as pd

                        hour_df = pd.DataFrame(
                            [
                                {"hour": f"{h:02d}:00", "count": hour_counts.get(h, 0)}
                                for h in range(24)
                            ]
                        )
                        st.bar_chart(hour_df.set_index("hour"), height=200)
                    else:
                        st.info("No hour data yet")

                # Refresh button
                st.markdown("---")
                if st.button("🔄 Refresh Facets"):
                    get_collection_field_stats.clear()
                    st.rerun()

        # ─────────────────────────────────────────────────────────────────
        # TAB: RAW ACCESS — Direct Qdrant operations
        # ─────────────────────────────────────────────────────────────────
        with tab_raw:
            st.markdown("### 🔧 Raw Qdrant Access")
            st.warning("⚠️ Advanced — use with caution")

            if not qdrant_client:
                st.error("Qdrant not connected.")
            else:
                raw_tabs = st.tabs(
                    ["📥 Scroll", "🔍 Get Point", "🔢 Count", "📊 Random Sample"]
                )

                with raw_tabs[0]:
                    st.markdown("**Scroll through all points**")
                    scroll_limit = st.number_input(
                        "Limit", 1, 1000, 10, key="scroll_limit"
                    )

                    if st.button("Execute Scroll"):
                        points, next_offset = qdrant_client.client.scroll(
                            collection_name=qdrant_client.collection_name,
                            limit=scroll_limit,
                            with_payload=True,
                            with_vectors=False,
                        )
                        st.success(f"Retrieved {len(points)} points")

                        for p in points:
                            with st.expander(f"Point: {p.id}"):
                                st.json(p.payload)

                with raw_tabs[1]:
                    st.markdown("**Get point by ID**")
                    point_id = st.text_input("Point ID (UUID)")

                    if st.button("Get Point") and point_id:
                        try:
                            points = qdrant_client.client.retrieve(
                                collection_name=qdrant_client.collection_name,
                                ids=[point_id],
                                with_payload=True,
                                with_vectors=True,
                            )
                            if points:
                                st.json(
                                    {
                                        "id": str(points[0].id),
                                        "payload": points[0].payload,
                                        "vector_dims": (
                                            len(points[0].vector)
                                            if points[0].vector
                                            else 0
                                        ),
                                    }
                                )
                            else:
                                st.warning("Point not found")
                        except Exception as ex:
                            st.error(f"Error: {ex}")

                with raw_tabs[2]:
                    st.markdown("**Count points with filter**")
                    count_category = st.selectbox(
                        "Category",
                        [
                            "",
                            "work",
                            "personal",
                            "meeting",
                            "deep_work",
                            "break",
                            "reflection",
                            "idea",
                            "unknown",
                        ],
                        key="count_cat",
                    )

                    if st.button("Count"):
                        from qdrant_client.models import (
                            Filter,
                            FieldCondition,
                            MatchValue,
                        )

                        count_filter = None
                        if count_category:
                            count_filter = Filter(
                                must=[
                                    FieldCondition(
                                        key="category",
                                        match=MatchValue(value=count_category),
                                    )
                                ]
                            )

                        result = qdrant_client.client.count(
                            collection_name=qdrant_client.collection_name,
                            count_filter=count_filter,
                            exact=True,
                        )
                        st.metric("Count", result.count)

                with raw_tabs[3]:
                    st.markdown("**Random sample**")
                    sample_size = st.number_input(
                        "Sample size", 1, 100, 5, key="sample_size"
                    )

                    if st.button("Get Random Sample"):
                        try:
                            # Use random query
                            from qdrant_client.models import SampleQuery

                            results = qdrant_client.client.query_points(
                                collection_name=qdrant_client.collection_name,
                                query=SampleQuery(sample="random"),
                                limit=sample_size,
                                with_payload=True,
                            ).points

                            st.success(f"Got {len(results)} random points")
                            for r in results:
                                st.markdown(
                                    f"**{r.payload.get('category')}**: {r.payload.get('clean_text', '')[:100]}..."
                                )
                        except Exception as ex:
                            st.error(f"Error: {ex}")

    except Exception as ex:
        st.error(f"Error loading timeline: {ex}")
        import traceback

        st.code(traceback.format_exc())
    finally:
        session.close()


# ---------------------------------------------------------------------------
# PAGE: SETTINGS
# ---------------------------------------------------------------------------


def page_settings(settings, status: Dict[str, Any]):
    """Configuration and diagnostics."""
    st.header("⚙️ Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        config_data = {
            "GEMINI_API_KEY": "✓ Set" if status["gemini"] else "✗ Missing",
            "GEMINI_API_VERSION": settings.gemini_api_version,
            "CHRONOS_CLEANING_MODEL": settings.chronos_cleaning_model,
            "CHRONOS_EMBEDDING_MODEL": settings.chronos_embedding_model,
            "CHRONOS_ANALYST_MODEL": settings.chronos_analyst_model,
            "QDRANT_URL": settings.qdrant_url,
            "QDRANT_COLLECTION": settings.qdrant_collection_name,
            "PLAUD_OAUTH": "✓ Configured" if status["plaud"] else "✗ Not configured",
        }
        for key, val in config_data.items():
            st.markdown(f"**{key}:** `{val}`")

    with col2:
        st.subheader("System Status")
        st.metric("Qdrant", "Connected" if status["qdrant"] else "Disconnected")
        st.metric("Indexed Points", status["points_count"])
        st.metric("Local Recordings", status["recordings_count"])

        if st.button("Open Qdrant Dashboard"):
            st.markdown(
                f"[{settings.qdrant_url}/dashboard]({settings.qdrant_url}/dashboard)"
            )

    st.markdown("---")
    st.subheader("Commands Reference")
    st.code(
        """
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run pipeline
python scripts/chronos_pipeline.py --full --limit 10

# Launch UI
streamlit run chronos_app.py

# Setup Plaud OAuth
python plaud_setup.py

# Run tests
python -m pytest tests/ -q
""",
        language="bash",
    )

    st.markdown("---")
    st.subheader("Error Logs")
    logs = st.session_state.get("error_logs", [])
    if logs:
        st.text("\n".join(logs))
    else:
        st.info("No errors logged this session.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def render_statusbar(status: Dict[str, Any]):
    """Render the bottom status bar."""
    latency = st.session_state.get("last_latency_ms")
    latency_txt = f"{latency:.0f}ms" if latency else "—"

    qdrant_status = "OK" if status["qdrant"] else "DOWN"
    gemini_status = "OK" if status["gemini"] else "N/A"

    st.markdown(
        f"""
    <div class="statusbar">
        <b>Status</b> ·
        Qdrant: <code>{qdrant_status}</code> ·
        Gemini: <code>{gemini_status}</code> ·
        Points: <code>{status["points_count"]}</code> ·
        Latency: <code>{latency_txt}</code>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application entry point."""
    settings = get_settings()
    init_db()

    # Get system status
    status = get_system_status(settings)

    # Sidebar navigation — SINGLE CLEAN LIST
    with st.sidebar:
        st.markdown("## 🕰️ Chronos")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            [
                "🏠 Home",
                "� Timeline",
                "�🔍 Search",
                "📚 Library",
                "⚡ Pipeline",
                "📱 Plaud",
                "⚙️ Settings",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick status in sidebar
        st.markdown("### Status")
        if status["qdrant"]:
            st.markdown("✅ Qdrant connected")
        else:
            st.markdown("❌ Qdrant offline")
        if status["gemini"]:
            st.markdown("✅ Gemini ready")
        else:
            st.markdown("⚠️ No Gemini key")
        if status["plaud"]:
            st.markdown("✅ Plaud configured")
        else:
            st.markdown("⚠️ No Plaud OAuth")

        st.markdown("---")
        st.caption("Chronos v2.2.0")

    # Route to page
    if page == "🏠 Home":
        page_home(settings, status)
    elif page == "� Timeline":
        page_timeline(settings, status)
    elif page == "�🔍 Search":
        page_search(settings, status)
    elif page == "📚 Library":
        page_library(settings, status)
    elif page == "⚡ Pipeline":
        page_pipeline(settings, status)
    elif page == "📱 Plaud":
        page_plaud(settings, status)
    elif page == "⚙️ Settings":
        page_settings(settings, status)

    # Status bar
    render_statusbar(status)


def _running_in_streamlit() -> bool:
    """Check if running under streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not _running_in_streamlit():
        print("Run with: streamlit run chronos_app.py", file=sys.stderr)
        raise SystemExit(1)
    main()
