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
    """Browse and manage all recordings."""
    st.header("📚 Recording Library")

    init_db()
    session = SessionLocal()

    try:
        # Filters
        filter_cols = st.columns([1, 1, 2])
        with filter_cols[0]:
            status_filter = st.multiselect(
                "Status",
                ["pending", "processing", "completed", "failed"],
                default=["pending", "completed"],
            )
        with filter_cols[1]:
            days_back = st.number_input("Days back", 1, 365, 30)
        with filter_cols[2]:
            search_q = st.text_input("Search", placeholder="title or ID...")

        # Query
        q = session.query(ChronosRecordingDB)
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
            )

        recs = q.order_by(ChronosRecordingDB.created_at.desc()).limit(200).all()

        if not recs:
            st.info("No recordings found. Go to Pipeline to fetch from Plaud.")
            return

        # Display as table
        rows = []
        for r in recs:
            event_count = (
                session.query(ChronosEventDB)
                .filter_by(recording_id=r.recording_id)
                .count()
            )
            rows.append(
                {
                    "ID": (
                        r.recording_id[:16] + "..."
                        if len(r.recording_id) > 16
                        else r.recording_id
                    ),
                    "Title": r.title or "—",
                    "Created": (
                        r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—"
                    ),
                    "Duration": (
                        f"{r.duration_seconds // 60}m" if r.duration_seconds else "—"
                    ),
                    "Status": r.processing_status,
                    "Events": event_count,
                    "Transcript": "✓" if r.transcript else "—",
                }
            )

        st.dataframe(rows, width="stretch", hide_index=True)

        # Detail view
        st.markdown("---")
        st.subheader("Recording Detail")

        options = [
            f"{r.title or r.recording_id[:20]} ({r.processing_status})" for r in recs
        ]
        selected_idx = st.selectbox(
            "Select recording", range(len(options)), format_func=lambda i: options[i]
        )

        if selected_idx is not None:
            rec = recs[selected_idx]

            # Metadata
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.markdown(f"**ID:** `{rec.recording_id}`")
                st.markdown(f"**Status:** {rec.processing_status}")
            with meta_cols[1]:
                st.markdown(f"**Created:** {rec.created_at}")
                st.markdown(f"**Duration:** {rec.duration_seconds}s")
            with meta_cols[2]:
                st.markdown(f"**Device:** {rec.device_id or 'Unknown'}")
                if rec.error_message:
                    st.error(f"Error: {rec.error_message}")

            # Actions
            action_cols = st.columns(4)
            with action_cols[0]:
                if st.button(
                    "🧠 Process",
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
                    "📤 Index", width="stretch", disabled=not status["gemini"]
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
                        # Extract transcript from source_list
                        import json

                        for source in file_details.get("source_list", []):
                            if source.get("data_type") == "transaction":
                                segments = json.loads(source.get("data_content", "[]"))
                                transcript = " ".join(
                                    s.get("content", "") for s in segments
                                )
                                if transcript.strip():
                                    set_chronos_recording_transcript(
                                        session, rec.recording_id, transcript
                                    )
                                    st.success(f"Cached {len(transcript):,} chars")
                                    st.rerun()
                                    break
                        else:
                            st.warning("No transcript found in Plaud")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            # Show transcript if available
            if rec.transcript:
                with st.expander("📜 Transcript", expanded=False):
                    st.text_area("Content", rec.transcript, height=200)

            # Show events
            events = (
                session.query(ChronosEventDB)
                .filter_by(recording_id=rec.recording_id)
                .order_by(ChronosEventDB.start_ts.asc())
                .all()
            )

            if events:
                with st.expander(f"🎯 Events ({len(events)})", expanded=True):
                    for e in events[:20]:
                        st.markdown(
                            f"**{e.category}** ({e.start_ts.strftime('%H:%M')}) — {e.clean_text[:200]}..."
                        )

    finally:
        session.close()


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
        st.error("Plaud not configured. Run `python plaud_setup.py` first.")
        st.info(
            "After setup, you'll be able to:\n"
            "- Auto-detect devices when plugged in via USB\n"
            "- Monitor device battery, storage, and sync status\n"
            "- Automatically sync recordings to Chronos"
        )
        return

    # Connection check
    try:
        client = PlaudClient()
        user = client.get_user_info()
        st.success(
            f"✅ Connected as **{user.get('email', user.get('id', 'Unknown'))}**"
        )
    except Exception as e:
        st.warning(f"API connection issue: {e}")
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
# PAGE: TIMELINE (Super Robust Visual Timeline)
# ---------------------------------------------------------------------------


def page_timeline(settings, status: Dict[str, Any]):
    """
    Beautiful, interactive timeline that shows EVERYTHING.

    Features:
    - Multi-scale timeline (zoom from months → hours)
    - Heatmap: day-of-week × hour patterns
    - Category color-coding
    - Sentiment overlay
    - Drill-down detail panel
    """
    st.header("📅 Timeline")
    st.markdown(
        '<p class="subtitle">Your complete cognitive timeline — zoom, filter, explore</p>',
        unsafe_allow_html=True,
    )

    init_db()
    session = SessionLocal()

    try:
        # Load all events
        events = (
            session.query(ChronosEventDB).order_by(ChronosEventDB.start_ts.desc()).all()
        )
        recordings = session.query(ChronosRecordingDB).all()

        if not events:
            st.info(
                "No events yet. Go to **Pipeline** to fetch and process recordings first."
            )
            return

        # Convert to dicts for processing
        events_data = []
        for e in events:
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
                }
            )

        # Build recordings lookup
        rec_lookup = {r.recording_id: r for r in recordings}

        # --------------- FILTERS ---------------
        st.markdown("### 🎛️ Filters")
        filter_cols = st.columns([2, 2, 2, 1])

        with filter_cols[0]:
            # Date range
            min_date = min(e["start_ts"].date() for e in events_data)
            max_date = max(e["start_ts"].date() for e in events_data)
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date

        with filter_cols[1]:
            # Categories
            all_categories = sorted(set(e["category"] for e in events_data))
            selected_categories = st.multiselect(
                "Categories",
                all_categories,
                default=all_categories,
            )

        with filter_cols[2]:
            # Days of week
            all_days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            selected_days = st.multiselect(
                "Days",
                all_days,
                default=all_days,
            )

        with filter_cols[3]:
            # Hour range
            hour_range = st.slider("Hours", 0, 23, (0, 23))

        # Apply filters
        filtered_events = [
            e
            for e in events_data
            if (e["start_ts"].date() >= start_date and e["start_ts"].date() <= end_date)
            and e["category"] in selected_categories
            and e["day_of_week"] in selected_days
            and e["hour_of_day"] >= hour_range[0]
            and e["hour_of_day"] <= hour_range[1]
        ]

        st.markdown(
            f"**Showing {len(filtered_events):,} of {len(events_data):,} events**"
        )

        # --------------- TABS ---------------
        tab_timeline, tab_heatmap, tab_categories, tab_sentiment, tab_list = st.tabs(
            ["📊 Timeline", "🔥 Heatmap", "📁 Categories", "💭 Sentiment", "📋 List"]
        )

        # Category colors
        CATEGORY_COLORS = {
            "work": "#4A90D9",
            "personal": "#9B59B6",
            "meeting": "#E67E22",
            "deep_work": "#2ECC71",
            "break": "#95A5A6",
            "reflection": "#1ABC9C",
            "idea": "#F1C40F",
            "unknown": "#7F8C8D",
        }

        # --------------- TAB: INTERACTIVE TIMELINE ---------------
        with tab_timeline:
            st.markdown("#### Interactive Timeline")
            st.markdown(
                "*Zoom with scroll wheel, pan by dragging, click items for details*"
            )

            # Generate vis-timeline items
            timeline_items = []
            for i, e in enumerate(filtered_events[:500]):  # Limit for performance
                color = CATEGORY_COLORS.get(e["category"], "#7F8C8D")
                # Truncate text for timeline display
                display_text = (
                    e["clean_text"][:80] + "..."
                    if len(e["clean_text"]) > 80
                    else e["clean_text"]
                )
                timeline_items.append(
                    {
                        "id": i,
                        "content": display_text.replace('"', '\\"').replace("\n", " "),
                        "start": e["start_ts"].isoformat(),
                        "end": (
                            e["end_ts"].isoformat()
                            if e["end_ts"] != e["start_ts"]
                            else None
                        ),
                        "group": e["category"],
                        "style": f"background-color: {color}; border-color: {color};",
                        "event_id": e["event_id"],
                    }
                )

            # Generate groups (categories)
            groups = [
                {
                    "id": cat,
                    "content": cat.replace("_", " ").title(),
                    "style": f"color: {CATEGORY_COLORS.get(cat, '#7F8C8D')}",
                }
                for cat in sorted(selected_categories)
            ]

            # Build the timeline HTML
            timeline_html = f"""
            <link rel="stylesheet" href="app/static/lib/vis-timeline/vis-timeline-graph2d.min.css">
            <script src="app/static/lib/vis-timeline/vis-timeline-graph2d.min.js"></script>
            <style>
                #timeline-container {{
                    width: 100%;
                    height: 500px;
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    background: rgba(30, 30, 46, 0.8);
                }}
                .vis-item {{
                    border-radius: 4px;
                    font-size: 11px;
                    padding: 2px 6px;
                }}
                .vis-item.vis-selected {{
                    border-width: 2px;
                    box-shadow: 0 0 10px rgba(255,255,255,0.3);
                }}
                .vis-label {{
                    color: #cdd6f4;
                    font-weight: 600;
                }}
                .vis-time-axis .vis-text {{
                    color: #a6adc8;
                }}
                .vis-panel.vis-background {{
                    background: rgba(30, 30, 46, 0.95);
                }}
            </style>
            <div id="timeline-container"></div>
            <script>
                var items = new vis.DataSet({json.dumps(timeline_items)});
                var groups = new vis.DataSet({json.dumps(groups)});
                var container = document.getElementById('timeline-container');
                var options = {{
                    height: '500px',
                    stack: true,
                    showCurrentTime: true,
                    zoomMin: 1000 * 60 * 60,  // 1 hour
                    zoomMax: 1000 * 60 * 60 * 24 * 365,  // 1 year
                    orientation: 'top',
                    groupOrder: 'content'
                }};
                var timeline = new vis.Timeline(container, items, groups, options);
            </script>
            """

            # Group events by date for a simple bar chart
            from collections import defaultdict

            daily_counts = defaultdict(lambda: defaultdict(int))
            for e in filtered_events:
                date_str = e["start_ts"].strftime("%Y-%m-%d")
                daily_counts[date_str][e["category"]] += 1

            if daily_counts:
                import pandas as pd

                # Create DataFrame for stacked bar
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

                st.bar_chart(df, width="stretch", height=400)

            # Simple day-by-day event timeline
            st.markdown("#### Daily Event Flow")

            # Group by date
            events_by_date = defaultdict(list)
            for e in filtered_events:
                date_str = e["start_ts"].strftime("%Y-%m-%d (%A)")
                events_by_date[date_str].append(e)

            # Show recent days with expandable details
            for date_str in sorted(events_by_date.keys(), reverse=True)[:14]:
                day_events = events_by_date[date_str]
                with st.expander(
                    f"📅 {date_str} — {len(day_events)} events", expanded=False
                ):
                    for e in sorted(day_events, key=lambda x: x["start_ts"]):
                        time_str = e["start_ts"].strftime("%H:%M")
                        cat_color = CATEGORY_COLORS.get(e["category"], "#7F8C8D")
                        sentiment_emoji = (
                            "😊"
                            if e["sentiment"] > 0.3
                            else "😐" if e["sentiment"] > -0.3 else "😔"
                        )

                        st.markdown(
                            f"""<div style="
                                border-left: 4px solid {cat_color};
                                padding: 8px 12px;
                                margin: 4px 0;
                                background: rgba(49, 50, 68, 0.6);
                                border-radius: 0 8px 8px 0;
                            ">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span><b>{time_str}</b> · <span style="color: {cat_color}">{e['category'].replace('_', ' ').title()}</span></span>
                                    <span>{sentiment_emoji} {e['sentiment']:.2f}</span>
                                </div>
                                <div style="margin-top: 6px; font-size: 0.9rem;">{e['clean_text'][:300]}{'...' if len(e['clean_text']) > 300 else ''}</div>
                                <div style="margin-top: 4px; opacity: 0.6; font-size: 0.75rem;">
                                    {', '.join(e['keywords'][:5]) if e['keywords'] else '—'}
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

        # --------------- TAB: HEATMAP ---------------
        with tab_heatmap:
            st.markdown("#### Activity Heatmap")
            st.markdown("*When do you record? Discover your temporal patterns.*")

            import pandas as pd
            import numpy as np

            # Build heatmap matrix: hour (0-23) × day (Mon-Sun)
            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            heatmap_data = np.zeros((24, 7))

            for e in filtered_events:
                day_idx = (
                    day_order.index(e["day_of_week"])
                    if e["day_of_week"] in day_order
                    else 0
                )
                hour_idx = e["hour_of_day"]
                heatmap_data[hour_idx, day_idx] += 1

            # Create DataFrame
            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=[f"{h:02d}:00" for h in range(24)],
                columns=day_order,
            )

            # Display as heatmap using Streamlit
            st.markdown("##### Events by Hour × Day of Week")

            # Use plotly for proper heatmap
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                fig = px.imshow(
                    heatmap_df.values,
                    labels=dict(x="Day of Week", y="Hour of Day", color="Event Count"),
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
                # Fallback to simple table with color coding
                st.dataframe(
                    heatmap_df.style.background_gradient(cmap="viridis", axis=None),
                    width="stretch",
                )

            # Summary stats
            stat_cols = st.columns(4)
            with stat_cols[0]:
                busiest_hour = heatmap_data.sum(axis=1).argmax()
                st.metric("🔥 Busiest Hour", f"{busiest_hour:02d}:00")
            with stat_cols[1]:
                busiest_day_idx = heatmap_data.sum(axis=0).argmax()
                st.metric("📅 Busiest Day", day_order[busiest_day_idx])
            with stat_cols[2]:
                total_events = len(filtered_events)
                total_days = len(set(e["start_ts"].date() for e in filtered_events))
                avg_per_day = total_events / max(total_days, 1)
                st.metric("📊 Avg Events/Day", f"{avg_per_day:.1f}")
            with stat_cols[3]:
                peak_count = int(heatmap_data.max())
                st.metric("⚡ Peak Hour Count", peak_count)

        # --------------- TAB: CATEGORIES ---------------
        with tab_categories:
            st.markdown("#### Category Distribution")

            import pandas as pd
            from collections import Counter

            # Count by category
            cat_counts = Counter(e["category"] for e in filtered_events)
            cat_df = pd.DataFrame(
                [
                    {
                        "Category": cat.replace("_", " ").title(),
                        "Count": count,
                        "color": CATEGORY_COLORS.get(cat, "#7F8C8D"),
                    }
                    for cat, count in cat_counts.most_common()
                ]
            )

            if not cat_df.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    try:
                        import plotly.express as px

                        fig = px.pie(
                            cat_df,
                            values="Count",
                            names="Category",
                            color="Category",
                            color_discrete_map={
                                cat.replace("_", " ").title(): CATEGORY_COLORS.get(
                                    cat, "#7F8C8D"
                                )
                                for cat in cat_counts.keys()
                            },
                            hole=0.4,
                        )
                        fig.update_layout(
                            height=400,
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#cdd6f4"),
                        )
                        st.plotly_chart(fig, width="stretch")
                    except ImportError:
                        st.bar_chart(cat_df.set_index("Category")["Count"])

                with col2:
                    st.markdown("##### Breakdown")
                    for _, row in cat_df.iterrows():
                        pct = row["Count"] / len(filtered_events) * 100
                        st.markdown(
                            f"**{row['Category']}**: {row['Count']} ({pct:.1f}%)"
                        )

            # Category timeline (stacked area)
            st.markdown("##### Category Trends Over Time")
            from collections import defaultdict

            cat_by_date = defaultdict(lambda: defaultdict(int))
            for e in filtered_events:
                date_str = e["start_ts"].strftime("%Y-%m-%d")
                cat_by_date[date_str][e["category"]] += 1

            if cat_by_date:
                dates = sorted(cat_by_date.keys())
                trend_data = []
                for date in dates:
                    row = {"date": date}
                    for cat in selected_categories:
                        row[cat.replace("_", " ").title()] = cat_by_date[date].get(
                            cat, 0
                        )
                    trend_data.append(row)

                trend_df = pd.DataFrame(trend_data)
                trend_df["date"] = pd.to_datetime(trend_df["date"])
                trend_df = trend_df.set_index("date")

                st.area_chart(trend_df, width="stretch", height=300)

        # --------------- TAB: SENTIMENT ---------------
        with tab_sentiment:
            st.markdown("#### Sentiment Analysis")
            st.markdown("*Track your emotional patterns over time*")

            import pandas as pd

            # Sentiment over time
            sent_data = [
                {
                    "date": e["start_ts"],
                    "sentiment": e["sentiment"],
                    "category": e["category"],
                    "text": e["clean_text"][:100],
                }
                for e in filtered_events
                if e["sentiment"] is not None
            ]

            if sent_data:
                sent_df = pd.DataFrame(sent_data)

                # Rolling average
                sent_df = sent_df.sort_values("date")
                sent_df["rolling_avg"] = (
                    sent_df["sentiment"].rolling(window=10, min_periods=1).mean()
                )

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown("##### Sentiment Timeline (Rolling Avg)")
                    chart_df = sent_df.set_index("date")[["sentiment", "rolling_avg"]]
                    st.line_chart(chart_df, width="stretch", height=300)

                with col2:
                    avg_sent = sent_df["sentiment"].mean()
                    emoji = (
                        "😊" if avg_sent > 0.2 else "😐" if avg_sent > -0.2 else "😔"
                    )
                    st.metric("Average Sentiment", f"{emoji} {avg_sent:.2f}")

                    pos_pct = (sent_df["sentiment"] > 0.2).sum() / len(sent_df) * 100
                    st.metric("Positive Events", f"{pos_pct:.1f}%")

                    neg_pct = (sent_df["sentiment"] < -0.2).sum() / len(sent_df) * 100
                    st.metric("Negative Events", f"{neg_pct:.1f}%")

                # Sentiment by category
                st.markdown("##### Sentiment by Category")
                cat_sent = (
                    sent_df.groupby("category")["sentiment"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                cat_sent.columns = ["Category", "Avg Sentiment", "Count"]
                cat_sent["Category"] = cat_sent["Category"].apply(
                    lambda x: x.replace("_", " ").title()
                )
                st.dataframe(cat_sent, width="stretch", hide_index=True)

                # Most positive/negative
                st.markdown("##### Extremes")
                extremes_col1, extremes_col2 = st.columns(2)

                with extremes_col1:
                    st.markdown("**Most Positive Events**")
                    top_pos = sent_df.nlargest(3, "sentiment")
                    for _, row in top_pos.iterrows():
                        st.success(f"😊 {row['sentiment']:.2f} — {row['text']}...")

                with extremes_col2:
                    st.markdown("**Most Negative Events**")
                    top_neg = sent_df.nsmallest(3, "sentiment")
                    for _, row in top_neg.iterrows():
                        st.error(f"😔 {row['sentiment']:.2f} — {row['text']}...")
            else:
                st.info(
                    "No sentiment data available. Process recordings to extract sentiment."
                )

        # --------------- TAB: LIST VIEW ---------------
        with tab_list:
            st.markdown("#### Full Event List")

            import pandas as pd

            # Search within events
            search_text = st.text_input("🔍 Search events", placeholder="keyword...")

            display_events = filtered_events
            if search_text:
                display_events = [
                    e
                    for e in filtered_events
                    if search_text.lower() in e["clean_text"].lower()
                    or search_text.lower() in " ".join(e["keywords"]).lower()
                ]
                st.markdown(
                    f"*Found {len(display_events)} events matching '{search_text}'*"
                )

            # Pagination
            events_per_page = 25
            total_pages = max(
                1, (len(display_events) + events_per_page - 1) // events_per_page
            )
            page_num = st.number_input("Page", 1, total_pages, 1)

            start_idx = (page_num - 1) * events_per_page
            end_idx = start_idx + events_per_page
            page_events = display_events[start_idx:end_idx]

            # Display as cards
            for e in page_events:
                cat_color = CATEGORY_COLORS.get(e["category"], "#7F8C8D")
                time_str = e["start_ts"].strftime("%Y-%m-%d %H:%M")
                duration = (e["end_ts"] - e["start_ts"]).total_seconds()

                rec = rec_lookup.get(e["recording_id"])
                rec_title = rec.title if rec and rec.title else e["recording_id"][:16]

                st.markdown(
                    f"""<div class="event-card">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <span class="status-pill" style="background: {cat_color}20; border-color: {cat_color};">{e['category'].replace('_', ' ').title()}</span>
                                <span class="muted">{time_str}</span>
                                <span class="muted">· {duration:.0f}s</span>
                            </div>
                            <div>
                                <span style="font-size: 0.8rem;">{'😊' if e['sentiment'] > 0.3 else '😐' if e['sentiment'] > -0.3 else '😔'} {e['sentiment']:.2f}</span>
                            </div>
                        </div>
                        <div style="margin-top: 8px;">{e['clean_text']}</div>
                        <div style="margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap;">
                            {''.join(f'<span style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem;">{kw}</span>' for kw in e['keywords'][:5])}
                        </div>
                        <div class="muted" style="margin-top: 6px;">
                            📎 {rec_title}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"*Page {page_num} of {total_pages} ({len(display_events)} events)*"
            )

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
