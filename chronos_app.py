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

import subprocess
import sys
import time
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
    """Run pipeline subprocess and stream output to UI."""
    st.subheader(header)
    output_area = st.empty()
    lines: List[str] = []

    proc = subprocess.Popen(
        [sys.executable, "scripts/chronos_pipeline.py", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        lines = lines[-300:]
        output_area.code("\n".join(lines), language="text")

    return proc.wait()


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
            use_container_width=True,
            disabled=not status["plaud"],
        ):
            code = run_pipeline_command(["--ingest", "--limit", "25"], "Fetching...")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with action_cols[1]:
        if st.button(
            "🧠 Process Pending",
            use_container_width=True,
            disabled=not status["gemini"],
        ):
            code = run_pipeline_command(["--process", "--limit", "10"], "Processing...")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with action_cols[2]:
        if st.button(
            "📤 Index to Qdrant",
            use_container_width=True,
            disabled=not (status["qdrant"] and status["gemini"]),
        ):
            code = run_pipeline_command(["--index", "--limit", "50"], "Indexing...")
            st.success("Done!" if code == 0 else f"Failed (code {code})")

    with action_cols[3]:
        if st.button(
            "🚀 Run Full Pipeline",
            use_container_width=True,
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

        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

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

        st.dataframe(rows, use_container_width=True, hide_index=True)

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
                    use_container_width=True,
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
                    "📤 Index", use_container_width=True, disabled=not status["gemini"]
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
                    use_container_width=True,
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
                    use_container_width=True,
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
            use_container_width=True,
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
            use_container_width=True,
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
            use_container_width=True,
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
        use_container_width=True,
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
            if st.button("Custom Ingest", use_container_width=True):
                args = ["--ingest", "--limit", str(ingest_limit)]
                if fetch_all:
                    args.append("--fetch-all")
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Ingest")

        with cmd_cols[1]:
            if st.button("Custom Process", use_container_width=True):
                args = ["--process", "--limit", str(process_limit)]
                if recording_id:
                    args += ["--recording-id", recording_id]
                if force:
                    args.append("--force")
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Process")

        with cmd_cols[2]:
            if st.button("Custom Index", use_container_width=True):
                args = ["--index", "--limit", str(index_limit)]
                if recording_id:
                    args += ["--recording-id", recording_id]
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Index")

        with cmd_cols[3]:
            if st.button("Custom Graph", use_container_width=True):
                args = ["--graph", "--limit", str(graph_limit)]
                if recording_id:
                    args += ["--recording-id", recording_id]
                st.code(f"python scripts/chronos_pipeline.py {' '.join(args)}")
                code = run_pipeline_command(args, "Custom Graph")

        st.markdown("### Diagnostics")
        diag_cols = st.columns(2)
        with diag_cols[0]:
            if st.button("Preflight Check", use_container_width=True):
                code = run_pipeline_command(["--preflight"], "Preflight")
        with diag_cols[1]:
            if st.button("Smoke Test (with API call)", use_container_width=True):
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
            if st.button("🔄 Fetch Latest Recordings", use_container_width=True):
                code = run_pipeline_command(
                    ["--ingest", "--limit", "25"], "Fetching..."
                )
                st.rerun() if code == 0 else None
        with quick_cols[1]:
            if st.button("📜 Get Recording Stats", use_container_width=True):
                try:
                    stats = client.get_recording_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"Failed: {e}")
        with quick_cols[2]:
            if st.button("👤 User Info", use_container_width=True):
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
                "🔍 Search",
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
    elif page == "🔍 Search":
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
