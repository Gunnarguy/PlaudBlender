"""Chronos Streamlit UI.

This is the Streamlit replacement for the legacy Tkinter GUI.

Goal: keep the *nuance* and "under the hood" visibility Gunnar likes:
- show system readiness (Qdrant/Gemini)
- show latency and provenance
- keep filters and controls discoverable
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import subprocess
import sys

# Increase Python's integer string conversion limit (default 4300 digits).
# This prevents "Exceeds the limit (4300 digits)" errors when parsing JSON
# responses from Gemini that contain very large numbers (e.g., token counts).
if sys.version_info >= (3, 11):
    sys.set_int_max_str_digits(0)  # 0 = no limit (safe for local processing)

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


# Page config
st.set_page_config(
    page_title="Chronos",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean aesthetics
st.markdown(
    """<style>
    /* Hide Streamlit chrome (keeps app feeling like a real product) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0.25rem 0 0.25rem 0;
    }
    .subheader {
        opacity: 0.85;
        margin-bottom: 1.25rem;
    }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        border: 1px solid rgba(137, 180, 250, 0.25);
        background: rgba(137, 180, 250, 0.10);
    }
    .event-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(49, 50, 68, 0.60);
        padding: 0.9rem 1rem;
        margin: 0.55rem 0;
        border-radius: 12px;
    }
    .event-time {
        opacity: 0.85;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .muted {
        opacity: 0.75;
        font-size: 0.85rem;
    }
    .statusbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(30, 30, 46, 0.92);
        border-top: 1px solid rgba(255,255,255,0.10);
        padding: 0.35rem 1rem;
        z-index: 1000;
        font-size: 0.85rem;
    }
    .statusbar code {
        font-size: 0.85rem;
    }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_qdrant_client() -> ChronosQdrantClient:
    """Initialize the Qdrant client (cached)."""
    return ChronosQdrantClient()


@st.cache_resource
def get_embedder() -> ChronosEmbeddingService:
    """Initialize the Gemini embedder (cached).

    NOTE: This will raise if GEMINI_API_KEY is not set. We intentionally keep it
    separate from Qdrant init so the UI can still load in a "browse-only" mode.
    """

    return ChronosEmbeddingService()


def _set_last_latency(ms: float):
    st.session_state.last_latency_ms = float(ms)


def _set_last_error(message: str):
    # Keep a small rolling buffer of recent errors for the Logs page.
    logs = st.session_state.get("ui_logs", [])
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    st.session_state.ui_logs = logs[-200:]


def render_header(settings, gemini_available: bool):
    st.markdown('<div class="main-header">Chronos</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader">Temporal narrative reconstruction from voice recordings</div>',
        unsafe_allow_html=True,
    )

    # Tiny badges like the old GUI had in the top chrome.
    st.markdown(
        """
        <span class="pill">Vector DB: Qdrant</span>
        <span class="pill">Gemini: {gemini}</span>
        <span class="pill">Collection: {collection}</span>
        """.format(
            gemini="enabled" if gemini_available else "missing key",
            collection=settings.qdrant_collection_name,
        ),
        unsafe_allow_html=True,
    )


def render_status_bar(settings, gemini_available: bool):
    latency = st.session_state.get("last_latency_ms")
    latency_txt = f"{latency:.0f} ms" if latency is not None else "—"
    st.markdown(
        (
            '<div class="statusbar">'
            f"<b>Status</b> · Qdrant: <code>{settings.qdrant_url}</code> · "
            f"Gemini: <code>{'OK' if gemini_available else 'MISSING GEMINI_API_KEY'}</code> · "
            f"Last latency: <code>{latency_txt}</code>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def format_event_card(event: Dict[str, Any]) -> str:
    payload = event.get("payload", {})
    category = payload.get("category", "unknown")

    score = event.get("score")
    score_txt = "" if score is None else f" · score: {score:.4f}"

    return (
        '<div class="event-card">'
        f"<div><span class='pill'>{category}</span><span class='muted'>{payload.get('recording_id', '')}</span></div>"
        f"<div class='event-time'>{payload.get('start_ts', '')} → {payload.get('end_ts', '')}{score_txt}</div>"
        f"<div style='margin-top: 0.45rem;'>{payload.get('clean_text', '')}</div>"
        "<div class='muted' style='margin-top: 0.6rem;'>"
        f"Sentiment: {payload.get('sentiment', 0):.2f} · "
        f"Duration: {payload.get('duration_seconds', 0):.0f}s"
        "</div>"
        "</div>"
    )


def page_dashboard(qdrant: ChronosQdrantClient, settings, gemini_available: bool):
    st.header("📊 Dashboard")

    colA, colB, colC = st.columns([1.2, 1.2, 1])
    with colA:
        st.subheader("System")
        st.write(
            {
                "qdrant_url": settings.qdrant_url,
                "collection": settings.qdrant_collection_name,
                "gemini": "enabled" if gemini_available else "missing GEMINI_API_KEY",
                "qdrant_timeout_seconds": getattr(
                    settings, "qdrant_timeout_seconds", None
                ),
            }
        )
    with colB:
        st.subheader("Quick actions")
        st.caption(
            "These mirror the old GUI's command bar — fast paths to common workflows."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Run preflight", use_container_width=True, disabled=not gemini_available
            ):
                import subprocess
                import sys

                with st.spinner("Running preflight…"):
                    try:
                        t0 = time.perf_counter()
                        proc = subprocess.run(
                            [
                                sys.executable,
                                "scripts/chronos_pipeline.py",
                                "--preflight",
                            ],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        _set_last_latency((time.perf_counter() - t0) * 1000)
                        st.code(
                            (proc.stdout or "") + (proc.stderr or ""), language="text"
                        )
                    except Exception as e:
                        _set_last_error(f"Preflight failed: {e}")
                        st.error(f"Preflight failed: {e}")
        with c2:
            st.link_button(
                "Open Qdrant dashboard",
                url=settings.qdrant_url.rstrip("/") + "/dashboard",
                use_container_width=True,
            )

        st.code("python scripts/chronos_pipeline.py --full", language="bash")
        if not gemini_available:
            st.info("Set `GEMINI_API_KEY` to enable embeddings + semantic search.")
    with colC:
        st.subheader("Collection stats")
        try:
            t0 = time.perf_counter()
            stats = qdrant.get_stats()
            _set_last_latency((time.perf_counter() - t0) * 1000)

            st.metric("Points", stats.get("points_count", 0))
            st.metric("Indexed", stats.get("indexed_vectors_count", 0))
            st.metric("Status", stats.get("status", "unknown"))
        except Exception as e:
            _set_last_error(f"Dashboard stats failed: {e}")
            st.warning(f"Could not fetch statistics: {e}")
            st.info("Run `python scripts/chronos_pipeline.py --full` to populate data")


def page_search(qdrant: ChronosQdrantClient, settings, gemini_available: bool):
    st.header("🔍 Search")
    st.caption(
        "Semantic + temporal filters, with debug visibility like the legacy GUI."
    )

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Query")
        if not gemini_available:
            st.warning(
                "Semantic search is disabled because `GEMINI_API_KEY` is not set. "
                "You can still run filter-only searches."
            )

        query_text = st.text_area(
            "Semantic query",
            placeholder="What do I think about anxiety?",
            help="Embeds this query and retrieves similar events (requires GEMINI_API_KEY).",
            disabled=not gemini_available,
            height=80,
            key="search_query_text",
        )

        st.subheader("Temporal filters")
        filter_mode = st.radio(
            "Mode",
            ["Date Range", "Day of Week", "Hour of Day"],
            horizontal=False,
            key="search_filter_mode",
        )

        temporal_filter: Optional[TemporalFilter] = None
        if filter_mode == "Date Range":
            start_date = st.date_input(
                "Start date",
                value=datetime.now() - timedelta(days=7),
                key="search_start_date",
            )
            end_date = st.date_input(
                "End date",
                value=datetime.now(),
                key="search_end_date",
            )
            temporal_filter = TemporalFilter(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
            )
        elif filter_mode == "Day of Week":
            days = st.multiselect(
                "Days",
                options=[d.value for d in DayOfWeek],
                default=[],
                key="search_days",
            )
            temporal_filter = TemporalFilter(days_of_week=[DayOfWeek(d) for d in days])
        else:
            hours = st.slider(
                "Hour range",
                min_value=0,
                max_value=23,
                value=(9, 17),
                help="Select hour range (0-23)",
                key="search_hours",
            )
            temporal_filter = TemporalFilter(
                hours_of_day=list(range(hours[0], hours[1] + 1))
            )

        categories = st.multiselect(
            "Categories",
            options=[c.value for c in EventCategory],
            default=[],
            key="search_categories",
        )

        limit = st.slider(
            "Max results",
            min_value=5,
            max_value=200,
            value=50,
            step=5,
            key="search_limit",
        )

        debug = st.checkbox(
            "Debug mode (show raw payloads)", value=False, key="search_debug"
        )

        run = st.button("Run search", type="primary", use_container_width=True)

        with st.expander("Saved searches", expanded=False):
            saved = st.session_state.get("saved_searches", {})
            if saved:
                pick = st.selectbox("Load", options=["(select)"] + sorted(saved.keys()))
                if (
                    pick
                    and pick != "(select)"
                    and st.button("Apply", use_container_width=True)
                ):
                    snap = saved[pick]
                    # Update widget values via session_state keys.
                    st.session_state.search_query_text = snap.get("query_text", "")
                    st.session_state.search_filter_mode = snap.get(
                        "filter_mode", "Date Range"
                    )
                    st.session_state.search_categories = snap.get("categories", [])
                    st.session_state.search_limit = snap.get("limit", 50)
                    st.session_state.search_debug = snap.get("debug", False)
                    st.session_state.search_start_date = snap.get(
                        "start_date", datetime.now() - timedelta(days=7)
                    )
                    st.session_state.search_end_date = snap.get(
                        "end_date", datetime.now()
                    )
                    st.session_state.search_days = snap.get("days", [])
                    st.session_state.search_hours = snap.get("hours", (9, 17))
                    st.rerun()
            else:
                st.caption("No saved searches yet.")

            name = st.text_input(
                "Save current as", placeholder="e.g. anxiety_last_week"
            )
            if st.button(
                "Save", use_container_width=True, disabled=not bool(name.strip())
            ):
                saved = dict(st.session_state.get("saved_searches", {}))
                saved[name.strip()] = {
                    "query_text": query_text,
                    "filter_mode": filter_mode,
                    "start_date": st.session_state.get(
                        "search_start_date", datetime.now() - timedelta(days=7)
                    ),
                    "end_date": st.session_state.get("search_end_date", datetime.now()),
                    "days": st.session_state.get("search_days", []),
                    "hours": st.session_state.get("search_hours", (9, 17)),
                    "categories": categories,
                    "limit": int(limit),
                    "debug": bool(debug),
                }
                st.session_state.saved_searches = saved
                st.success(f"Saved: {name.strip()}")

    with right:
        if not run:
            st.info("Configure filters on the left, then run a search.")
            return

        query_vector = None
        if query_text and query_text.strip():
            if not gemini_available:
                st.error("`GEMINI_API_KEY` is not set, so semantic search can't run.")
                return

            with st.spinner("Embedding query…"):
                t0 = time.perf_counter()
                embedder = get_embedder()
                query_vector = embedder.embed_text(
                    query_text.strip(), task_type="RETRIEVAL_QUERY"
                )
                _set_last_latency((time.perf_counter() - t0) * 1000)

        with st.spinner("Searching Qdrant…"):
            try:
                t0 = time.perf_counter()
                results = qdrant.search_hybrid(
                    query_vector=query_vector,
                    temporal_filter=temporal_filter,
                    categories=categories or None,
                    limit=int(limit),
                )
                _set_last_latency((time.perf_counter() - t0) * 1000)
            except Exception as e:
                _set_last_error(f"Search failed: {e}")
                st.error(f"Search failed: {e}")
                st.exception(e)
                return

        st.subheader(f"Results ({len(results)})")
        if not results:
            st.warning("No events found.")
            return

        for i, event in enumerate(results, start=1):
            with st.expander(
                f"Event {i}: {event.get('payload', {}).get('category', 'unknown')} · {event.get('payload', {}).get('start_ts', '')[:10]}",
                expanded=(i <= 3),
            ):
                st.markdown(format_event_card(event), unsafe_allow_html=True)
                if debug:
                    st.json(event)


def page_timeline(qdrant: ChronosQdrantClient):
    st.header("🗓 Timeline")
    st.caption(
        "A timeline view (like the old GUI) is the fastest way to see narrative flow."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        days = st.slider("Days back", min_value=1, max_value=60, value=14)
        limit = st.slider(
            "Max events", min_value=50, max_value=2000, value=300, step=50
        )
        build = st.button("Build timeline", type="primary", use_container_width=True)

    if not build:
        st.info("Pick a window and click “Build timeline”.")
        return

    tf = TemporalFilter(
        start_date=datetime.now() - timedelta(days=int(days)),
        end_date=datetime.now(),
    )
    try:
        t0 = time.perf_counter()
        results = qdrant.search_hybrid(
            query_vector=None, temporal_filter=tf, categories=None, limit=int(limit)
        )
        _set_last_latency((time.perf_counter() - t0) * 1000)
    except Exception as e:
        _set_last_error(f"Timeline load failed: {e}")
        st.error(f"Timeline load failed: {e}")
        st.exception(e)
        return

    if not results:
        st.warning("No events found in that window.")
        return

    # Try to render a true timeline if streamlit-timeline is installed.
    try:
        from streamlit_timeline import timeline  # type: ignore

        items = []
        for ev in results:
            p = ev.get("payload", {})
            start = p.get("start_ts")
            end = p.get("end_ts")
            content = (p.get("clean_text") or "").strip()
            if len(content) > 180:
                content = content[:177] + "…"
            items.append(
                {
                    "id": str(ev.get("event_id")),
                    "content": f"{p.get('category', 'event')}: {content}",
                    "start": start,
                    "end": end,
                }
            )

        timeline(items, height=520)
        st.divider()
    except Exception:
        st.info("(Optional) Install `streamlit-timeline` for a richer timeline widget.")

    # Always provide the raw list view (good for auditing).
    st.subheader("Events")
    for i, ev in enumerate(results[:200], start=1):
        st.markdown(format_event_card(ev), unsafe_allow_html=True)


def page_settings(settings, gemini_available: bool):
    st.header("⚙️ Settings & Diagnostics")
    st.caption("Visibility-first: show exactly what Chronos is configured to use.")

    st.subheader("Environment")
    st.write(
        {
            "GEMINI_API_KEY": "set" if gemini_available else "missing",
            "GEMINI_API_VERSION": settings.gemini_api_version,
            "CHRONOS_CLEANING_MODEL": settings.chronos_cleaning_model,
            "CHRONOS_ANALYST_MODEL": settings.chronos_analyst_model,
            "CHRONOS_EMBEDDING_MODEL": settings.chronos_embedding_model,
            "CHRONOS_THINKING_LEVEL": settings.chronos_thinking_level,
            "QDRANT_URL": settings.qdrant_url,
            "QDRANT_COLLECTION_NAME": settings.qdrant_collection_name,
            "QDRANT_TIMEOUT_SECONDS": getattr(settings, "qdrant_timeout_seconds", None),
        }
    )

    st.subheader("Help")
    st.markdown(
        """
        - Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
        - Preflight (models + config): `python scripts/chronos_pipeline.py --preflight`
        - Full pipeline: `python scripts/chronos_pipeline.py --full`
        - **First-time setup:** Use Controls → Ingest with "Fetch all recordings" checked to grab your entire Plaud history
        - **Ongoing:** Run Ingest without "Fetch all" to just pull the 100 most recent
        """
    )


def _extract_transcript_from_plaud_file_details(file_details: dict) -> Optional[str]:
    """Extract transcript text from Plaud `get_recording` response.

    Mirrors the logic used by `TranscriptProcessor` but avoids requiring Gemini.
    """

    import json

    source_list = file_details.get("source_list", [])
    for source in source_list:
        if source.get("data_type") == "transaction":
            try:
                segments = json.loads(source.get("data_content", "[]"))
                texts = [seg.get("content", "") for seg in segments]
                txt = " ".join(texts).strip()
                return txt or None
            except json.JSONDecodeError:
                return None
    return None


def _run_pipeline_with_live_logs(args: list[str], *, header: str) -> int:
    """Run the CLI pipeline as a subprocess and stream logs into the UI."""

    st.subheader(header)
    st.caption("Streaming logs live (so you can *see stuff happen*).")

    out = st.empty()
    lines: list[str] = []

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
        # Keep the last ~400 lines for readability.
        lines = lines[-400:]
        out.code("\n".join(lines), language="text")

    return int(proc.wait())


def page_controls(settings, gemini_available: bool):
    st.header("🧰 Controls")
    st.caption(
        "Button-driven ingest/process/index — with live logs and command previews so you know exactly what's about to run."
    )

    st.subheader("Pipeline limits")
    ingest_col, process_col, index_col, graph_col = st.columns(4)
    with ingest_col:
        ingest_limit = st.number_input(
            "Ingest limit",
            min_value=1,
            max_value=500,
            value=25,
            step=5,
            key="ingest_limit",
            help="Max recordings to fetch per ingest pass.",
        )
        fetch_all = st.checkbox(
            "Fetch all recordings",
            value=False,
            key="fetch_all",
            help="Paginate through your entire Plaud history (use only once when catching up).",
        )
    with process_col:
        process_limit = st.number_input(
            "Process limit",
            min_value=1,
            max_value=500,
            value=25,
            step=5,
            key="process_limit",
            help="Number of pending recordings to process per run.",
        )
    with index_col:
        index_limit = st.number_input(
            "Index limit",
            min_value=1,
            max_value=500,
            value=25,
            step=5,
            key="index_limit",
            help="Max events to index per run (multiple events per recording).",
        )
    with graph_col:
        graph_limit = st.number_input(
            "Graph limit",
            min_value=1,
            max_value=500,
            value=25,
            step=5,
            key="graph_limit",
            help="Max indexed events to run through the knowledge-graph extractor.",
        )

    full_limit = st.number_input(
        "Full pipeline limit",
        min_value=1,
        max_value=500,
        value=25,
        step=5,
        key="full_limit",
        help="Limit applied to each phase when running the full pipeline.",
    )

    override_col1, override_col2 = st.columns([3, 1])
    with override_col1:
        recording_id = (
            st.text_input(
                "Single recording_id (optional)",
                placeholder="plaud_recording_id",
                help="If set, --process/--index/--graph will target just that recording.",
            ).strip()
            or None
        )
    with override_col2:
        force = st.checkbox(
            "Force reprocess",
            value=False,
            help="Delete existing events for that recording before reprocessing.",
        )

    st.subheader("Preflight & diagnostics")
    preflight_smoke = st.checkbox(
        "Smoke test (tiny embed call)",
        value=False,
        key="preflight_smoke",
        help=(
            "Runs `--preflight-smoke`: lists accessible models and performs a tiny embedding call to verify connectivity. "
            "Useful for debugging model/auth issues before running the pipeline."
        ),
    )
    preflight_args = ["--preflight-smoke"] if preflight_smoke else ["--preflight"]
    st.caption("Command that will run:")
    st.code(
        f"python scripts/chronos_pipeline.py {' '.join(preflight_args)}",
        language="bash",
    )
    if st.button("Preflight", key="preflight_button", use_container_width=True):
        code = _run_pipeline_with_live_logs(preflight_args, header="Preflight")
        if code != 0:
            st.error(f"Preflight failed (exit code {code}).")
        else:
            st.success("Preflight OK")
        return

    def _preview_command(base_cmd: list[str]) -> str:
        return f"python scripts/chronos_pipeline.py {' '.join(base_cmd)}"

    st.markdown("---")
    st.subheader("Ingest")
    ingest_args = ["--ingest", "--limit", str(int(ingest_limit))]
    if fetch_all:
        ingest_args.append("--fetch-all")
    st.caption("Command that will run:")
    st.code(_preview_command(ingest_args), language="bash")
    if st.button("Run ingest", key="run_ingest"):
        if fetch_all:
            st.warning(
                "⚠️ Fetch All will paginate through your entire Plaud account. This can take quite a while the first time."
            )
        code = _run_pipeline_with_live_logs(ingest_args, header="Ingest")
        if code == 0:
            st.success("Ingest finished")
        else:
            st.error(f"Ingest failed (exit code {code})")
        return

    st.markdown("---")
    st.subheader("Process (Gemini)")
    process_args = ["--process", "--limit", str(int(process_limit))]
    if recording_id:
        process_args += ["--recording-id", recording_id]
    if force:
        process_args.append("--force")
    st.caption("Command that will run:")
    st.code(_preview_command(process_args), language="bash")
    if st.button("Run process", key="run_process", disabled=not gemini_available):
        code = _run_pipeline_with_live_logs(process_args, header="Process")
        if code == 0:
            st.success("Process finished")
        else:
            st.error(f"Process failed (exit code {code})")
        return

    st.markdown("---")
    st.subheader("Index (Qdrant)")
    index_args = ["--index", "--limit", str(int(index_limit))]
    if recording_id:
        index_args += ["--recording-id", recording_id]
    st.caption("Command that will run:")
    st.code(_preview_command(index_args), language="bash")
    if st.button("Run index", key="run_index", disabled=not gemini_available):
        code = _run_pipeline_with_live_logs(index_args, header="Index")
        if code == 0:
            st.success("Index finished")
        else:
            st.error(f"Index failed (exit code {code})")
        return

    st.markdown("---")
    st.subheader("Graph extraction")
    graph_args = ["--graph", "--limit", str(int(graph_limit))]
    if recording_id:
        graph_args += ["--recording-id", recording_id]
    st.caption("Command that will run:")
    st.code(_preview_command(graph_args), language="bash")
    if st.button("Run graph", key="run_graph", disabled=not gemini_available):
        code = _run_pipeline_with_live_logs(graph_args, header="Graph")
        if code == 0:
            st.success("Graph extraction finished")
        else:
            st.error(f"Graph extraction failed (exit code {code})")
        return

    st.markdown("---")
    st.subheader("Full pipeline")
    full_args = ["--full", "--limit", str(int(full_limit))]
    if fetch_all:
        full_args.append("--fetch-all")
    st.caption("Command that will run:")
    st.code(_preview_command(full_args), language="bash")
    if st.button("Run full pipeline", key="run_full", disabled=not gemini_available):
        code = _run_pipeline_with_live_logs(full_args, header="Full pipeline")
        if code == 0:
            st.success("Full pipeline finished")
        else:
            st.error(f"Full pipeline failed (exit code {code})")


def page_recordings(settings, gemini_available: bool):
    st.header("📚 Recordings library")
    st.caption(
        "Browse everything Chronos knows about your Plaud recordings: metadata, cached transcripts, and extracted events."
    )

    init_db()
    session = SessionLocal()
    try:
        statuses = ["pending", "processing", "completed", "failed"]
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
        with col1:
            status_filter = st.multiselect(
                "Status",
                options=statuses,
                default=statuses,
            )
        with col2:
            days_back = st.number_input(
                "Days back", min_value=1, max_value=365, value=30
            )
        with col3:
            text_q = st.text_input(
                "Search (title/id)", placeholder="meeting / grocery / ..."
            ).strip()
        with col4:
            refresh = st.button("↻ Refresh", use_container_width=True)

        # Query recordings.
        q = session.query(ChronosRecordingDB)
        if status_filter:
            q = q.filter(ChronosRecordingDB.processing_status.in_(status_filter))
        if days_back:
            q = q.filter(
                ChronosRecordingDB.created_at
                >= (datetime.utcnow() - timedelta(days=int(days_back)))
            )
        if text_q:
            like = f"%{text_q}%"
            q = q.filter(
                (ChronosRecordingDB.recording_id.ilike(like))
                | (ChronosRecordingDB.title.ilike(like))
            )

        recs = q.order_by(ChronosRecordingDB.created_at.desc()).limit(500).all()

        rows: list[dict[str, Any]] = []
        for r in recs:
            event_count = (
                session.query(ChronosEventDB)
                .filter(ChronosEventDB.recording_id == r.recording_id)
                .count()
            )
            rows.append(
                {
                    "recording_id": r.recording_id,
                    "title": r.title,
                    "created_at": r.created_at,
                    "duration_s": r.duration_seconds,
                    "status": r.processing_status,
                    "events": int(event_count),
                    "has_transcript": bool((r.transcript or "").strip()),
                    "error": r.error_message,
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Bulk selection section
        st.divider()
        st.subheader("📋 Bulk operations")
        st.caption("Select multiple recordings and perform batch actions.")

        # Initialize session state for bulk checkboxes if not exists
        if "bulk_selections" not in st.session_state:
            st.session_state.bulk_selections = {}

        bulk_cols = st.columns([0.5, 2, 1, 1, 1])
        with bulk_cols[0]:
            select_all = st.checkbox("All", key="select_all_recordings", value=False)
        with bulk_cols[1]:
            st.caption("**Select recordings:**")
        with bulk_cols[2]:
            st.caption("")  # spacer
        with bulk_cols[3]:
            st.caption("")  # spacer
        with bulk_cols[4]:
            st.caption("")  # spacer

        # Update bulk selections based on "select all"
        if select_all:
            for r in recs:
                st.session_state.bulk_selections[r.recording_id] = True
        elif not st.session_state.get("bulk_selections"):
            # Initialize empty if not already set
            for r in recs:
                st.session_state.bulk_selections[r.recording_id] = False

        # Display checkboxes and recording info
        selected_recordings: list[str] = []
        for r in recs[:50]:  # Show checkboxes for first 50 (performance)
            bulk_col1, bulk_col2, bulk_col3, bulk_col4, bulk_col5 = st.columns(
                [0.5, 2, 1, 1, 1]
            )
            with bulk_col1:
                is_selected = st.checkbox(
                    label="",
                    value=st.session_state.bulk_selections.get(r.recording_id, False),
                    key=f"bulk_checkbox_{r.recording_id}",
                    label_visibility="collapsed",
                )
                st.session_state.bulk_selections[r.recording_id] = is_selected
                if is_selected:
                    selected_recordings.append(r.recording_id)
            with bulk_col2:
                label = r.recording_id
                if r.title:
                    label = f"{r.title} · {r.recording_id}"
                st.caption(label)
            with bulk_col3:
                st.caption(r.processing_status)
            with bulk_col4:
                event_count = (
                    session.query(ChronosEventDB)
                    .filter(ChronosEventDB.recording_id == r.recording_id)
                    .count()
                )
                st.caption(f"{int(event_count)} events")
            with bulk_col5:
                st.caption("✓" if (r.transcript or "").strip() else "—")

        if selected_recordings:
            st.info(f"**{len(selected_recordings)} recording(s) selected**")

            # Bulk action buttons
            bulk_action_cols = st.columns(4)
            with bulk_action_cols[0]:
                if st.button(
                    "🔄 Process selected",
                    use_container_width=True,
                    disabled=not gemini_available,
                ):
                    for rec_id in selected_recordings:
                        st.info(f"Processing {rec_id}…")
                        code = _run_pipeline_with_live_logs(
                            ["--process", "--recording-id", rec_id, "--limit", "1"],
                            header=f"Process {rec_id}",
                        )
                        if code == 0:
                            st.success(f"✓ {rec_id}")
                        else:
                            st.error(f"✗ {rec_id} (exit code {code})")
                    st.rerun()

            with bulk_action_cols[1]:
                if st.button(
                    "⚡ Index selected",
                    use_container_width=True,
                    disabled=not gemini_available,
                ):
                    for rec_id in selected_recordings:
                        st.info(f"Indexing {rec_id}…")
                        code = _run_pipeline_with_live_logs(
                            ["--index", "--recording-id", rec_id, "--limit", "25"],
                            header=f"Index {rec_id}",
                        )
                        if code == 0:
                            st.success(f"✓ {rec_id}")
                        else:
                            st.error(f"✗ {rec_id} (exit code {code})")
                    st.rerun()

            with bulk_action_cols[2]:
                if st.button(
                    "🔂 Force reprocess",
                    use_container_width=True,
                    disabled=not gemini_available,
                ):
                    for rec_id in selected_recordings:
                        st.info(f"Force reprocessing {rec_id}…")
                        code = _run_pipeline_with_live_logs(
                            [
                                "--process",
                                "--recording-id",
                                rec_id,
                                "--force",
                                "--limit",
                                "1",
                            ],
                            header=f"Force reprocess {rec_id}",
                        )
                        if code == 0:
                            st.success(f"✓ {rec_id}")
                        else:
                            st.error(f"✗ {rec_id} (exit code {code})")
                    st.rerun()

            with bulk_action_cols[3]:
                if st.button(
                    "📝 Fetch transcripts",
                    use_container_width=True,
                ):
                    for rec_id in selected_recordings:
                        st.info(f"Fetching transcript for {rec_id}…")
                        try:
                            found_rec = (
                                session.query(ChronosRecordingDB)
                                .filter_by(recording_id=rec_id)
                                .first()
                            )
                            if found_rec and not (found_rec.transcript or "").strip():
                                file_details = PlaudClient().get_recording(rec_id)
                                transcript = (
                                    _extract_transcript_from_plaud_file_details(
                                        file_details
                                    )
                                )
                                if transcript:
                                    set_chronos_recording_transcript(
                                        session, rec_id, transcript
                                    )
                                    st.success(
                                        f"✓ {rec_id} ({len(transcript):,} chars)"
                                    )
                                else:
                                    st.warning(f"✗ {rec_id} (no transcript in Plaud)")
                            else:
                                st.caption(f"↪ {rec_id} (already cached)")
                        except Exception as e:
                            st.error(f"✗ {rec_id}: {e}")
                    st.rerun()

        if not recs:
            st.info("No Chronos recordings found yet. Try Controls → Ingest.")
            return

        options = []
        for r in recs:
            label = r.recording_id
            if r.title:
                label = f"{r.title} · {r.recording_id}"
            options.append(label)

        pick = st.selectbox(
            "Select recording",
            options=options,
            index=0,
        )
        selected_id = pick.split(" · ")[-1].strip()
        rec = (
            session.query(ChronosRecordingDB)
            .filter_by(recording_id=selected_id)
            .first()
        )
        if not rec:
            st.warning("Could not load selected recording.")
            return

        st.divider()
        st.subheader("Recording detail")
        meta1, meta2, meta3 = st.columns([1, 1, 2])
        with meta1:
            st.write(
                {
                    "recording_id": rec.recording_id,
                    "title": rec.title,
                    "status": rec.processing_status,
                }
            )
        with meta2:
            st.write(
                {
                    "created_at": rec.created_at,
                    "duration_seconds": rec.duration_seconds,
                    "device_id": rec.device_id,
                }
            )
        with meta3:
            st.write(
                {
                    "transcript_cached_at": rec.transcript_cached_at,
                    "local_audio_path": rec.local_audio_path or "(transcript-only)",
                    "error_message": rec.error_message,
                }
            )

        actions1, actions2, actions3, actions4 = st.columns(4)
        with actions1:
            fetch_tx = st.button("Fetch + cache transcript", use_container_width=True)
        with actions2:
            process_one = st.button(
                "Process (Gemini)",
                use_container_width=True,
                disabled=not gemini_available,
            )
        with actions3:
            index_one = st.button(
                "Index (Qdrant)",
                use_container_width=True,
                disabled=not gemini_available,
            )
        with actions4:
            force_one = st.button(
                "Force reprocess",
                use_container_width=True,
                disabled=not gemini_available,
                help="Deletes existing DB events first, then reprocesses.",
            )

        if fetch_tx:
            with st.spinner("Fetching transcript from Plaud…"):
                try:
                    file_details = PlaudClient().get_recording(rec.recording_id)
                    transcript = _extract_transcript_from_plaud_file_details(
                        file_details
                    )
                    if not transcript:
                        st.warning("No transcript found in Plaud source_list.")
                    else:
                        set_chronos_recording_transcript(
                            session, rec.recording_id, transcript
                        )
                        st.success(f"Cached transcript ({len(transcript):,} chars)")
                        st.rerun()
                except Exception as e:
                    st.error(f"Transcript fetch failed: {e}")

        if process_one:
            code = _run_pipeline_with_live_logs(
                ["--process", "--recording-id", rec.recording_id, "--limit", "1"],
                header=f"Process {rec.recording_id}",
            )
            (
                st.success("Process finished")
                if code == 0
                else st.error(f"Process failed (exit code {code})")
            )
            st.rerun()

        if force_one:
            code = _run_pipeline_with_live_logs(
                [
                    "--process",
                    "--recording-id",
                    rec.recording_id,
                    "--force",
                    "--limit",
                    "1",
                ],
                header=f"Force reprocess {rec.recording_id}",
            )
            (
                st.success("Force reprocess finished")
                if code == 0
                else st.error(f"Force reprocess failed (exit code {code})")
            )
            st.rerun()

        if index_one:
            code = _run_pipeline_with_live_logs(
                ["--index", "--recording-id", rec.recording_id, "--limit", "25"],
                header=f"Index {rec.recording_id}",
            )
            (
                st.success("Index finished")
                if code == 0
                else st.error(f"Index failed (exit code {code})")
            )
            st.rerun()

        st.divider()
        st.subheader("Cached transcript")
        transcript_txt = (rec.transcript or "").strip()
        if not transcript_txt:
            st.info("No cached transcript yet. Use “Fetch + cache transcript”.")
        else:
            st.text_area(
                "Transcript",
                value=transcript_txt,
                height=260,
            )

        st.divider()
        st.subheader("Events (SQLite)")
        events = (
            session.query(ChronosEventDB)
            .filter(ChronosEventDB.recording_id == rec.recording_id)
            .order_by(ChronosEventDB.start_ts.asc())
            .limit(500)
            .all()
        )
        if not events:
            st.info("No events yet. Use “Process (Gemini)”.")
            return

        ev_rows: list[dict[str, Any]] = []
        for e in events:
            ev_rows.append(
                {
                    "start": e.start_ts,
                    "end": e.end_ts,
                    "category": e.category,
                    "sentiment": e.sentiment,
                    "text": (e.clean_text or "")[:220],
                    "qdrant": bool(e.qdrant_point_id),
                }
            )
        st.dataframe(ev_rows, use_container_width=True, hide_index=True)
    finally:
        session.close()


def page_logs():
    st.header("📋 Logs")
    st.caption("Recent UI-level errors and diagnostics.")

    logs = st.session_state.get("ui_logs", [])
    if not logs:
        st.info("No UI errors recorded in this session.")
        return
    st.text("\n".join(logs))


def main():
    """Main Streamlit application."""

    settings = get_settings()
    gemini_available = bool(settings.gemini_api_key)

    # Ensure DB + lightweight migrations are applied before any views.
    init_db()

    # Initialize Qdrant client (Gemini is optional for browse-only mode)
    try:
        qdrant = get_qdrant_client()
    except Exception as e:
        _set_last_error(f"Failed to initialize Qdrant client: {e}")
        st.error(f"Failed to initialize Qdrant client: {e}")
        st.info("Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`")
        return

    render_header(settings, gemini_available)

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "",
            [
                "Dashboard",
                "Recordings",
                "Controls",
                "Search",
                "Timeline",
                "Settings",
                "Logs",
            ],
            index=0,
            label_visibility="collapsed",
        )

        with st.expander("Diagnostics", expanded=False):
            st.write(
                {
                    "gemini": "OK" if gemini_available else "MISSING GEMINI_API_KEY",
                    "qdrant_url": settings.qdrant_url,
                    "collection": settings.qdrant_collection_name,
                }
            )

    if page == "Dashboard":
        page_dashboard(qdrant, settings, gemini_available)
    elif page == "Recordings":
        page_recordings(settings, gemini_available)
    elif page == "Controls":
        page_controls(settings, gemini_available)
    elif page == "Search":
        page_search(qdrant, settings, gemini_available)
    elif page == "Timeline":
        page_timeline(qdrant)
    elif page == "Settings":
        page_settings(settings, gemini_available)
    else:
        page_logs()

    render_status_bar(settings, gemini_available)


def _running_in_streamlit() -> bool:
    """Best-effort detection for whether we're running under `streamlit run`."""

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not _running_in_streamlit():
        import sys

        print(
            "This is a Streamlit app. Run it with:\n\n  streamlit run chronos_app.py\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    main()
