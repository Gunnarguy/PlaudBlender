"""Chronos Streamlit UI - Master-Detail Timeline Interface

The visual layer for interacting with Chronos temporal narratives.
Provides timeline navigation, event details, and graph-based exploration.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.database import SessionLocal
from src.database.chronos_repository import (
    get_chronos_events_by_date_range,
    get_chronos_events_by_day,
)
from src.chronos.qdrant_client import ChronosQdrantClient
from src.chronos.embedding_service import ChronosEmbeddingService
from src.models.chronos_schemas import DayOfWeek, EventCategory, TemporalFilter


# Page config
st.set_page_config(
    page_title="Chronos",
    page_icon="🕰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean aesthetics
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 300;
        margin-bottom: 1rem;
    }
    .event-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .event-time {
        color: #666;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .event-category {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .category-work { background-color: #e3f2fd; color: #1565c0; }
    .category-personal { background-color: #f3e5f5; color: #6a1b9a; }
    .category-meeting { background-color: #fff3e0; color: #e65100; }
    .category-deep_work { background-color: #e8f5e9; color: #2e7d32; }
    .category-break { background-color: #fce4ec; color: #c2185b; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_clients():
    """Initialize database and vector clients (cached)."""
    qdrant = ChronosQdrantClient()
    embedder = ChronosEmbeddingService()
    return qdrant, embedder


def format_event_card(event: Dict[str, Any]) -> str:
    """Format event as HTML card."""
    payload = event.get("payload", {})
    category = payload.get("category", "unknown")

    return f"""
    <div class="event-card">
        <span class="event-category category-{category}">{category}</span>
        <div class="event-time">
            {payload.get('start_ts', '')} → {payload.get('end_ts', '')}
        </div>
        <p style="margin-top: 0.5rem;">{payload.get('clean_text', '')}</p>
        <small style="color: #999;">
            Sentiment: {payload.get('sentiment', 0):.2f} | 
            Duration: {payload.get('duration_seconds', 0):.0f}s
        </small>
    </div>
    """


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">Chronos</h1>', unsafe_allow_html=True)
    st.markdown("*Temporal narrative reconstruction from voice recordings*")

    # Initialize clients
    try:
        qdrant, embedder = get_clients()
    except Exception as e:
        st.error(f"Failed to initialize clients: {e}")
        st.info("Make sure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`")
        return

    # Sidebar: Search & Filters
    with st.sidebar:
        st.header("🔍 Search & Filter")

        # Semantic search
        query_text = st.text_input(
            "Semantic query",
            placeholder="What do I think about anxiety?",
            help="Find events similar to this query",
        )

        # Temporal filters
        st.subheader("Temporal Filters")

        filter_mode = st.radio(
            "Filter by",
            ["Date Range", "Day of Week", "Hour of Day"],
            horizontal=True,
        )

        temporal_filter = None

        if filter_mode == "Date Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start date",
                    value=datetime.now() - timedelta(days=7),
                )
            with col2:
                end_date = st.date_input(
                    "End date",
                    value=datetime.now(),
                )

            temporal_filter = TemporalFilter(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
            )

        elif filter_mode == "Day of Week":
            days = st.multiselect(
                "Select days",
                options=[d.value for d in DayOfWeek],
                default=["Monday"],
            )
            temporal_filter = TemporalFilter(
                days_of_week=[DayOfWeek(d) for d in days],
            )

        elif filter_mode == "Hour of Day":
            hours = st.slider(
                "Hour range",
                min_value=0,
                max_value=23,
                value=(9, 17),
                help="Select hour range (0-23)",
            )
            temporal_filter = TemporalFilter(
                hours_of_day=list(range(hours[0], hours[1] + 1)),
            )

        # Category filter
        categories = st.multiselect(
            "Categories",
            options=[c.value for c in EventCategory],
            default=None,
        )

        # Search button
        search_clicked = st.button(
            "🔎 Search", type="primary", use_container_width=True
        )

    # Main content area
    if search_clicked or query_text:
        st.header("📊 Results")

        # Prepare query
        query_vector = None
        if query_text:
            with st.spinner("Generating query embedding..."):
                query_vector = embedder.embed_text(
                    query_text, task_type="RETRIEVAL_QUERY"
                )

        # Execute hybrid search
        with st.spinner("Searching..."):
            try:
                results = qdrant.search_hybrid(
                    query_vector=query_vector,
                    temporal_filter=temporal_filter,
                    categories=categories,
                    limit=50,
                )

                st.success(f"Found {len(results)} events")

                # Display results
                if results:
                    for i, event in enumerate(results):
                        with st.expander(
                            f"Event {i+1}: {event['payload'].get('category', 'unknown')} "
                            f"({event['payload'].get('start_ts', '')[:10]})",
                            expanded=(i < 3),  # Expand first 3
                        ):
                            st.markdown(
                                format_event_card(event), unsafe_allow_html=True
                            )

                            # Show raw payload in debug mode
                            if st.checkbox(f"Show raw data {i+1}", key=f"raw_{i}"):
                                st.json(event)
                else:
                    st.warning("No events found matching your query.")

            except Exception as e:
                st.error(f"Search failed: {e}")
                st.exception(e)

    else:
        # Default view: Show statistics
        st.header("📈 Overview")

        try:
            stats = qdrant.get_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", stats.get("points_count", 0))
            with col2:
                st.metric("Indexed Vectors", stats.get("indexed_vectors_count", 0))
            with col3:
                st.metric("Collection Status", stats.get("status", "Unknown"))

            st.info("👈 Use the sidebar to search and filter events")

            # Quick links
            st.subheader("Quick Queries")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("What happens on Mondays?", use_container_width=True):
                    st.session_state.quick_query = "Monday"
                if st.button("Show all deep work sessions", use_container_width=True):
                    st.session_state.quick_query = "deep_work"

            with col2:
                if st.button(
                    "What do I think about on Thursdays?", use_container_width=True
                ):
                    st.session_state.quick_query = "Thursday"
                if st.button("Show recent reflections", use_container_width=True):
                    st.session_state.quick_query = "reflection"

        except Exception as e:
            st.warning(f"Could not fetch statistics: {e}")
            st.info("Run `python scripts/chronos_pipeline.py --full` to populate data")


if __name__ == "__main__":
    main()
