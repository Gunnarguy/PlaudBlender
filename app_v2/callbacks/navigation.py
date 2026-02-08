"""Navigation callbacks - view switching and main content updates."""

from dash import Input, Output, State, callback, ctx, html, no_update, ALL, dcc
from dash.exceptions import PreventUpdate
import logging

from app_v2.services import get_data_service
from app_v2.components import (
    create_day_view,
    create_topics_grid,
    create_stats_view,
    create_graph_view,
    create_topic_timeline_view,
    create_search_results,
)

logger = logging.getLogger(__name__)


def create_sync_view(service) -> html.Div:
    """Create the sync view with full pipeline controls."""
    stats = service.get_stats()
    db_stats = service.get_recording_db_stats()

    pending = db_stats.get("pending", 0)
    processing = db_stats.get("processing", 0)
    failed = db_stats.get("failed", 0)
    completed = db_stats.get("completed", 0)
    total = pending + processing + failed + completed

    return html.Div(
        className="sync-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("🔄 Sync & Process", className="view-title"),
                    html.P(
                        "Fetch, process, and index your Plaud recordings",
                        className="view-subtitle",
                    ),
                ],
            ),
            # Pipeline status dashboard
            html.Div(
                className="sync-status-card",
                children=[
                    html.H4("Pipeline Status"),
                    html.Div(
                        className="status-stats",
                        children=[
                            html.Div(
                                [
                                    html.Span(str(total), className="big-number"),
                                    html.Span("Total", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        str(completed),
                                        className="big-number",
                                        style={"color": "#10b981"},
                                    ),
                                    html.Span("Completed", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        str(pending),
                                        className="big-number",
                                        style={"color": "#f59e0b"},
                                    ),
                                    html.Span("Pending", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        str(processing),
                                        className="big-number",
                                        style={"color": "#3b82f6"},
                                    ),
                                    html.Span("Processing", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        str(failed),
                                        className="big-number",
                                        style={"color": "#ef4444"},
                                    ),
                                    html.Span("Failed", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                        ],
                    ),
                    html.Div(
                        className="status-stats",
                        style={
                            "marginTop": "10px",
                            "borderTop": "1px solid #334155",
                            "paddingTop": "10px",
                        },
                        children=[
                            html.Div(
                                [
                                    html.Span(
                                        str(stats.total_events), className="big-number"
                                    ),
                                    html.Span(
                                        "Events in Qdrant", className="stat-label"
                                    ),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        str(stats.total_days), className="big-number"
                                    ),
                                    html.Span("Days", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        f"{stats.total_duration_hours:.1f}",
                                        className="big-number",
                                    ),
                                    html.Span("Hours Recorded", className="stat-label"),
                                ],
                                className="status-stat",
                            ),
                        ],
                    ),
                ],
            ),
            # Action buttons
            html.Div(
                className="sync-options",
                children=[
                    html.H4("Actions"),
                    # Full pipeline sync
                    html.Div(
                        className="sync-action-group",
                        children=[
                            html.Label("Days to fetch back:"),
                            dcc.Slider(
                                id="sync-days-slider",
                                min=1,
                                max=30,
                                step=1,
                                value=7,
                                marks={1: "1", 7: "7", 14: "14", 30: "30"},
                                className="sync-slider",
                            ),
                            html.Button(
                                id="do-sync-btn",
                                className="sync-action-btn",
                                children=[
                                    html.Span("🚀", className="btn-icon"),
                                    html.Span(
                                        "Full Sync (Fetch → Process → Index)",
                                        className="btn-text",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Reset stuck recordings
                    html.Div(
                        className="sync-action-group",
                        style={"marginTop": "15px"},
                        children=[
                            html.Button(
                                id="reset-stuck-btn",
                                className="sync-action-btn secondary",
                                children=[
                                    html.Span("🔧", className="btn-icon"),
                                    html.Span(
                                        f"Reset Stuck Recordings ({processing} stuck)",
                                        className="btn-text",
                                    ),
                                ],
                                disabled=(processing == 0),
                            ),
                        ],
                    ),
                    html.Div(id="sync-result", className="sync-result"),
                ],
            ),
        ],
    )


def create_settings_view() -> html.Div:
    """Create the settings view with real connectivity checks."""
    # Check Plaud
    plaud_status, plaud_detail = "❌ Not Connected", ""
    try:
        from src.plaud_client import PlaudClient

        pc = PlaudClient()
        user_info = pc.get_user()
        if user_info:
            plaud_status = "✅ Connected"
            plaud_detail = f"User: {user_info.get('nickname', 'unknown')}"
    except Exception as e:
        plaud_detail = str(e)[:80]

    # Check Gemini
    gemini_status, gemini_detail = "❌ Not Connected", ""
    try:
        from src.config import get_settings

        settings = get_settings()
        if settings.gemini_api_key:
            gemini_status = "✅ API Key Set"
            gemini_detail = (
                f"Model: {settings.chronos_cleaning_model or 'gemini-3-flash-preview'}"
            )
    except Exception as e:
        gemini_detail = str(e)[:80]

    # Check Qdrant
    qdrant_status, qdrant_detail = "❌ Not Connected", ""
    try:
        from qdrant_client import QdrantClient

        qc = QdrantClient(host="localhost", port=6333, timeout=3)
        info = qc.get_collection("chronos_events")
        points = info.points_count
        qdrant_status = "✅ Connected"
        qdrant_detail = f"{points} events indexed"
    except Exception as e:
        qdrant_detail = str(e)[:80]

    return html.Div(
        className="settings-view",
        children=[
            html.Div(
                className="view-header",
                children=[
                    html.H2("⚙️ Settings", className="view-title"),
                    html.P(
                        "System status and configuration", className="view-subtitle"
                    ),
                ],
            ),
            html.Div(
                className="settings-section",
                children=[
                    html.H4("🔑 Service Connections"),
                    html.Div(
                        className="setting-row",
                        children=[
                            html.Label("Plaud API:"),
                            html.Span(plaud_status, className="status-badge"),
                            html.Span(plaud_detail, className="status-detail"),
                        ],
                    ),
                    html.Div(
                        className="setting-row",
                        children=[
                            html.Label("Gemini AI:"),
                            html.Span(gemini_status, className="status-badge"),
                            html.Span(gemini_detail, className="status-detail"),
                        ],
                    ),
                    html.Div(
                        className="setting-row",
                        children=[
                            html.Label("Qdrant:"),
                            html.Span(qdrant_status, className="status-badge"),
                            html.Span(qdrant_detail, className="status-detail"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="settings-section",
                children=[
                    html.H4("🧠 Processing"),
                    html.P(
                        "Processing settings are configured via environment variables.",
                        className="setting-note",
                    ),
                    html.Code("See .env file for configuration options."),
                ],
            ),
            html.Div(
                className="settings-section",
                children=[
                    html.H4("ℹ️ About"),
                    html.P("Chronos v2.0 — Recording Lifecycle Intelligence"),
                    html.P(
                        "Transform your Plaud voice recordings into searchable knowledge.",
                        className="about-desc",
                    ),
                ],
            ),
        ],
    )


def register_navigation_callbacks(app):
    """Register navigation-related callbacks."""

    @app.callback(
        Output("content-container", "children"),
        Output("current-view", "data"),
        Output("detail-panel", "children"),
        Output("detail-panel", "className"),
        Input({"type": "nav-item", "view": ALL}, "n_clicks"),
        Input("selected-recording", "data"),
        Input("selected-topic", "data"),
        Input("search-query", "data"),
        Input("auto-refresh", "n_intervals"),
        State("current-view", "data"),
        prevent_initial_call=False,
    )
    def update_main_content(
        nav_clicks,
        selected_recording,
        selected_topic,
        search_query,
        n_intervals,
        current_view,
    ):
        """Update main content based on navigation and state."""
        triggered = ctx.triggered_id
        service = get_data_service()

        logger.info(f"Navigation callback triggered by: {triggered}")
        logger.info(f"selected_recording: {selected_recording}")

        # Determine what triggered the callback
        view = current_view or "days"

        if isinstance(triggered, dict) and triggered.get("type") == "nav-item":
            view = triggered.get("view", "days")

        # Handle search query
        if search_query and triggered == "search-query":
            results = service.search(search_query)
            return (
                create_search_results(results, search_query),
                "search",
                [],
                "detail-panel",
            )

        # Handle topic selection
        if selected_topic and triggered == "selected-topic":
            timeline = service.get_topic_timeline(selected_topic)
            return (
                create_topic_timeline_view(timeline),
                "topic-detail",
                [],
                "detail-panel",
            )

        # Handle recording selection
        detail_content = []
        detail_class = "detail-panel"

        if selected_recording:
            from app_v2.components import create_recording_detail

            logger.info(
                f"Fetching recording detail for: {selected_recording.get('id')}"
            )
            detail = service.get_recording_detail(selected_recording.get("id"))
            if detail:
                logger.info(f"Got detail with {len(detail.events)} events")
                transcript = service.get_transcript(selected_recording.get("id"))
                detail_content = create_recording_detail(
                    detail, selected_recording.get("date", ""), transcript=transcript
                )
                detail_class = "detail-panel open"
            else:
                logger.warning("No detail returned!")

        # Render main content based on view
        if view == "days":
            days = service.get_days()
            content = create_day_view(days)
        elif view == "topics":
            topics = service.get_all_topics()
            content = create_topics_grid(topics)
        elif view == "graph":
            graph_data = service.get_graph_data()
            content = create_graph_view(graph_data)
        elif view == "stats":
            stats = service.get_stats()
            content = create_stats_view(stats)
        elif view == "sync":
            content = create_sync_view(service)
        elif view == "settings":
            content = create_settings_view()
        else:
            days = service.get_days()
            content = create_day_view(days)

        return content, view, detail_content, detail_class

    @app.callback(
        Output({"type": "nav-item", "view": ALL}, "className"),
        Input("current-view", "data"),
        State({"type": "nav-item", "view": ALL}, "id"),
    )
    def update_nav_active(current_view, nav_ids):
        """Update active state of navigation items."""
        if not nav_ids:
            raise PreventUpdate

        classes = []
        for nav_id in nav_ids:
            view = nav_id.get("view")
            base_class = "nav-item"
            # Add sync-btn class for sync button
            if view == "sync":
                base_class = "nav-item sync-btn"

            if view == current_view:
                classes.append(f"{base_class} active")
            else:
                classes.append(base_class)

        return classes

    @app.callback(
        Output("sync-result", "children"),
        Input("do-sync-btn", "n_clicks"),
        Input("reset-stuck-btn", "n_clicks"),
        State("sync-days-slider", "value"),
        prevent_initial_call=True,
    )
    def perform_sync(sync_clicks, reset_clicks, days_back):
        """Perform full pipeline sync or reset stuck recordings."""
        triggered = ctx.triggered_id

        if triggered == "reset-stuck-btn" and reset_clicks:
            try:
                service = get_data_service()
                count = service.reset_stuck_recordings()
                return html.Div(
                    className="sync-success",
                    children=[
                        html.Span("🔧 Reset Complete!", className="success-icon"),
                        html.P(f"Reset {count} stuck recordings to pending."),
                        html.P("Run Full Sync to process them.", className="sync-note"),
                    ],
                )
            except Exception as e:
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Reset Failed", className="error-icon"),
                        html.P(str(e)),
                    ],
                )

        if triggered == "do-sync-btn" and sync_clicks:
            try:
                from src.chronos.ingest_service import ChronosIngestService
                from src.chronos.transcript_processor import TranscriptProcessor
                from src.chronos.embedding_service import ChronosEmbeddingService
                from src.chronos.qdrant_client import ChronosQdrantClient
                from src.database.engine import SessionLocal
                from src.database.chronos_repository import (
                    get_pending_chronos_recordings,
                    get_chronos_events_by_recording,
                )
                from src.database.models import ChronosEvent as ChronosEventModel

                db = SessionLocal()
                steps = []
                try:
                    # Phase 1: Ingest from Plaud
                    ingest_svc = ChronosIngestService(db_session=db)
                    success, failed = ingest_svc.ingest_recent_recordings(
                        days_back=days_back or 7, fetch_all_pages=True
                    )
                    steps.append(f"📥 Ingested: {success} new, {failed} failed")

                    # Phase 2: Process pending through Gemini
                    pending = get_pending_chronos_recordings(db)
                    if pending:
                        processor = TranscriptProcessor(db_session=db)
                        processed = 0
                        proc_failed = 0
                        for rec in pending:
                            try:
                                ok = processor.process_recording_id(
                                    str(rec.recording_id)
                                )
                                if ok:
                                    processed += 1
                                else:
                                    proc_failed += 1
                            except Exception as e:
                                logger.error(f"Process error: {e}")
                                proc_failed += 1
                        steps.append(
                            f"🧠 Processed: {processed} recordings ({proc_failed} failed)"
                        )
                    else:
                        steps.append("🧠 No pending recordings to process")

                    # Phase 3: Index to Qdrant
                    try:
                        embedder = ChronosEmbeddingService()
                        qdrant = ChronosQdrantClient()

                        # Find events not yet in Qdrant
                        all_events = (
                            db.query(ChronosEventModel)
                            .filter(ChronosEventModel.qdrant_point_id.is_(None))
                            .all()
                        )
                        unindexed = all_events

                        if unindexed:
                            texts = [str(e.clean_text) for e in unindexed]
                            vectors = embedder.embed_batch(texts)

                            from src.models.chronos_schemas import ChronosEvent as CE

                            indexed = 0
                            for event, vector in zip(unindexed, vectors):
                                try:
                                    schema_event = CE(
                                        event_id=str(event.event_id),
                                        recording_id=str(event.recording_id),
                                        start_ts=event.start_ts,  # type: ignore[arg-type]
                                        end_ts=event.end_ts,  # type: ignore[arg-type]
                                        day_of_week=str(event.day_of_week),  # type: ignore[arg-type]
                                        hour_of_day=int(event.hour_of_day),  # type: ignore[arg-type]
                                        clean_text=str(event.clean_text),
                                        category=str(event.category),  # type: ignore[arg-type]
                                        sentiment=float(event.sentiment or 0.0),  # type: ignore[arg-type]
                                        keywords=list(event.keywords or []),  # type: ignore[arg-type]
                                        speaker=str(event.speaker or "unknown"),  # type: ignore[arg-type]
                                        raw_transcript_snippet=str(event.raw_transcript_snippet) if event.raw_transcript_snippet else None,  # type: ignore[truthy-bool]
                                        gemini_reasoning=str(event.gemini_reasoning) if event.gemini_reasoning else None,  # type: ignore[truthy-bool]
                                    )
                                    point_id = qdrant.upsert_event(schema_event, vector)
                                    event.qdrant_point_id = point_id  # type: ignore[assignment]
                                    db.commit()
                                    indexed += 1
                                except Exception as e:
                                    logger.error(f"Index error: {e}")
                            steps.append(f"📊 Indexed: {indexed} events to Qdrant")
                        else:
                            steps.append("📊 All events already indexed")
                    except Exception as e:
                        steps.append(f"📊 Indexing error: {str(e)[:60]}")

                    # Refresh the data service cache
                    service = get_data_service()
                    service.refresh_cache()

                    return html.Div(
                        className="sync-success",
                        children=[
                            html.Span(
                                "✅ Full Pipeline Complete!", className="success-icon"
                            ),
                        ]
                        + [html.P(step) for step in steps]
                        + [
                            html.P(
                                "Refresh the page to see updated data.",
                                className="sync-note",
                            ),
                        ],
                    )
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Sync error: {e}")
                return html.Div(
                    className="sync-error",
                    children=[
                        html.Span("❌ Pipeline Failed", className="error-icon"),
                        html.P(str(e)),
                    ],
                )

        raise PreventUpdate
