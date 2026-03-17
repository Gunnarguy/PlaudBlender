"""Chronos MCP Server — Production MCP server for Plaud knowledge timeline.

Exposes the full Chronos system as MCP tools for OpenAI-compatible clients.
Runs over stdio transport (standard for MCP).

Tools:
  - search_events: Semantic search across all events
  - get_recording: Get recording details + events
  - list_recordings: List all recordings with filters
  - get_timeline: Get events for a date range
  - get_stats: System statistics
  - get_topics: List all topic categories
  - run_pipeline: Trigger ingest/process/index pipeline
  - get_graph: Get knowledge graph entities and relationships
  - system_status: Health check for all services
  - ask_chronos: RAG — semantic search + Gemini answer synthesis

Run with:
    python -m scripts.mcp_server
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set integer string conversion limit early
if sys.version_info >= (3, 11):
    sys.set_int_max_str_digits(0)

from dotenv import load_dotenv

load_dotenv()

from mcp.server import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronos.mcp")

server = FastMCP("chronos-mcp")

# ---------------------------------------------------------------------------
# Lazy service initialization
# ---------------------------------------------------------------------------

_services: Dict[str, Any] = {}


def _get_db_session():
    """Get a fresh database session."""
    from src.database import SessionLocal

    return SessionLocal()


def _get_qdrant():
    """Get Qdrant client."""
    if "qdrant" not in _services:
        from src.chronos.qdrant_client import ChronosQdrantClient

        _services["qdrant"] = ChronosQdrantClient()
    return _services["qdrant"]


def _get_data_service():
    """Get the Dash data service for aggregated views."""
    if "data" not in _services:
        from app_v2.services.data_service import ChronosDataService

        _services["data"] = ChronosDataService()
    return _services["data"]


# ---------------------------------------------------------------------------
# Timeout protection for MCP tools
# ---------------------------------------------------------------------------

MCP_TOOL_TIMEOUT = 30  # seconds — prevents hung API calls from blocking MCP


def _with_timeout(timeout_sec: int = MCP_TOOL_TIMEOUT):
    """Decorator: wraps an async MCP tool with asyncio.wait_for timeout."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_sec)
            except asyncio.TimeoutError:
                name = fn.__name__
                logger.error("MCP tool '%s' timed out after %ds", name, timeout_sec)
                return json.dumps(
                    {"error": f"Tool '{name}' timed out after {timeout_sec}s"}
                )

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@server.tool()
@_with_timeout(5)
async def ping() -> str:
    """Health check — returns pong if server is alive."""
    return "pong"


@server.tool()
@_with_timeout()
async def search_events(
    query: str,
    category: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 10,
) -> str:
    """Search Chronos events using semantic vector search.

    Finds events from voice recordings that match your query. Supports
    filtering by category and date range.

    Args:
        query: Natural language search query (e.g. "budget meeting with team")
        category: Filter by category — work, meeting, personal, health, travel, idea, errand, social, reflection, or leave empty for all
        date_from: Start date filter (YYYY-MM-DD), optional
        date_to: End date filter (YYYY-MM-DD), optional
        limit: Max results to return (1-50, default 10)
    """
    try:
        limit = max(1, min(50, limit))
        ds = _get_data_service()

        categories = [category] if category else None
        results = ds.search(query, limit=limit, categories=categories)

        if not results:
            return json.dumps(
                {"results": [], "message": f"No events found for: {query}"}
            )

        output = []
        for r in results:
            e = r.event
            date_str = e.start_ts.strftime("%Y-%m-%d")
            if date_from and date_str < date_from:
                continue
            if date_to and date_str > date_to:
                continue
            time_str = f"{e.start_ts.strftime('%H:%M')}-{e.end_ts.strftime('%H:%M')}"
            # Title: first sentence or first 80 chars of clean_text
            title = e.clean_text.split(".")[0][:80] if e.clean_text else "Untitled"
            output.append(
                {
                    "title": title,
                    "summary": e.clean_text[:300] if e.clean_text else "",
                    "date": date_str,
                    "time": time_str,
                    "category": e.category,
                    "score": round(r.score, 3) if r.score else None,
                    "recording_id": e.recording_id,
                }
            )

        return json.dumps({"results": output[:limit], "total": len(output)}, indent=2)
    except Exception as e:
        logger.exception("search_events failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def get_recording(recording_id: str) -> str:
    """Get full details for a specific recording including all extracted events.

    Args:
        recording_id: The recording ID (full or partial match)
    """
    try:
        ds = _get_data_service()
        detail = ds.get_recording_detail(recording_id)

        if not detail:
            return json.dumps({"error": f"Recording {recording_id} not found"})

        events = []
        for ev in detail.events:
            title = ev.clean_text.split(".")[0][:80] if ev.clean_text else "Untitled"
            events.append(
                {
                    "title": title,
                    "summary": ev.clean_text[:300] if ev.clean_text else "",
                    "category": ev.category,
                    "start_time": ev.start_ts.isoformat(),
                    "end_time": ev.end_ts.isoformat(),
                }
            )

        s = detail.summary
        result = {
            "recording_id": s.recording_id,
            "date": s.start_time.strftime("%Y-%m-%d"),
            "duration": s.duration_formatted,
            "event_count": len(events),
            "top_category": s.top_category,
            "categories": detail.category_percentages,
            "events": events,
        }

        transcript = ds.get_transcript(recording_id)
        if transcript:
            result["transcript_preview"] = transcript[:2000]
            result["transcript_length"] = len(transcript)

        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("get_recording failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def list_recordings(
    status: str = "completed",
    limit: int = 20,
    offset: int = 0,
) -> str:
    """List all Chronos recordings with their status and event counts.

    Args:
        status: Filter by status — completed, processing, failed, pending, or 'all'
        limit: Max recordings to return (1-100, default 20)
        offset: Pagination offset (default 0)
    """
    try:
        from sqlalchemy import text

        db = _get_db_session()
        try:
            params = {}
            query_str = """
                SELECT r.recording_id, r.title, r.created_at, r.duration_seconds,
                       r.processing_status, r.source,
                       COUNT(e.id) as event_count
                FROM chronos_recordings r
                LEFT JOIN chronos_events e ON e.recording_id = r.recording_id
            """
            if status != "all":
                query_str += " WHERE r.processing_status = :status"
                params["status"] = status
            query_str += " GROUP BY r.recording_id ORDER BY r.created_at DESC"
            query_str += f" LIMIT {min(100, limit)} OFFSET {offset}"

            rows = db.execute(text(query_str), params).fetchall()

            recordings = []
            for row in rows:
                dur_s = row[3] or 0
                recordings.append(
                    {
                        "recording_id": row[0],
                        "title": row[1] or "Untitled",
                        "date": str(row[2])[:10] if row[2] else None,
                        "duration_minutes": round(dur_s / 60, 1),
                        "status": row[4],
                        "source": row[5],
                        "event_count": row[6],
                    }
                )

            return json.dumps(
                {"recordings": recordings, "count": len(recordings)}, indent=2
            )
        finally:
            db.close()
    except Exception as e:
        logger.exception("list_recordings failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def get_timeline(date: str = "", days: int = 1) -> str:
    """Get the event timeline for a specific date or date range.

    Returns all events grouped by recording for the given date(s).

    Args:
        date: Date to query (YYYY-MM-DD). Defaults to today.
        days: Number of days to include (1 = just that date, 7 = week)
    """
    try:
        ds = _get_data_service()
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        base_date = datetime.strptime(date, "%Y-%m-%d")
        all_days = []
        for i in range(days):
            d = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            detail = ds.get_day_detail(d)
            if detail:
                day_data = {"date": d, "recordings": []}
                for rec in detail.recordings:
                    # Fetch actual events for this recording
                    rec_events = ds.get_events_for_recording(rec.recording_id)
                    events = []
                    for ev in rec_events:
                        title = (
                            ev.clean_text.split(".")[0][:80]
                            if ev.clean_text
                            else "Untitled"
                        )
                        events.append(
                            {
                                "title": title,
                                "summary": ev.clean_text[:300] if ev.clean_text else "",
                                "category": ev.category,
                                "time": f"{ev.start_ts.strftime('%H:%M')}-{ev.end_ts.strftime('%H:%M')}",
                            }
                        )
                    day_data["recordings"].append(
                        {
                            "recording_id": rec.recording_id,
                            "duration": rec.duration_formatted,
                            "event_count": rec.event_count,
                            "top_category": rec.top_category,
                            "events": events,
                        }
                    )
                all_days.append(day_data)

        if not all_days:
            return json.dumps({"message": f"No events found for {date}", "days": []})

        return json.dumps({"days": all_days}, indent=2)
    except Exception as e:
        logger.exception("get_timeline failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def get_stats() -> str:
    """Get comprehensive Chronos system statistics.

    Returns recording counts, event totals, category breakdown,
    time distribution, and productivity metrics.
    """
    try:
        ds = _get_data_service()
        stats = ds.get_stats()
        db_stats = ds.get_recording_db_stats()

        avg_per_day = (
            round(stats.total_events / max(stats.total_days, 1), 1)
            if stats.total_days
            else 0
        )

        result = {
            "recordings": {
                "total": sum(db_stats.values()),
                "completed": db_stats.get("completed", 0),
                "failed": db_stats.get("failed", 0),
                "pending": db_stats.get("pending", 0),
            },
            "events": {
                "total": stats.total_events,
                "total_hours": round(stats.total_duration_hours, 1),
                "categories": stats.categories,
            },
            "top_day": stats.most_productive_day,
            "avg_events_per_day": avg_per_day,
            "total_days": stats.total_days,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception("get_stats failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def get_topics() -> str:
    """List all topic categories with event counts sorted by frequency."""
    try:
        ds = _get_data_service()
        topics = ds.get_all_topics()
        result = [{"category": t[0], "count": t[1]} for t in topics]
        return json.dumps({"topics": result}, indent=2)
    except Exception as e:
        logger.exception("get_topics failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def get_graph(max_nodes: int = 50, entity_types: str = "") -> str:
    """Get knowledge graph data — entities and relationships extracted from recordings.

    Args:
        max_nodes: Maximum number of nodes to return (default 50)
        entity_types: Comma-separated entity types to filter (person, project, topic, action, organization). Empty for all.
    """
    try:
        ds = _get_data_service()
        graph_data = ds.get_graph_data()

        if not graph_data or not graph_data.nodes:
            return json.dumps(
                {
                    "message": "No graph data available. Run pipeline to build graph.",
                    "nodes": [],
                    "edges": [],
                }
            )

        type_filter = set(entity_types.split(",")) if entity_types else None

        nodes = []
        for node in graph_data.nodes[:max_nodes]:
            if type_filter and node.get("type", "") not in type_filter:
                continue
            nodes.append(
                {
                    "id": node.get("id", ""),
                    "label": node.get("label", ""),
                    "type": node.get("type", ""),
                    "weight": node.get("weight", 1),
                }
            )

        edges = []
        node_ids = {n["id"] for n in nodes}
        for edge in graph_data.edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src in node_ids and tgt in node_ids:
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "type": edge.get("type", ""),
                        "weight": edge.get("weight", 1),
                    }
                )

        return json.dumps(
            {
                "nodes": nodes,
                "edges": edges,
                "total_nodes": len(graph_data.nodes),
                "total_edges": len(graph_data.edges),
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("get_graph failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout(660)
async def run_pipeline(stage: str = "full") -> str:
    """Run the Chronos pipeline to sync and process recordings.

    Stages:
    - ingest: Download new recordings from Plaud API
    - process: Process pending recordings through Gemini AI
    - index: Index processed events to Qdrant vector store
    - full: Run all stages sequentially

    Args:
        stage: Pipeline stage to run — ingest, process, index, or full
    """
    try:
        valid_stages = {"ingest", "process", "index", "full"}
        if stage not in valid_stages:
            return json.dumps(
                {
                    "error": f"Invalid stage: {stage}. Must be one of: {list(valid_stages)}"
                }
            )

        cmd = [sys.executable, "scripts/chronos_pipeline.py", f"--{stage}"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).parent.parent),
        )

        output = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
        errors = (
            result.stderr[-1000:]
            if result.stderr and len(result.stderr) > 1000
            else result.stderr
        )

        return json.dumps(
            {
                "stage": stage,
                "exit_code": result.returncode,
                "output": output,
                "errors": errors or None,
            },
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Pipeline timed out after 10 minutes"})
    except Exception as e:
        logger.exception("run_pipeline failed")
        return json.dumps({"error": str(e)})


@server.tool()
@_with_timeout()
async def system_status() -> str:
    """Check health of all Chronos services — database, Qdrant, Plaud API, Gemini."""
    status = {}

    # Database
    try:
        from sqlalchemy import text

        db = _get_db_session()
        try:
            count = db.execute(text("SELECT COUNT(*) FROM chronos_recordings")).scalar()
            status["database"] = {"status": "ok", "recordings": count}
        finally:
            db.close()
    except Exception as e:
        status["database"] = {"status": "error", "message": str(e)}

    # Qdrant
    try:
        qdrant = _get_qdrant()
        stats = qdrant.get_stats()
        status["qdrant"] = {
            "status": "ok",
            "points": stats.get("points_count", 0),
            "indexed": stats.get("indexed_vectors_count", 0),
        }
    except Exception as e:
        status["qdrant"] = {"status": "error", "message": str(e)}

    # Gemini
    try:
        from src.chronos.genai_helpers import get_genai_client

        client = get_genai_client()
        models = list(client.models.list())
        status["gemini"] = {"status": "ok", "models_available": len(models)}
    except Exception as e:
        status["gemini"] = {"status": "error", "message": str(e)}

    # Plaud
    try:
        from src.config import get_settings

        settings = get_settings()
        if settings.plaud_client_id and settings.plaud_client_secret:
            status["plaud"] = {"status": "ok", "oauth_configured": True}
        else:
            status["plaud"] = {
                "status": "warning",
                "message": "Plaud OAuth credentials not configured",
            }
    except Exception as e:
        status["plaud"] = {"status": "error", "message": str(e)}

    return json.dumps(status, indent=2)


@server.tool()
@_with_timeout(60)
async def ask_chronos(question: str, reasoning: str = "none") -> str:
    """Ask a natural language question about your recordings and get an AI answer.

    Uses semantic search to find relevant events, then synthesizes an answer.
    Prefers OpenAI Responses API when available, falls back to Gemini.

    Args:
        question: Natural language question about your recordings
        reasoning: Reasoning effort level for GPT-5.4: none, low, medium, high, xhigh
    """
    try:
        from src.config import get_settings

        settings = get_settings()

        ds = _get_data_service()
        results = ds.search(question, limit=15, task_type="QUESTION_ANSWERING")

        if not results:
            return json.dumps(
                {
                    "answer": "I couldn't find any relevant events for your question.",
                    "sources": [],
                }
            )

        # Build context from search results
        context_events = []
        sources = []
        for r in results:
            e = r.event
            date_str = e.start_ts.strftime("%Y-%m-%d")
            time_str = f"{e.start_ts.strftime('%H:%M')}-{e.end_ts.strftime('%H:%M')}"
            title = e.clean_text.split(".")[0][:80] if e.clean_text else "Untitled"
            context_events.append(
                {
                    "date": date_str,
                    "time": time_str,
                    "category": e.category,
                    "text": e.clean_text,
                }
            )
            sources.append(
                {
                    "date": date_str,
                    "title": title,
                    "category": e.category,
                    "score": round(r.score, 3) if r.score else None,
                }
            )

        # Try OpenAI Responses API first (avoids Gemini free tier cap)
        if settings.openai_api_key:
            try:
                from src.chronos.openai_service import OpenAIResponseService

                openai_svc = OpenAIResponseService()
                result = openai_svc.ask(question, context_events, reasoning=reasoning)
                if "error" not in result:
                    return json.dumps(
                        {
                            "answer": result["answer"],
                            "sources": sources[:5],
                            "events_searched": len(results),
                            "model": result.get("model", settings.openai_model),
                            "usage": result.get("usage"),
                        },
                        indent=2,
                    )
                logger.warning(
                    f"OpenAI failed, falling back to Gemini: {result['error']}"
                )
            except Exception as oe:
                logger.warning(f"OpenAI unavailable, falling back to Gemini: {oe}")

        # Fallback: Gemini
        if not settings.gemini_api_key:
            return json.dumps(
                {
                    "error": "No AI provider configured (set OPENAI_API_KEY or GEMINI_API_KEY)"
                }
            )

        from src.chronos.genai_helpers import get_genai_client

        context = "\n".join(
            f"[{evt['date']} {evt['time']}] ({evt['category']}) {evt['text']}"
            for evt in context_events
        )

        prompt = f"""Based on the following events from the user's voice recordings,
answer their question concisely and accurately. Only reference information that
appears in the provided events. If the events don't contain enough information
to fully answer, say so.

EVENTS:
{context}

QUESTION: {question}

ANSWER:"""

        client = get_genai_client()
        response = client.models.generate_content(
            model=settings.chronos_analyst_model,
            contents=prompt,
        )
        answer = response.text if response.text else "Unable to generate answer."

        return json.dumps(
            {
                "answer": answer,
                "sources": sources[:5],
                "events_searched": len(results),
                "model": settings.chronos_analyst_model,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("ask_chronos failed")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Start the Chronos MCP server over stdio."""
    logger.info("Starting Chronos MCP server (stdio transport)...")
    logger.info(
        "Tools: search_events, get_recording, list_recordings, get_timeline, "
        "get_stats, get_topics, get_graph, run_pipeline, system_status, ask_chronos"
    )
    await server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
