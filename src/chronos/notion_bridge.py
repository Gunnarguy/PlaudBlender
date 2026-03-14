"""Notion → Chronos Bridge.

Imports Notion recordings into the Chronos pipeline so they get the
full AI treatment: Gemini cleaning → event extraction → Qdrant indexing
→ knowledge graph. After processing, optionally writes enriched
metadata BACK to Notion (categories, sentiment, keywords).

This is the critical link that turns raw Notion pages into first-class
Chronos citizens — searchable, graphable, analyzable.
"""

import logging
import time as _time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from src.config import get_settings
from src.database.chronos_repository import (
    add_chronos_events,
    get_chronos_recording,
    mark_chronos_recording_status,
    set_chronos_recording_transcript,
    upsert_chronos_recording,
)
from src.database.models import ChronosEvent as ChronosEventDB, ChronosRecording
from src.notion_service import NotionRecording, get_notion_service

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Smart Matching — fuzzy title + date alignment
# ═══════════════════════════════════════════════════════════════════


def match_notion_to_chronos(
    notion_recordings: List[NotionRecording],
    session: Session,
) -> Dict[str, Optional[str]]:
    """Match Notion recordings to Chronos recordings using fuzzy logic.

    Returns: {notion_page_id → chronos_recording_id or None}
    """
    from app_v2.services.xray import xray_log

    # Build a lookup of Chronos recordings by date → list of (id, title, created_at)
    chronos_recs = session.query(ChronosRecording).all()
    by_date: Dict[str, List[Tuple[str, str, datetime]]] = {}
    for rec in chronos_recs:
        if rec.created_at:
            date_key = rec.created_at.strftime("%Y-%m-%d") if isinstance(rec.created_at, datetime) else str(rec.created_at)[:10]
            by_date.setdefault(date_key, []).append(
                (rec.recording_id, rec.title or "", rec.created_at)
            )

    matches: Dict[str, Optional[str]] = {}
    matched_count = 0

    for nrec in notion_recordings:
        best_match = None
        best_score = 0.0

        notion_date = nrec.date or (nrec.created_time[:10] if nrec.created_time else "")
        notion_title = (nrec.title or "").lower().strip()

        # Phase 1: Same-date fuzzy title match (strongest signal)
        candidates = by_date.get(notion_date, [])
        for cid, ctitle, _ in candidates:
            ctitle_lower = (ctitle or "").lower().strip()
            if not notion_title or not ctitle_lower:
                # If either has no title, same-date is a weaker match
                score = 0.4
            else:
                score = SequenceMatcher(None, notion_title, ctitle_lower).ratio()

            if score > best_score:
                best_score = score
                best_match = cid

        # Phase 2: Adjacent-date fuzzy match (±1 day, lower threshold)
        if best_score < 0.5 and notion_date:
            try:
                from datetime import timedelta
                nd = datetime.strptime(notion_date, "%Y-%m-%d")
                for delta in [-1, 1]:
                    adj_date = (nd + timedelta(days=delta)).strftime("%Y-%m-%d")
                    for cid, ctitle, _ in by_date.get(adj_date, []):
                        ctitle_lower = (ctitle or "").lower().strip()
                        if notion_title and ctitle_lower:
                            score = SequenceMatcher(None, notion_title, ctitle_lower).ratio() * 0.8  # penalize date mismatch
                            if score > best_score:
                                best_score = score
                                best_match = cid
            except (ValueError, TypeError):
                pass

        # Threshold: need >0.45 similarity to consider a match
        if best_score >= 0.45 and best_match:
            matches[nrec.page_id] = best_match
            matched_count += 1
        else:
            matches[nrec.page_id] = None

    xray_log(
        "data", "notion-match",
        f"Matched {matched_count} of {len(notion_recordings)} Notion pages to Chronos recordings"
    )
    return matches


# ═══════════════════════════════════════════════════════════════════
# Import Pipeline — Notion → Chronos
# ═══════════════════════════════════════════════════════════════════


def import_notion_recording(
    page_id: str,
    session: Session,
    *,
    process: bool = True,
    index: bool = True,
) -> Tuple[bool, str]:
    """Import a single Notion page into the Chronos pipeline.

    Steps:
    1. Fetch page from Notion (content + properties)
    2. Create ChronosRecording (source="notion")
    3. Cache transcript text
    4. (Optional) Process through Gemini → extract events
    5. (Optional) Index events to Qdrant

    Returns: (success, message)
    """
    from app_v2.services.xray import xray_log

    try:
        svc = get_notion_service()

        # Step 1: Fetch the page
        xray_log("data", "notion-import", f"Pulling page from Notion...")
        recordings = svc.fetch_recordings(limit=1000)
        page = None
        for r in recordings:
            if r.page_id == page_id:
                page = r
                break

        if not page:
            return False, f"Page {page_id} not found in Notion database"

        # Get full body content
        body_text = svc.fetch_page_content(page_id)

        # Build transcript: prefer explicit transcript field, fall back to body
        transcript = page.transcript or body_text or ""
        if not transcript.strip():
            return False, "No transcript or body text found in Notion page"

        # Step 2: Create ChronosRecording
        created_at = _parse_iso(page.created_time)
        # Estimate duration from transcript length (~150 words/minute)
        word_count = len(transcript.split())
        estimated_duration = max(60, int(word_count / 2.5))  # ~150wpm speaking rate

        rec = upsert_chronos_recording(
            session=session,
            recording_id=f"notion:{page_id}",
            title=page.title,
            created_at=created_at,
            duration_seconds=estimated_duration,
            local_audio_path="",  # No audio for Notion imports
            source="notion",
            device_id="notion",
        )
        xray_log("data", "notion-import", f"Created Chronos recording for '{page.title}'")

        # Step 3: Cache transcript
        set_chronos_recording_transcript(session, rec.recording_id, transcript)

        if not process:
            mark_chronos_recording_status(session, rec.recording_id, "pending")
            return True, f"Imported '{page.title}' — ready for processing"

        # Step 4: Process through Gemini
        xray_log("gemini", "notion-process", f"Sending '{page.title}' to Gemini for analysis...")
        from src.chronos.transcript_processor import TranscriptProcessor

        processor = TranscriptProcessor(db_session=session)

        recording_date = created_at.strftime("%Y-%m-%d") if created_at else ""

        # Use Notion summary as context to improve Gemini's categorization
        plaud_context = page.summary if page.summary else None

        _t0 = _time.perf_counter()
        output = processor.process_transcript_text(
            transcript,
            rec.recording_id,
            recording_date=recording_date,
            plaud_context=plaud_context,
        )
        _ms = (_time.perf_counter() - _t0) * 1000

        if not output or not output.events:
            mark_chronos_recording_status(
                session, rec.recording_id, "failed",
                error_message="No events extracted by Gemini",
            )
            return False, f"Gemini couldn't extract events from '{page.title}'"

        # Save events to SQLite
        event_models = []
        for ev in output.events:
            event_models.append(
                ChronosEventDB(
                    event_id=str(uuid.uuid4()),
                    recording_id=rec.recording_id,
                    start_ts=ev.start_ts,
                    end_ts=ev.end_ts,
                    day_of_week=ev.day_of_week.value if hasattr(ev.day_of_week, "value") else str(ev.day_of_week),
                    hour_of_day=ev.hour_of_day,
                    clean_text=ev.clean_text,
                    category=ev.category.value if hasattr(ev.category, "value") else str(ev.category),
                    category_confidence=getattr(ev, "category_confidence", None),
                    sentiment=ev.sentiment,
                    keywords=ev.keywords,
                    speaker=ev.speaker.value if hasattr(ev.speaker, "value") else str(ev.speaker),
                    raw_transcript_snippet=getattr(ev, "raw_transcript_snippet", None),
                    gemini_reasoning=getattr(ev, "gemini_reasoning", None),
                )
            )
        add_chronos_events(session, event_models)
        mark_chronos_recording_status(session, rec.recording_id, "completed")

        xray_log(
            "gemini", "notion-process",
            f"Extracted {len(event_models)} moments from '{page.title}'",
            duration_ms=round(_ms, 1),
        )

        if not index:
            return True, f"Processed '{page.title}' → {len(event_models)} events (not yet indexed)"

        # Step 5: Index to Qdrant
        indexed = _index_recording_events(session, rec.recording_id)
        xray_log("qdrant", "notion-index", f"Indexed {indexed} events to Qdrant for '{page.title}'")

        return True, f"Imported '{page.title}' → {len(event_models)} events, {indexed} indexed to Qdrant"

    except Exception as e:
        logger.error(f"Error importing Notion page {page_id}: {e}", exc_info=True)
        return False, f"Error: {str(e)}"


def import_all_unmatched(
    session: Session,
    *,
    process: bool = True,
    index: bool = True,
    progress_callback=None,
) -> Tuple[int, int, List[str]]:
    """Import all Notion recordings that aren't already in Chronos.

    Returns: (success_count, failure_count, error_messages)
    """
    from app_v2.services.xray import xray_log

    svc = get_notion_service()
    recordings = svc.fetch_recordings(limit=1000)

    if not recordings:
        return 0, 0, ["No recordings found in Notion"]

    # Find which ones are already imported
    existing = set()
    for rec in session.query(ChronosRecording).filter(
        ChronosRecording.source == "notion"
    ).all():
        # Extract page_id from "notion:<page_id>" format
        if rec.recording_id.startswith("notion:"):
            existing.add(rec.recording_id[7:])

    # Also check fuzzy matches
    matches = match_notion_to_chronos(recordings, session)

    to_import = []
    for nrec in recordings:
        if nrec.page_id in existing:
            continue  # Already imported directly
        if matches.get(nrec.page_id):
            continue  # Already in Chronos via Plaud
        to_import.append(nrec)

    if not to_import:
        xray_log("data", "notion-import", "All Notion recordings are already in Chronos!")
        return 0, 0, []

    xray_log(
        "data", "notion-import",
        f"Importing {len(to_import)} unmatched Notion recordings into Chronos..."
    )

    successes = 0
    failures = 0
    errors = []

    for i, nrec in enumerate(to_import):
        if progress_callback:
            progress_callback(i + 1, len(to_import), nrec.title)

        ok, msg = import_notion_recording(
            nrec.page_id, session,
            process=process,
            index=index,
        )
        if ok:
            successes += 1
        else:
            failures += 1
            errors.append(msg)

        xray_log(
            "pipeline", "notion-import",
            f"Imported {i + 1}/{len(to_import)}: {'✓' if ok else '✗'} {nrec.title[:40]}"
        )

    xray_log(
        "pipeline", "notion-import",
        f"Batch import complete: {successes} succeeded, {failures} failed"
    )
    return successes, failures, errors


# ═══════════════════════════════════════════════════════════════════
# Write-back — Push Chronos insights to Notion
# ═══════════════════════════════════════════════════════════════════


def write_back_to_notion(
    page_id: str,
    session: Session,
) -> Tuple[bool, str]:
    """Push Chronos AI enrichments back to a Notion page.

    Updates Notion properties with:
    - Category (most common event category)
    - Sentiment (average across events)
    - Keywords (union of all event keywords)
    - Event count
    """
    from app_v2.services.xray import xray_log

    try:
        svc = get_notion_service()
        recording_id = f"notion:{page_id}"

        # Get events for this recording
        events = session.query(ChronosEventDB).filter_by(recording_id=recording_id).all()
        if not events:
            return False, "No Chronos events found for this page"

        # Aggregate insights
        from collections import Counter
        categories = Counter()
        sentiments = []
        all_keywords = set()

        for ev in events:
            cat = ev.user_category_override or ev.category or "unknown"
            categories[cat] += 1
            if ev.sentiment is not None:
                sentiments.append(ev.sentiment)
            if ev.keywords:
                all_keywords.update(ev.keywords)

        top_category = categories.most_common(1)[0][0] if categories else "unknown"
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        sentiment_label = "positive" if avg_sentiment > 0.2 else ("negative" if avg_sentiment < -0.2 else "neutral")
        top_keywords = sorted(all_keywords)[:10]

        # Build Notion properties update
        properties = {}

        # Try to update known property types
        # We update rich_text properties for category/sentiment/keywords
        # Property names are best-effort — user's schema may vary
        client = svc._get_client()

        # First check what properties exist on this page
        page = client.pages.retrieve(page_id=page_id)
        existing_props = page.get("properties", {})
        schema_types = {name: prop.get("type") for name, prop in existing_props.items()}

        # Smart property mapping: find writable properties for our data
        def _set_rich_text(prop_name: str, text: str):
            if prop_name in schema_types and schema_types[prop_name] == "rich_text":
                properties[prop_name] = {
                    "rich_text": [{"text": {"content": text[:2000]}}]
                }

        def _set_select(prop_name: str, value: str):
            if prop_name in schema_types and schema_types[prop_name] == "select":
                properties[prop_name] = {"select": {"name": value}}

        def _set_multi_select(prop_name: str, values: list):
            if prop_name in schema_types and schema_types[prop_name] == "multi_select":
                properties[prop_name] = {
                    "multi_select": [{"name": v} for v in values[:10]]
                }

        def _set_number(prop_name: str, value: float):
            if prop_name in schema_types and schema_types[prop_name] == "number":
                properties[prop_name] = {"number": value}

        # Try common property names
        for name in ["Category", "category", "Type", "type"]:
            _set_select(name, top_category)
        for name in ["Sentiment", "sentiment", "Mood", "mood"]:
            _set_select(name, sentiment_label)
            _set_rich_text(name, f"{sentiment_label} ({avg_sentiment:+.2f})")
        for name in ["Keywords", "keywords", "Tags", "tags", "Topics", "topics"]:
            _set_multi_select(name, top_keywords)
        for name in ["Events", "events", "Event Count", "event_count"]:
            _set_number(name, len(events))

        if not properties:
            return False, "No matching writable properties found in Notion page schema"

        # Update the page
        client.pages.update(page_id=page_id, properties=properties)

        updated_props = list(properties.keys())
        xray_log(
            "data", "notion-writeback",
            f"Enriched Notion page with: {', '.join(updated_props)}"
        )
        return True, f"Updated {len(properties)} properties: {', '.join(updated_props)}"

    except Exception as e:
        logger.error(f"Error writing back to Notion page {page_id}: {e}", exc_info=True)
        return False, f"Write-back error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# Coverage Analysis
# ═══════════════════════════════════════════════════════════════════


def get_coverage_calendar(
    session: Session,
    days: int = 90,
) -> List[Dict]:
    """Build a coverage calendar showing data presence by source per day.

    Returns list of dicts: [{date, has_chronos, has_notion, chronos_count, notion_count}]
    """
    from datetime import timedelta

    today = datetime.utcnow().date()
    start = today - timedelta(days=days - 1)

    # Get Chronos recording dates
    chronos_dates: Dict[str, int] = {}
    for rec in session.query(ChronosRecording).all():
        if rec.created_at:
            d = rec.created_at.strftime("%Y-%m-%d") if isinstance(rec.created_at, datetime) else str(rec.created_at)[:10]
            if rec.source != "notion":
                chronos_dates[d] = chronos_dates.get(d, 0) + 1

    # Get Notion recording dates
    notion_dates: Dict[str, int] = {}
    try:
        svc = get_notion_service()
        recordings = svc.fetch_recordings(limit=1000)
        for r in recordings:
            d = r.date or (r.created_time[:10] if r.created_time else "")
            if d:
                notion_dates[d] = notion_dates.get(d, 0) + 1
    except Exception as e:
        logger.warning(f"Could not fetch Notion dates for calendar: {e}")

    # Also count notion-imported recordings
    notion_imported_dates: Dict[str, int] = {}
    for rec in session.query(ChronosRecording).filter(ChronosRecording.source == "notion").all():
        if rec.created_at:
            d = rec.created_at.strftime("%Y-%m-%d") if isinstance(rec.created_at, datetime) else str(rec.created_at)[:10]
            notion_imported_dates[d] = notion_imported_dates.get(d, 0) + 1

    # Build calendar
    calendar = []
    current = start
    while current <= today:
        date_str = current.strftime("%Y-%m-%d")
        c_count = chronos_dates.get(date_str, 0)
        n_count = notion_dates.get(date_str, 0)
        ni_count = notion_imported_dates.get(date_str, 0)

        calendar.append({
            "date": date_str,
            "day_of_week": current.strftime("%a"),
            "has_chronos": c_count > 0,
            "has_notion": n_count > 0,
            "has_both": c_count > 0 and n_count > 0,
            "imported": ni_count > 0,
            "chronos_count": c_count,
            "notion_count": n_count,
            "imported_count": ni_count,
            "total": c_count + n_count,
        })
        current += timedelta(days=1)

    return calendar


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp, handling Notion's format."""
    if not ts:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.utcnow()


def _index_recording_events(session: Session, recording_id: str) -> int:
    """Index all un-indexed events for a recording to Qdrant."""
    from src.chronos.qdrant_client import ChronosQdrantClient
    from src.chronos.embedding_service import ChronosEmbeddingService
    from src.models.chronos_schemas import (
        ChronosEvent as ChronosEventSchema,
        DayOfWeek,
        EventCategory,
        SpeakerMode,
    )

    qdrant = ChronosQdrantClient()
    embedder = ChronosEmbeddingService()

    # Get un-indexed events
    events = session.query(ChronosEventDB).filter(
        ChronosEventDB.recording_id == recording_id,
        ChronosEventDB.qdrant_point_id.is_(None),
    ).all()

    if not events:
        return 0

    indexed = 0
    for db_event in events:
        try:
            # Convert to Pydantic
            pydantic_event = ChronosEventSchema(
                event_id=db_event.event_id,
                recording_id=db_event.recording_id,
                start_ts=db_event.start_ts,
                end_ts=db_event.end_ts,
                day_of_week=DayOfWeek(db_event.day_of_week),
                hour_of_day=db_event.hour_of_day,
                clean_text=db_event.clean_text,
                category=EventCategory(db_event.category),
                category_confidence=db_event.category_confidence,
                sentiment=db_event.sentiment,
                keywords=db_event.keywords or [],
                speaker=SpeakerMode(db_event.speaker) if db_event.speaker else SpeakerMode.SELF_TALK,
                raw_transcript_snippet=db_event.raw_transcript_snippet,
                gemini_reasoning=db_event.gemini_reasoning,
            )

            # Embed
            vector = embedder.embed_text(
                pydantic_event.clean_text,
                task_type="RETRIEVAL_DOCUMENT",
            )

            # Upsert to Qdrant
            point_id = qdrant.upsert_event(pydantic_event, vector)

            # Update SQLite with point ID
            db_event.qdrant_point_id = point_id
            session.commit()
            indexed += 1

        except Exception as e:
            logger.error(f"Failed to index event {db_event.event_id}: {e}")
            continue

    return indexed
