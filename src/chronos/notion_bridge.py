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
    delete_chronos_events_by_recording,
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


def _extract_date_from_title(title: str, year_hint: str = "") -> Optional[str]:
    """Extract date from Notion title.

    Supports:
    - MM-DD prefix: '03-13 Operational Briefing...'
    - YYYY-MM-DD prefix: '2026-02-11 22:39:55'

    Returns YYYY-MM-DD string or None.
    """
    import re

    # Try YYYY-MM-DD first (timestamp titles like '2026-02-11 22:39:55')
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[\s T]", title)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                from datetime import date as _date

                _date(year, month, day)
                return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass

    # Try MM-DD prefix
    m = re.match(r"^(\d{2})-(\d{2})\s", title)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    year = (
        int(year_hint[:4])
        if year_hint and len(year_hint) >= 4
        else datetime.utcnow().year
    )
    try:
        from datetime import date as _date

        _date(year, month, day)  # validate
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return None


def match_notion_to_chronos(
    notion_recordings: List[NotionRecording],
    session: Session,
) -> Dict[str, Optional[str]]:
    """Match Notion recordings to Chronos recordings using fuzzy logic.

    Strategy:
    1. Parse real recording date from Notion title prefix (MM-DD format)
    2. Same-date fuzzy title match (strongest signal)
    3. Adjacent-date (+-1 day) fuzzy title match (penalized)
    4. Prevent many-to-one: each Chronos recording matches at most one Notion page

    Returns: {notion_page_id -> chronos_recording_id or None}
    """
    from app_v2.services.xray import xray_log

    # Build a lookup of Chronos recordings by date -> list of (id, title, created_at)
    chronos_recs = session.query(ChronosRecording).all()
    by_date: Dict[str, List[Tuple[str, str, datetime]]] = {}
    for rec in chronos_recs:
        if rec.created_at:
            date_key = rec.created_at.strftime("%Y-%m-%d") if isinstance(rec.created_at, datetime) else str(rec.created_at)[:10]
            by_date.setdefault(date_key, []).append(
                (rec.recording_id, rec.title or "", rec.created_at)
            )

    # Score all candidates, then assign greedily (best score first, no duplicates)
    scored_pairs: List[Tuple[float, str, str]] = (
        []
    )  # (score, notion_page_id, chronos_id)

    for nrec in notion_recordings:
        notion_title = (nrec.title or "").lower().strip()

        # Determine the true recording date: prefer title-embedded date over page date
        fallback_date = nrec.date or (
            nrec.created_time[:10] if nrec.created_time else ""
        )
        title_date = _extract_date_from_title(nrec.title or "", fallback_date)
        notion_date = title_date or fallback_date

        # Phase 1: Same-date candidates
        candidates = by_date.get(notion_date, [])
        for cid, ctitle, _ in candidates:
            ctitle_lower = (ctitle or "").lower().strip()
            if not notion_title or not ctitle_lower:
                score = 0.4  # weak date-only match
            else:
                score = SequenceMatcher(None, notion_title, ctitle_lower).ratio()
            if score >= 0.45:
                scored_pairs.append((score, nrec.page_id, cid))

        # Phase 2: Adjacent-date fuzzy match (+-1 day, penalized)
        if notion_date:
            try:
                from datetime import timedelta
                nd = datetime.strptime(notion_date, "%Y-%m-%d")
                for delta in [-1, 1]:
                    adj_date = (nd + timedelta(days=delta)).strftime("%Y-%m-%d")
                    for cid, ctitle, _ in by_date.get(adj_date, []):
                        ctitle_lower = (ctitle or "").lower().strip()
                        if notion_title and ctitle_lower:
                            score = (
                                SequenceMatcher(
                                    None, notion_title, ctitle_lower
                                ).ratio()
                                * 0.8
                            )
                            if score >= 0.45:
                                scored_pairs.append((score, nrec.page_id, cid))
            except (ValueError, TypeError):
                pass

    # Greedy assignment: best scores first, each side matched at most once
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    used_notion: Set[str] = set()
    used_chronos: Set[str] = set()
    matches: Dict[str, Optional[str]] = {}

    for score, pid, cid in scored_pairs:
        if pid in used_notion or cid in used_chronos:
            continue
        matches[pid] = cid
        used_notion.add(pid)
        used_chronos.add(cid)

    # Fill in unmatched Notion pages as None
    for nrec in notion_recordings:
        if nrec.page_id not in matches:
            matches[nrec.page_id] = None

    matched_count = sum(1 for v in matches.values() if v)
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
    prefetched: Optional["NotionRecording"] = None,
) -> Tuple[bool, str]:
    """Import a single Notion page into the Chronos pipeline.

    Idempotent: safe to call multiple times on the same page.
    - If already completed, skips and returns success.
    - If previously failed, cleans up old events and retries.
    - If new, creates recording + processes + indexes.

    Args:
        page_id: Notion page ID
        session: SQLAlchemy session
        process: Whether to run Gemini processing
        index: Whether to index events to Qdrant
        prefetched: Pre-fetched NotionRecording to avoid redundant API calls

    Returns: (success, message)
    """
    from app_v2.services.xray import xray_log

    recording_id = f"notion:{page_id}"

    try:
        # Check if already completed — skip entirely
        existing = get_chronos_recording(session, recording_id)
        if existing and existing.processing_status == "completed":
            return True, f"Already imported '{existing.title}'"

        # If previously failed, clean up old events before retrying
        if existing and existing.processing_status == "failed":
            cleaned = delete_chronos_events_by_recording(session, recording_id)
            if cleaned:
                xray_log(
                    "data",
                    "notion-import",
                    f"Cleaned {cleaned} stale events from failed import",
                )

        svc = get_notion_service()

        # Step 1: Get the page data (use prefetched when available)
        page = prefetched
        if not page:
            xray_log("data", "notion-import", f"Pulling page from Notion...")
            recordings = svc.fetch_recordings(limit=1000)
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

        # Step 2: Create/update ChronosRecording
        created_at = _parse_iso(page.created_time)
        word_count = len(transcript.split())
        estimated_duration = max(60, int(word_count / 2.5))

        rec = upsert_chronos_recording(
            session=session,
            recording_id=recording_id,
            title=page.title,
            created_at=created_at,
            duration_seconds=estimated_duration,
            local_audio_path="",
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
        # Mark as failed so resume knows to retry
        try:
            mark_chronos_recording_status(
                session,
                recording_id,
                "failed",
                error_message=str(e)[:500],
            )
        except Exception:
            pass
        return False, f"Error: {str(e)}"


# ── Batch Progress Persistence ────────────────────────────────

import json
import os

_PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "notion_import_progress.json",
)


def _load_progress() -> Dict:
    """Load batch import progress from disk."""
    try:
        with open(_PROGRESS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_progress(data: Dict) -> None:
    """Persist batch import progress to disk."""
    os.makedirs(os.path.dirname(_PROGRESS_FILE), exist_ok=True)
    with open(_PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _clear_progress() -> None:
    """Remove the progress file when the batch is complete."""
    try:
        os.remove(_PROGRESS_FILE)
    except FileNotFoundError:
        pass


def get_import_progress() -> Optional[Dict]:
    """Get current import progress for the UI.

    Returns None if no import is running/paused, otherwise:
    {
        "total": int,
        "completed": int,
        "failed": int,
        "skipped": int,
        "current_title": str,
        "status": "running" | "paused" | "done",
        "errors": [str],
    }
    """
    data = _load_progress()
    if not data:
        return None
    return data


def import_all_unmatched(
    session: Session,
    *,
    process: bool = True,
    index: bool = True,
    progress_callback=None,
    batch_size: int = 0,
) -> Tuple[int, int, List[str]]:
    """Import Notion recordings not already in Chronos.

    Resume-safe: tracks progress to disk. On re-run:
    - Completed recordings are skipped instantly
    - Failed recordings are retried (events cleaned up first)
    - New recordings are imported normally

    Args:
        batch_size: Max recordings to process in this run (0 = all)

    Returns: (success_count, failure_count, error_messages)
    """
    from app_v2.services.xray import xray_log

    svc = get_notion_service()
    recordings = svc.fetch_recordings(limit=1000)

    if not recordings:
        return 0, 0, ["No recordings found in Notion"]

    # Build set of completed notion imports (skip these entirely)
    completed_notion = set()
    failed_notion = set()
    for rec in (
        session.query(ChronosRecording)
        .filter(ChronosRecording.source == "notion")
        .all()
    ):
        if rec.recording_id.startswith("notion:"):
            pid = rec.recording_id[7:]
            if rec.processing_status == "completed":
                completed_notion.add(pid)
            elif rec.processing_status == "failed":
                failed_notion.add(pid)

    # Also check fuzzy matches (recordings already in Chronos via Plaud)
    matches = match_notion_to_chronos(recordings, session)

    to_import = []
    for nrec in recordings:
        if nrec.page_id in completed_notion:
            continue  # Already fully imported
        if matches.get(nrec.page_id):
            continue  # Already in Chronos via Plaud
        to_import.append(nrec)

    # Sort newest first — prioritize recent recordings
    to_import.sort(
        key=lambda n: n.date or n.created_time or "",
        reverse=True,
    )

    if not to_import:
        xray_log("data", "notion-import", "All Notion recordings are already in Chronos!")
        _clear_progress()
        return 0, 0, []

    # Apply batch size limit
    if batch_size > 0:
        to_import = to_import[:batch_size]

    total = len(to_import)
    retrying = sum(1 for n in to_import if n.page_id in failed_notion)
    xray_log(
        "data",
        "notion-import",
        f"Importing {total} recordings ({retrying} retries) into Chronos...",
    )

    # Initialize progress
    progress = {
        "total": total,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current_title": "",
        "current_index": 0,
        "status": "running",
        "errors": [],
    }
    _save_progress(progress)

    successes = 0
    failures = 0
    errors: List[str] = []

    for i, nrec in enumerate(to_import):
        # Update progress before each item
        progress["current_index"] = i + 1
        progress["current_title"] = (nrec.title or "Untitled")[:60]
        progress["status"] = "running"
        _save_progress(progress)

        if progress_callback:
            progress_callback(i + 1, total, nrec.title)

        ok, msg = import_notion_recording(
            nrec.page_id,
            session,
            process=process,
            index=index,
            prefetched=nrec,
        )
        if ok:
            successes += 1
            progress["completed"] = successes
        else:
            failures += 1
            errors.append(msg)
            progress["failed"] = failures
            progress["errors"] = errors[-5:]  # keep last 5

        _save_progress(progress)

        xray_log(
            "pipeline",
            "notion-import",
            f"[{i + 1}/{total}] {'✓' if ok else '✗'} {nrec.title[:40]}",
        )

    # Mark batch done
    progress["status"] = "done"
    _save_progress(progress)

    xray_log(
        "pipeline",
        "notion-import",
        f"Batch complete: {successes} succeeded, {failures} failed out of {total}",
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

        # Richer write-back: AI summary and cleaned transcript
        rec = get_chronos_recording(session, recording_id)
        if rec:
            # Build AI summary from event clean texts
            ai_summary = ""
            if rec.plaud_ai_summary:
                ai_summary = str(rec.plaud_ai_summary)
            elif events:
                ai_summary = " | ".join(
                    str(ev.clean_text)[:200] for ev in events[:5] if ev.clean_text
                )
            if ai_summary:
                for name in ["Summary", "summary", "ChatGPT Summary", "AI Summary"]:
                    _set_rich_text(name, ai_summary)

            # Cleaned transcript text
            if rec.transcript:
                clean_transcript = str(rec.transcript)
                for name in ["Transcript", "transcript", "Text", "text"]:
                    _set_rich_text(name, clean_transcript)

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


def write_back_all_matched(
    match_map: Dict[str, Optional[str]],
    session: Session,
) -> Tuple[int, int, List[str]]:
    """Write back Chronos enrichments to ALL matched Notion pages.

    Args:
        match_map: {notion_page_id → chronos_recording_id or None}
        session: SQLAlchemy session

    Returns:
        (success_count, fail_count, error_messages)
    """
    matched_page_ids = [pid for pid, cid in match_map.items() if cid]
    if not matched_page_ids:
        return 0, 0, ["No matched recordings to write back"]

    success = 0
    failed = 0
    errors = []

    for page_id in matched_page_ids:
        ok, msg = write_back_to_notion(page_id, session)
        if ok:
            success += 1
        else:
            failed += 1
            errors.append(f"{page_id[:8]}…: {msg}")

    logger.info(
        f"Batch write-back: {success} succeeded, {failed} failed out of {len(matched_page_ids)}"
    )
    return success, failed, errors


# ═══════════════════════════════════════════════════════════════════
# Change Detection — Stale Import Detection
# ═══════════════════════════════════════════════════════════════════


def detect_stale_imports(
    recordings: List,
    match_map: Dict[str, Optional[str]],
    session: Session,
) -> Dict[str, bool]:
    """Detect Notion pages that were edited after their Chronos import.

    Returns: {notion_page_id → True if stale (Notion edited after import)}
    """
    stale_map: Dict[str, bool] = {}

    # Build lookup: notion_page_id → ChronosRecording.ingested_at
    import_times: Dict[str, datetime] = {}
    for rec in (
        session.query(ChronosRecording)
        .filter(
            ChronosRecording.source == "notion",
            ChronosRecording.processing_status == "completed",
        )
        .all()
    ):
        pid = str(rec.recording_id)[7:]  # strip "notion:" prefix
        if rec.ingested_at:
            import_times[pid] = (
                rec.ingested_at
                if isinstance(rec.ingested_at, datetime)
                else datetime.utcnow()
            )

    for nrec in recordings:
        if hasattr(nrec, "page_id"):
            page_id = nrec.page_id
            edited = nrec.last_edited_time
        elif isinstance(nrec, dict):
            page_id = nrec.get("page_id", "")
            edited = nrec.get("last_edited_time", "")
        else:
            continue

        if not match_map.get(page_id):
            continue  # Not imported

        ingested = import_times.get(page_id)
        if not ingested or not edited:
            continue

        try:
            edited_dt = datetime.fromisoformat(edited.replace("Z", "+00:00")).replace(
                tzinfo=None
            )
            if edited_dt > ingested:
                stale_map[page_id] = True
        except (ValueError, TypeError):
            pass

    return stale_map


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
