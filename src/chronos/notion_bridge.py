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

    # Phase 0: Direct match — pages already imported via notion:{page_id}
    already_imported: Dict[str, str] = {}
    notion_rec_ids = {
        rec.recording_id: rec.recording_id
        for rec in chronos_recs
        if rec.recording_id.startswith("notion:")
    }
    for nrec in notion_recordings:
        direct_id = f"notion:{nrec.page_id}"
        if direct_id in notion_rec_ids:
            already_imported[nrec.page_id] = direct_id

    # Score all candidates, then assign greedily (best score first, no duplicates)
    scored_pairs: List[Tuple[float, str, str]] = (
        []
    )  # (score, notion_page_id, chronos_id)

    for nrec in notion_recordings:
        if nrec.page_id in already_imported:
            continue  # Already matched by direct import
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

    # Merge direct imports into matches
    matches.update(already_imported)

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
                existing = existing_props.get(prop_name, {}).get("rich_text", [])
                if not existing:
                    properties[prop_name] = {"rich_text": _build_rich_text_chunks(text)}

        def _set_select(prop_name: str, value: str):
            if prop_name in schema_types and schema_types[prop_name] == "select":
                existing = existing_props.get(prop_name, {}).get("select")
                if not existing:
                    properties[prop_name] = {"select": {"name": value}}

        def _set_multi_select(prop_name: str, values: list):
            if prop_name in schema_types and schema_types[prop_name] == "multi_select":
                existing = existing_props.get(prop_name, {}).get("multi_select", [])
                if not existing:
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
    from app_v2.services.xray import xray_log

    matched_page_ids = [pid for pid, cid in match_map.items() if cid]
    if not matched_page_ids:
        return 0, 0, ["No matched recordings to write back"]

    total = len(matched_page_ids)
    xray_log(
        "data",
        "notion-writeback",
        f"Writing back Chronos insights to {total} Notion pages...",
    )

    success = 0
    failed = 0
    errors = []

    for i, page_id in enumerate(matched_page_ids):
        ok, msg = write_back_to_notion(page_id, session)
        if ok:
            success += 1
        else:
            failed += 1
            errors.append(f"{page_id[:8]}…: {msg}")
            xray_log(
                "data",
                "notion-writeback",
                f"[{i+1}/{total}] ✗ {page_id[:8]}…: {msg[:50]}",
                level="warn",
            )

    xray_log(
        "data",
        "notion-writeback",
        f"Write-back complete: {success} succeeded, {failed} failed out of {total}",
    )
    logger.info(
        f"Batch write-back: {success} succeeded, {failed} failed out of {total}"
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
    from app_v2.services.xray import xray_log

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

    if stale_map:
        xray_log(
            "data",
            "notion-stale",
            f"Found {len(stale_map)} pages edited in Notion after import — may need re-import",
            level="warn",
        )
    else:
        xray_log("data", "notion-stale", "All imported pages are up to date")

    return stale_map


# ═══════════════════════════════════════════════════════════════════
# Coverage Analysis
# ═══════════════════════════════════════════════════════════════════


def get_coverage_calendar(
    session: Session,
    days: int = 90,
    notion_recordings: List = None,
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

    # Get Notion recording dates (use pre-fetched if available)
    notion_dates: Dict[str, int] = {}
    try:
        recordings = notion_recordings
        if recordings is None:
            svc = get_notion_service()
            recordings = svc.fetch_recordings(limit=1000)
        for r in recordings:
            if hasattr(r, "date"):
                d = r.date or (r.created_time[:10] if r.created_time else "")
            elif isinstance(r, dict):
                d = r.get("date") or (r.get("created_time", "")[:10])
            else:
                continue
            if d:
                notion_dates[d] = notion_dates.get(d, 0) + 1
    except Exception as e:
        logger.warning(f"Could not process Notion dates for calendar: {e}")

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


# ═══════════════════════════════════════════════════════════════════
# Notion Sync & Reformat Engine
# ═══════════════════════════════════════════════════════════════════

_REFORMAT_PROGRESS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "notion_reformat_progress.json",
)


def _save_reformat_progress(data: Dict) -> None:
    os.makedirs(os.path.dirname(_REFORMAT_PROGRESS_FILE), exist_ok=True)
    with open(_REFORMAT_PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_reformat_progress() -> Dict:
    try:
        with open(_REFORMAT_PROGRESS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _clear_reformat_progress() -> None:
    try:
        os.remove(_REFORMAT_PROGRESS_FILE)
    except FileNotFoundError:
        pass


def get_reformat_progress() -> Optional[Dict]:
    """Get current reformat/push progress for the UI."""
    data = _load_reformat_progress()
    return data if data else None


# ── Title Normalizer ─────────────────────────────────────────────


def normalize_notion_title(title: str, created_time: str = "") -> str:
    """Normalize a Notion page title to MM-DD-YYYY prefix format.

    Rules:
    1. Parse existing date from title (YYYY-MM-DD, MM-DD, or MM-DD-YYYY)
    2. If no date found, fall back to created_time (ISO 8601)
    3. Strip any existing date prefix, prepend MM-DD-YYYY
    4. If title IS just a date/timestamp, keep as MM-DD-YYYY only
    """
    import re
    from datetime import date as _date

    cleaned_title = (title or "").strip()
    parsed_date = None

    # Try YYYY-MM-DD prefix (e.g. '2026-02-11 22:39:55')
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[\s T]?", cleaned_title)
    if m:
        try:
            parsed_date = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            # Strip the date (and optional timestamp) prefix
            rest = cleaned_title[m.end() :]
            # Also strip trailing timestamp if the whole title was a timestamp
            rest = re.sub(r"^\d{2}:\d{2}(:\d{2})?\s*", "", rest).strip()
            cleaned_title = rest
        except ValueError:
            pass

    # Try MM-DD-YYYY prefix (already in target format)
    if not parsed_date:
        m = re.match(r"^(\d{2})-(\d{2})-(\d{4})\s*", cleaned_title)
        if m:
            try:
                parsed_date = _date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                cleaned_title = cleaned_title[m.end() :].strip()
            except ValueError:
                pass

    # Try MM-DD prefix (e.g. '03-13 Operational Briefing')
    if not parsed_date:
        m = re.match(r"^(\d{2})-(\d{2})\s+", cleaned_title)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                # Infer year from created_time
                year = datetime.utcnow().year
                if created_time and len(created_time) >= 4:
                    try:
                        year = int(created_time[:4])
                    except ValueError:
                        pass
                try:
                    parsed_date = _date(year, month, day)
                    cleaned_title = cleaned_title[m.end() :].strip()
                except ValueError:
                    pass

    # Fallback: use created_time
    if not parsed_date and created_time:
        try:
            ct = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
            parsed_date = ct.date()
        except (ValueError, TypeError):
            pass

    # Last resort: today
    if not parsed_date:
        parsed_date = datetime.utcnow().date()

    date_prefix = parsed_date.strftime("%m-%d-%Y")

    if not cleaned_title:
        return date_prefix

    return f"{date_prefix} {cleaned_title}"


# ── Rich Text Chunking ──────────────────────────────────────────


def _build_rich_text_chunks(text: str, max_chunk: int = 2000) -> list:
    """Split text into Notion rich_text array chunks at word boundaries.

    Notion rich_text elements are limited to 2000 chars each,
    but you can have multiple in an array for unlimited total length.
    """
    if not text:
        return [{"text": {"content": ""}}]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chunk:
            chunks.append({"text": {"content": remaining}})
            break
        # Find a word boundary to split at
        split_at = remaining.rfind(" ", 0, max_chunk)
        if split_at <= 0:
            split_at = max_chunk  # No space found, hard split
        chunks.append({"text": {"content": remaining[:split_at]}})
        remaining = remaining[split_at:].lstrip()

    return chunks


# ── Page Reformatter ─────────────────────────────────────────────


def reformat_notion_page(
    page_id: str,
    notion_rec,
    session: Session,
    match_map: Dict[str, Optional[str]],
    dry_run: bool = True,
) -> Dict:
    """Compute (and optionally apply) reformatting for a single Notion page.

    Returns a diff dict:
    {
        "page_id": str,
        "changes": {property_name: {"old": ..., "new": ...}},
        "applied": bool,
        "error": str or None,
    }
    """
    from app_v2.services.xray import xray_log

    result = {"page_id": page_id, "changes": {}, "applied": False, "error": None}

    try:
        svc = get_notion_service()
        client = svc._get_client()
        old_title = (
            notion_rec.title
            if hasattr(notion_rec, "title")
            else str(notion_rec.get("title", ""))
        )
        created_time = (
            notion_rec.created_time
            if hasattr(notion_rec, "created_time")
            else str(notion_rec.get("created_time", ""))
        )

        # 1. Normalize title
        new_title = normalize_notion_title(old_title, created_time)
        if new_title != old_title:
            result["changes"]["Title"] = {"old": old_title, "new": new_title}

        # 2. Parse date for Date property
        import re

        date_match = re.match(r"^(\d{2})-(\d{2})-(\d{4})", new_title)
        iso_date = None
        if date_match:
            m, d, y = (
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
            iso_date = f"{y:04d}-{m:02d}-{d:02d}"

        # 3. Check matched Chronos recording for AI enrichment
        chronos_id = match_map.get(page_id)
        enrichments = {}
        if chronos_id:
            events = (
                session.query(ChronosEventDB).filter_by(recording_id=chronos_id).all()
            )
            rec = get_chronos_recording(session, chronos_id)
            if events:
                from collections import Counter

                categories = Counter()
                sentiments_list = []
                all_keywords = set()
                for ev in events:
                    cat = ev.user_category_override or ev.category or "unknown"
                    categories[cat] += 1
                    if ev.sentiment is not None:
                        sentiments_list.append(ev.sentiment)
                    if ev.keywords:
                        all_keywords.update(ev.keywords)

                enrichments["category"] = (
                    categories.most_common(1)[0][0] if categories else "unknown"
                )
                avg_sent = (
                    sum(sentiments_list) / len(sentiments_list)
                    if sentiments_list
                    else 0
                )
                enrichments["sentiment"] = (
                    "positive"
                    if avg_sent > 0.2
                    else ("negative" if avg_sent < -0.2 else "neutral")
                )
                enrichments["keywords"] = sorted(all_keywords)[:10]
                enrichments["event_count"] = len(events)

                # AI summary
                if rec and rec.plaud_ai_summary:
                    enrichments["summary"] = str(rec.plaud_ai_summary)
                elif events:
                    enrichments["summary"] = " | ".join(
                        str(ev.clean_text)[:200] for ev in events[:5] if ev.clean_text
                    )

                # Full cleaned transcript
                if rec and rec.transcript:
                    enrichments["transcript"] = str(rec.transcript)

        # Now retrieve page to check existing properties
        page = client.pages.retrieve(page_id=page_id)
        existing_props = page.get("properties", {})
        schema_types = {name: prop.get("type") for name, prop in existing_props.items()}

        properties_update = {}

        # Title update — find the title property
        title_prop_name = None
        for name, ptype in schema_types.items():
            if ptype == "title":
                title_prop_name = name
                break

        if title_prop_name and new_title != old_title:
            properties_update[title_prop_name] = {
                "title": [{"text": {"content": new_title}}]
            }

        # Date property — find date-type property
        if iso_date:
            date_prop_name = None
            for name in ["Date", "date", "Recording Date", "Created"]:
                if name in schema_types and schema_types[name] == "date":
                    date_prop_name = name
                    break
            if date_prop_name:
                # Check existing date value
                existing_date_val = existing_props.get(date_prop_name, {})
                existing_date = None
                if existing_date_val.get("date"):
                    existing_date = existing_date_val["date"].get("start")

                if existing_date != iso_date:
                    properties_update[date_prop_name] = {"date": {"start": iso_date}}
                    result["changes"]["Date"] = {"old": existing_date, "new": iso_date}

        # Enrichment properties — only fill if field is empty/missing
        if enrichments:
            for name in ["Category", "category", "Type", "type"]:
                if name in schema_types and schema_types[name] == "select":
                    existing_val = existing_props.get(name, {}).get("select")
                    if not existing_val:
                        properties_update[name] = {
                            "select": {"name": enrichments["category"]}
                        }
                        result["changes"]["Category"] = {
                            "old": None,
                            "new": enrichments["category"],
                        }
                    break

            for name in ["Sentiment", "sentiment", "Mood", "mood"]:
                if name in schema_types and schema_types[name] == "select":
                    existing_val = existing_props.get(name, {}).get("select")
                    if not existing_val:
                        properties_update[name] = {
                            "select": {"name": enrichments["sentiment"]}
                        }
                        result["changes"]["Sentiment"] = {
                            "old": None,
                            "new": enrichments["sentiment"],
                        }
                    break
                if name in schema_types and schema_types[name] == "rich_text":
                    existing_val = existing_props.get(name, {}).get("rich_text", [])
                    if not existing_val:
                        properties_update[name] = {
                            "rich_text": [
                                {"text": {"content": enrichments["sentiment"]}}
                            ]
                        }
                        result["changes"]["Sentiment"] = {
                            "old": None,
                            "new": enrichments["sentiment"],
                        }
                    break

            for name in ["Keywords", "keywords", "Tags", "tags", "Topics", "topics"]:
                if name in schema_types and schema_types[name] == "multi_select":
                    existing_val = existing_props.get(name, {}).get("multi_select", [])
                    if not existing_val and enrichments.get("keywords"):
                        properties_update[name] = {
                            "multi_select": [
                                {"name": v} for v in enrichments["keywords"]
                            ]
                        }
                        result["changes"]["Keywords"] = {
                            "old": [],
                            "new": enrichments["keywords"],
                        }
                    break

            for name in ["Events", "events", "Event Count", "event_count"]:
                if name in schema_types and schema_types[name] == "number":
                    properties_update[name] = {"number": enrichments["event_count"]}
                    result["changes"]["Event Count"] = {
                        "old": None,
                        "new": enrichments["event_count"],
                    }
                    break

            if enrichments.get("summary"):
                for name in ["Summary", "summary", "ChatGPT Summary", "AI Summary"]:
                    if name in schema_types and schema_types[name] == "rich_text":
                        existing_val = existing_props.get(name, {}).get("rich_text", [])
                        if not existing_val:
                            properties_update[name] = {
                                "rich_text": _build_rich_text_chunks(
                                    enrichments["summary"]
                                )
                            }
                            result["changes"]["Summary"] = {
                                "old": None,
                                "new": enrichments["summary"][:100] + "...",
                            }
                        break

            if enrichments.get("transcript"):
                for name in ["Transcript", "transcript", "Text", "text"]:
                    if name in schema_types and schema_types[name] == "rich_text":
                        old_rt = existing_props.get(name, {}).get("rich_text", [])
                        old_len = sum(
                            len(c.get("text", {}).get("content", "")) for c in old_rt
                        )
                        if not old_rt:
                            properties_update[name] = {
                                "rich_text": _build_rich_text_chunks(
                                    enrichments["transcript"]
                                )
                            }
                            new_len = len(enrichments["transcript"])
                            result["changes"]["Transcript"] = {
                                "old": f"{old_len} chars",
                                "new": f"{new_len} chars",
                            }
                        break

        if not properties_update:
            return result  # Nothing to change

        if dry_run:
            return result

        # Execute the update
        client.pages.update(page_id=page_id, properties=properties_update)
        result["applied"] = True
        xray_log("data", "notion-reformat", f"Reformatted: {new_title[:50]}")

    except Exception as e:
        logger.error(f"Error reformatting page {page_id}: {e}", exc_info=True)
        result["error"] = str(e)

    return result


# ── Batch Reformat ───────────────────────────────────────────────


def reformat_all_notion_pages(
    session: Session,
    match_map: Dict[str, Optional[str]],
    dry_run: bool = True,
) -> Dict:
    """Reformat all Notion pages for consistent titles, dates, and enrichment.

    In dry_run mode: returns proposed changes without writing.
    In execute mode: applies changes with progress tracking and rate limiting.

    Returns: {total, reformatted, skipped, errors, changes_by_type, backup_file, diffs}
    """
    from app_v2.services.xray import xray_log

    svc = get_notion_service()
    recordings = svc.fetch_recordings(limit=1000)

    if not recordings:
        return {
            "total": 0,
            "reformatted": 0,
            "skipped": 0,
            "errors": [],
            "changes_by_type": {},
            "diffs": [],
        }

    total = len(recordings)
    reformatted = 0
    skipped = 0
    errors = []
    changes_by_type: Dict[str, int] = {}
    diffs = []

    # Backup originals before any writes (execute mode only)
    backup_file = None
    if not dry_run:
        backup_data = []
        for r in recordings:
            backup_data.append(
                {
                    "page_id": r.page_id,
                    "title": r.title,
                    "created_time": r.created_time,
                    "date": r.date,
                }
            )
        backup_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            f"notion_reformat_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)
        xray_log(
            "data", "notion-reformat", f"Backup saved: {os.path.basename(backup_file)}"
        )

    progress = {
        "total": total,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current_title": "",
        "current_index": 0,
        "status": "running",
        "mode": "dry-run" if dry_run else "execute",
        "errors": [],
    }
    _save_reformat_progress(progress)

    for i, nrec in enumerate(recordings):
        progress["current_index"] = i + 1
        progress["current_title"] = (nrec.title or "Untitled")[:60]
        _save_reformat_progress(progress)

        diff = reformat_notion_page(
            nrec.page_id, nrec, session, match_map, dry_run=dry_run
        )

        title_short = (nrec.title or "Untitled")[:40]
        if diff.get("error"):
            errors.append(f"{title_short}: {diff['error']}")
            progress["failed"] += 1
            xray_log(
                "data",
                "notion-reformat",
                f"[{i+1}/{total}] ✗ {title_short}: {diff['error'][:60]}",
                level="warn",
            )
        elif diff["changes"]:
            reformatted += 1
            progress["completed"] = reformatted
            for change_key in diff["changes"]:
                changes_by_type[change_key] = changes_by_type.get(change_key, 0) + 1
            diffs.append(diff)
            change_list = ", ".join(diff["changes"][:3])
            xray_log(
                "data",
                "notion-reformat",
                f"[{i+1}/{total}] ✓ {title_short} — {change_list}",
            )
        else:
            skipped += 1
            progress["skipped"] = skipped
            xray_log(
                "data",
                "notion-reformat",
                f"[{i+1}/{total}] · {title_short} — already OK",
            )

        _save_reformat_progress(progress)

        # Rate limiting — only in execute mode (dry run just reads)
        if not dry_run and diff["changes"]:
            _time.sleep(0.35)  # ~3 req/sec

    progress["status"] = "done"
    _save_reformat_progress(progress)

    mode_label = "Preview" if dry_run else "Reformat"
    xray_log(
        "data",
        "notion-reformat",
        f"{mode_label} complete: {reformatted} changed, {skipped} already OK, {len(errors)} errors",
    )

    return {
        "total": total,
        "reformatted": reformatted,
        "skipped": skipped,
        "errors": errors,
        "changes_by_type": changes_by_type,
        "backup_file": backup_file,
        "diffs": diffs[:50],  # Cap preview diffs at 50
    }


# ── Push Chronos-Only Recordings to Notion ───────────────────────


def push_chronos_to_notion(
    recording_id: str,
    session: Session,
    dry_run: bool = True,
) -> Dict:
    """Create a new Notion page for a Chronos recording.

    Returns: {recording_id, title, properties, page_id (if created), error}
    """
    from app_v2.services.xray import xray_log

    result = {
        "recording_id": recording_id,
        "title": "",
        "properties": {},
        "page_id": None,
        "error": None,
        "dry_run": dry_run,
    }

    try:
        rec = get_chronos_recording(session, recording_id)
        if not rec:
            result["error"] = f"Recording {recording_id} not found"
            return result

        events = (
            session.query(ChronosEventDB).filter_by(recording_id=recording_id).all()
        )

        # Build title
        rec_title = rec.title or "Untitled"
        created_at = rec.created_at
        created_iso = ""
        if created_at:
            if isinstance(created_at, datetime):
                created_iso = created_at.isoformat()
            else:
                created_iso = str(created_at)
        title = normalize_notion_title(rec_title, created_iso)
        result["title"] = title

        # Parse date for Date property
        import re

        date_match = re.match(r"^(\d{2})-(\d{2})-(\d{4})", title)
        iso_date = None
        if date_match:
            mm, dd, yyyy = (
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
            iso_date = f"{yyyy:04d}-{mm:02d}-{dd:02d}"

        # Aggregate event data
        from collections import Counter

        categories = Counter()
        sentiments_list = []
        all_keywords = set()
        for ev in events:
            cat = ev.user_category_override or ev.category or "unknown"
            categories[cat] += 1
            if ev.sentiment is not None:
                sentiments_list.append(ev.sentiment)
            if ev.keywords:
                all_keywords.update(ev.keywords)

        top_category = categories.most_common(1)[0][0] if categories else "unknown"
        avg_sent = sum(sentiments_list) / len(sentiments_list) if sentiments_list else 0
        sentiment_label = (
            "positive"
            if avg_sent > 0.2
            else ("negative" if avg_sent < -0.2 else "neutral")
        )
        top_keywords = sorted(all_keywords)[:10]

        # Summary
        summary = ""
        if rec.plaud_ai_summary:
            summary = str(rec.plaud_ai_summary)
        elif events:
            summary = " | ".join(
                str(ev.clean_text)[:200] for ev in events[:5] if ev.clean_text
            )

        # Transcript
        transcript = str(rec.transcript) if rec.transcript else ""

        # Build Notion properties
        settings = get_settings()
        db_id = settings.notion_database_id
        if not db_id:
            result["error"] = "NOTION_DATABASE_ID not configured"
            return result

        # Discover the database schema to build properties correctly
        svc = get_notion_service()
        client = svc._get_client()

        # Retrieve DB schema
        db_info = client.databases.retrieve(database_id=db_id)
        db_schema = db_info.get("properties", {})
        schema_types = {name: prop.get("type") for name, prop in db_schema.items()}

        properties = {}

        # Title property (required)
        title_prop_name = None
        for name, ptype in schema_types.items():
            if ptype == "title":
                title_prop_name = name
                break
        if title_prop_name:
            properties[title_prop_name] = {"title": [{"text": {"content": title}}]}

        # Date
        for name in ["Date", "date", "Recording Date", "Created"]:
            if name in schema_types and schema_types[name] == "date" and iso_date:
                properties[name] = {"date": {"start": iso_date}}
                break

        # Category
        for name in ["Category", "category", "Type", "type"]:
            if name in schema_types and schema_types[name] == "select":
                properties[name] = {"select": {"name": top_category}}
                break

        # Sentiment
        for name in ["Sentiment", "sentiment", "Mood", "mood"]:
            if name in schema_types and schema_types[name] == "select":
                properties[name] = {"select": {"name": sentiment_label}}
                break
            if name in schema_types and schema_types[name] == "rich_text":
                properties[name] = {
                    "rich_text": [
                        {"text": {"content": f"{sentiment_label} ({avg_sent:+.2f})"}}
                    ]
                }
                break

        # Keywords
        for name in ["Keywords", "keywords", "Tags", "tags", "Topics", "topics"]:
            if name in schema_types and schema_types[name] == "multi_select":
                properties[name] = {"multi_select": [{"name": v} for v in top_keywords]}
                break

        # Event count
        for name in ["Events", "events", "Event Count", "event_count"]:
            if name in schema_types and schema_types[name] == "number":
                properties[name] = {"number": len(events)}
                break

        # Summary
        if summary:
            for name in ["Summary", "summary", "ChatGPT Summary", "AI Summary"]:
                if name in schema_types and schema_types[name] == "rich_text":
                    properties[name] = {"rich_text": _build_rich_text_chunks(summary)}
                    break

        # Transcript
        if transcript:
            for name in ["Transcript", "transcript", "Text", "text"]:
                if name in schema_types and schema_types[name] == "rich_text":
                    properties[name] = {
                        "rich_text": _build_rich_text_chunks(transcript)
                    }
                    break

        result["properties"] = {k: "set" for k in properties}

        if dry_run:
            return result

        # Create the page
        new_page = client.pages.create(
            parent={"database_id": db_id},
            properties=properties,
        )
        result["page_id"] = new_page["id"]
        xray_log("data", "notion-push", f"Created Notion page: {title[:50]}")

    except Exception as e:
        logger.error(f"Error pushing {recording_id} to Notion: {e}", exc_info=True)
        result["error"] = str(e)

    return result


def push_all_chronos_only(
    session: Session,
    match_map: Dict[str, Optional[str]],
    dry_run: bool = True,
) -> Dict:
    """Push all Chronos-only recordings (not in Notion) as new Notion pages.

    Only pushes source="plaud" recordings (won't re-push Notion imports).
    Sorted by created_at ascending for chronological insertion.

    Returns: {total, created, skipped, errors, pages}
    """
    from app_v2.services.xray import xray_log

    # Get all matched Chronos recording IDs (already in Notion)
    matched_chronos_ids = set(cid for cid in match_map.values() if cid)

    # Also include notion-imported recordings (source="notion")
    notion_imported_ids = set()
    for rec in (
        session.query(ChronosRecording)
        .filter(ChronosRecording.source == "notion")
        .all()
    ):
        notion_imported_ids.add(rec.recording_id)

    # Find Chronos recordings NOT matched to Notion
    all_recordings = (
        session.query(ChronosRecording)
        .filter(
            ChronosRecording.source != "notion",
        )
        .all()
    )

    to_push = []
    for rec in all_recordings:
        if rec.recording_id in matched_chronos_ids:
            continue
        if rec.processing_status != "completed":
            continue
        to_push.append(rec)

    # Sort oldest first for chronological Notion insertion
    to_push.sort(key=lambda r: r.created_at or datetime.min)

    if not to_push:
        return {"total": 0, "created": 0, "skipped": 0, "errors": [], "pages": []}

    total = len(to_push)
    created = 0
    skipped = 0
    errors = []
    pages = []

    progress = {
        "total": total,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "current_title": "",
        "current_index": 0,
        "status": "running",
        "mode": "dry-run" if dry_run else "push",
        "errors": [],
    }
    _save_reformat_progress(progress)

    xray_log(
        "data",
        "notion-push",
        f"{'Preview' if dry_run else 'Pushing'} {total} Chronos recordings to Notion...",
    )

    for i, rec in enumerate(to_push):
        progress["current_index"] = i + 1
        progress["current_title"] = (rec.title or "Untitled")[:60]
        _save_reformat_progress(progress)

        result = push_chronos_to_notion(rec.recording_id, session, dry_run=dry_run)

        title_short = (rec.title or "Untitled")[:40]
        if result.get("error"):
            errors.append(f"{title_short}: {result['error']}")
            progress["failed"] += 1
            xray_log(
                "data",
                "notion-push",
                f"[{i+1}/{total}] ✗ {title_short}: {result['error'][:60]}",
                level="warn",
            )
        elif result.get("page_id") or dry_run:
            created += 1
            progress["completed"] = created
            pages.append(result)
            xray_log("data", "notion-push", f"[{i+1}/{total}] ✓ {title_short}")
        else:
            skipped += 1
            progress["skipped"] = skipped
            xray_log(
                "data", "notion-push", f"[{i+1}/{total}] · {title_short} — skipped"
            )

        _save_reformat_progress(progress)

        # Rate limiting in execute mode
        if not dry_run and not result.get("error"):
            _time.sleep(0.35)

    progress["status"] = "done"
    _save_reformat_progress(progress)

    mode_label = "Preview" if dry_run else "Push"
    xray_log(
        "data",
        "notion-push",
        f"{mode_label} complete: {created} pages, {skipped} skipped, {len(errors)} errors",
    )

    return {
        "total": total,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "pages": pages[:50],
    }
