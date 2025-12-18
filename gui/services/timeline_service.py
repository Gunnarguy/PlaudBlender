"""Timeline + storyboard services.

This is the "time-first" layer that sits above raw vector search.

Core idea:
- Vector search is great for "find similar text".
- Timeline questions are calendar queries first (date range, weekday, cadence),
  then summarization/aggregation.

This module keeps the first iteration SQLite-first and deterministic:
- Query recordings by created_at
- Group by local date
- Render a chronological report

Optionally, the GUI can feed the grouped evidence into an LLM to generate a
narrative/storyboard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from sqlalchemy import select

from src.database.engine import SessionLocal, init_db
from src.database.models import Recording


WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass(frozen=True)
class TimelineItem:
    day: date
    recordings: List[Dict]


def _coerce_tz(tz_name: Optional[str]) -> timezone:
    """Return a tzinfo.

    - If ZoneInfo is available and tz_name is provided, use it.
    - Otherwise fall back to local timezone (best-effort) or UTC.
    """

    if tz_name and ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)  # type: ignore[return-value]
        except Exception:
            pass

    # Local tz if possible
    try:
        return datetime.now().astimezone().tzinfo or timezone.utc
    except Exception:
        return timezone.utc


def _to_local_date(dt: datetime, tz) -> date:
    """Convert a stored datetime to a local date.

    Note: DB timestamps are often stored naive in this project.
    Strategy:
    - If naive: treat as UTC (consistent with datetime.utcnow default)
    - Convert to tz, then take .date()
    """

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).date()


def fetch_recordings_in_year(
    year: int,
    tz_name: Optional[str] = None,
) -> List[Recording]:
    """Fetch recordings whose created_at falls within the given year.

    We do a broad DB filter by created_at year bounds, then do exact local-date
    calculations in Python to support timezone-based grouping.
    """

    init_db()
    session = SessionLocal()
    try:
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)
        rows = (
            session.execute(
                select(Recording)
                .where(Recording.created_at >= start)
                .where(Recording.created_at < end)
                .order_by(Recording.created_at.asc())
            )
            .scalars()
            .all()
        )
        return list(rows)
    finally:
        session.close()


def query_weekday_in_year(
    year: int,
    weekday: int,
    tz_name: Optional[str] = None,
) -> List[TimelineItem]:
    """Return recordings grouped by local date for a weekday in a given year.

    Args:
        year: e.g. 2025
        weekday: Python weekday (0=Monday..6=Sunday)
        tz_name: IANA tz name, e.g. "America/Los_Angeles" (optional)

    Returns:
        Chronologically sorted list of TimelineItem.
    """

    if weekday < 0 or weekday > 6:
        raise ValueError("weekday must be 0..6")

    tz = _coerce_tz(tz_name)
    recordings = fetch_recordings_in_year(year, tz_name=tz_name)

    grouped: Dict[date, List[Dict]] = defaultdict(list)

    for rec in recordings:
        created_at = getattr(rec, "created_at", None)
        if not isinstance(created_at, datetime):
            continue
        day = _to_local_date(created_at, tz)
        if day.weekday() != weekday:
            continue

        extra = rec.extra if isinstance(rec.extra, dict) else {}
        grouped[day].append(
            {
                "id": str(rec.id),
                "title": rec.title or rec.filename or "Untitled",
                "created_at": created_at,
                "local_date": day.isoformat(),
                # Keep transcript available for downstream summarization.
                "transcript": rec.transcript or "",
                "themes": extra.get("themes"),
            }
        )

    items = [TimelineItem(day=d, recordings=grouped[d]) for d in sorted(grouped.keys())]
    return items


def render_weekday_report(
    year: int,
    weekday: int,
    tz_name: Optional[str] = None,
    include_snippets: bool = True,
    snippet_chars: int = 280,
) -> str:
    """Render a human-readable report for weekday-in-year queries."""

    items = query_weekday_in_year(year=year, weekday=weekday, tz_name=tz_name)
    weekday_name = WEEKDAYS[weekday]

    if not items:
        return f"No recordings found for {weekday_name}s in {year}."

    lines: List[str] = []
    lines.append("═" * 66)
    lines.append(f"🗓 TIMELINE: What happened every {weekday_name} in {year}")
    if tz_name:
        lines.append(f"   Timezone: {tz_name}")
    lines.append(f"   Days with recordings: {len(items)}")
    total_recs = sum(len(it.recordings) for it in items)
    lines.append(f"   Total recordings: {total_recs}")
    lines.append("═" * 66)
    lines.append("")

    for it in items:
        lines.append(f"## {it.day.isoformat()} ({weekday_name})")
        for r in it.recordings:
            lines.append(f"- 🎧 {r.get('title','Untitled')}  (id={r.get('id')})")
            if include_snippets:
                t = (r.get("transcript") or "").strip()
                if t:
                    snippet = (t[:snippet_chars] + "…") if len(t) > snippet_chars else t
                    # keep report readable
                    snippet = " ".join(snippet.split())
                    lines.append(f"    • snippet: {snippet}")
        lines.append("")

    return "\n".join(lines)


def build_storyboard_prompt(
    items: List[TimelineItem],
    weekday_name: str,
    year: int,
) -> str:
    """Create an LLM-friendly prompt for narrative/storyboard generation."""

    # Keep it bounded: include short evidence per day.
    evidence_lines: List[str] = []
    for it in items:
        evidence_lines.append(f"DATE: {it.day.isoformat()} ({weekday_name})")
        for r in it.recordings:
            title = r.get("title") or "Untitled"
            transcript = (r.get("transcript") or "").strip()
            snippet = transcript[:1200] if transcript else ""
            evidence_lines.append(f"- {title} (id={r.get('id')})")
            if snippet:
                evidence_lines.append(f"  SNIPPET: {snippet}")
        evidence_lines.append("")

    return (
        "You are building a chronological storyboard from voice recordings.\n"
        f"Task: summarize what happened every {weekday_name} in {year}.\n\n"
        "Requirements:\n"
        "- Keep chronology.\n"
        "- Identify recurring threads across weeks (projects, people, topics).\n"
        "- Note changes over time (progression).\n"
        "- Produce two outputs:\n"
        "  1) A per-date bullet storyboard (2-5 bullets per date)\n"
        "  2) A roll-up narrative (5-12 paragraphs) that connects the dots\n"
        "- If evidence is thin, say so and avoid hallucination.\n\n"
        "EVIDENCE:\n" + "\n".join(evidence_lines)
    )
