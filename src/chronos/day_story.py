"""The Day Story: a short written account of one day, made from its moments.

The Timeline shows a day's moments one by one; the owner asked to "understand what
happened in each and every day". This turns a day's moments, and its recordings'
Plaud summaries, into a headline, a few paragraphs in time order and the threads
left open. Gemini writes it through the AGY subscription (no API cost). Stories are
cached per day in chronos_day_stories and rewritten only when the day's moments
change. Writing takes up to a few minutes, so it runs in a background thread and
the endpoint answers "pending" until it is done (2026-10-06).
"""
from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from src.config import get_local_timezone

logger = logging.getLogger(__name__)

LEASE = timedelta(minutes=20)  # a pending story older than this is started again
MAX_MOMENTS = 450
MAX_SUMMARY_CHARS = 700

SYSTEM = """You write the daily log of one person's recorded day, for that person to read later.
You are given the day's recordings and the moments extracted from them, in time order.

Write:
- "headline": one sentence, at most 16 words, saying what the day was mostly about.
- "story": two to five short paragraphs in time order (morning, afternoon, evening only
  where there are moments then), in second person and past tense ("You started in ...").
  Say who was involved, where, what was decided and what happened. Plain, specific words.
- "open_threads": things left open at the end of the day: follow-ups, promises, questions,
  pending decisions. Empty when there are none.

Rules: use only what the moments and summaries say; never invent names, places, numbers or
outcomes. Keep names exactly as written. When something is unclear, say less rather than
guess. At most 260 words in total. Answer with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "story": {"type": "string"},
        "open_threads": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "story", "open_threads"],
}


def _clock(ts: datetime) -> str:
    """A moment's local clock time; moments are stored in UTC (naive means UTC)."""
    aware = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return aware.astimezone(get_local_timezone()).strftime("%H:%M")


def _recording_clock(ts: datetime) -> str:
    """Recording start times from data_service are already local (naive means local)."""
    return (ts.astimezone(get_local_timezone()) if ts.tzinfo else ts).strftime("%H:%M")


def day_inputs(svc, day: str) -> Optional[dict]:
    """The day's recordings and moments, oldest first; None when the day is unknown."""
    detail = svc.get_day_detail(day)
    if detail is None:
        return None
    recordings = sorted(detail.recordings or [], key=lambda r: r.start_time)
    ids = {r.recording_id for r in recordings}
    moments = sorted((e for e in svc._get_all_events() if e.recording_id in ids), key=lambda e: e.start_ts)
    return {"display": detail.date_display, "recordings": recordings, "moments": moments}


def fingerprint(moments) -> str:
    digest = hashlib.sha256()
    for moment in sorted(str(m.id) for m in moments):
        digest.update(moment.encode())
    return digest.hexdigest()


def build_prompt(display: str, recordings, moments, summaries: dict[str, str]) -> str:
    lines = [f"Day: {display}", "", "Recordings:"]
    for rec in recordings:
        minutes = int((rec.duration_seconds or 0) // 60)
        length = f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"
        lines.append(f"- {_recording_clock(rec.start_time)}, {length if minutes else 'length unknown'}, {rec.event_count} moments")
        summary = (summaries.get(rec.recording_id) or "").strip()
        if summary:
            lines.append(f"  Plaud summary: {summary[:MAX_SUMMARY_CHARS]}")
    shown = moments[:MAX_MOMENTS]
    width = 300 if len(moments) <= 200 else 170
    lines += ["", f"Moments ({len(moments)}{f', first {MAX_MOMENTS} shown' if len(moments) > MAX_MOMENTS else ''}):"]
    for m in shown:
        speaker = f" {m.speaker}:" if m.speaker and not str(m.speaker).lower().startswith("unknown") else ""
        text = " ".join(str(m.clean_text or "").split())[:width]
        lines.append(f"{_clock(m.start_ts)} [{m.category}, mood {m.sentiment:+.1f}]{speaker} {text}")
    return "\n".join(lines)


class DayStoryService:
    """Reads and writes chronos_day_stories; writing runs in the background."""

    _starting = threading.Lock()

    def __init__(self, session_factory: Callable, agy=None, summaries: Optional[Callable] = None,
                 run_inline: bool = False):
        self.session_factory = session_factory
        self._agy = agy
        self._summaries = summaries or _plaud_summaries
        self.run_inline = run_inline  # tests

    @property
    def agy(self):
        if self._agy is None:
            from src.chronos.agy_service import AgyBridgeService
            self._agy = AgyBridgeService()
        return self._agy

    def get(self, svc, day: str, *, force: bool = False) -> Optional[dict]:
        """The day's story, starting a new one when there is none, its moments changed,
        or `force`. None when the day has no data at all."""
        from src.database.models import ChronosDayStory

        inputs = day_inputs(svc, day)
        if inputs is None:
            return None
        moments = inputs["moments"]
        if not moments:
            return {"date": day, "status": "empty", "moment_count": 0, "stale": False}
        current = fingerprint(moments)
        now = datetime.utcnow()

        with self.session_factory() as session:
            row = session.get(ChronosDayStory, day)
            stale = bool(row and row.fingerprint and row.fingerprint != current)
            recent = bool(row and row.requested_at and now - row.requested_at < LEASE)
            busy = bool(row and row.status == "pending" and recent)
            # A failed story waits out the lease before it is tried again (force skips the wait).
            wanted = force or row is None or stale or (row.status == "pending" and not busy) or (
                row.status == "failed" and not recent)
            start = False
            if wanted and not busy:
                if not self.agy.available:
                    return {**self._out(row, day, stale), "status": "unavailable",
                            "error": "The AGY bridge is not set up on this server."}
                with self._starting:
                    if row is None:
                        row = ChronosDayStory(day=day, status="pending", moment_count=len(moments))
                        session.add(row)
                    row.status = "pending"
                    row.requested_at = now
                    row.error = None
                    session.commit()
                    start = True
            out = self._out(row, day, stale and not force)
            if start:
                out["status"] = "pending"

        if start:
            job = (inputs["display"], inputs["recordings"], moments, current)
            if self.run_inline:
                self._write(day, *job)
            else:
                threading.Thread(target=self._write, args=(day, *job), daemon=True,
                                 name=f"day-story-{day}").start()
        return out

    @staticmethod
    def _out(row, day: str, stale: bool) -> dict:
        if row is None:
            return {"date": day, "status": "pending", "stale": False}
        return {
            "date": day,
            "status": row.status,
            "headline": row.headline,
            "story": row.story,
            "open_threads": list(row.open_threads or []),
            "moment_count": row.moment_count,
            "model": row.model,
            "generated_at": row.generated_at.replace(tzinfo=timezone.utc).isoformat() if row.generated_at else None,
            "error": row.error,
            "stale": stale,
        }

    def _write(self, day: str, display: str, recordings, moments, current: str) -> None:
        from src.chronos.agy_service import parse_json_objects
        from src.database.models import ChronosDayStory

        prompt = build_prompt(display, recordings, moments, self._summaries([r.recording_id for r in recordings]))
        result = self.agy.complete(SYSTEM, prompt, SCHEMA)
        parsed = None
        if result.get("ok"):
            for candidate in reversed(parse_json_objects(result.get("text") or "")):
                if isinstance(candidate.get("story"), str) and candidate["story"].strip():
                    parsed = candidate
                    break
        with self.session_factory() as session:
            row = session.get(ChronosDayStory, day) or ChronosDayStory(day=day)
            session.add(row)
            if parsed is None:
                row.status = "failed"
                row.error = str(result.get("error") or "AGY returned no story")[:500]
                if row.fingerprint is None:  # keep an older story readable
                    row.moment_count = len(moments)
            else:
                row.status = "ready"
                row.headline = str(parsed.get("headline") or "").strip() or None
                row.story = parsed["story"].strip()
                row.open_threads = [str(t).strip() for t in parsed.get("open_threads") or [] if str(t).strip()]
                row.fingerprint = current
                row.moment_count = len(moments)
                row.model = f"agy/{result.get('model') or self.agy.model}"
                row.generated_at = datetime.utcnow()
                row.error = None
            session.commit()
        if parsed is not None:
            try:
                from src.chronos.cost_tracker import track_usage

                usage = result.get("usage") or {}
                track_usage(f"agy/{result.get('model') or self.agy.model}", "day_story",
                            input_tokens=int(usage.get("input_tokens") or 0),
                            output_tokens=int(usage.get("output_tokens") or 0))
            except Exception:  # accounting never fails a story
                logger.debug("day story usage not tracked", exc_info=True)
        logger.info("Day story %s: %s", day, "ready" if parsed else f"failed ({str(result.get('error'))[:120]})")


def _plaud_summaries(recording_ids: list[str]) -> dict[str, str]:
    from src.database.engine import SessionLocal
    from src.database.models import ChronosRecording

    with SessionLocal() as session:
        rows = (session.query(ChronosRecording.recording_id, ChronosRecording.plaud_ai_summary)
                .filter(ChronosRecording.recording_id.in_(recording_ids)).all())
    return {rid: summary for rid, summary in rows if summary}
