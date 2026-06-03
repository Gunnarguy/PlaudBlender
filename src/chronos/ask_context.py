"""Shared evidence builders for Ask Chronos flows.

These helpers keep the API route, Dash UI, and MCP tool aligned so they all
feed the same style of evidence into the language model.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class QuestionProfile:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    broad_summary: bool = False
    recent_bias: bool = False
    raw_search_limit: int = 12
    selected_hit_limit: int = 12
    max_hits_per_date: int = 3
    max_relevant_dates: int = 4
    day_recording_limit: int = 5
    search_snippet_limit: int = 320
    neighbor_snippet_limit: int = 160
    recording_preview_limit: int = 180
    day_summary_limit: int = 260
    context_char_budget: int = 8000
    max_day_summaries: int = 4


def format_timestamp_parts(value) -> tuple[str, str]:
    raw = str(value or "")
    if not raw:
        return "?", ""
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed.date().isoformat(), parsed.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return raw[:10], raw


def snippet(text: str | None, limit: int = 240) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _window_from_days(anchor: date, days: int) -> tuple[str, str]:
    start = anchor - timedelta(days=max(days - 1, 0))
    return start.isoformat(), anchor.isoformat()


def infer_question_date_window(
    question: str,
    *,
    now: Optional[datetime] = None,
) -> tuple[Optional[str], Optional[str], bool]:
    """Infer a practical date window from a broad natural-language question."""
    anchor = (now or datetime.now()).date()
    normalized = " ".join((question or "").lower().split())

    iso_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", normalized)
    if iso_dates:
        parsed = sorted(
            {
                datetime.strptime(value, "%Y-%m-%d").date()
                for value in iso_dates
            }
        )
        return parsed[0].isoformat(), parsed[-1].isoformat(), False

    if "today" in normalized:
        day = anchor.isoformat()
        return day, day, True

    if "yesterday" in normalized:
        day = (anchor - timedelta(days=1)).isoformat()
        return day, day, True

    if "this week" in normalized:
        start = anchor - timedelta(days=anchor.weekday())
        return start.isoformat(), anchor.isoformat(), True

    if "last week" in normalized:
        this_week_start = anchor - timedelta(days=anchor.weekday())
        last_week_end = this_week_start - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        return last_week_start.isoformat(), last_week_end.isoformat(), True

    if "this month" in normalized:
        start = anchor.replace(day=1)
        return start.isoformat(), anchor.isoformat(), True

    if "last month" in normalized:
        this_month_start = anchor.replace(day=1)
        last_month_end = this_month_start - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start.isoformat(), last_month_end.isoformat(), True

    for marker, days in (
        ("last few days", 7),
        ("past few days", 7),
        ("last few weeks", 28),
        ("past few weeks", 28),
        ("last several weeks", 28),
        ("past several weeks", 28),
        ("last few months", 90),
        ("past few months", 90),
    ):
        if marker in normalized:
            start_date, end_date = _window_from_days(anchor, days)
            return start_date, end_date, True

    quantity_match = re.search(
        r"\b(?:last|past|previous|over the last)\s+(\d+)\s+(day|days|week|weeks|month|months)\b",
        normalized,
    )
    if quantity_match:
        count = int(quantity_match.group(1))
        unit = quantity_match.group(2)
        multiplier = 1 if unit.startswith("day") else 7 if unit.startswith("week") else 30
        start_date, end_date = _window_from_days(anchor, count * multiplier)
        return start_date, end_date, True

    if any(marker in normalized for marker in ("recently", "lately", "these days", "recent ")):
        start_date, end_date = _window_from_days(anchor, 21)
        return start_date, end_date, True

    return None, None, False


def _is_broad_summary_question(question: str) -> bool:
    normalized = " ".join((question or "").lower().split())
    return any(
        marker in normalized
        for marker in (
            "what have i been up to",
            "big themes",
            "overall",
            "patterns",
            "trends",
            "summarize",
            "summary",
            "across",
            "lately",
            "recently",
            "over the last",
            "last few weeks",
            "past few weeks",
        )
    )


def infer_question_profile(
    question: str,
    *,
    limit: int = 12,
    now: Optional[datetime] = None,
) -> QuestionProfile:
    start_date, end_date, recent_bias = infer_question_date_window(question, now=now)
    broad_summary = _is_broad_summary_question(question)

    if broad_summary or recent_bias:
        selected_hit_limit = min(limit, 8)
        return QuestionProfile(
            start_date=start_date,
            end_date=end_date,
            broad_summary=broad_summary,
            recent_bias=recent_bias,
            raw_search_limit=max(limit * 2, 24),
            selected_hit_limit=selected_hit_limit,
            max_hits_per_date=2,
            max_relevant_dates=2,
            day_recording_limit=3,
            search_snippet_limit=220,
            neighbor_snippet_limit=110,
            recording_preview_limit=120,
            day_summary_limit=180,
            context_char_budget=4200,
            max_day_summaries=2,
        )

    return QuestionProfile(
        start_date=start_date,
        end_date=end_date,
        broad_summary=False,
        recent_bias=recent_bias,
        raw_search_limit=max(limit, min(limit * 2, 20)),
        selected_hit_limit=limit,
    )


def build_neighbor_context(
    events: Iterable[Any],
    target_id: str,
    radius: int = 1,
) -> tuple[str | None, str | None]:
    events = list(events or [])
    if not events:
        return None, None

    try:
        idx = next(i for i, evt in enumerate(events) if getattr(evt, "id", None) == target_id)
    except StopIteration:
        return None, None

    before = []
    for evt in events[max(0, idx - radius):idx]:
        text = snippet(getattr(evt, "clean_text", ""), 160)
        if text:
            before.append(text)

    after = []
    for evt in events[idx + 1:idx + 1 + radius]:
        text = snippet(getattr(evt, "clean_text", ""), 160)
        if text:
            after.append(text)

    return (" | ".join(before) or None, " | ".join(after) or None)


def build_context_from_results(
    svc,
    results,
    *,
    question_profile: Optional[QuestionProfile] = None,
) -> list[dict[str, Any]]:
    profile = question_profile or QuestionProfile()
    context: list[dict[str, Any]] = []
    date_scores: Counter[str] = Counter()
    recording_events_cache: dict[str, list[Any]] = {}

    for rank, result in enumerate(results or [], start=1):
        event = getattr(result, "event", result)
        event_date, event_time = format_timestamp_parts(getattr(event, "start_ts", None))
        score = round(getattr(result, "score", 0.0) or 0.0, 4)

        if event_date and event_date != "?":
            date_scores[event_date] += score

        recording_id = getattr(event, "recording_id", None)
        if recording_id and recording_id not in recording_events_cache:
            try:
                recording_events_cache[recording_id] = svc.get_events_for_recording(recording_id)
            except Exception:
                recording_events_cache[recording_id] = []

        neighbors = recording_events_cache.get(recording_id, [])
        context_before = getattr(result, "context_before", None)
        context_after = getattr(result, "context_after", None)
        if not context_before and not context_after:
            context_before, context_after = build_neighbor_context(
                neighbors,
                getattr(event, "id", ""),
            )

        parts = [
            f"Exact moment: {snippet(getattr(event, 'clean_text', ''), profile.search_snippet_limit)}"
        ]
        if context_before:
            parts.append(
                f"Just before: {snippet(context_before, profile.neighbor_snippet_limit)}"
            )
        if context_after:
            parts.append(
                f"Just after: {snippet(context_after, profile.neighbor_snippet_limit)}"
            )

        context.append(
            {
                "kind": "search_hit",
                "rank": rank,
                "score": score,
                "date": event_date,
                "time": event_time,
                "category": getattr(event, "category", "unknown"),
                "text": "\n".join(parts),
            }
        )

    def _date_sort_key(item: tuple[str, float]) -> tuple[float, int]:
        date_key, score = item
        try:
            parsed = datetime.strptime(date_key, "%Y-%m-%d").date()
            recency_rank = -parsed.toordinal() if profile.recent_bias else parsed.toordinal()
        except Exception:
            recency_rank = 0
        return (-score, recency_rank)

    relevant_dates = [
        date_key
        for date_key, _score in sorted(date_scores.items(), key=_date_sort_key)[: profile.max_relevant_dates]
    ]

    for relevant_date in relevant_dates:
        try:
            day = svc.get_day_detail(relevant_date)
        except Exception:
            day = None

        if not day:
            continue

        recording_lines = []
        for recording in (getattr(day, "recordings", None) or [])[: profile.day_recording_limit]:
            title = (
                getattr(recording, "title", None)
                or getattr(recording, "time_range_formatted", None)
                or getattr(recording, "recording_id", "Recording")
            )
            details = [title]
            time_range = getattr(recording, "time_range_formatted", None)
            if time_range and time_range != title:
                details.append(time_range)
            top_category = getattr(recording, "top_category", None)
            if top_category:
                details.append(top_category)
            event_count = getattr(recording, "event_count", None)
            if event_count is not None:
                details.append(f"{event_count} events")
            preview = getattr(recording, "plaud_ai_summary", None) or getattr(recording, "preview_text", None)
            line = " · ".join(details)
            if preview:
                line += f": {snippet(preview, profile.recording_preview_limit)}"
            recording_lines.append(f"- {line}")

        day_parts = [
            f"{getattr(day, 'recording_count', 0)} recordings, {getattr(day, 'event_count', 0)} events",
        ]
        if getattr(day, "ai_summary", None):
            day_parts.append(
                f"Day summary: {snippet(day.ai_summary, profile.day_summary_limit)}"
            )
        if getattr(day, "top_keywords", None):
            day_parts.append("Top keywords: " + ", ".join(day.top_keywords[:8]))
        if recording_lines:
            day_parts.append("Recordings:\n" + "\n".join(recording_lines))

        context.append(
            {
                "kind": "expanded_day",
                "date": relevant_date,
                "time": "",
                "category": getattr(day, "top_category", None) or "day_summary",
                "text": "\n".join(day_parts),
            }
        )

    return _trim_context_to_budget(context, profile)


def _event_day_key(event: Any) -> Optional[str]:
    day_key, _ = format_timestamp_parts(getattr(event, "start_ts", None))
    return None if day_key == "?" else day_key


def _result_priority(result: Any, profile: QuestionProfile) -> float:
    event = getattr(result, "event", result)
    score = float(getattr(result, "score", 0.0) or 0.0)
    if not profile.recent_bias:
        return score

    day_key = _event_day_key(event)
    if not day_key:
        return score

    try:
        event_day = datetime.strptime(day_key, "%Y-%m-%d").date()
    except Exception:
        return score

    age_days = max((datetime.now().date() - event_day).days, 0)
    recency_bonus = max(0.0, 0.18 - min(age_days, 60) * 0.003)
    return score + recency_bonus


def _select_results_for_context(
    results: list[Any],
    profile: QuestionProfile,
) -> list[Any]:
    if not results:
        return []

    ranked = sorted(results, key=lambda item: _result_priority(item, profile), reverse=True)
    selected: list[Any] = []
    selected_ids: set[int] = set()
    per_date_counts: Counter[str] = Counter()

    for result in ranked:
        event = getattr(result, "event", result)
        day_key = _event_day_key(event)
        if day_key and per_date_counts[day_key] >= profile.max_hits_per_date:
            continue
        selected.append(result)
        selected_ids.add(id(result))
        if day_key:
            per_date_counts[day_key] += 1
        if len(selected) >= profile.selected_hit_limit:
            return selected

    for result in ranked:
        if id(result) in selected_ids:
            continue
        selected.append(result)
        if len(selected) >= profile.selected_hit_limit:
            break

    return selected


def _trim_context_to_budget(
    context: list[dict[str, Any]],
    profile: QuestionProfile,
) -> list[dict[str, Any]]:
    if not context:
        return context

    trimmed: list[dict[str, Any]] = []
    total_chars = 0
    day_summary_count = 0

    for item in context:
        kind = item.get("kind", "search_hit")
        if kind == "expanded_day" and day_summary_count >= profile.max_day_summaries:
            continue

        text = item.get("text", "") or ""
        item_chars = len(text)
        if trimmed and total_chars + item_chars > profile.context_char_budget:
            continue

        trimmed.append(item)
        total_chars += item_chars
        if kind == "expanded_day":
            day_summary_count += 1

    return trimmed or context[:1]


def build_sources_from_results(results, *, limit: int = 5) -> list[dict[str, Any]]:
    sources = []
    for result in list(results or [])[:limit]:
        event = getattr(result, "event", result)
        event_date, _ = format_timestamp_parts(getattr(event, "start_ts", None))
        title = snippet((getattr(event, "clean_text", "") or "").split(".")[0], 80) or "Untitled"
        sources.append(
            {
                "date": event_date,
                "title": title,
                "category": getattr(event, "category", "unknown"),
                "score": round(getattr(result, "score", 0.0) or 0.0, 3),
            }
        )
    return sources


def build_ask_context(
    svc,
    question: str,
    *,
    limit: int = 12,
    task_type: str = "QUESTION_ANSWERING",
    now: Optional[datetime] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    profile = infer_question_profile(question, limit=limit, now=now)
    search_kwargs: dict[str, Any] = {
        "query": question,
        "limit": profile.raw_search_limit,
        "task_type": task_type,
    }
    if profile.start_date:
        search_kwargs["start_date"] = profile.start_date
    if profile.end_date:
        search_kwargs["end_date"] = profile.end_date

    raw_results = svc.search(**search_kwargs)
    if not raw_results and (profile.start_date or profile.end_date):
        raw_results = svc.search(query=question, limit=profile.raw_search_limit, task_type=task_type)

    results = _select_results_for_context(list(raw_results or []), profile)
    return results, build_context_from_results(svc, results, question_profile=profile)
