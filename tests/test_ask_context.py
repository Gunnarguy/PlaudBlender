from collections import Counter
from datetime import datetime
from types import SimpleNamespace

from src.chronos.ask_context import build_ask_context, infer_question_profile


def _make_result(date_key: str, score: float, text: str, *, suffix: str) -> SimpleNamespace:
    event = SimpleNamespace(
        id=f"evt-{suffix}",
        start_ts=datetime.fromisoformat(f"{date_key}T10:00:00"),
        recording_id=f"rec-{date_key}-{suffix}",
        category="work",
        clean_text=text,
    )
    return SimpleNamespace(
        event=event,
        score=score,
        context_before=None,
        context_after=None,
    )


class _FakeService:
    def __init__(self, results):
        self._results = results
        self.search_calls = []

    def search(self, query, limit, task_type, start_date=None, end_date=None):
        self.search_calls.append(
            {
                "query": query,
                "limit": limit,
                "task_type": task_type,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return self._results

    def get_events_for_recording(self, recording_id):
        return []

    def get_day_detail(self, date_key):
        return SimpleNamespace(
            recordings=[
                SimpleNamespace(
                    title=f"Recording for {date_key}",
                    time_range_formatted="10:00 AM - 10:30 AM",
                    top_category="work",
                    event_count=4,
                    plaud_ai_summary="Worked through several tasks and follow-ups.",
                    preview_text=None,
                    recording_id=f"rec-{date_key}",
                )
            ],
            recording_count=1,
            event_count=4,
            ai_summary=f"Summary for {date_key}",
            top_keywords=["theme", "project", "follow-up"],
            top_category="work",
        )


def test_infer_question_profile_for_last_few_weeks():
    profile = infer_question_profile(
        "What have I been up to lately and what are the big themes over the last few weeks?",
        now=datetime(2026, 5, 20, 12, 0, 0),
    )

    assert profile.start_date == "2026-04-23"
    assert profile.end_date == "2026-05-20"
    assert profile.broad_summary is True
    assert profile.recent_bias is True
    assert profile.selected_hit_limit == 8


def test_build_ask_context_filters_recent_questions_and_diversifies_dates():
    results = [
        _make_result("2026-05-19", 0.97, "Recent project planning and review.", suffix="1"),
        _make_result("2026-05-19", 0.96, "Another recent work block.", suffix="2"),
        _make_result("2026-05-19", 0.95, "A third hit on the same day that should be trimmed.", suffix="3"),
        _make_result("2026-05-18", 0.94, "Kept working through follow-ups.", suffix="4"),
        _make_result("2026-05-18", 0.93, "Another strong recent hit.", suffix="5"),
        _make_result("2026-05-17", 0.92, "A different recent date.", suffix="6"),
        _make_result("2026-05-16", 0.91, "Yet another recent date.", suffix="7"),
        _make_result("2026-05-15", 0.90, "Still recent and relevant.", suffix="8"),
        _make_result("2026-05-14", 0.89, "Recent enough to compete for inclusion.", suffix="9"),
        _make_result("2026-05-13", 0.88, "A lower-ranked result that may be dropped.", suffix="10"),
    ]
    svc = _FakeService(results)
    question = "What have I been up to lately and what are the big themes over the last few weeks?"
    profile = infer_question_profile(question, now=datetime(2026, 5, 20, 12, 0, 0))

    selected_results, context = build_ask_context(
        svc,
        question,
        now=datetime(2026, 5, 20, 12, 0, 0),
    )

    assert svc.search_calls[0]["start_date"] == "2026-04-23"
    assert svc.search_calls[0]["end_date"] == "2026-05-20"
    assert len(selected_results) <= profile.selected_hit_limit

    selected_dates = Counter(
        result.event.start_ts.date().isoformat() for result in selected_results
    )
    assert selected_dates["2026-05-19"] == 2

    search_hits = [item for item in context if item["kind"] == "search_hit"]
    expanded_days = [item for item in context if item["kind"] == "expanded_day"]
    assert len(search_hits) <= profile.selected_hit_limit
    assert len(expanded_days) <= profile.max_day_summaries
    assert sum(len((item.get("text") or "")) for item in context) <= profile.context_char_budget
