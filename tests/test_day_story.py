"""Day Story: cached per day, rewritten when moments change, never invented (2026-10-06)."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.chronos import day_story
from src.chronos.day_story import DayStoryService, build_prompt
from src.database.models import Base, ChronosDayStory


def make_factory():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def moment(mid, minute, text, category="work"):
    return SimpleNamespace(id=mid, recording_id="r1", start_ts=datetime(2026, 10, 5, 15, minute),
                           clean_text=text, category=category, sentiment=0.2, speaker="")


class FakeData:
    def __init__(self, moments):
        self.moments = moments

    def get_day_detail(self, day):
        rec = SimpleNamespace(recording_id="r1", start_time=datetime(2026, 10, 5, 8, 0),
                              duration_seconds=3600, event_count=len(self.moments))
        return SimpleNamespace(date_display="Monday, Oct 05", recordings=[rec])

    def _get_all_events(self):
        return list(self.moments)


class FakeAgy:
    model = "gemini-test"
    available = True

    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def complete(self, system, user, schema=None):
        self.calls.append(user)
        if not self.ok:
            return {"ok": False, "error": "bridge down"}
        story = {"headline": "Staff meeting", "story": "You met the team.", "open_threads": ["Order scopes"]}
        return {"ok": True, "text": json.dumps(story), "model": "gemini-test", "usage": {}}


def service(agy, factory):
    return DayStoryService(factory, agy=agy, summaries=lambda ids: {"r1": "Plaud says: a meeting"}, run_inline=True)


def test_writes_once_then_serves_the_cache():
    factory, agy = make_factory(), FakeAgy()
    data = FakeData([moment("a", 1, "Met the team"), moment("b", 5, "Agreed to order scopes")])
    stories = service(agy, factory)

    first = stories.get(data, "2026-10-05")
    assert first["status"] == "pending"
    second = stories.get(data, "2026-10-05")
    assert second["status"] == "ready"
    assert second["headline"] == "Staff meeting"
    assert second["open_threads"] == ["Order scopes"]
    assert second["moment_count"] == 2 and second["stale"] is False
    assert len(agy.calls) == 1


def test_new_moments_rewrite_the_story_and_keep_the_old_one_meanwhile():
    factory, agy = make_factory(), FakeAgy()
    data = FakeData([moment("a", 1, "Met the team")])
    stories = DayStoryService(factory, agy=agy, summaries=lambda ids: {}, run_inline=True)
    stories.get(data, "2026-10-05")
    data.moments.append(moment("b", 9, "Called the vendor"))
    stories.run_inline = False
    stories._write = lambda *args: None  # keep it pending
    out = stories.get(data, "2026-10-05")
    assert out["status"] == "pending" and out["stale"] is True and out["story"] == "You met the team."


def test_an_empty_day_costs_nothing():
    factory, agy = make_factory(), FakeAgy()
    out = service(agy, factory).get(FakeData([]), "2026-10-05")
    assert out["status"] == "empty" and agy.calls == []


def test_a_failure_waits_out_the_lease_unless_forced():
    factory, agy = make_factory(), FakeAgy(ok=False)
    data = FakeData([moment("a", 1, "Met the team")])
    stories = service(agy, factory)
    stories.get(data, "2026-10-05")
    again = stories.get(data, "2026-10-05")
    assert again["status"] == "failed" and "bridge down" in again["error"] and len(agy.calls) == 1
    stories.get(data, "2026-10-05", force=True)
    assert len(agy.calls) == 2
    with factory() as session:  # after the lease it is tried again on its own
        row = session.get(ChronosDayStory, "2026-10-05")
        row.requested_at = datetime.utcnow() - day_story.LEASE - timedelta(minutes=1)
        session.commit()
    stories.get(data, "2026-10-05")
    assert len(agy.calls) == 3


def test_prompt_is_in_time_order_with_local_clock_times(monkeypatch):
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(day_story, "get_local_timezone", lambda: ZoneInfo("America/Los_Angeles"))
    rec = SimpleNamespace(recording_id="r1", start_time=datetime(2026, 10, 5, 8, 0), duration_seconds=11_377, event_count=2)
    prompt = build_prompt("Monday, Oct 05", [rec], [moment("a", 1, "Met the team"), moment("b", 5, "Ordered scopes")],
                          {"r1": "A staff meeting"})
    assert "- 08:00, 3 h 9 min, 2 moments" in prompt
    assert "Plaud summary: A staff meeting" in prompt
    assert prompt.index("08:01 [work") < prompt.index("08:05 [work")
