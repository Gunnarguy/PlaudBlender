"""
src/chronos/event_timing.py: events are placed inside the real recording.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chronos.event_timing import (  # noqa: E402
    EventTiming,
    TimedLine,
    _words,
    load_transcript_lines,
    place_events,
)

LA = ZoneInfo("America/Los_Angeles")
# 2026-09-04 00:25:35 UTC is 2026-09-03 17:25:35 in Los Angeles (PDT, -7)
REC_UTC = datetime(2026, 9, 4, 0, 25, 35)
REC_LOCAL = datetime(2026, 9, 3, 17, 25, 35)


def line(start_s, text):
    return TimedLine(start_s * 1000, start_s * 1000 + 5000, _words(text))


TRANSCRIPT = [
    line(0, "welcome everyone to the open intelligence walkthrough for today"),
    line(60, "module zero covers system architecture and the boundaries we keep"),
    line(300, "now we pause because food is being ordered at the drive through"),
    line(600, "resuming module one ingestion control identity and recovery work"),
]


def ev(start, snippet=None):
    return EventTiming(start, start + timedelta(minutes=5), snippet)


def test_transcript_matches_give_exact_times_in_local_time():
    events = [
        ev(datetime(2026, 9, 4, 9, 0), "Welcome everyone to the Open Intelligence walkthrough for today."),
        ev(datetime(2026, 9, 4, 9, 30), "Module zero covers system architecture and the boundaries"),
        ev(datetime(2026, 9, 4, 10, 0), "resuming module one ingestion control identity and recovery"),
    ]
    placed = place_events(events, REC_UTC, 900, LA, TRANSCRIPT)
    assert [p.method for p in placed] == ["transcript"] * 3
    assert [p.start for p in placed] == [REC_LOCAL, REC_LOCAL + timedelta(seconds=60), REC_LOCAL + timedelta(seconds=600)]


def test_month_slip_is_undone_and_order_kept():
    # The model wrote 12:52 as 2026-12-04; it belongs to the same session.
    events = [ev(datetime(2026, 9, 4, 12, 38)), ev(datetime(2026, 12, 4, 12, 52)), ev(datetime(2026, 9, 4, 12, 24))]
    placed = place_events(events, REC_UTC, 1800, LA)
    assert all(REC_LOCAL <= p.start <= REC_LOCAL + timedelta(seconds=1800) for p in placed)
    assert placed[2].start < placed[0].start < placed[1].start


def test_unmatched_events_interpolate_between_matches():
    events = [
        ev(datetime(2026, 9, 4, 9, 0), "welcome everyone to the open intelligence walkthrough"),
        ev(datetime(2026, 9, 4, 9, 10), "something the model paraphrased beyond recognition entirely"),
        ev(datetime(2026, 9, 4, 9, 20), "resuming module one ingestion control identity and recovery"),
    ]
    placed = place_events(events, REC_UTC, 900, LA, TRANSCRIPT)
    assert placed[1].method == "interpolated"
    assert placed[0].start < placed[1].start < placed[2].start


def test_no_transcript_fits_model_spacing_into_real_length():
    events = [ev(datetime(2026, 9, 4, 7, 30) + timedelta(hours=h)) for h in range(8)]
    placed = place_events(events, REC_UTC, 33 * 60, LA)
    assert {p.method for p in placed} == {"proportional"}
    assert placed[0].start == REC_LOCAL
    assert all(p.end <= REC_LOCAL + timedelta(minutes=33) for p in placed)
    assert [p.start for p in placed] == sorted(p.start for p in placed)


def test_every_event_ends_after_it_starts_and_inside_the_recording():
    events = [ev(datetime(2026, 9, 4, 9, 0))] * 4
    placed = place_events(events, REC_UTC, 120, LA)
    for p in placed:
        assert p.end > p.start
        assert p.start >= REC_LOCAL and p.end <= REC_LOCAL + timedelta(seconds=120)


def test_unknown_duration_uses_transcript_length():
    placed = place_events([ev(datetime(2026, 9, 4, 9, 0)), ev(datetime(2026, 9, 4, 9, 5))], REC_UTC, None, LA, TRANSCRIPT)
    assert placed[-1].end <= REC_LOCAL + timedelta(seconds=605)


def test_ambiguous_snippet_is_not_matched():
    repeated = [line(0, "yes okay yes okay yes okay"), line(100, "yes okay yes okay yes okay")]
    placed = place_events([ev(datetime(2026, 9, 4, 9, 0), "yes okay yes okay yes okay")], REC_UTC, 200, LA, repeated)
    assert placed[0].method == "proportional"


def test_load_transcript_lines(tmp_path):
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "TRANSCRIPT.json").write_text(json.dumps([
        {"start_time": 5000, "end_time": 9000, "content": "Second."},
        {"start_time": 310, "end_time": 4000, "content": "First line"},
        {"bad": True},
    ]))
    lines = load_transcript_lines(tmp_path, "r1")
    assert [l.start_ms for l in lines] == [310, 5000]
    assert load_transcript_lines(tmp_path, "missing") == []


def test_processor_anchors_new_events_before_saving(monkeypatch):
    monkeypatch.setattr("src.config.get_local_timezone", lambda: LA)
    from types import SimpleNamespace

    from src.chronos.transcript_processor import TranscriptProcessor
    from src.models.chronos_schemas import ChronosEvent

    def schema_event(start):
        return ChronosEvent(
            event_id="placeholder", recording_id="r-new", start_ts=start, end_ts=start + timedelta(minutes=3),
            day_of_week=start.strftime("%A"), hour_of_day=start.hour, clean_text="Something happened in the meeting.",
            category="work", sentiment=0.0, keywords=["meeting"], speaker="self_talk",
        )

    events = [schema_event(datetime(2026, 12, 4, 9, 0)), schema_event(datetime(2026, 12, 4, 9, 30))]
    rec = SimpleNamespace(recording_id="r-new-no-artifact", duration_seconds=600, source="plaud")
    TranscriptProcessor._anchor_event_times(None, events, rec, REC_UTC)
    assert events[0].start_ts == REC_LOCAL
    assert all(REC_LOCAL <= e.start_ts < e.end_ts <= REC_LOCAL + timedelta(seconds=600) for e in events)
    assert events[0].day_of_week.value == "Thursday" and events[0].hour_of_day == 17


def test_processor_leaves_notion_recordings_alone():
    from types import SimpleNamespace

    from src.chronos.transcript_processor import TranscriptProcessor

    marker = object()
    events = [SimpleNamespace(start_ts=marker)]
    TranscriptProcessor._anchor_event_times(None, events, SimpleNamespace(source="notion"), REC_UTC)
    assert events[0].start_ts is marker
