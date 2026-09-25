"""Place extracted events at their real time inside the recording.

The processing prompt only gives the model the recording's date, so the
wall-clock times it returns are invented (09:00 and 07:30 are the favourite
starts) and sometimes slip a month when the hour reaches 10-12. Measured on
2026-09-24: 92% of events sat more than an hour outside their recording.

What the model does get right is order. So each event is placed by, in turn:

1. transcript   - its raw_transcript_snippet matched word-for-word to a line
                  of Plaud's timed TRANSCRIPT artifact: the true offset.
2. interpolated - spaced by order between the nearest transcript matches.
3. proportional - no matches in this pass: the model's relative spacing is
                  kept and fitted into the real recording length.

Times are returned as naive local wall-clock, which is how event start_ts
is stored and displayed (recording created_at is naive UTC).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

_WORD = re.compile(r"[a-z0-9']+")
MATCH_WORDS = 6


def _words(text: Optional[str]) -> list[str]:
    return _WORD.findall((text or "").lower())


@dataclass
class TimedLine:
    start_ms: int
    end_ms: int
    words: list[str]


@dataclass
class EventTiming:
    """What placement needs to know about one extracted event."""

    llm_start: datetime
    llm_end: datetime
    snippet: Optional[str] = None


@dataclass
class Placement:
    start: datetime
    end: datetime
    method: str


def load_transcript_lines(artifact_dir: Path, recording_id: str) -> list[TimedLine]:
    """Plaud 4.0 TRANSCRIPT artifact: [{start_time, end_time (ms), content}, ...]."""
    try:
        data = json.loads((Path(artifact_dir) / recording_id / "TRANSCRIPT.json").read_text())
    except (OSError, ValueError):
        return []
    lines = []
    for item in data if isinstance(data, list) else []:
        try:
            lines.append(TimedLine(int(item["start_time"]), int(item["end_time"]), _words(item.get("content"))))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(lines, key=lambda line: line.start_ms)


class _TranscriptIndex:
    def __init__(self, lines: Sequence[TimedLine]):
        self.stream: list[str] = []
        self.line_of: list[int] = []
        self.lines = list(lines)
        for index, line in enumerate(self.lines):
            self.stream.extend(line.words)
            self.line_of.extend([index] * len(line.words))
        self.text = " " + " ".join(self.stream) + " "
        # character offset in self.text -> word index
        self._char_to_word: dict[int, int] = {}
        pos = 1
        for i, word in enumerate(self.stream):
            self._char_to_word[pos] = i
            pos += len(word) + 1

    def offset_ms(self, snippet: Optional[str]) -> Optional[int]:
        """Start time of the transcript line where the snippet begins."""
        words = _words(snippet)
        if len(words) < MATCH_WORDS or not self.stream:
            return None
        # Try the opening words, then shifted windows (snippets are often
        # trimmed or lightly cleaned at the start). Unique matches only.
        for skip in range(0, min(6, len(words) - MATCH_WORDS + 1)):
            needle = " " + " ".join(words[skip : skip + MATCH_WORDS]) + " "
            first = self.text.find(needle)
            if first < 0 or self.text.find(needle, first + 1) >= 0:
                continue
            word_index = self._char_to_word.get(first + 1)
            if word_index is None:
                continue
            return self.lines[self.line_of[word_index]].start_ms
        return None


def _normalized_llm_seconds(events: Sequence[EventTiming]) -> list[float]:
    """Model times as seconds from the pass's first event, with month/day slips undone."""
    days = Counter(e.llm_start.date() for e in events)
    home = days.most_common(1)[0][0]

    def on_home_day(ts: datetime) -> datetime:
        return datetime.combine(home, ts.time())

    starts = [on_home_day(e.llm_start) for e in events]
    origin = min(starts)
    return [(s - origin).total_seconds() for s in starts]


def _longest_non_decreasing(pairs: list[tuple[int, int]]) -> set[int]:
    """Indices (first element) of the longest run of non-decreasing offsets."""
    if not pairs:
        return set()
    n = len(pairs)
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if pairs[j][1] <= pairs[i][1] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(n), key=lambda k: best[k])
    keep = set()
    while i >= 0:
        keep.add(pairs[i][0])
        i = prev[i]
    return keep


def utc_to_local_naive(value: datetime, tz) -> datetime:
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return aware.astimezone(tz).replace(tzinfo=None)


def place_events(
    events: Sequence[EventTiming],
    recording_start_utc: datetime,
    duration_seconds: Optional[float],
    local_tz,
    transcript: Sequence[TimedLine] = (),
) -> list[Placement]:
    """Placements for one processing pass, in the same order as `events`."""
    if not events:
        return []
    start_local = utc_to_local_naive(recording_start_utc, local_tz)
    duration = float(duration_seconds or 0)
    if duration <= 0 and transcript:
        duration = max(line.end_ms for line in transcript) / 1000.0
    llm_seconds = _normalized_llm_seconds(events)
    if duration <= 0:
        # Nothing to fit into: keep the model's spacing, anchored at the real start.
        duration = max(llm_seconds) + 60.0

    order = sorted(range(len(events)), key=lambda i: (llm_seconds[i], i))
    offsets: dict[int, float] = {}
    method: dict[int, str] = {}

    if transcript:
        index = _TranscriptIndex(transcript)
        matched = []
        for i in order:
            ms = index.offset_ms(events[i].snippet)
            if ms is not None and ms / 1000.0 <= duration + 60:
                matched.append((i, ms))
        for i in _longest_non_decreasing(matched):
            offsets[i] = min(dict(matched)[i] / 1000.0, duration)
            method[i] = "transcript"

    if offsets:
        # Evenly space the rest between the nearest matched neighbours.
        positions = [k for k, i in enumerate(order) if i in offsets]
        bounds = [(-1, 0.0)] + [(k, offsets[order[k]]) for k in positions] + [(len(order), duration)]
        for (lo_k, lo_t), (hi_k, hi_t) in zip(bounds, bounds[1:]):
            gap = hi_k - lo_k
            for step, k in enumerate(range(lo_k + 1, hi_k), start=1):
                offsets[order[k]] = lo_t + (hi_t - lo_t) * step / gap
                method[order[k]] = "interpolated"
    else:
        span = max(llm_seconds)
        for rank, i in enumerate(order):
            if span > 0:
                offsets[i] = llm_seconds[i] / span * duration * (len(order) - 1) / len(order)
            else:
                offsets[i] = duration * rank / len(order)
            method[i] = "proportional"

    placements: dict[int, Placement] = {}
    for k, i in enumerate(order):
        start = min(max(offsets[i], 0.0), duration)
        nxt = offsets[order[k + 1]] if k + 1 < len(order) else duration
        end = min(nxt if nxt > start else start + 60.0, duration)
        if end <= start:
            end = start + 1.0
        placements[i] = Placement(
            start=start_local + timedelta(seconds=round(start)),
            end=start_local + timedelta(seconds=round(end)),
            method=method[i],
        )
    return [placements[i] for i in range(len(events))]
