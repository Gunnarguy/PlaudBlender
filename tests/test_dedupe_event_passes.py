"""
scripts/dedupe_event_passes.py: pass splitting and the same-stretch test.
"""

import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.dedupe_event_passes import same_stretch, split_passes  # noqa: E402

T = datetime(2026, 9, 1, 12, 0)


def rows(*minutes):
    return [{"created_at": str(T + timedelta(minutes=m))} for m in minutes]


def test_events_inserted_close_together_are_one_pass():
    assert [len(p) for p in split_passes(rows(0, 1, 2, 9))] == [4]


def test_reprocessing_starts_a_new_pass():
    assert [len(p) for p in split_passes(rows(0, 1, 60 * 24 * 7, 60 * 24 * 7 + 1))] == [2, 2]


def test_same_stretch_needs_real_overlap():
    whole = (T, T + timedelta(minutes=30))
    assert same_stretch((T + timedelta(minutes=2), T + timedelta(minutes=28)), whole)
    assert not same_stretch((T + timedelta(minutes=40), T + timedelta(minutes=50)), whole)
    assert not same_stretch(None, whole)
