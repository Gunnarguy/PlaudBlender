"""
scripts/plaud_v4_artifacts.py: only re-read recordings whose version moved.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.plaud_v4_artifacts as artifacts  # noqa: E402


@pytest.fixture
def harness(tmp_path, monkeypatch):
    items = [
        {"file_id": "f_a", "version_ms": 1},
        {"file_id": "f_b", "version_ms": 1},
    ]
    client = MagicMock()
    client.has_session = True
    client.iter_recordings.side_effect = lambda: iter([dict(i) for i in items])
    client.file_detail.return_value = {"objects": []}
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []
    session_cm = MagicMock()
    session_cm.__enter__.return_value = session

    monkeypatch.setattr(artifacts, "PlaudV4Client", lambda: client)
    monkeypatch.setattr(artifacts, "init_db", lambda: None)
    monkeypatch.setattr(artifacts, "SessionLocal", lambda: session_cm)
    monkeypatch.setattr(artifacts, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(artifacts, "state_path", lambda: tmp_path / "seen.json")
    monkeypatch.setattr(artifacts.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["plaud_v4_artifacts.py"])
    return client, items


def test_first_run_checks_everything(harness):
    client, _ = harness
    artifacts.main()
    assert client.file_detail.call_count == 2


def test_second_run_skips_unchanged(harness):
    client, _ = harness
    artifacts.main()
    artifacts.main()
    assert client.file_detail.call_count == 2


def test_version_bump_is_rechecked(harness):
    client, items = harness
    artifacts.main()
    items[1]["version_ms"] = 2
    artifacts.main()
    assert client.file_detail.call_count == 3
    assert client.file_detail.call_args.args[0] == "f_b"


def test_failed_detail_is_retried(harness):
    client, _ = harness
    client.file_detail.side_effect = [artifacts.PlaudV4Error("boom"), {"objects": []}, {"objects": []}]
    artifacts.main()
    artifacts.main()
    assert client.file_detail.call_count == 3


def test_full_sweep_after_24h(harness, monkeypatch):
    client, _ = harness
    artifacts.main()
    now = artifacts.time.time()
    monkeypatch.setattr(artifacts.time, "time", lambda: now + artifacts.FULL_SWEEP_SECONDS + 1)
    artifacts.main()
    assert client.file_detail.call_count == 4
