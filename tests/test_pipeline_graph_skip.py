"""
run_graph: skip the per-event LLM rebuild when its inputs are unchanged.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.chronos_pipeline as pipeline  # noqa: E402


@pytest.fixture
def harness(tmp_path, monkeypatch):
    extractor = MagicMock()
    extractor.extract_from_events.return_value = ([], nx.Graph())
    extractor.detect_communities.return_value = {}
    extractor.last_failed_events = 0
    monkeypatch.setattr("src.chronos.graph_service.ChronosGraphExtractor", lambda: extractor)
    monkeypatch.setattr(
        "src.config.get_settings",
        lambda: SimpleNamespace(chronos_graph_cache_dir=str(tmp_path)),
    )
    monkeypatch.setattr(pipeline, "pipeline_progress", MagicMock())

    events = [SimpleNamespace(event_id="e1", clean_text="hello"), SimpleNamespace(event_id="e2", clean_text="world")]
    session = MagicMock()
    q = session.query.return_value.filter.return_value
    q.limit.return_value.all.side_effect = lambda: list(events)
    return SimpleNamespace(extractor=extractor, events=events, session=session, dir=tmp_path)


def test_first_run_builds_and_records_fingerprint(harness):
    pipeline.run_graph(harness.session)
    assert harness.extractor.extract_from_events.call_count == 1
    assert (harness.dir / "knowledge_graph.fingerprint").exists()


def test_unchanged_inputs_skip_rebuild(harness):
    pipeline.run_graph(harness.session)
    assert pipeline.run_graph(harness.session) == 0
    assert harness.extractor.extract_from_events.call_count == 1


def test_changed_text_rebuilds(harness):
    pipeline.run_graph(harness.session)
    harness.events[0].clean_text = "edited"
    pipeline.run_graph(harness.session)
    assert harness.extractor.extract_from_events.call_count == 2


def test_per_recording_build_is_skipped_and_leaves_graph_alone(harness):
    pipeline.run_graph(harness.session)
    before = (harness.dir / "knowledge_graph.pkl").read_bytes()
    assert pipeline.run_graph(harness.session, recording_id="r1") == 0
    assert harness.extractor.extract_from_events.call_count == 1
    assert (harness.dir / "knowledge_graph.pkl").read_bytes() == before
    assert (harness.dir / "knowledge_graph.fingerprint").exists()


def test_full_build_still_skips_after_per_recording_run(harness):
    pipeline.run_graph(harness.session)
    pipeline.run_graph(harness.session, recording_id="r1")
    pipeline.run_graph(harness.session)
    assert harness.extractor.extract_from_events.call_count == 1


def test_failed_build_is_retried(harness):
    harness.extractor.last_failed_events = 3
    pipeline.run_graph(harness.session)
    assert not (harness.dir / "knowledge_graph.fingerprint").exists()
    pipeline.run_graph(harness.session)
    assert harness.extractor.extract_from_events.call_count == 2
