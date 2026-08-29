"""Guards that the test suite never reads or writes the real database.

src/database/engine.py creates its `engine` and `SessionLocal` at import time,
and much of the codebase reaches for those globals rather than accepting a
session from its caller -- src/chronos/trace_service.py does it on every traced
run. That means patching a session inside one test does not contain it, and a
test that forgets to patch writes straight into data/brain.db. tests/conftest.py
redirects the engine to a scratch file before any of that is imported; these
tests fail loudly if that ever stops working.
"""

import os

from src.chronos.trace_service import finish_trace_run, start_trace_run
from src.database.engine import DB_PATH, SessionLocal, engine, resolve_database_url
from src.database.models import ChronosExecutionRun


def test_engine_points_at_the_scratch_database():
    """The engine must be open on conftest's temporary file, not data/brain.db."""
    url = str(engine.url)

    assert url == resolve_database_url(), "engine drifted from the configured URL"
    assert "plaudblender-tests-" in url, f"suite is not on a scratch database: {url}"
    assert DB_PATH not in url, "suite is pointed at the real database"


def test_the_real_database_is_never_opened():
    """The real file must not be what any module-level session resolves to."""
    assert os.path.abspath(DB_PATH) not in os.path.abspath(
        str(engine.url).replace("sqlite:///", "")
    )


def test_traced_runs_land_in_the_scratch_database():
    """A traced run writes through the engine global, so it proves the redirect.

    This is the exact path that leaked before: trace_service imports SessionLocal
    itself, so pipeline tests that patched their own session still appended real
    rows to chronos_execution_runs.
    """
    run_id = start_trace_run(
        trigger="cli", source="test-isolation", title="isolation guard", emit_xray=False
    )
    finish_trace_run(run_id, emit_xray=False)

    with SessionLocal() as session:
        run = session.get(ChronosExecutionRun, run_id)

    assert run is not None, "traced run did not reach the scratch database"
    assert run.source == "test-isolation"
    assert run.status == "completed"
