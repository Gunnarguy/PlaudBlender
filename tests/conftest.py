import atexit
import os
import shutil
import sys
import tempfile

import pytest

# Ensure project root is on path for test imports
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Point the database at a scratch file before anything imports the engine.
#
# src/database/engine.py builds its module-level `engine` and `SessionLocal` at
# import time, and plenty of code reaches for those globals directly instead of
# using whatever session a caller passes in -- src/chronos/trace_service.py does
# it on every traced run. Monkeypatching a session in an individual test does not
# reach those, so the suite used to write real rows into data/brain.db: one run
# added six chronos_execution_runs. That is noise on a dev checkout and real
# corruption on the machine holding the actual recordings.
#
# This has to happen at import time, above any `src.` import, because rebinding
# after the engine module loads would leave every `from ... import SessionLocal`
# still pointing at the real database.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="plaudblender-tests-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TEST_DB_DIR, "test-brain.db")
atexit.register(shutil.rmtree, _TEST_DB_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _schema_in_scratch_database():
    """Create the schema in the scratch database the whole session shares.

    Uses create_all rather than init_db(): a database built from the current
    models already has every column the additive SQLite migrations backfill, and
    init_db() also migrates data/notion_matches.json and renames the file, which
    a test run has no business doing to the real data directory.
    """
    from src.database.engine import engine
    from src.database.models import Base

    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def test_settings_factory():
    """Factory fixture to create Settings instances with overrides for testing."""
    from src.config import Settings

    def _create_settings(**kwargs):
        settings = Settings()
        # Mock credentials/API status to prevent loading real keys
        settings.gemini_api_key = "test-gemini"
        settings.openai_api_key_configured = False
        settings.qdrant_api_key = None
        settings.notion_token = None

        for k, v in kwargs.items():
            setattr(settings, k, v)
        return settings

    return _create_settings

