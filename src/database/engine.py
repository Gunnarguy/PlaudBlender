"""Database engine and session helpers.

This centralizes engine creation so both the app and tests can share the
same configuration. By default we store the SQLite database under the
project root in `data/brain.db`.
"""

import os
import re
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .models import Base

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "brain.db")


def get_engine(database_url: Optional[str] = None) -> Engine:
    """Return a SQLAlchemy engine, creating data dir as needed."""
    url = database_url or f"sqlite:///{DB_PATH}"
    if url.startswith("sqlite:///"):
        db_location = url.replace("sqlite:///", "")
        if db_location != ":memory:" and db_location:
            os.makedirs(os.path.dirname(db_location), exist_ok=True)
    engine = create_engine(url, echo=False, future=True)

    # Enable WAL mode + busy_timeout for SQLite to prevent lock contention
    # between readers (UI) and writers (auto_sync, pipeline)
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA mmap_size=268435456")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA cache_size=-10000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


# Default engine/session for application code
engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(engine_override: Optional[Engine] = None) -> None:
    """Create tables if they don't exist.

    This project is intentionally lightweight (no Alembic migrations). For a few
    safety-critical UX fields we perform tiny, additive schema upgrades on SQLite
    so existing local `data/brain.db` files keep working.
    """
    eng = engine_override or engine
    Base.metadata.create_all(eng)

    # Best-effort additive migrations for SQLite.
    if eng.dialect.name == "sqlite":
        _ensure_sqlite_additive_schema(eng)
        _migrate_json_notion_matches_to_db(eng)


def _migrate_json_notion_matches_to_db(eng: Engine) -> None:
    """Read data/notion_matches.json if it exists, insert into SQLite, and safely rename/back it up."""
    import json
    import logging

    logger = logging.getLogger(__name__)
    json_path = os.path.join(PROJECT_ROOT, "data", "notion_matches.json")
    if not os.path.exists(json_path):
        return

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and data:
            with eng.connect() as conn:
                transaction = conn.begin()
                try:
                    for page_id, rec_id in data.items():
                        conn.execute(
                            text(
                                "INSERT OR REPLACE INTO notion_match_overrides "
                                "(notion_page_id, chronos_recording_id) "
                                "VALUES (:page_id, :rec_id)"
                            ),
                            {"page_id": page_id, "rec_id": rec_id}
                        )
                    transaction.commit()
                except Exception as db_exc:
                    transaction.rollback()
                    raise db_exc

        # After successful insertion, backup the file to prevent re-running
        backup_path = json_path + ".bak"
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(json_path, backup_path)
        logger.info(f"Successfully migrated JSON notion matches to database and renamed to {backup_path}")
    except Exception as exc:
        logger.error(f"Failed to migrate JSON notion matches: {exc}")


def _ensure_sqlite_additive_schema(eng: Engine) -> None:
    """Apply best-effort additive schema changes for SQLite.

    SQLite's ALTER TABLE support is limited, but adding nullable columns is safe.
    We use this to keep existing local dev DBs compatible with new UI features.
    """

    def _columns_for(table: str) -> set[str]:
        if not re.match(r"^[a-zA-Z0-9_]+$", table):
            raise ValueError(f"Invalid table name: {table}")
        with eng.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM pragma_table_info(:table)"), {"table": table}
            ).fetchall()
        # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
        return {str(r[1]) for r in rows}

    def _add_column(table: str, column_sql: str) -> None:
        if not re.match(r"^[a-zA-Z0-9_]+$", table):
            raise ValueError(f"Invalid table name: {table}")
        if not re.match(r"^[a-zA-Z0-9_ ]+$", column_sql):
            raise ValueError(f"Invalid column definition: {column_sql}")
        with eng.connect() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column_sql}"))
            conn.commit()

    try:
        cols = _columns_for("chronos_recordings")
        if not cols:
            return

        # ChronosRecording optional metadata + transcript cache (transcript-first mode)
        if "title" not in cols:
            _add_column("chronos_recordings", "title VARCHAR")
        if "transcript" not in cols:
            _add_column("chronos_recordings", "transcript TEXT")
        if "transcript_cached_at" not in cols:
            _add_column("chronos_recordings", "transcript_cached_at DATETIME")
        if "plaud_ai_summary" not in cols:
            _add_column("chronos_recordings", "plaud_ai_summary TEXT")

        # Plaud Cloud Workflow tracking columns
        if "plaud_workflow_id" not in cols:
            _add_column("chronos_recordings", "plaud_workflow_id VARCHAR")
        if "plaud_workflow_status" not in cols:
            _add_column("chronos_recordings", "plaud_workflow_status VARCHAR")
        if "plaud_workflow_submitted_at" not in cols:
            _add_column("chronos_recordings", "plaud_workflow_submitted_at DATETIME")
        if "plaud_workflow_completed_at" not in cols:
            _add_column("chronos_recordings", "plaud_workflow_completed_at DATETIME")
        if "plaud_workflow_template_id" not in cols:
            _add_column("chronos_recordings", "plaud_workflow_template_id VARCHAR")
        if "plaud_workflow_error" not in cols:
            _add_column("chronos_recordings", "plaud_workflow_error TEXT")
        if "plaud_extracted_data" not in cols:
            _add_column("chronos_recordings", "plaud_extracted_data JSON")
        if "time_is_estimated" not in cols:
            _add_column("chronos_recordings", "time_is_estimated BOOLEAN")
        if "time_estimate_reason" not in cols:
            _add_column("chronos_recordings", "time_estimate_reason TEXT")

        # ChronosEvent: user-editable category override
        event_cols = _columns_for("chronos_events")
        if event_cols and "user_category_override" not in event_cols:
            _add_column("chronos_events", "user_category_override VARCHAR")
        if event_cols and "category_confidence" not in event_cols:
            _add_column("chronos_events", "category_confidence REAL")

        # Add lease columns to chronos_recordings table
        rec_cols = _columns_for("chronos_recordings")
        if rec_cols:
            if "processing_started_at" not in rec_cols:
                _add_column("chronos_recordings", "processing_started_at DATETIME")
            if "heartbeat_at" not in rec_cols:
                _add_column("chronos_recordings", "heartbeat_at DATETIME")
            if "lease_expires_at" not in rec_cols:
                _add_column("chronos_recordings", "lease_expires_at DATETIME")
            if "worker_id" not in rec_cols:
                _add_column("chronos_recordings", "worker_id VARCHAR")
            if "attempt_count" not in rec_cols:
                _add_column("chronos_recordings", "attempt_count INTEGER DEFAULT 0")

        # Ensure indexes exist for frequently-queried columns
        with eng.connect() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chronos_events_recording_id "
                    "ON chronos_events (recording_id)"
                )
            )
            conn.commit()
    except Exception:
        # Never block app startup due to a best-effort migration.
        return


def get_db(engine_override: Optional[Engine] = None) -> Generator:
    """Session generator for dependency-style usage."""
    eng = engine_override or engine
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    db = Session()
    try:
        yield db
    finally:
        db.close()
