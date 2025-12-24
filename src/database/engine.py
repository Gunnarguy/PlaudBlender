"""Database engine and session helpers.

This centralizes engine creation so both the app and tests can share the
same configuration. By default we store the SQLite database under the
project root in `data/brain.db`.
"""

import os
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
    return create_engine(url, echo=False, future=True)


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


def _ensure_sqlite_additive_schema(eng: Engine) -> None:
    """Apply best-effort additive schema changes for SQLite.

    SQLite's ALTER TABLE support is limited, but adding nullable columns is safe.
    We use this to keep existing local dev DBs compatible with new UI features.
    """

    def _columns_for(table: str) -> set[str]:
        with eng.connect() as conn:
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
        return {str(r[1]) for r in rows}

    def _add_column(table: str, column_sql: str) -> None:
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
