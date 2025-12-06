"""Database engine and session helpers.

This centralizes engine creation so both the app and tests can share the
same configuration. By default we store the SQLite database under the
project root in `data/brain.db`.
"""
import os
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .models import Base

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
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
    """Create tables if they don't exist."""
    eng = engine_override or engine
    Base.metadata.create_all(eng)


def get_db(engine_override: Optional[Engine] = None) -> Generator:
    """Session generator for dependency-style usage."""
    eng = engine_override or engine
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    db = Session()
    try:
        yield db
    finally:
        db.close()
