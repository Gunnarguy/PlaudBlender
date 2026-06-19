"""
FastAPI dependency injection — service singletons.

All route handlers use these via `Depends(...)`.
"""

from functools import lru_cache
from fastapi import Depends
from typing import Generator
from src.database.engine import get_db

from app_v2.services.data_service import ChronosDataService, get_data_service
from src.config import Settings, get_settings


@lru_cache(maxsize=1)
def get_service() -> ChronosDataService:
    """Return the singleton ChronosDataService."""
    return get_data_service()


def get_config() -> Settings:
    """Return current Settings (reads .env each time for hot-reload in dev)."""
    return get_settings()

def get_database() -> Generator:
    """FastAPI dependency for the database session."""
    yield from get_db()
