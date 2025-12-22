"""Centralized configuration loading.

Loads environment variables once and provides a typed Settings object
for the rest of the codebase. Call load_env() early in entrypoints
(gui, CLI, workers) to ensure .env is respected.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


# Load .env once at import time
load_dotenv()


@dataclass
class Settings:
    # Plaud / OAuth
    plaud_client_id: Optional[str] = os.getenv("PLAUD_CLIENT_ID")
    plaud_client_secret: Optional[str] = os.getenv("PLAUD_CLIENT_SECRET")
    plaud_redirect_uri: Optional[str] = os.getenv("PLAUD_REDIRECT_URI")

    # LLM
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Chronos: Gemini Model Selection
    chronos_cleaning_model: str = os.getenv(
        "CHRONOS_CLEANING_MODEL", "gemini-2.0-flash-exp"
    )
    chronos_embedding_model: str = os.getenv(
        "CHRONOS_EMBEDDING_MODEL", "text-embedding-004"
    )
    chronos_analyst_model: str = os.getenv(
        "CHRONOS_ANALYST_MODEL", "gemini-2.0-flash-thinking-exp-1219"
    )

    # Pinecone (legacy - for backward compatibility)
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "transcripts")

    # Qdrant (primary for Chronos)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "chronos_events")

    # Database (root-level data directory by default)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///"
        + os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "brain.db",
        ),
    )

    # Chronos: Data Directories
    chronos_raw_audio_dir: str = os.getenv(
        "CHRONOS_RAW_AUDIO_DIR",
        os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "raw",
        ),
    )
    chronos_processed_dir: str = os.getenv(
        "CHRONOS_PROCESSED_DIR",
        os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "processed",
        ),
    )
    chronos_graph_cache_dir: str = os.getenv(
        "CHRONOS_GRAPH_CACHE_DIR",
        os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "data",
            "cache",
            "graphs",
        ),
    )

    # Logging
    log_level: str = os.getenv("PB_LOG_LEVEL", "INFO")
    verbose: bool = os.getenv("PB_VERBOSE", "0") == "1"


def get_settings() -> Settings:
    """Return a new Settings instance (cheap dataclass construction)."""
    return Settings()
