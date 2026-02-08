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

    # Plaud Webhook (for async notifications)
    plaud_webhook_secret: Optional[str] = os.getenv("PLAUD_WEBHOOK_SECRET")
    plaud_webhook_url: Optional[str] = os.getenv("PLAUD_WEBHOOK_URL")

    # Plaud API Settings
    plaud_api_base_url: str = os.getenv(
        "PLAUD_API_BASE_URL", "https://api.plaud.ai/api"
    )
    plaud_default_language: str = os.getenv("PLAUD_DEFAULT_LANGUAGE", "en")
    plaud_enable_diarization: bool = os.getenv("PLAUD_ENABLE_DIARIZATION", "1") == "1"
    plaud_workflow_timeout: int = int(os.getenv("PLAUD_WORKFLOW_TIMEOUT", "600"))

    # LLM
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Gemini API version (google-genai defaults to v1beta; allow override for preview features)
    gemini_api_version: str = os.getenv("GEMINI_API_VERSION", "v1beta")

    # ─────────────────────────────────────────────────────────────────────────
    # Chronos: Gemini Model Selection (Feb 2026 — Latest Available Models)
    # ─────────────────────────────────────────────────────────────────────────
    # Model Hierarchy (best to fastest):
    #   gemini-3-pro-preview    → Best reasoning, Batch only (no free standard tier)
    #   gemini-3-flash-preview  → Best FREE model, great for processing ✅
    #   gemini-2.5-pro          → Stable thinking model, FREE standard tier
    #   gemini-2.5-flash        → Stable fast model, FREE standard tier
    #   gemini-2.0-flash        → ⚠️ DEPRECATED (shutdown March 31, 2026)
    #
    # Embeddings:
    #   gemini-embedding-001    → Current stable embedding model, FREE
    # ─────────────────────────────────────────────────────────────────────────
    chronos_cleaning_model: str = os.getenv(
        # Gemini 3 Flash Preview — FREE on standard tier, excellent for processing
        "CHRONOS_CLEANING_MODEL",
        "gemini-3-flash-preview",
    )
    chronos_embedding_model: str = os.getenv(
        # Gemini Embedding 001 — stable, FREE, 768 dims
        "CHRONOS_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )
    chronos_embedding_dim: int = int(os.getenv("CHRONOS_EMBEDDING_DIM", "768"))
    chronos_analyst_model: str = os.getenv(
        # Gemini 3 Pro Preview — Best reasoning model (Batch only, no free standard)
        # Falls back to gemini-3-flash-preview if Pro unavailable
        "CHRONOS_ANALYST_MODEL",
        "gemini-3-pro-preview",
    )

    # Gemini 3 thinking level (applies when supported by the selected model)
    # Flash supports: minimal/low/medium/high. Pro supports: low/high.
    chronos_thinking_level: str = os.getenv("CHRONOS_THINKING_LEVEL", "high")

    # Qdrant (primary vector store)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "chronos_events")
    # Prevent long "hangs" if Qdrant is down or the URL is misconfigured.
    # Qdrant client expects seconds.
    qdrant_timeout_seconds: float = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "5"))

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
