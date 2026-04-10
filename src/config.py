"""Centralized configuration loading.

Loads environment variables once and provides a typed Settings object
for the rest of the codebase. Call load_env() early in entrypoints
(gui, CLI, workers) to ensure .env is respected.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv


# Load .env once at import time, override stale shell env vars
load_dotenv(override=True)


_OPENAI_MODEL_ALIASES = {
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt-5-nano": "gpt-5.4-nano",
}


def normalize_openai_model_name(model: Optional[str]) -> str:
    """Map legacy OpenAI model aliases onto the current GPT-5.4 family."""
    raw = (model or "gpt-5.4").strip()
    return _OPENAI_MODEL_ALIASES.get(raw, raw)


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
    # Chronos: Gemini Model Selection (March 2026 — Latest Available Models)
    # ─────────────────────────────────────────────────────────────────────────
    # Chronos: OpenAI Model Selection (April 2026)
    # ─────────────────────────────────────────────────────────────────────────
    # Model Hierarchy (best to fastest):
    #   gpt-5.4       → Flagship ($2.50/$15 MTok), 1M context, 128K output
    #   gpt-5.4-mini  → Strong mini ($0.75/$4.50 MTok), 400K context
    #   gpt-5.4-nano  → Cheapest ($0.20/$1.25 MTok), 400K context
    #
    # Embeddings:
    #   text-embedding-3-large → Best quality (3072 native, MRL to any dim)
    #   text-embedding-3-small → Fast & cheap (1536 native, MRL to any dim)
    #   Both support `dimensions` param for Matryoshka dimensionality reduction.
    #   8192 token input limit. Vectors are L2-normalized by the API.
    # ─────────────────────────────────────────────────────────────────────────
    chronos_cleaning_model: str = os.getenv(
        # gpt-5.4-mini — strong structured extraction at low cost
        "CHRONOS_CLEANING_MODEL",
        "gpt-5.4-mini",
    )
    chronos_embedding_model: str = os.getenv(
        # text-embedding-3-large with MRL at 768 dims — best accuracy
        "CHRONOS_EMBEDDING_MODEL",
        "text-embedding-3-large",
    )
    chronos_embedding_dim: int = int(os.getenv("CHRONOS_EMBEDDING_DIM", "768"))
    chronos_analyst_model: str = os.getenv(
        # gpt-5.4 — flagship for hard tasks, fallback, RAG
        "CHRONOS_ANALYST_MODEL",
        "gpt-5.4",
    )
    chronos_processing_provider: str = (
        os.getenv(
            "CHRONOS_PROCESSING_PROVIDER",
            "openai",
        )
        .strip()
        .lower()
    )
    gemini_billing_tier: str = os.getenv("GEMINI_BILLING_TIER", "paid").strip().lower()
    chronos_allow_paid_gemini_fallback: bool = (
        os.getenv("CHRONOS_ALLOW_PAID_GEMINI_FALLBACK", "0") == "1"
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

    # ─────────────────────────────────────────────────────────────────────────
    # OpenAI: Responses API (for RAG / conversational queries)
    # ─────────────────────────────────────────────────────────────────────────
    # Model Hierarchy (March 2026 — Latest Available):
    #   gpt-5.4       → Frontier flagship ($2.50/$15 MTok), 1.05M context, 128K output
    #   gpt-5.4-pro   → Smart/precise version of 5.4
    #   gpt-5.4-mini  → Strong mini model ($0.75/$4.50 MTok), 400K context
    #   gpt-5.4-nano  → Cheapest GPT-5.4 model ($0.20/$1.25 MTok), 400K context
    #   gpt-5         → Previous reasoning model
    #   gpt-4.1       → Smartest non-reasoning model (legacy)
    # Reasoning effort (gpt-5.4): none (default), low, medium, high, xhigh
    # ─────────────────────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = normalize_openai_model_name(
        os.getenv("OPENAI_MODEL", "gpt-5.4")
    )
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # ─────────────────────────────────────────────────────────────────────────
    # Notion: OAuth + Direct API Integration
    # ─────────────────────────────────────────────────────────────────────────
    notion_token: Optional[str] = os.getenv(
        "NOTION_TOKEN"
    )  # Static internal token (fallback)
    notion_database_id: Optional[str] = os.getenv("NOTION_DATABASE_ID")
    notion_client_id: Optional[str] = os.getenv("NOTION_CLIENT_ID")
    notion_client_secret: Optional[str] = os.getenv("NOTION_CLIENT_SECRET")
    notion_redirect_uri: Optional[str] = os.getenv("NOTION_REDIRECT_URI")
    notion_weekday_start_time: str = os.getenv("NOTION_WEEKDAY_START_TIME", "07:30")
    notion_weekend_start_time: str = os.getenv("NOTION_WEEKEND_START_TIME", "12:00")

    # Logging
    log_level: str = os.getenv("PB_LOG_LEVEL", "INFO")
    verbose: bool = os.getenv("PB_VERBOSE", "0") == "1"


def get_settings() -> Settings:
    """Return a new Settings instance (cheap dataclass construction)."""
    return Settings()


@lru_cache(maxsize=1)
def get_local_timezone():
    """Return the machine's IANA timezone when available.

    macOS often reports a fixed-offset tzinfo like PDT, which breaks historical
    DST conversions. Prefer a real zone database entry such as
    America/Los_Angeles.
    """

    def _load_zone(name: Optional[str]):
        if not name:
            return None
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            return None

    for candidate in [os.getenv("TZ")]:
        zone = _load_zone(candidate)
        if zone is not None:
            return zone

    localtime_path = os.path.realpath("/etc/localtime")
    for marker in ["/zoneinfo/", "/zoneinfo.default/"]:
        if marker in localtime_path:
            zone_name = localtime_path.split(marker, 1)[1]
            zone = _load_zone(zone_name)
            if zone is not None:
                return zone

    tzinfo = datetime.now().astimezone().tzinfo
    if hasattr(tzinfo, "key"):
        zone = _load_zone(getattr(tzinfo, "key", None))
        if zone is not None:
            return zone

    return tzinfo
