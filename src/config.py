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

# Set native DNS resolver for gRPC/HTTP client on macOS to prevent hangs
os.environ["GRPC_DNS_RESOLVER"] = "native"


_OPENAI_MODEL_ALIASES = {
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt-5-nano": "gpt-5.4-nano",
}


def normalize_openai_model_name(model: Optional[str]) -> str:
    """Normalize legacy OpenAI aliases onto concrete model IDs."""
    raw = (model or "gpt-5.5").strip()
    return _OPENAI_MODEL_ALIASES.get(raw, raw)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _resolve_chronos_gemini_api_key() -> Optional[str]:
    dedicated = (os.getenv("CHRONOS_GEMINI_API_KEY") or "").strip()
    if dedicated:
        return dedicated

    if _env_flag("CHRONOS_ALLOW_SHARED_GEMINI_KEY", "0"):
        shared = (os.getenv("GEMINI_API_KEY") or "").strip()
        return shared or None

    return None


def _openai_key_configured() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _resolve_openai_api_key() -> Optional[str]:
    # We resolve the key directly if configured, bypassing forced toggle checking on load.
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return key or None


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
    # Chronos should use its own Gemini key by default so PlaudBlender usage
    # stays isolated from any other Gemini-heavy projects on the machine.
    chronos_allow_shared_gemini_key: bool = _env_flag(
        "CHRONOS_ALLOW_SHARED_GEMINI_KEY", "0"
    )
    gemini_api_key: Optional[str] = _resolve_chronos_gemini_api_key()

    # Gemini API version (google-genai defaults to v1beta; allow override for preview features)
    gemini_api_version: str = os.getenv("GEMINI_API_VERSION", "v1beta")

    # ─────────────────────────────────────────────────────────────────────────
    # Chronos: Model Selection (May 2026)
    # ─────────────────────────────────────────────────────────────────────────
    # Model Hierarchy (best to fastest):
    #   gpt-5.5       → Flagship ($5.00/$30 MTok)
    #   gpt-5.5-pro   → Highest-quality GPT-5.5 variant ($30/$180 MTok)
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
        # Gemini 3.5 Flash — flagship high-performance reasoning model
        "CHRONOS_CLEANING_MODEL",
        "gemini-3.5-flash",
    )
    chronos_embedding_model: str = os.getenv(
        # Gemini Embedding 2 — current multimodal embedding path for Chronos
        "CHRONOS_EMBEDDING_MODEL",
        "gemini-embedding-2",
    )
    chronos_embedding_dim: int = int(os.getenv("CHRONOS_EMBEDDING_DIM", "768"))
    chronos_analyst_model: str = os.getenv(
        # Gemini 3.5 Flash — flagship low-cost analyst path
        "CHRONOS_ANALYST_MODEL",
        "gemini-3.5-flash",
    )
    chronos_processing_provider: str = (
        os.getenv(
            "CHRONOS_PROCESSING_PROVIDER",
            "gemini",
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

    # Optional local LLM sidecar. This is intentionally disabled by default on
    # the Raspberry Pi and should be used for tiny helper tasks first (JSON
    # repair, classification, short entity extraction), not full-day transcript
    # extraction unless explicitly opted in later.
    chronos_local_llm_enabled: bool = _env_flag("CHRONOS_LOCAL_LLM_ENABLED", "0")
    chronos_local_llm_provider: str = os.getenv(
        "CHRONOS_LOCAL_LLM_PROVIDER", "ollama"
    ).strip().lower()
    chronos_local_llm_base_url: str = os.getenv(
        "CHRONOS_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434"
    ).strip().rstrip("/")
    chronos_local_llm_model: str = os.getenv(
        "CHRONOS_LOCAL_LLM_MODEL", "qwen2.5:0.5b"
    ).strip()
    chronos_local_llm_max_context: int = int(
        os.getenv("CHRONOS_LOCAL_LLM_MAX_CONTEXT", "4096")
    )
    chronos_local_llm_allowed_tasks: str = os.getenv(
        "CHRONOS_LOCAL_LLM_ALLOWED_TASKS", "json_repair,entity_extract,classify,ask"
    )
    chronos_local_embed_batch_size: int = int(
        os.getenv("CHRONOS_LOCAL_EMBED_BATCH_SIZE", "4")
    )
    chronos_local_embed_timeout_seconds: float = float(
        os.getenv("CHRONOS_LOCAL_EMBED_TIMEOUT_SECONDS", "300")
    )
    chronos_ollama_keep_alive: str = os.getenv("CHRONOS_OLLAMA_KEEP_ALIVE", "0s").strip()
    chronos_poll_interval: int = int(os.getenv("CHRONOS_POLL_INTERVAL", "1800"))
    chronos_enable_notion_import: bool = _env_flag(
        "CHRONOS_ENABLE_NOTION_IMPORT", "1"
    )
    chronos_notion_import_batch_size: int = int(
        os.getenv("CHRONOS_NOTION_IMPORT_BATCH_SIZE", "25")
    )
    chronos_self_heal_limit: int = int(os.getenv("CHRONOS_SELF_HEAL_LIMIT", "10"))
    chronos_embed_batch_size: int = int(
        os.getenv("CHRONOS_EMBED_BATCH_SIZE", "20")
    )
    chronos_index_events_per_limit: int = int(
        os.getenv("CHRONOS_INDEX_EVENTS_PER_LIMIT", "0")
    )
    chronos_autosync_process_limit: int = int(
        os.getenv("CHRONOS_AUTOSYNC_PROCESS_LIMIT", "10")
    )
    chronos_autosync_index_limit: int = int(
        os.getenv("CHRONOS_AUTOSYNC_INDEX_LIMIT", "10")
    )
    chronos_autosync_index_timeout: int = int(
        os.getenv("CHRONOS_AUTOSYNC_INDEX_TIMEOUT", "900")
    )
    chronos_autosync_graph_limit: int = int(
        os.getenv("CHRONOS_AUTOSYNC_GRAPH_LIMIT", "10")
    )
    chronos_autosync_max_load_avg: float = float(
        os.getenv("CHRONOS_AUTOSYNC_MAX_LOAD_AVG", "3.5")
    )
    chronos_autosync_min_available_mb: int = int(
        os.getenv("CHRONOS_AUTOSYNC_MIN_AVAILABLE_MB", "700")
    )
    chronos_autosync_max_swap_used_mb: int = int(
        os.getenv("CHRONOS_AUTOSYNC_MAX_SWAP_USED_MB", "512")
    )
    chronos_autosync_defer_seconds: int = int(
        os.getenv("CHRONOS_AUTOSYNC_DEFER_SECONDS", "90")
    )
    chronos_stats_enable_plaud_cloud: bool = _env_flag(
        "CHRONOS_STATS_ENABLE_PLAUD_CLOUD", "0"
    )

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
    # Model Hierarchy (May 2026 — Latest Available):
    #   gpt-5.5       → Frontier flagship ($5/$30 MTok)
    #   gpt-5.5-pro   → Maximum quality ($30/$180 MTok)
    #   gpt-5.4       → Previous frontier flagship
    #   gpt-5.4-pro   → Smart/precise version of 5.4
    #   gpt-5.4-mini  → Strong mini model ($0.75/$4.50 MTok), 400K context
    #   gpt-5.4-nano  → Cheapest GPT-5.4 model ($0.20/$1.25 MTok), 400K context
    #   gpt-5         → Previous reasoning model
    #   gpt-4.1       → Smartest non-reasoning model (legacy)
    # Reasoning effort (GPT-5 family): none (default), low, medium, high, xhigh
    # ─────────────────────────────────────────────────────────────────────────
    # Hard kill switch: a stored OPENAI_API_KEY is inert unless this is opted in.
    chronos_openai_enabled: bool = (
        os.getenv("CHRONOS_OPENAI_ENABLED", "1" if _openai_key_configured() else "0")
        == "1"
    )
    openai_api_key_configured: bool = _openai_key_configured()
    openai_api_key: Optional[str] = _resolve_openai_api_key()
    openai_model: str = normalize_openai_model_name(
        os.getenv("OPENAI_MODEL", "gpt-5.5")
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
