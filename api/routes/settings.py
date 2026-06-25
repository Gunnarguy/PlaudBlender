"""Server settings endpoints for mobile and web clients."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException

from api.auth.jwt import require_auth
from api.schemas.responses import (
    ServerSettingsFlagsOut,
    ServerSettingsOut,
    ServerSettingsUpdateRequest,
    SuccessResponse,
)
from src.config import get_settings
from src.notion_oauth import NotionOAuthClient

router = APIRouter(
    prefix="/api/v1/settings",
    tags=["settings"],
    dependencies=[Depends(require_auth)],
)

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

_FIELD_TO_ENV = {
    "processing_provider": "CHRONOS_PROCESSING_PROVIDER",
    "cleaning_model": "CHRONOS_CLEANING_MODEL",
    "analyst_model": "CHRONOS_ANALYST_MODEL",
    "embedding_model": "CHRONOS_EMBEDDING_MODEL",
    "openai_model": "OPENAI_MODEL",
    "thinking_level": "CHRONOS_THINKING_LEVEL",
    "openai_temperature": "OPENAI_TEMPERATURE",
    "embedding_dim": "CHRONOS_EMBEDDING_DIM",
    "plaud_language": "PLAUD_DEFAULT_LANGUAGE",
    "plaud_diarization": "PLAUD_ENABLE_DIARIZATION",
    "log_level": "PB_LOG_LEVEL",
    "custom_categories": "CHRONOS_CUSTOM_CATEGORIES",
    "notion_weekday_start": "NOTION_WEEKDAY_START_TIME",
    "notion_weekend_start": "NOTION_WEEKEND_START_TIME",
    "qdrant_url": "QDRANT_URL",
    "qdrant_collection_name": "QDRANT_COLLECTION_NAME",
    "chronos_openai_enabled": "CHRONOS_OPENAI_ENABLED",
    "chronos_local_llm_enabled": "CHRONOS_LOCAL_LLM_ENABLED",
    "chronos_local_llm_provider": "CHRONOS_LOCAL_LLM_PROVIDER",
    "chronos_local_llm_base_url": "CHRONOS_LOCAL_LLM_BASE_URL",
    "chronos_local_llm_model": "CHRONOS_LOCAL_LLM_MODEL",
    "chronos_local_llm_max_context": "CHRONOS_LOCAL_LLM_MAX_CONTEXT",
    "chronos_local_llm_allowed_tasks": "CHRONOS_LOCAL_LLM_ALLOWED_TASKS",
    "chronos_poll_interval": "CHRONOS_POLL_INTERVAL",
    "chronos_enable_notion_import": "CHRONOS_ENABLE_NOTION_IMPORT",
    "chronos_notion_import_batch_size": "CHRONOS_NOTION_IMPORT_BATCH_SIZE",
    "chronos_self_heal_limit": "CHRONOS_SELF_HEAL_LIMIT",
    "chronos_embed_batch_size": "CHRONOS_EMBED_BATCH_SIZE",
    "chronos_index_events_per_limit": "CHRONOS_INDEX_EVENTS_PER_LIMIT",
    "chronos_autosync_process_limit": "CHRONOS_AUTOSYNC_PROCESS_LIMIT",
    "chronos_autosync_index_limit": "CHRONOS_AUTOSYNC_INDEX_LIMIT",
    "chronos_autosync_index_timeout": "CHRONOS_AUTOSYNC_INDEX_TIMEOUT",
    "chronos_autosync_graph_limit": "CHRONOS_AUTOSYNC_GRAPH_LIMIT",
    "chronos_autosync_max_load_avg": "CHRONOS_AUTOSYNC_MAX_LOAD_AVG",
    "chronos_autosync_min_available_mb": "CHRONOS_AUTOSYNC_MIN_AVAILABLE_MB",
    "chronos_autosync_max_swap_used_mb": "CHRONOS_AUTOSYNC_MAX_SWAP_USED_MB",
    "chronos_autosync_defer_seconds": "CHRONOS_AUTOSYNC_DEFER_SECONDS",
    "chronos_stats_enable_plaud_cloud": "CHRONOS_STATS_ENABLE_PLAUD_CLOUD",
}


def _valid_hhmm(value: str) -> bool:
    try:
        hour_text, minute_text = str(value).strip().split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        return 0 <= hour <= 23 and 0 <= minute <= 59
    except (AttributeError, TypeError, ValueError):
        return False


def _env_string(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _write_env_updates(updates: dict[str, str]) -> int:
    lines = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text().splitlines()

    existing_keys: set[str] = set()
    changed = 0
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, _, old_val = stripped.partition("=")
            key = key.strip()
            old_val = old_val.strip()
            if key in updates:
                new_val = updates[key]
                new_lines.append(f"{key}={new_val}")
                existing_keys.add(key)
                if old_val != new_val:
                    changed += 1
                continue
        new_lines.append(line)

    for key, value in updates.items():
        if key not in existing_keys:
            new_lines.append(f"{key}={value}")
            changed += 1

    _ENV_PATH.write_text("\n".join(new_lines) + "\n")
    for key, value in updates.items():
        os.environ[key] = value
    load_dotenv(_ENV_PATH, override=True)
    return changed


@router.get("", response_model=ServerSettingsOut)
async def get_server_settings():
    settings = get_settings()
    notion_oauth = NotionOAuthClient()

    return ServerSettingsOut(
        processing_provider=settings.chronos_processing_provider,
        cleaning_model=settings.chronos_cleaning_model,
        analyst_model=settings.chronos_analyst_model,
        embedding_model=settings.chronos_embedding_model,
        openai_model=settings.openai_model,
        thinking_level=settings.chronos_thinking_level,
        openai_temperature=settings.openai_temperature,
        embedding_dim=settings.chronos_embedding_dim,
        plaud_language=settings.plaud_default_language,
        plaud_diarization=settings.plaud_enable_diarization,
        log_level=settings.log_level,
        custom_categories=os.getenv("CHRONOS_CUSTOM_CATEGORIES", ""),
        notion_weekday_start=settings.notion_weekday_start_time,
        notion_weekend_start=settings.notion_weekend_start_time,
        qdrant_url=settings.qdrant_url,
        qdrant_collection_name=settings.qdrant_collection_name,
        chronos_openai_enabled=settings.chronos_openai_enabled,
        chronos_local_llm_enabled=settings.chronos_local_llm_enabled,
        chronos_local_llm_provider=settings.chronos_local_llm_provider,
        chronos_local_llm_base_url=settings.chronos_local_llm_base_url,
        chronos_local_llm_model=settings.chronos_local_llm_model,
        chronos_local_llm_max_context=settings.chronos_local_llm_max_context,
        chronos_local_llm_allowed_tasks=settings.chronos_local_llm_allowed_tasks,
        chronos_poll_interval=settings.chronos_poll_interval,
        chronos_enable_notion_import=settings.chronos_enable_notion_import,
        chronos_notion_import_batch_size=settings.chronos_notion_import_batch_size,
        chronos_self_heal_limit=settings.chronos_self_heal_limit,
        chronos_embed_batch_size=settings.chronos_embed_batch_size,
        chronos_index_events_per_limit=settings.chronos_index_events_per_limit,
        chronos_autosync_process_limit=settings.chronos_autosync_process_limit,
        chronos_autosync_index_limit=settings.chronos_autosync_index_limit,
        chronos_autosync_index_timeout=settings.chronos_autosync_index_timeout,
        chronos_autosync_graph_limit=settings.chronos_autosync_graph_limit,
        chronos_autosync_max_load_avg=settings.chronos_autosync_max_load_avg,
        chronos_autosync_min_available_mb=settings.chronos_autosync_min_available_mb,
        chronos_autosync_max_swap_used_mb=settings.chronos_autosync_max_swap_used_mb,
        chronos_autosync_defer_seconds=settings.chronos_autosync_defer_seconds,
        chronos_stats_enable_plaud_cloud=settings.chronos_stats_enable_plaud_cloud,
        flags=ServerSettingsFlagsOut(
            has_gemini_api_key=bool(settings.gemini_api_key),
            has_openai_api_key=bool(getattr(settings, "openai_api_key_configured", False)),
            has_qdrant_api_key=bool(settings.qdrant_api_key),
            has_notion_token=bool(settings.notion_token),
            has_notion_oauth=bool(notion_oauth.access_token),
        ),
    )


@router.put("", response_model=SuccessResponse)
async def update_server_settings(body: ServerSettingsUpdateRequest):
    payload = body.model_dump(exclude_none=True)
    if not payload:
        return SuccessResponse(message="No changes supplied")

    weekday = payload.get("notion_weekday_start")
    weekend = payload.get("notion_weekend_start")

    if weekday is not None:
        if not _valid_hhmm(weekday):
            raise HTTPException(
                status_code=400, detail="Weekday fallback start must be HH:MM"
            )
        weekday_hour, weekday_minute = [
            int(part) for part in str(weekday).strip().split(":", 1)
        ]
        if (weekday_hour, weekday_minute) > (8, 0):
            raise HTTPException(
                status_code=400,
                detail="Weekday fallback start must be 08:00 or earlier",
            )

    if weekend is not None and not _valid_hhmm(weekend):
        raise HTTPException(
            status_code=400, detail="Weekend fallback start must be HH:MM"
        )

    for field, minimum, detail in (
        ("chronos_poll_interval", 1, "Poll interval must be at least 1 second"),
        (
            "chronos_notion_import_batch_size",
            1,
            "Notion import batch size must be at least 1",
        ),
        (
            "chronos_self_heal_limit",
            0,
            "Self-heal limit cannot be negative",
        ),
        (
            "chronos_embed_batch_size",
            1,
            "Embedding batch size must be at least 1",
        ),
        (
            "chronos_index_events_per_limit",
            0,
            "Index events-per-limit cannot be negative",
        ),
        (
            "chronos_autosync_process_limit",
            1,
            "Autosync process limit must be at least 1",
        ),
        (
            "chronos_autosync_index_limit",
            1,
            "Autosync index limit must be at least 1",
        ),
        (
            "chronos_autosync_index_timeout",
            60,
            "Autosync index timeout must be at least 60 seconds",
        ),
        (
            "chronos_autosync_graph_limit",
            1,
            "Autosync graph limit must be at least 1",
        ),
        (
            "chronos_autosync_max_load_avg",
            0.1,
            "Autosync max load average must be greater than 0",
        ),
        (
            "chronos_autosync_min_available_mb",
            0,
            "Autosync minimum available RAM cannot be negative",
        ),
        (
            "chronos_autosync_max_swap_used_mb",
            0,
            "Autosync max swap used cannot be negative",
        ),
        (
            "chronos_autosync_defer_seconds",
            0,
            "Autosync defer seconds cannot be negative",
        ),
    ):
        value = payload.get(field)
        if value is not None and value < minimum:
            raise HTTPException(status_code=400, detail=detail)

    updates = {
        _FIELD_TO_ENV[field]: _env_string(value)
        for field, value in payload.items()
        if field in _FIELD_TO_ENV
    }

    changed = _write_env_updates(updates)
    if changed == 0:
        return SuccessResponse(message="Settings unchanged — nothing to save")
    return SuccessResponse(
        message=f"Saved {changed} setting{'s' if changed != 1 else ''} to .env"
    )
