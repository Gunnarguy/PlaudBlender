"""Health / connectivity check endpoints."""

from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, Depends

from api.schemas.responses import HealthResponse, SystemStatusOut
from src.config import Settings

router = APIRouter(prefix="/api/v1", tags=["health"])


def _openai_enabled_for_current_routing(settings: Settings) -> bool:
    if not getattr(settings, "chronos_openai_enabled", False):
        return False

    provider = (settings.chronos_processing_provider or "").strip().lower()
    if provider in {"openai", "auto"}:
        return True

    routed_models = [
        settings.chronos_cleaning_model,
        settings.chronos_analyst_model,
        settings.chronos_embedding_model,
    ]
    return any(
        str(model or "").strip().lower().startswith(("gpt-", "o", "text-embedding"))
        for model in routed_models
    )


def _gemini_enabled_for_current_routing(settings: Settings) -> bool:
    provider = (settings.chronos_processing_provider or "").strip().lower()
    if provider in {"gemini", "auto"}:
        return True
    if provider in {"local", "ollama", "llama.cpp", "llamacpp"}:
        return False

    routed_models = [
        settings.chronos_cleaning_model,
        settings.chronos_analyst_model,
        settings.chronos_embedding_model,
    ]
    return any(str(model or "").strip().lower().startswith("gemini") for model in routed_models)


def _qdrant_candidate_urls(url: str) -> list[str]:
    candidates = [url]
    parsed = urlparse(url)
    if parsed.hostname == "localhost":
        netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
        candidates.append(urlunparse(parsed._replace(netloc=netloc)))
    return candidates


@router.get("/health", response_model=HealthResponse)
async def health():
    """Basic liveness probe."""
    return HealthResponse()


@router.get("/status", response_model=SystemStatusOut)
async def system_status():
    """Deep connectivity check — database, Qdrant, Gemini, OpenAI, Plaud, Notion."""
    from src.config import get_settings

    settings = get_settings()
    result = {}

    # Database
    try:
        from src.database import SessionLocal

        with SessionLocal() as session:
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
        result["database"] = {"ok": True, "url": settings.database_url}
    except Exception as e:
        result["database"] = {"ok": False, "error": str(e)}

    # Qdrant
    try:
        from qdrant_client import QdrantClient as QC

        last_error = None
        for candidate_url in _qdrant_candidate_urls(settings.qdrant_url):
            try:
                qc = QC(url=candidate_url, timeout=3)
                collections = qc.get_collections()
                result["qdrant"] = {
                    "ok": True,
                    "url": settings.qdrant_url,
                    "collections": len(collections.collections),
                }
                break
            except Exception as e:  # pragma: no cover - defensive fallback
                last_error = e
        else:
            raise last_error or RuntimeError("Unknown Qdrant connection failure")
    except Exception as e:
        result["qdrant"] = {"ok": False, "error": str(e)}

    # Gemini. Treat disabled cloud routing as a safe skipped provider so UIs do
    # not show red for a provider that local/Ollama mode intentionally avoids.
    gemini_configured = bool(settings.gemini_api_key)
    gemini_enabled = _gemini_enabled_for_current_routing(settings)
    if not gemini_enabled:
        result["gemini"] = {
            "ok": True,
            "configured": gemini_configured,
            "enabled": False,
            "skipped": True,
            "detail": "Disabled by current local/Ollama Chronos routing; no Gemini quota will be used",
        }
    else:
        result["gemini"] = {
            "ok": gemini_configured,
            "configured": gemini_configured,
            "enabled": True,
            "detail": "Configured" if gemini_configured else "CHRONOS_GEMINI_API_KEY not set",
        }

    # OpenAI. Avoid generative readiness probes unless the active routing can
    # actually use OpenAI; status polling should not spend tokens.
    try:
        openai_configured = bool(getattr(settings, "openai_api_key_configured", False))
        if not getattr(settings, "chronos_openai_enabled", False):
            result["openai"] = {
                "ok": True,
                "configured": openai_configured,
                "enabled": False,
                "skipped": True,
                "detail": "Disabled by CHRONOS_OPENAI_ENABLED=0; no OpenAI quota will be used",
            }
        elif not settings.openai_api_key:
            result["openai"] = {"ok": False, "configured": False, "enabled": True, "detail": "OPENAI_API_KEY not set"}
        elif not _openai_enabled_for_current_routing(settings):
            result["openai"] = {
                "ok": True,
                "configured": True,
                "enabled": False,
                "skipped": True,
                "detail": "Configured but not used by current Chronos routing; probe skipped",
            }
        else:
            from src.chronos.openai_service import OpenAIResponseService

            svc = OpenAIResponseService()
            ok, detail = svc.check_connection(quick=True)
            result["openai"] = {"ok": ok, "configured": True, "enabled": True, "detail": detail}
    except Exception as e:
        result["openai"] = {"ok": False, "configured": bool(getattr(settings, "openai_api_key_configured", False)), "error": str(e)}

    # Plaud
    try:
        from src.plaud_oauth import PlaudOAuthClient

        pc = PlaudOAuthClient()
        status = dict(pc.token_status)
        status["recovery_attempted"] = False

        if status.get("has_access_token") or status.get("has_refresh_token"):
            try:
                pc.ensure_valid_token()
                status = dict(pc.token_status)
                status["recovery_attempted"] = True
            except Exception as exc:
                status = dict(pc.token_status)
                status["recovery_attempted"] = True
                status["recovery_error"] = str(exc)

        result["plaud"] = status
    except Exception as e:
        result["plaud"] = {"ok": False, "error": str(e)}

    # Notion
    try:
        from src.notion_oauth import NotionOAuthClient

        nc = NotionOAuthClient()
        result["notion"] = nc.token_status
    except Exception as e:
        result["notion"] = {"ok": False, "error": str(e)}

    # Optional local LLM sidecar (Ollama / llama.cpp-compatible HTTP)
    try:
        from src.chronos.local_llm_service import LocalLLMService

        result["local_llm"] = LocalLLMService(settings=settings).status()
    except Exception as e:
        result["local_llm"] = {
            "enabled": bool(getattr(settings, "chronos_local_llm_enabled", False)),
            "ok": False,
            "error": str(e),
        }

    return SystemStatusOut(**result)
