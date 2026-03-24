"""Health / connectivity check endpoints."""

from fastapi import APIRouter, Depends

from api.schemas.responses import HealthResponse, SystemStatusOut
from src.config import Settings

router = APIRouter(prefix="/api/v1", tags=["health"])


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

        qc = QC(url=settings.qdrant_url, timeout=3)
        collections = qc.get_collections()
        result["qdrant"] = {
            "ok": True,
            "url": settings.qdrant_url,
            "collections": len(collections.collections),
        }
    except Exception as e:
        result["qdrant"] = {"ok": False, "error": str(e)}

    # Gemini
    result["gemini"] = {"configured": bool(settings.gemini_api_key)}

    # OpenAI
    try:
        from src.chronos.openai_service import OpenAIResponseService

        svc = OpenAIResponseService()
        ok, detail = svc.check_connection()
        result["openai"] = {"ok": ok, "detail": detail}
    except Exception as e:
        result["openai"] = {"ok": False, "error": str(e)}

    # Plaud
    try:
        from src.plaud_oauth import PlaudOAuthClient

        pc = PlaudOAuthClient()
        result["plaud"] = {"is_authenticated": pc.is_authenticated}
    except Exception as e:
        result["plaud"] = {"ok": False, "error": str(e)}

    # Notion
    try:
        from src.notion_oauth import NotionOAuthClient

        nc = NotionOAuthClient()
        result["notion"] = nc.token_status
    except Exception as e:
        result["notion"] = {"ok": False, "error": str(e)}

    return SystemStatusOut(**result)
