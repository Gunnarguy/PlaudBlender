"""Semantic search and AI Q&A endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from api.auth.jwt import require_auth
from api.dependencies import get_service
from api.schemas.responses import (
    AIAnswerOut,
    AskRequest,
    SearchRequest,
    SearchResponse,
    SearchResultOut,
)
from api.routes.recordings import _event_to_out
from app_v2.services.data_service import ChronosDataService

router = APIRouter(
    prefix="/api/v1/search",
    tags=["search"],
    dependencies=[Depends(require_auth)],
)


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """Semantic search across all events."""
    results = svc.search(
        query=body.query,
        limit=body.limit,
        categories=body.categories,
        start_date=body.start_date,
        end_date=body.end_date,
    )
    out = []
    for r in results:
        out.append(
            SearchResultOut(
                event=(
                    _event_to_out(r.event) if hasattr(r, "event") else _event_to_out(r)
                ),
                score=getattr(r, "score", 0.0),
                context_before=getattr(r, "context_before", None),
                context_after=getattr(r, "context_after", None),
            )
        )
    return SearchResponse(results=out, total=len(out))


@router.post("/ask", response_model=AIAnswerOut)
async def ask_ai(body: AskRequest):
    """AI-powered Q&A over the knowledge timeline using GPT-5.4."""
    try:
        from src.chronos.openai_service import OpenAIResponseService

        ai = OpenAIResponseService()
        if not ai.available:
            raise HTTPException(status_code=503, detail="OpenAI not configured")

        # Get context events from recent search
        from api.dependencies import get_service

        svc = get_service()
        results = svc.search(query=body.question, limit=10)
        context = []
        for r in results:
            ev = r.event
            context.append(
                {
                    "date": str(ev.start_ts),
                    "time": str(ev.start_ts),
                    "category": ev.category,
                    "text": ev.clean_text,
                }
            )

        response = ai.ask(
            question=body.question,
            context_events=context,
            reasoning=body.reasoning,
        )

        if "error" in response:
            raise HTTPException(status_code=502, detail=response["error"])

        return AIAnswerOut(
            answer=response.get("answer", ""),
            model=response.get("model", ""),
            response_id=response.get("response_id"),
            usage=response.get("usage"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
