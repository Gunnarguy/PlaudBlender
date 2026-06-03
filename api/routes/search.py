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
from src.chronos.ask_context import build_ask_context
from src.chronos.ask_service import ChronosAskService

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
async def ask_ai(
    body: AskRequest,
    svc: ChronosDataService = Depends(get_service),
):
    """AI-powered Q&A over the knowledge timeline using the configured model."""
    try:
        ai = ChronosAskService()
        if not ai.available:
            raise HTTPException(
                status_code=503,
                detail="No AI provider configured (set CHRONOS_GEMINI_API_KEY or OPENAI_API_KEY)",
            )

        results, context = build_ask_context(svc, body.question)
        if not results:
            return AIAnswerOut(
                answer="I couldn't find any relevant events for that question.",
                model="",
                response_id=None,
                reasoning_summary=None,
                config={"provider": "none"},
                usage={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                },
            )

        response = ai.ask(
            question=body.question,
            context_events=context,
            previous_response_id=body.previous_response_id,
            model=body.model,
            reasoning=body.reasoning,
            reasoning_summary=body.reasoning_summary,
            temperature=body.temperature,
            top_p=body.top_p,
            max_output_tokens=body.max_output_tokens,
            verbosity=body.verbosity,
            service_tier=body.service_tier,
        )

        if "error" in response:
            raise HTTPException(status_code=502, detail=response["error"])

        return AIAnswerOut(
            answer=response.get("answer", ""),
            model=response.get("model", ""),
            response_id=response.get("response_id"),
            reasoning_summary=response.get("reasoning_summary"),
            config=response.get("config"),
            usage=response.get("usage"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
