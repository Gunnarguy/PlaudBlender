"""API cost tracking endpoints."""

from fastapi import APIRouter, Depends

from api.schemas.responses import CostHistoryOut, SessionCostOut
from api.routes._ttl_cache import TTLCache

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/costs",
    tags=["costs"],
    dependencies=[Depends(require_auth)],
)

_cache = TTLCache()
_SESSION_COST_TTL_SECONDS = 3.0
_COST_HISTORY_TTL_SECONDS = 30.0
_PRICING_TTL_SECONDS = 600.0


@router.get("/session", response_model=SessionCostOut)
async def session_costs():
    """Current session API costs."""
    from src.chronos.cost_tracker import get_session_cost

    data = _cache.get_or_compute(
        ("session",),
        _SESSION_COST_TTL_SECONDS,
        get_session_cost,
    )
    return SessionCostOut(
        total_cost_usd=data.get("total_cost_usd", 0.0),
        total_calls=data.get("total_calls", 0),
        total_input_tokens=data.get("total_input_tokens", 0),
        total_output_tokens=data.get("total_output_tokens", 0),
        by_model=data.get("by_model", {}),
        by_type=data.get("by_type", {}),
        session_minutes=data.get("session_minutes", 0.0),
    )


@router.get("/history", response_model=CostHistoryOut)
async def cost_history(days: int = 30):
    """Historical API costs (last N days)."""
    from src.chronos.cost_tracker import get_cost_summary

    days = max(int(days), 1)
    data = _cache.get_or_compute(
        ("history", days),
        _COST_HISTORY_TTL_SECONDS,
        lambda: get_cost_summary(days=days),
    )
    return CostHistoryOut(
        days=data.get("days", days),
        total_cost_usd=data.get("total_cost_usd", 0.0),
        total_calls=data.get("total_calls", 0),
        by_model=data.get("by_model", {}),
        by_day=data.get("by_day"),
    )


@router.get("/pricing")
async def model_pricing():
    """Model pricing table."""
    from src.chronos.cost_tracker import get_model_pricing_table

    models = _cache.get_or_compute(
        ("pricing",),
        _PRICING_TTL_SECONDS,
        get_model_pricing_table,
    )
    return {"models": models}
