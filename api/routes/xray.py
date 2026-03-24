"""X-Ray Activity Monitor endpoints."""

from fastapi import APIRouter, Depends

from api.schemas.responses import XRayEventsResponse, XRayEventOut

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/xray",
    tags=["xray"],
    dependencies=[Depends(require_auth)],
)


@router.get("/events", response_model=XRayEventsResponse)
async def get_events(since_seq: int = 0, limit: int = 50):
    """Poll X-Ray events (incremental via since_seq)."""
    from app_v2.services.xray import get_recent_events

    raw = get_recent_events(limit=limit, since_seq=since_seq)
    events = [
        XRayEventOut(
            seq=e["seq"],
            ts=e["ts"],
            source=e["source"],
            op=e["op"],
            message=e["message"],
            duration_ms=e.get("duration_ms"),
            detail=e.get("detail"),
            level=e.get("level", "info"),
        )
        for e in raw
    ]
    latest = events[0].seq if events else since_seq
    return XRayEventsResponse(events=events, latest_seq=latest)


@router.get("/throughput")
async def throughput(buckets: int = 30):
    """Per-second event throughput (last N seconds)."""
    from app_v2.services.xray import get_throughput

    return {"buckets": get_throughput(buckets=buckets)}


@router.post("/clear")
async def clear_events():
    """Clear all X-Ray events."""
    from app_v2.services.xray import clear_events as _clear

    _clear()
    return {"success": True}
