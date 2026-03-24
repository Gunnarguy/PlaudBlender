"""Topic aggregation endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.responses import TopicOut, TopicOccurrenceOut, TopicTimelineOut
from app_v2.services.data_service import ChronosDataService

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/topics",
    tags=["topics"],
    dependencies=[Depends(require_auth)],
)


@router.get("", response_model=list[TopicOut])
async def list_topics(svc: ChronosDataService = Depends(get_service)):
    """All topics with event counts."""
    topics = svc.get_all_topics()
    return [TopicOut(name=name, count=count) for name, count in topics]


@router.get("/{topic_name}", response_model=TopicTimelineOut)
async def topic_timeline(
    topic_name: str,
    svc: ChronosDataService = Depends(get_service),
):
    """Timeline of occurrences for a specific topic/keyword."""
    tl = svc.get_topic_timeline(topic_name)
    occurrences = []
    if tl and hasattr(tl, "occurrences"):
        for occ in tl.occurrences:
            occurrences.append(
                TopicOccurrenceOut(
                    event_id=occ.event_id,
                    recording_id=occ.recording_id,
                    timestamp=str(occ.timestamp),
                    text_snippet=occ.text_snippet,
                    category=occ.category,
                )
            )
    return TopicTimelineOut(
        topic=topic_name,
        total_occurrences=tl.total_occurrences if tl else 0,
        recording_count=tl.recording_count if tl else 0,
        occurrences=occurrences,
    )
