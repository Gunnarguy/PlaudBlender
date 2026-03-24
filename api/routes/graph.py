"""Knowledge-graph endpoints."""

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.responses import GraphDataOut
from app_v2.services.data_service import ChronosDataService

from api.auth.jwt import require_auth

router = APIRouter(
    prefix="/api/v1/graph",
    tags=["graph"],
    dependencies=[Depends(require_auth)],
)


@router.get("", response_model=GraphDataOut)
async def get_graph(svc: ChronosDataService = Depends(get_service)):
    """Full knowledge graph (nodes + edges) for rendering."""
    data = svc.get_graph_data()
    if data is None:
        return GraphDataOut(nodes=[], edges=[])
    return GraphDataOut(
        nodes=data.nodes if hasattr(data, "nodes") else getattr(data, "nodes", []),
        edges=data.edges if hasattr(data, "edges") else getattr(data, "edges", []),
    )
