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


@router.get("/entities")
async def get_entities(type: str | None = None, min_mentions: int = 1, limit: int = 150):
    """Top entities from the knowledge graph (people, places, orgs, projects, topics)
    and the edges among them. Empty until the pipeline has built the graph."""
    from src.chronos.entity_graph import load_entity_graph, node_payload

    graph = load_entity_graph()
    if graph is None:
        return {"nodes": [], "edges": [], "events": 0}
    limit = max(1, min(limit, 2000))
    picked = sorted(
        (n for n in graph.nodes.values()
         if (not type or n.get("type") == type) and int(n.get("mentions") or 0) >= min_mentions),
        key=lambda n: -int(n.get("mentions") or 0),
    )[:limit]
    ids = {n["id"] for n in picked}
    edges = [e for e in graph.edges if e["source"] in ids and e["target"] in ids]
    return {"nodes": [node_payload(n) for n in picked], "edges": edges, "events": graph.events}


@router.get("/entities/search")
async def search_entities(q: str, type: str | None = None, limit: int = 20):
    """Entities whose name or alias contains q, most mentioned first."""
    from src.chronos.entity_graph import load_entity_graph, node_payload

    graph = load_entity_graph()
    if graph is None:
        return {"results": []}
    return {"results": [node_payload(n) for n in graph.search(q, limit=max(1, min(limit, 200)), entity_type=type)]}


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: str, svc: ChronosDataService = Depends(get_service)):
    """One entity: its profile, its strongest links (with evidence) and its latest moments."""
    from fastapi import HTTPException

    from src.chronos.entity_graph import load_entity_graph, node_payload

    graph = load_entity_graph()
    node = graph.nodes.get(entity_id) if graph else None
    if node is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    neighbors = [
        {"node": node_payload(other), "type": edge.get("type"), "weight": edge.get("weight"),
         "evidence": edge.get("evidence") or []}
        for edge, other in graph.neighbors(entity_id, limit=30)
    ]
    wanted = list(node.get("events") or [])[:10]
    moments = {}
    if wanted:
        for event in svc._get_all_events():
            if getattr(event, "id", None) in wanted:
                moments[event.id] = event
    recent = [
        {"id": eid, "start_ts": getattr(moments[eid], "start_ts", None),
         "text": getattr(moments[eid], "clean_text", "")}
        for eid in wanted if eid in moments
    ]
    return {"entity": node_payload(node), "neighbors": neighbors, "recent_events": recent}


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
