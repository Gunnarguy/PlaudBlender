"""Knowledge-graph endpoints."""

from datetime import date as date_cls

from fastapi import APIRouter, Depends, HTTPException

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


def _day(value: str | None, name: str) -> str | None:
    """A YYYY-MM-DD query value, checked; 422 when it isn't a date."""
    if value is None or value == "":
        return None
    try:
        return date_cls.fromisoformat(value).isoformat()
    except ValueError:
        raise HTTPException(status_code=422, detail=f"{name} must be a date as YYYY-MM-DD") from None


@router.get("/constellation")
async def get_constellation(
    limit: int = 800,
    types: str | None = None,
    since: str | None = None,
    until: str | None = None,
):
    """The entity map: people, places, organizations, projects and topics with fixed x/y in
    [-1, 1], their community and moments per ISO week, the communities as regions, and the
    links among the entities shown.

    limit: most-mentioned entities to return (1-5000). types: comma-separated entity types.
    since/until (YYYY-MM-DD): only entities active in that window (by ISO week). Edges: at
    most 4 per entity shown, stated relationships first. meta says what was cut and why
    (meta.truncation); empty lists when the graph hasn't been built.
    """
    from src.chronos.entity_graph import constellation, load_entity_graph

    wanted = {t.strip() for t in (types or "").split(",") if t.strip()} or None
    return constellation(
        load_entity_graph(),
        limit=max(1, min(limit, 5000)),
        types=wanted,
        since=_day(since, "since"),
        until=_day(until, "until"),
    )


@router.get("/recordings/{recording_id}/entities")
async def get_recording_entities(recording_id: str):
    """Entities named in one recording, with how many of its moments name each and where
    they sit on the map; most moments first. Empty when the recording is unknown."""
    from src.chronos.entity_graph import entities_with_counts, load_entity_graph, load_entity_index

    counts = load_entity_index()["recordings"].get(recording_id) or {}
    entities = entities_with_counts(load_entity_graph(), counts)
    return {"recording_id": recording_id, "entities": entities, "total": len(entities)}


@router.get("/days/{date}/entities")
async def get_day_entities(date: str):
    """Entities named on one day (YYYY-MM-DD, the day /api/v1/timeline/days lists the
    recording under), with moment counts and map positions; most moments first."""
    from src.chronos.entity_graph import entities_with_counts, load_entity_graph, load_entity_index

    day = _day(date, "date")
    counts = load_entity_index()["days"].get(day) or {}
    entities = entities_with_counts(load_entity_graph(), counts)
    return {"date": day, "entities": entities, "total": len(entities)}


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: str, svc: ChronosDataService = Depends(get_service)):
    """One entity: its profile, its strongest links (with evidence) and its latest moments.

    The entity also carries weeks ({"YYYY-Www": moments}), community, x and y from the
    constellation (null/empty before the first build with it); each recent moment carries
    recording_id, category and start_ts.
    """
    from src.chronos.entity_graph import load_entity_graph, node_payload

    graph = load_entity_graph()
    node = graph.nodes.get(entity_id) if graph else None
    if node is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    neighbors = [
        {"node": node_payload(other), "type": edge.get("type"), "weight": edge.get("weight"),
         "evidence": edge.get("evidence") or [],
         # "out": this entity -type-> node; "in": node -type-> this entity
         "direction": "out" if edge.get("source") == entity_id else "in"}
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
         "text": getattr(moments[eid], "clean_text", ""),
         "recording_id": getattr(moments[eid], "recording_id", None),
         "category": getattr(moments[eid], "category", None)}
        for eid in wanted if eid in moments
    ]
    entity = {**node_payload(node), "weeks": dict(node.get("weeks") or {}), "community": node.get("community"),
              "x": node.get("x"), "y": node.get("y")}
    return {"entity": entity, "neighbors": neighbors, "recent_events": recent}


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
