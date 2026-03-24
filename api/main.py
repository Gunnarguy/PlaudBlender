"""
Chronos FastAPI Backend — Main Application

Wraps the existing ChronosDataService and core engine services
as a REST API for the iOS app (and any other clients).

Run:
    python scripts/launch_api.py
    # or directly:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    init_db()
    yield


app = FastAPI(
    title="Chronos API",
    description="REST API for the Chronos knowledge timeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow iOS app and local development
# The iOS simulator can't reach localhost, so it uses the Mac's LAN IP.
# allow_origin_regex covers any 192.168.x.x / 10.x.x.x / 172.16-31.x.x address.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8050",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"^http://(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.).*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ────────────────────────────────────────
from api.routes.health import router as health_router  # noqa: E402
from api.routes.timeline import router as timeline_router  # noqa: E402
from api.routes.recordings import router as recordings_router  # noqa: E402
from api.routes.search import router as search_router  # noqa: E402
from api.routes.topics import router as topics_router  # noqa: E402
from api.routes.graph import router as graph_router  # noqa: E402
from api.routes.stats import router as stats_router  # noqa: E402
from api.routes.sync import router as sync_router  # noqa: E402
from api.routes.notion import router as notion_router  # noqa: E402
from api.routes.xray import router as xray_router  # noqa: E402
from api.routes.costs import router as costs_router  # noqa: E402
from api.routes.auth import router as auth_router  # noqa: E402

app.include_router(health_router)
app.include_router(timeline_router)
app.include_router(recordings_router)
app.include_router(search_router)
app.include_router(topics_router)
app.include_router(graph_router)
app.include_router(stats_router)
app.include_router(sync_router)
app.include_router(notion_router)
app.include_router(xray_router)
app.include_router(costs_router)
app.include_router(auth_router)
