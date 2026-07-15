import sys

if sys.platform in ("linux", "darwin"):
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

"""
Chronos FastAPI Backend — Main Application

Wraps the existing ChronosDataService and core engine services
as a REST API for the iOS app (and any other clients).

Run:
    python scripts/launch_api.py
    # or directly:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager  # noqa: E402
import logging  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402

from src.database import init_db  # noqa: E402


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    init_db()
    
    async def run_startup_tasks():
        # 1. Clean up stale processing recordings in background
        try:
            import asyncio
            from app_v2.services.data_service import ChronosDataService
            
            svc = ChronosDataService()
            await asyncio.to_thread(svc._reconcile_stale_processing_recordings)
            logger.info("Stale processing recordings check complete.")
        except Exception as exc:
            logger.warning("Stale recordings cleanup on startup failed: %s", exc)

        # 2. Warm up Plaud API credentials
        try:
            import asyncio
            from src.plaud_oauth import PlaudOAuthClient

            status = await asyncio.to_thread(
                PlaudOAuthClient().token_status_with_recovery, 
                attempt_recovery=True
            )
            if status.get("is_authenticated"):
                logger.info("Plaud auth is ready")
            else:
                logger.warning(
                    "Plaud auth not connected on startup (has_refresh_token=%s)",
                    status.get("has_refresh_token"),
                )
        except Exception as exc:
            logger.warning("Plaud auth warmup failed: %s", exc)

    import asyncio
    asyncio.create_task(run_startup_tasks())
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
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8050",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"^https?://(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
        "ngrok-skip-browser-warning",
        "X-Request-ID",
    ],
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
from api.routes.settings import router as settings_router  # noqa: E402
from api.routes.admin import router as admin_router  # noqa: E402
from api.routes.plaud_integrations import router as plaud_integrations_router  # noqa: E402

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
app.include_router(settings_router)
app.include_router(admin_router)
app.include_router(plaud_integrations_router)
