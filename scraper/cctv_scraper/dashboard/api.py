import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cctv_scraper.dashboard.history import HistorySampler, HistoryStore, path_is_writable
from cctv_scraper.dashboard.security import build_auth_dependency
from cctv_scraper.dashboard.service import DashboardService
from cctv_scraper.dashboard.settings import DashboardConfig, load_dashboard_config

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("dashboard")


def build_history(config: DashboardConfig) -> HistoryStore | None:
    if not config.history_enabled:
        return None
    if not path_is_writable(config.history_db):
        logger.warning(
            "History database directory %s is not writable; continuing without long-range "
            "history. Mount it read-write or set DASHBOARD_HISTORY_ENABLED=false.",
            config.history_db.parent,
        )
        return None
    try:
        return HistoryStore(config)
    except Exception:
        logger.exception("Could not open the history database; continuing without history.")
        return None


def create_app(config: DashboardConfig | None = None, *, run_sampler: bool = True) -> FastAPI:
    settings = config or load_dashboard_config()
    store = build_history(settings)
    service = DashboardService(settings, store)
    require_auth = build_auth_dependency(settings)
    stop_event = threading.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        sampler: HistorySampler | None = None
        if store is not None and run_sampler:
            sampler = HistorySampler(settings, store, stop_event)
            sampler.start()
        try:
            yield
        finally:
            stop_event.set()
            if sampler is not None:
                sampler.join(timeout=5)

    app = FastAPI(
        title="CCTV Scraper Monitor",
        description="Read-only monitoring for the 24/7 CCTV recorder.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = settings
    app.state.service = service
    app.state.history = store

    @app.middleware("http")
    async def no_store_and_hardening(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # The page loads only its own assets; block anything else outright.
        # Scripts stay strict; style-src allows inline because the charts set
        # bar widths and heat colours through element.style.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    def known_camera(camera: str) -> str:
        # Also the path-traversal guard: only names the collectors discovered.
        if camera not in service.cameras():
            raise HTTPException(status_code=404, detail="Unknown camera.")
        return camera

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/overview", dependencies=[Depends(require_auth)])
    async def overview() -> JSONResponse:
        return JSONResponse(service.overview())

    @app.get("/api/cameras/{camera}", dependencies=[Depends(require_auth)])
    async def camera_detail(camera: str, days: int = Query(7, ge=1, le=60)) -> JSONResponse:
        return JSONResponse(service.camera_detail(known_camera(camera), days=days))

    @app.get("/api/coverage", dependencies=[Depends(require_auth)])
    async def coverage(days: int = Query(7, ge=1, le=60)) -> JSONResponse:
        return JSONResponse(service.coverage(days=days))

    @app.get("/api/history/uptime", dependencies=[Depends(require_auth)])
    async def uptime_history(
        hours: int = Query(24, ge=1, le=24 * 90),
        buckets: int = Query(96, ge=8, le=480),
        camera: str | None = Query(None),
    ) -> JSONResponse:
        if store is None:
            return JSONResponse({"bucket_seconds": 0, "points": []})
        name = known_camera(camera) if camera else None
        return JSONResponse(store.uptime_series(hours, buckets, camera=name))

    @app.get("/api/history/disk", dependencies=[Depends(require_auth)])
    async def disk_history(hours: int = Query(72, ge=1, le=24 * 90)) -> JSONResponse:
        if store is None:
            return JSONResponse([])
        return JSONResponse(store.disk_series(hours))

    @app.get("/api/logs", dependencies=[Depends(require_auth)])
    async def logs(lines: int = Query(200, ge=10, le=2000)) -> JSONResponse:
        return JSONResponse({"lines": service.app_log(lines)})

    @app.get("/", dependencies=[Depends(require_auth)])
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
