from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI

from .routes import auth, datasets, stats, tasks
from .config import get_settings
from .services.backup import idle_backup_worker


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Fish Disease Annotation Platform API",
        version="0.1.0",
        root_path=settings.root_path or "",
    )

    app.include_router(auth.router)
    app.include_router(datasets.router)
    app.include_router(tasks.router)
    app.include_router(stats.router)

    worker_task: Optional[asyncio.Task] = None

    @app.on_event("startup")
    async def _start_background() -> None:
        nonlocal worker_task
        settings = get_settings()
        if settings.idle_backup_enabled:
            worker_task = asyncio.create_task(idle_backup_worker(settings))

    @app.on_event("shutdown")
    async def _stop_background() -> None:
        nonlocal worker_task
        if worker_task is not None:
            worker_task.cancel()
            try:
                await worker_task
            except Exception:
                pass

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
