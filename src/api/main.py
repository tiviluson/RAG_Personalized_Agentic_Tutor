"""FastAPI application factory."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import health, query, upload
from src.db.job_store import cleanup_old_jobs
from src.db.qdrant import get_qdrant_client
from src.ingestion.embedders.dense import _get_model as _get_dense_model
from src.ingestion.storage import create_collections
from src.retrieval.session import cleanup_expired_sessions

JOB_CLEANUP_INTERVAL_SECONDS = 3600


async def _periodic_cleanup() -> None:
    """Run job store and session cleanup every ``JOB_CLEANUP_INTERVAL_SECONDS``."""
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        removed_jobs = cleanup_old_jobs()
        if removed_jobs:
            logger.info("Job cleanup: removed {} old jobs", removed_jobs)
        removed_sessions = cleanup_expired_sessions()
        if removed_sessions:
            logger.info("Session cleanup: removed {} expired sessions", removed_sessions)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ensure Qdrant collections exist on startup, warm up models, and start background cleanup."""
    try:
        client = get_qdrant_client()
        create_collections(client, recreate=False)
        logger.info("Qdrant collections verified")
    except Exception as e:
        logger.warning("Qdrant not available on startup: {}", e)

    async def _load_dense_model() -> None:
        try:
            await asyncio.to_thread(_get_dense_model)
        except Exception as e:
            logger.warning("Dense embedding model failed to load on startup: {}", e)

    model_task = asyncio.create_task(_load_dense_model())
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    model_task.cancel()
    cleanup_task.cancel()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured ``FastAPI`` instance with routers mounted
        and CORS middleware enabled.
    """
    app = FastAPI(title="RAG Ingestion API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(upload.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    app.include_router(query.router, prefix="/api")

    return app


app = create_app()
