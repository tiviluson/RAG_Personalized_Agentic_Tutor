"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import health, upload
from src.db.qdrant import get_qdrant_client
from src.ingestion.storage import create_collections


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ensure Qdrant collections exist on startup."""
    try:
        client = get_qdrant_client()
        create_collections(client, recreate=False)
        logger.info("Qdrant collections verified")
    except Exception as e:
        logger.warning("Qdrant not available on startup: {}", e)
    yield


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

    return app


app = create_app()
