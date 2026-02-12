"""Health and stats endpoints."""

from fastapi import APIRouter

from src.api.models.upload import CollectionStatsResponse
from src.db.qdrant import get_qdrant_client
from src.ingestion.storage import get_collection_stats

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Check API and Qdrant connectivity.

    Returns:
        A dict with ``status`` and ``qdrant`` connectivity info.
    """
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {e}"

    return {"status": "ok", "qdrant": qdrant_status}


@router.get("/collections/stats", response_model=CollectionStatsResponse)
async def collection_stats() -> CollectionStatsResponse:
    """Return point counts and status for ingestion collections.

    Returns:
        A ``CollectionStatsResponse`` with stats for each collection.
    """
    client = get_qdrant_client()
    stats = get_collection_stats(client)
    return CollectionStatsResponse(**stats)
