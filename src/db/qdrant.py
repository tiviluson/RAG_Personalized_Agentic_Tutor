"""Qdrant client singleton."""

from qdrant_client import QdrantClient

from src.config import settings

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a lazily-initialized Qdrant client singleton.

    Returns:
        QdrantClient: Connected Qdrant client instance.
    """
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )
    return _client
