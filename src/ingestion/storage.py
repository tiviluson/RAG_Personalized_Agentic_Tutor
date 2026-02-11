"""Qdrant storage operations for the ingestion pipeline."""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings

COLLECTIONS = {
    "course_content": "Lecturer-uploaded course materials",
    "student_notes": "Student-uploaded personal notes (namespace-isolated by student_id)",
}


def create_collections(
    client: QdrantClient, recreate: bool = False
) -> None:
    """Create the ``course_content`` and ``student_notes`` collections.

    Both collections use hybrid vector config: a dense named vector
    and a BM25 sparse named vector.

    Args:
        client: An initialised Qdrant client.
        recreate: If ``True``, delete and re-create existing
            collections (useful for schema changes).
    """
    existing = {c.name for c in client.get_collections().collections}

    for name, description in COLLECTIONS.items():
        if name in existing:
            if recreate:
                client.delete_collection(name)
                logger.info("Deleted existing collection: {}", name)
            else:
                logger.info("Collection already exists (skipping): {}", name)
                continue

        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.dense_embedding_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info("Created collection: {} -- {}", name, description)


def _build_payload(chunk: dict, doc_id: str, extra_meta: dict) -> dict:
    """Build the Qdrant point payload from a chunk dict.

    Strips vector fields (stored separately) and adds provenance
    metadata (``doc_id``, ``chunk_id``, ``uploaded_at``).

    Args:
        chunk: A chunk dict containing text and metadata.
        doc_id: The document-level unique identifier.
        extra_meta: Additional payload fields (e.g. ``course_id``,
            ``uploaded_by``, ``module_name``).

    Returns:
        A payload dict suitable for ``PointStruct``.
    """
    excluded = {"dense_vector", "sparse_vector", "used_vision"}
    payload = {k: v for k, v in chunk.items() if k not in excluded}
    payload.update({
        "doc_id": doc_id,
        "chunk_id": str(uuid.uuid4()),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        **extra_meta,
    })
    return payload


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    doc_id: str | None = None,
    extra_meta: dict | None = None,
) -> int:
    """Upsert embedded chunks into a Qdrant collection.

    Chunks must already have ``dense_vector`` and ``sparse_vector``
    keys (added by ``embed_chunks``).

    Args:
        client: An initialised Qdrant client.
        collection_name: Target collection name.
        chunks: A list of chunk dicts with embedding vectors.
        doc_id: Optional document-level ID. Generated if not provided.
        extra_meta: Additional payload fields to attach to every point.

    Returns:
        The total number of points upserted.
    """
    if not chunks:
        return 0

    if doc_id is None:
        doc_id = str(uuid.uuid4())
    if extra_meta is None:
        extra_meta = {}

    points = []
    for chunk in chunks:
        sparse = chunk["sparse_vector"]
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": chunk["dense_vector"],
                "bm25": SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload=_build_payload(chunk, doc_id, extra_meta),
        )
        points.append(point)

    total = 0
    for i in range(0, len(points), settings.upsert_batch_size):
        batch = points[i : i + settings.upsert_batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)

    logger.info("Upserted {} points to '{}'", total, collection_name)
    return total


def delete_chunks_by_doc_id(
    client: QdrantClient, collection_name: str, doc_id: str
) -> None:
    """Delete all points belonging to a document from a collection.

    Used for cleaning up old chunks before re-ingesting a document.

    Args:
        client: An initialised Qdrant client.
        collection_name: Target collection name.
        doc_id: The document ID whose chunks should be deleted.
    """
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    logger.info("Deleted chunks with doc_id='{}' from '{}'", doc_id, collection_name)


def file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file for deduplication.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_collection_stats(client: QdrantClient) -> dict:
    """Return basic statistics for all ingestion collections.

    Args:
        client: An initialised Qdrant client.

    Returns:
        A dict mapping collection name to its stats (point count,
        vector count, etc.).
    """
    stats = {}
    for name in COLLECTIONS:
        try:
            info = client.get_collection(name)
            stats[name] = {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value,
            }
        except Exception:
            stats[name] = {"points_count": 0, "vectors_count": 0, "status": "missing"}
    return stats
