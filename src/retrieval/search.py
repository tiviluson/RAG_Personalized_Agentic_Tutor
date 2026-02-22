"""Hybrid search with Qdrant native RRF across both collections."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from src.config import settings
from src.ingestion.embedders.dense import embed_texts_dense
from src.ingestion.embedders.sparse import embed_texts_sparse


@dataclass
class RetrievedChunk:
    """A single chunk returned from hybrid search.

    Attributes:
        text: The chunk text content.
        score: RRF fusion score from Qdrant.
        chunk_id: Unique chunk identifier from payload.
        collection: Which collection this came from.
        metadata: Full payload metadata dict.
    """

    text: str
    score: float
    chunk_id: str
    collection: str
    metadata: dict = field(default_factory=dict)


def _build_filter(
    filters: dict | None,
    collection: str,
    student_id: str | None,
) -> Filter | None:
    """Build a Qdrant Filter from the provided metadata filters.

    Args:
        filters: Dict of optional filter fields.
        collection: The target collection name.
        student_id: Student ID for scoping student_notes.

    Returns:
        A Qdrant Filter, or None if no conditions apply.
    """
    conditions = []

    if collection == "student_notes" and student_id:
        conditions.append(
            FieldCondition(key="student_id", match=MatchValue(value=student_id))
        )

    if filters:
        field_map = {
            "course_id": "course_id",
            "module_week": "module_week",
            "module_name": "module_name",
            "uploaded_by": "uploaded_by",
            "source_filename": "source_filename",
        }
        for param, qdrant_field in field_map.items():
            value = filters.get(param)
            if value is not None:
                conditions.append(
                    FieldCondition(key=qdrant_field, match=MatchValue(value=value))
                )

    return Filter(must=conditions) if conditions else None


def _search_single_collection(
    client: QdrantClient,
    collection: str,
    dense_vector: list[float],
    sparse_vector: SparseVector,
    query_filter: Filter | None,
    k: int,
) -> list[RetrievedChunk]:
    """Run hybrid RRF search on a single Qdrant collection.

    Args:
        client: Qdrant client instance.
        collection: Collection name to search.
        dense_vector: Dense embedding of the query.
        sparse_vector: BM25 sparse embedding of the query.
        query_filter: Optional metadata filter.
        k: Number of results to return.

    Returns:
        List of RetrievedChunk results.
    """
    results = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", limit=k),
            Prefetch(query=sparse_vector, using="bm25", limit=k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
        query_filter=query_filter,
    )

    chunks = []
    for point in results.points:
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                score=point.score if point.score is not None else 0.0,
                chunk_id=payload.get("chunk_id", ""),
                collection=collection,
                metadata=payload,
            )
        )
    return chunks


def hybrid_search(
    client: QdrantClient,
    query_text: str,
    *,
    student_id: str | None = None,
    filters: dict | None = None,
    collections: list[str] | None = None,
    k_per_collection: int | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid dense+BM25 search with RRF across collections.

    Queries both ``course_content`` and ``student_notes`` (if student_id
    is provided) in parallel, merges results into a single ranked list.

    Args:
        client: Qdrant client instance.
        query_text: The search query string.
        student_id: Student ID for scoping student_notes. If None,
            only course_content is searched.
        filters: Optional metadata filters dict.
        collections: Collections to search. Defaults to course_content
            plus student_notes (if student_id provided).
        k_per_collection: Number of results per collection.

    Returns:
        Merged list of RetrievedChunk, sorted by score descending.
    """
    if k_per_collection is None:
        k_per_collection = settings.retrieval_k_per_collection

    if collections is None:
        collections = ["course_content"]
        if student_id:
            collections.append("student_notes")

    # Embed query (dense + sparse)
    dense_vecs = embed_texts_dense([query_text])
    sparse_vecs = embed_texts_sparse([query_text])
    dense_vector = dense_vecs[0]
    sparse_raw = sparse_vecs[0]
    sparse_vector = SparseVector(
        indices=sparse_raw.indices.tolist(),
        values=sparse_raw.values.tolist(),
    )

    # Search collections in parallel
    all_chunks: list[RetrievedChunk] = []

    with ThreadPoolExecutor(max_workers=len(collections)) as pool:
        futures = {}
        for coll in collections:
            qf = _build_filter(filters, coll, student_id)
            future = pool.submit(
                _search_single_collection,
                client,
                coll,
                dense_vector,
                sparse_vector,
                qf,
                k_per_collection,
            )
            futures[future] = coll

        for future in as_completed(futures):
            coll_name = futures[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                logger.debug(
                    "Retrieved {} chunks from '{}'", len(chunks), coll_name
                )
            except Exception as e:
                logger.error("Search failed for '{}': {}", coll_name, e)

    # Sort by score descending
    all_chunks.sort(key=lambda c: c.score, reverse=True)
    logger.info(
        "Hybrid search returned {} total chunks from {} collection(s)",
        len(all_chunks),
        len(collections),
    )
    return all_chunks
