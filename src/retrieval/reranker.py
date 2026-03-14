"""Reranking via jina-reranker-v3."""

from __future__ import annotations

from loguru import logger
from transformers import AutoModel

from src.config import settings
from src.retrieval.search import RetrievedChunk

_reranker = None


def _get_reranker():
    """Return a lazily-initialised jina-reranker-v3 singleton.

    Returns:
        A jina-reranker-v3 model loaded via ``transformers.AutoModel``.
    """
    global _reranker
    if _reranker is None:
        model_id = settings.reranker_model
        logger.info("Loading reranker model: {}...", model_id)
        _reranker = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )
        _reranker.eval()
        logger.info("Reranker ready")
    return _reranker


def unload_model() -> None:
    """Release the reranker model and free memory."""
    global _reranker
    if _reranker is not None:
        logger.info("Unloading reranker model")
        del _reranker
        _reranker = None


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_results: int | None = None,
    min_results: int | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Rerank chunks using jina-reranker-v3 listwise scoring.

    Applies both confidence-based and count-based cutoffs, with a
    guaranteed minimum floor.

    Args:
        query: The search query string.
        chunks: Candidate chunks to rerank.
        max_results: Maximum results to return.
        min_results: Minimum results to return (floor), even if below
            min_score. Returns all if fewer exist.
        min_score: Minimum reranker score threshold.

    Returns:
        Reranked and filtered list of RetrievedChunk with updated scores.
    """
    if max_results is None:
        max_results = settings.reranker_max_results
    if min_results is None:
        min_results = settings.reranker_min_results
    if min_score is None:
        min_score = settings.reranker_min_score

    if not chunks:
        return []

    reranker = _get_reranker()
    documents = [c.text for c in chunks]

    logger.debug("Reranking {} candidates for query: {}...", len(documents), query[:80])
    results = reranker.rerank(query, documents)

    # Map reranker scores back to chunks
    scored: list[tuple[float, RetrievedChunk]] = []
    for result in results:
        idx = result["index"]
        score = result["relevance_score"]
        chunk = chunks[idx]
        chunk.score = score
        scored.append((score, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Apply cutoffs: keep above min_score up to max_results, but ensure min_results floor
    above_threshold = [(s, c) for s, c in scored if s >= min_score]

    if len(above_threshold) >= min_results:
        final = above_threshold[:max_results]
    else:
        # Not enough above threshold -- take at least min_results from all
        final = scored[: min(min_results, len(scored))]

    # Cap at max_results
    final = final[:max_results]
    result_chunks = [c for _, c in final]

    logger.info(
        "Reranked: {} -> {} results (min_score={}, score range={:.3f}-{:.3f})",
        len(chunks),
        len(result_chunks),
        min_score,
        result_chunks[-1].score if result_chunks else 0.0,
        result_chunks[0].score if result_chunks else 0.0,
    )
    return result_chunks
