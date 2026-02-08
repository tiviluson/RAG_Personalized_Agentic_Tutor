"""Embedding module for the ingestion pipeline."""

from loguru import logger

from src.ingestion.embedders.dense import embed_texts_dense
from src.ingestion.embedders.sparse import embed_texts_sparse


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add dense and sparse vectors to each chunk dict.

    Generates ``gemini-embedding-001`` dense vectors and BM25 sparse
    vectors for all chunks, then attaches them as ``dense_vector`` and
    ``sparse_vector`` keys. Modifies chunks in-place.

    Args:
        chunks: A list of chunk dicts, each containing a ``text`` key.

    Returns:
        The same list of chunk dicts, each now also containing
        ``dense_vector`` (``list[float]``) and ``sparse_vector``
        (``SparseEmbedding``) keys.
    """
    texts = [c["text"] for c in chunks]

    logger.info("Generating dense embeddings for {} chunks...", len(texts))
    dense_vecs = embed_texts_dense(texts)

    logger.info("Generating BM25 sparse embeddings for {} chunks...", len(texts))
    sparse_vecs = embed_texts_sparse(texts)

    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        chunk["dense_vector"] = dense
        chunk["sparse_vector"] = sparse

    logger.info(
        "Done. Dense dim: {}, BM25 non-zero (first chunk): {}",
        len(dense_vecs[0]),
        len(sparse_vecs[0].indices),
    )
    return chunks


__all__ = ["embed_chunks", "embed_texts_dense", "embed_texts_sparse"]
