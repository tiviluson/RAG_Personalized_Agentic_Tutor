"""Sparse BM25 embedding via fastembed."""

from fastembed import SparseTextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from loguru import logger

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    """Return a lazily-initialised BM25 model singleton.

    Downloads the model on first call and caches it at module level.

    Returns:
        A ``SparseTextEmbedding`` instance for the ``Qdrant/bm25`` model.
    """
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading BM25 model...")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 model ready")
    return _bm25_model


def embed_texts_sparse(texts: list[str]) -> list[SparseEmbedding]:
    """Generate BM25 sparse embeddings for a list of texts.

    Args:
        texts: The text strings to embed.

    Returns:
        A list of ``SparseEmbedding`` objects, each with ``.indices``
        and ``.values`` arrays.
    """
    model = _get_bm25_model()
    return list(model.embed(texts))
