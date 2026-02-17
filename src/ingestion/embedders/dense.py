"""Dense embedding via Qwen3-Embedding-8B (local, sentence-transformers)."""

from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return a lazily-initialised Qwen3-Embedding model singleton.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _model
    if _model is None:
        logger.info("Loading dense embedding model: {}...", settings.dense_embedding_model)
        _model = SentenceTransformer(
            settings.dense_embedding_model,
            truncate_dim=settings.dense_embedding_dim,
        )
        logger.info(
            "Dense model loaded (dim={}, device={})",
            settings.dense_embedding_dim,
            _model.device,
        )
    return _model


def embed_texts_dense(texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings for a list of texts.

    Args:
        texts (list[str]): The text strings to embed.

    Returns:
        list[list[float]]: A list of dense embedding vectors, each of length ``settings.dense_embedding_dim``.
    """
    model = _get_model()
    logger.info("Embedding {} texts with {}", len(texts), settings.dense_embedding_model)

    embeddings = model.encode(
        texts,
        batch_size=settings.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()
