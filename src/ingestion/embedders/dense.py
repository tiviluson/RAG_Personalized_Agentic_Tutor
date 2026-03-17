"""Dense embedding via Qwen3-Embedding-8B (local, sentence-transformers)."""

import threading

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Return a lazily-initialised Qwen3-Embedding model singleton. Thread-safe.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            logger.info("Loading dense embedding model: {}...", settings.dense_embedding_model)
            _model = SentenceTransformer(
                settings.dense_embedding_model,
                truncate_dim=settings.dense_embedding_dim,
                model_kwargs={"torch_dtype": torch.float16},
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

    # Free MPS GPU memory after each encode call to prevent OOM across batches
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return embeddings.tolist()


def unload_model() -> None:
    """Release the dense embedding model and free GPU memory."""
    global _model
    if _model is not None:
        logger.info("Unloading dense embedding model")
        del _model
        _model = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
