"""Dense embedding via Google Gemini embedding API."""

from google import genai
from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

_genai_client: genai.Client | None = None


def _get_genai_client() -> genai.Client:
    """Return a lazily-initialised Gemini client singleton.

    Returns:
        An initialised ``genai.Client``.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is not configured.
    """
    global _genai_client
    if _genai_client is None:
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for dense embedding. "
                "Set it in .env or as an environment variable."
            )
        _genai_client = genai.Client(api_key=settings.google_api_key)
    return _genai_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _embed_batch(client: genai.Client, batch: list[str]) -> list[list[float]]:
    """Embed a single batch of texts via the Gemini API.

    Args:
        client: An initialised ``genai.Client``.
        batch: A list of text strings (max ``embed_batch_size``).

    Returns:
        A list of dense embedding vectors.

    Raises:
        Exception: Propagated after 3 retry attempts.
    """
    result = client.models.embed_content(
        model=settings.dense_embedding_model,
        contents=batch,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in result.embeddings]


def embed_texts_dense(texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings for a list of texts.

    Batches requests according to ``settings.embed_batch_size``.

    Args:
        texts: The text strings to embed.

    Returns:
        A list of dense embedding vectors, each of length
        ``settings.dense_embedding_dim``.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is not configured.
    """
    client = _get_genai_client()
    logger.info("Embedding {} texts with Gemini", len(texts))
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        all_embeddings.extend(_embed_batch(client, batch))
    return all_embeddings
