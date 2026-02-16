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

from src.clients import get_genai_client
from src.config import settings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _embed_batch(client: genai.Client, batch: list[str]) -> list[list[float]]:
    """Embed a single batch of texts via the Gemini API.

    Args:
        client (genai.Client): An initialised Gemini client.
        batch (list[str]): A list of text strings (max ``embed_batch_size``).

    Returns:
        list[list[float]]: A list of dense embedding vectors.

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
        texts (list[str]): The text strings to embed.

    Returns:
        list[list[float]]: A list of dense embedding vectors, each of length ``settings.dense_embedding_dim``.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is not configured.
    """
    client = get_genai_client(required=True)
    logger.info("Embedding {} texts with Gemini", len(texts))
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        all_embeddings.extend(_embed_batch(client, batch))
    return all_embeddings
