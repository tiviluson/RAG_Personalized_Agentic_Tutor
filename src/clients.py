"""Shared API client singletons.

Provides lazily-initialised client instances used across the ingestion pipeline (loaders, embedders, etc.).
"""

from google import genai

from src.config import settings

_genai_client: genai.Client | None = None


def get_genai_client(*, required: bool = False) -> genai.Client | None:
    """Return a lazily-initialised Gemini client singleton.

    Args:
        required (bool): If ``True``, raise ``RuntimeError`` when ``GOOGLE_API_KEY`` is not configured instead of returning ``None``.

    Returns:
        genai.Client | None: An initialised ``genai.Client``, or ``None`` if the API key is missing and ``required`` is ``False``.

    Raises:
        RuntimeError: If ``required=True`` and no API key is set.
    """
    global _genai_client
    if _genai_client is None:
        if not settings.google_api_key:
            if required:
                raise RuntimeError(
                    "GOOGLE_API_KEY is required. "
                    "Set it in .env or as an environment variable."
                )
            return None
        _genai_client = genai.Client(api_key=settings.google_api_key)
    return _genai_client
