"""Shared utilities for document chunkers.

Provides a lazy-loaded tiktoken encoder singleton and a token-counting
helper used by all chunker modules.
"""

from __future__ import annotations

import tiktoken


_encoder: tiktoken.Encoding | None = None


def _get_tiktoken_encoder() -> tiktoken.Encoding:
    """Return a lazily-initialised tiktoken ``cl100k_base`` encoder.

    The encoder is created once and cached in a module-level variable so
    that repeated calls avoid the overhead of re-loading the BPE data.

    Returns:
        tiktoken.Encoding: The ``cl100k_base`` encoding instance.
    """
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _token_len(text: str) -> int:
    """Count the number of tokens in *text* using the ``cl100k_base`` encoder.

    Args:
        text (str): The string to tokenise.

    Returns:
        int: Number of tokens.
    """
    return len(_get_tiktoken_encoder().encode(text))
