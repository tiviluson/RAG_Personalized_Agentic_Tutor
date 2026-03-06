"""Streaming answer generation via Gemini Flash."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from google.genai import types

from src.clients import get_genai_client
from src.config import settings
from src.retrieval.context import ContextBlock
from src.retrieval.utils import format_history


def _build_user_message(
    query: str,
    context_block: ContextBlock,
    history: list[dict],
) -> str:
    """Build the user message with context and conversation history.

    Args:
        query: The student's query (rewritten).
        context_block: Assembled context with citation markers.
        history: Recent conversation turns.

    Returns:
        Formatted user message string.
    """
    parts: list[str] = []

    if history:
        parts.append("Previous conversation:")
        parts.append(format_history(history))
        parts.append("")

    if context_block.text:
        parts.append("Retrieved context:")
        parts.append(context_block.text)
        parts.append("")

    parts.append(f"Student's question: {query}")
    return "\n".join(parts)


async def stream_gemini(
    system_prompt: str,
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Yield text fragments from Gemini Flash as they stream in.

    Retries once after a 2-second wait on failure. A mid-stream failure
    propagates immediately without retry to avoid duplicate tokens.

    Args:
        system_prompt: The system instruction for the answer mode.
        user_message: The formatted user message with context.

    Yields:
        Text fragments in arrival order.

    Raises:
        Exception: Propagated after 2 retry attempts.
    """
    client = get_genai_client(required=True)
    last_exc: Exception | None = None

    for attempt in range(2):
        try:
            async for chunk in await client.aio.models.generate_content_stream(  # type: ignore[union-attr]
                model=settings.gemini_generation_model,
                contents=[user_message],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                ),
            ):
                if chunk.text:
                    yield chunk.text
            return  # stream completed successfully
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(2)

    raise last_exc  # type: ignore[misc]
