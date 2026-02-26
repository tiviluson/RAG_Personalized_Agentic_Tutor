"""Streaming answer generation via Gemini Flash."""

from __future__ import annotations

from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

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


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
)
def _call_gemini_stream(
    system_prompt: str, user_message: str
) -> list[str]:
    """Call Gemini Flash and collect streamed text chunks.

    Args:
        system_prompt: The system instruction for the answer mode.
        user_message: The formatted user message with context.

    Returns:
        List of text fragments from the stream.

    Raises:
        Exception: Propagated after 2 retry attempts.
    """
    client = get_genai_client(required=True)
    stream = client.models.generate_content_stream(  # type: ignore[union-attr]
        model=settings.gemini_generation_model,
        contents=[user_message],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    fragments = []
    for chunk in stream:
        if chunk.text:
            fragments.append(chunk.text)
    return fragments
