"""Shared utilities for the retrieval and generation pipeline."""

from __future__ import annotations


def format_history(conversation_history: list[dict]) -> str:
    """Format conversation turns into a readable string.

    Args:
        conversation_history: Recent turns, each with ``role`` and
            ``content`` keys.

    Returns:
        Formatted string of conversation history, or "No prior conversation."
    """
    if not conversation_history:
        return "No prior conversation."

    parts = []
    for turn in conversation_history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
