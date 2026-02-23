"""Query preprocessing via Gemini Flash.

Resolves conversation references, rewrites for retrieval, classifies
strategy, optionally generates expansion queries, and detects
out-of-scope/harmful input -- all in a single LLM call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

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
from src.retrieval.prompts import QUERY_REWRITE_PROMPT
from src.retrieval.utils import format_history


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing.

    Attributes:
        rewritten_query: Main search query, rewritten for retrieval.
        expansion_queries: 0-2 additional queries for ambiguous/complex cases.
        strategy: Classification of the query processing strategy used.
        is_out_of_scope: True if the query is unrelated, harmful, or
            attempts prompt injection.
        refusal_message: Short response when query is out of scope.
    """

    rewritten_query: str
    expansion_queries: list[str] = field(default_factory=list)
    strategy: Literal["simple", "multi_query", "decomposition", "step_back"] = "simple"
    is_out_of_scope: bool = False
    refusal_message: str | None = None


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
def _call_gemini_rewrite(user_prompt: str) -> dict:
    """Call Gemini Flash to preprocess a query.

    Args:
        user_prompt: Formatted prompt with query and history.

    Returns:
        Parsed JSON dict with rewrite results.

    Raises:
        Exception: Propagated after 2 retry attempts.
    """
    client = get_genai_client(required=True)
    response = client.models.generate_content(  # type: ignore[union-attr]
        model=settings.gemini_generation_model,
        contents=[user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=QUERY_REWRITE_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return json.loads(response.text)  # type: ignore[arg-type]


def preprocess_query(
    raw_query: str,
    conversation_history: list[dict] | None = None,
) -> PreprocessedQuery:
    """Preprocess a student query for retrieval.

    Makes a single Gemini Flash call that simultaneously resolves
    conversation references, rewrites the query, classifies the
    strategy, and optionally generates expansion queries.

    Args:
        raw_query: The student's raw input query.
        conversation_history: Recent conversation turns (last N), each
            a dict with ``role`` and ``content`` keys.

    Returns:
        PreprocessedQuery with rewritten query and metadata.
    """
    history = conversation_history or []
    history_text = format_history(history)

    user_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Student's query: {raw_query}"
    )

    try:
        result = _call_gemini_rewrite(user_prompt)
    except Exception:
        logger.warning(
            "Query preprocessing failed, falling back to raw query"
        )
        return PreprocessedQuery(rewritten_query=raw_query)

    # Parse and validate
    rewritten = result.get("rewritten_query", raw_query)
    if not rewritten or not rewritten.strip():
        rewritten = raw_query

    expansion = result.get("expansion_queries", [])
    if not isinstance(expansion, list):
        expansion = []
    expansion = [q for q in expansion if isinstance(q, str) and q.strip()][:2]

    strategy = result.get("strategy", "simple")
    valid_strategies = {"simple", "multi_query", "decomposition", "step_back"}
    if strategy not in valid_strategies:
        strategy = "simple"

    is_oos = bool(result.get("is_out_of_scope", False))
    refusal = result.get("refusal_message")
    if is_oos and not refusal:
        refusal = "I can only help with questions related to this course."

    preprocessed = PreprocessedQuery(
        rewritten_query=rewritten,
        expansion_queries=expansion,
        strategy=strategy,
        is_out_of_scope=is_oos,
        refusal_message=refusal,
    )

    logger.info(
        "Query preprocessed: strategy={}, expansions={}, oos={}",
        preprocessed.strategy,
        len(preprocessed.expansion_queries),
        preprocessed.is_out_of_scope,
    )
    return preprocessed
