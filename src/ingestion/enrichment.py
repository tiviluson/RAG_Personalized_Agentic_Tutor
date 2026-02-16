"""LLM-based metadata enrichment for chunks.

Extracts ``content_category`` and ``topic_tags`` per chunk using
batched Gemini calls for efficiency.
"""

import json

from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.clients import get_genai_client

ENRICH_BATCH_SIZE = 20

VALID_CATEGORIES = {
    "concept",
    "definition",
    "example",
    "exercise",
    "proof",
    "summary",
    "reference",
    "other",
}

_SYSTEM_PROMPT = """\
You are a metadata extractor for academic course materials. \
For each chunk of text, determine:
1. content_category: exactly one of \
[concept, definition, example, exercise, proof, summary, reference, other]
2. topic_tags: a list of 2-5 key academic concepts or topics covered. \
Use concise, canonical names.

Respond with a JSON array where each element has \
"index", "content_category", and "topic_tags". \
Return ONLY valid JSON, no markdown fences or commentary.

Example input:

--- Chunk 0 ---
A binary search tree (BST) is a binary tree where each node's left subtree \
contains only nodes with keys less than the node's key, and each node's right \
subtree contains only nodes with keys greater than the node's key.

--- Chunk 1 ---
Example 4.2: Insert the values 5, 3, 7, 1, 4 into an empty BST. \
Step 1: Insert 5 as the root. Step 2: 3 < 5, so insert as left child. \
Step 3: 7 > 5, so insert as right child...

--- Chunk 2 ---
Prove that the height of a balanced BST with n nodes is O(log n). \
Proof: By induction on n. Base case: n=1, height=0=O(log 1)...

Example output:

[
  {"index": 0, "content_category": "definition", "topic_tags": ["binary search tree", "tree data structure"]},
  {"index": 1, "content_category": "example", "topic_tags": ["binary search tree", "BST insertion"]},
  {"index": 2, "content_category": "proof", "topic_tags": ["balanced BST", "tree height", "asymptotic analysis"]}
]"""


def _build_batch_prompt(chunks: list[dict], start_idx: int) -> str:
    """Build a prompt containing multiple chunks for batch classification.

    Args:
        chunks (list[dict]): Chunk dicts, each with a ``text`` key.
        start_idx (int): Global index offset for the first chunk in this batch.

    Returns:
        str: Formatted prompt with numbered chunks.
    """
    parts = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"][:2000]  # truncate very long chunks to save tokens
        parts.append(f"--- Chunk {start_idx + i} ---\n{text}")
    return "\n\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _call_gemini_enrich(prompt: str) -> list[dict]:
    """Call Gemini to classify a batch of chunks.

    Args:
        prompt (str): The formatted prompt with chunk texts.

    Returns:
        list[dict]: Parsed JSON array of enrichment results.

    Raises:
        Exception: Propagated after 3 retry attempts.
    """
    client = get_genai_client(required=True)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return json.loads(response.text)


def _parse_enrichment(raw_results: list[dict], batch_size: int) -> list[dict]:
    """Validate and normalize LLM enrichment output.

    Args:
        raw_results (list[dict]): Parsed JSON from the LLM response.
        batch_size (int): Expected number of results.

    Returns:
        list[dict]: Normalized list of dicts with ``content_category`` and ``topic_tags``.
    """
    indexed = {}
    for item in raw_results:
        idx = item.get("index")
        if idx is not None:
            category = item.get("content_category", "other")
            if category not in VALID_CATEGORIES:
                category = "other"
            tags = item.get("topic_tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t) for t in tags[:10]]
            indexed[idx] = {"content_category": category, "topic_tags": tags}

    results = []
    for i in range(batch_size):
        results.append(indexed.get(i, {"content_category": "", "topic_tags": []}))
    return results


def enrich_chunks(chunks: list[dict]) -> list[dict]:
    """Add ``content_category`` and ``topic_tags`` to each chunk via batched LLM calls.

    Processes chunks in batches of ``ENRICH_BATCH_SIZE`` for efficiency.
    On failure, falls back to empty defaults so ingestion is not blocked.

    Args:
        chunks (list[dict]): Chunk dicts, each containing a ``text`` key.

    Returns:
        list[dict]: The same chunk dicts with ``content_category`` and ``topic_tags`` added.
    """
    if not chunks:
        return chunks

    logger.info("Enriching {} chunks with content_category and topic_tags...", len(chunks))
    total_enriched = 0

    for batch_start in range(0, len(chunks), ENRICH_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + ENRICH_BATCH_SIZE]
        try:
            prompt = _build_batch_prompt(batch, start_idx=0)
            raw = _call_gemini_enrich(prompt)
            enrichments = _parse_enrichment(raw, len(batch))

            for chunk, enrichment in zip(batch, enrichments):
                chunk["content_category"] = enrichment["content_category"]
                chunk["topic_tags"] = enrichment["topic_tags"]
            total_enriched += sum(1 for e in enrichments if e["content_category"])

        except Exception as e:
            logger.warning(
                "Enrichment failed for batch starting at {}: {}. Using defaults.",
                batch_start,
                e,
            )
            for chunk in batch:
                chunk.setdefault("content_category", "")
                chunk.setdefault("topic_tags", [])

    logger.info("Enriched {}/{} chunks successfully", total_enriched, len(chunks))
    return chunks
