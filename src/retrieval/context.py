"""Context assembly: dedup, truncate, format citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from loguru import logger

from src.config import settings
from src.ingestion.chunkers.base import _get_tiktoken_encoder
from src.retrieval.search import RetrievedChunk

# Maximum candidates that jina-reranker-v3 can handle in one listwise pass.
_RERANKER_MAX_CANDIDATES = 64


@dataclass
class Citation:
    """Source metadata for a single inline citation marker.

    Attributes:
        index: 1-based citation number (matches ``[N]`` in the context).
        source_filename: Original filename of the source document.
        page_num: Page number in the source document, if available.
        section: Section heading, if available.
        chapter: Chapter heading, if available.
        module_week: Module week label, if available.
        collection: Which Qdrant collection this came from.
        content_category: LLM-assigned content category.
        relevance_score: Reranker score.
        text_preview: First ~100 characters of the chunk text.
    """

    index: int
    source_filename: str
    page_num: int | None
    section: str | None
    chapter: str | None
    module_week: str | None
    collection: str
    content_category: str | None
    relevance_score: float
    text_preview: str


@dataclass
class ContextBlock:
    """Formatted context string with citation metadata.

    Attributes:
        text: Formatted context string with ``[N]`` citation markers.
        citations: List of Citation objects corresponding to each marker.
        total_tokens: Approximate token count of the formatted context.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    total_tokens: int = 0


def dedup_chunks(
    chunks: list[RetrievedChunk],
    *,
    similarity_threshold: float = 0.85,
    max_candidates: int = _RERANKER_MAX_CANDIDATES,
) -> list[RetrievedChunk]:
    """Remove duplicate chunks before reranking.

    Applies two dedup passes:
    1. Exact dedup by ``chunk_id``.
    2. Near-dedup by text similarity (SequenceMatcher ratio > threshold).

    If more than ``max_candidates`` remain after dedup, keeps the top
    candidates by RRF score (to stay within jina-v3's listwise limit).

    Args:
        chunks: Raw candidate chunks from hybrid search.
        similarity_threshold: Text similarity ratio above which two
            chunks are considered near-duplicates.
        max_candidates: Maximum candidates to keep (jina-v3 limit).

    Returns:
        Deduplicated list of RetrievedChunk, preserving original order.
    """
    if not chunks:
        return []

    # Pass 1: exact dedup by chunk_id
    seen_ids: set[str] = set()
    id_deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.chunk_id and chunk.chunk_id in seen_ids:
            continue
        if chunk.chunk_id:
            seen_ids.add(chunk.chunk_id)
        id_deduped.append(chunk)

    logger.debug(
        "Exact dedup: {} -> {} chunks", len(chunks), len(id_deduped)
    )

    # Pass 2: near-dedup by text similarity
    unique: list[RetrievedChunk] = []
    for chunk in id_deduped:
        is_near_dup = False
        for kept in unique:
            ratio = SequenceMatcher(None, chunk.text, kept.text).ratio()
            if ratio > similarity_threshold:
                is_near_dup = True
                break
        if not is_near_dup:
            unique.append(chunk)

    logger.debug(
        "Near-dedup (threshold={}): {} -> {} chunks",
        similarity_threshold,
        len(id_deduped),
        len(unique),
    )

    # Trim to max_candidates by score if still too many
    if len(unique) > max_candidates:
        unique.sort(key=lambda c: c.score, reverse=True)
        unique = unique[:max_candidates]
        logger.debug("Trimmed to {} candidates for reranker", max_candidates)

    return unique


def assemble_context(
    chunks: list[RetrievedChunk],
    *,
    max_tokens: int | None = None,
) -> ContextBlock:
    """Format reranked chunks into a context string with numbered citations.

    Each chunk is prefixed with ``[N]`` and separated by a blank line.
    Chunks are added until the token budget is exhausted.

    Args:
        chunks: Reranked chunks to assemble (already filtered/sorted).
        max_tokens: Maximum token budget for the context string.
            Defaults to ``settings.context_max_tokens``.

    Returns:
        A ContextBlock with the formatted text and citation metadata.
    """
    if max_tokens is None:
        max_tokens = settings.context_max_tokens

    if not chunks:
        return ContextBlock(text="", citations=[], total_tokens=0)

    enc = _get_tiktoken_encoder()
    parts: list[str] = []
    citations: list[Citation] = []
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        citation_idx = i + 1
        formatted = f"[{citation_idx}] {chunk.text}"
        chunk_tokens = len(enc.encode(formatted))

        if total_tokens + chunk_tokens > max_tokens:
            logger.debug(
                "Context budget reached at chunk {} ({}/{} tokens)",
                citation_idx,
                total_tokens,
                max_tokens,
            )
            break

        parts.append(formatted)
        total_tokens += chunk_tokens

        meta = chunk.metadata
        citations.append(
            Citation(
                index=citation_idx,
                source_filename=meta.get("source_filename", ""),
                page_num=meta.get("page_num"),
                section=meta.get("section"),
                chapter=meta.get("chapter"),
                module_week=str(mw) if (mw := meta.get("module_week")) is not None else None,
                collection=chunk.collection,
                content_category=meta.get("content_category"),
                relevance_score=chunk.score,
                text_preview=chunk.text,
            )
        )

    context_text = "\n\n".join(parts)

    logger.info(
        "Assembled context: {} chunks, {} tokens",
        len(citations),
        total_tokens,
    )
    return ContextBlock(
        text=context_text,
        citations=citations,
        total_tokens=total_tokens,
    )
