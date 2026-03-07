"""Query pipeline orchestrator: ties all retrieval + generation stages together."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from loguru import logger

from src.api.models.query import ChatEvent, PipelineMetrics
from src.config import settings
from src.db.qdrant import get_qdrant_client
from src.retrieval.context import ContextBlock, assemble_context, dedup_chunks
from src.retrieval.generator import _build_user_message, stream_gemini
from src.retrieval.prompts import ANSWER_MODE_PROMPTS
from src.retrieval.query_processor import PreprocessedQuery, preprocess_query
from src.retrieval.reranker import rerank
from src.retrieval.search import RetrievedChunk, hybrid_search
from src.retrieval.session import (
    add_turn,
    get_recent_history,
    get_session,
)
from src.api.models.query import Citation


@dataclass
class PipelineResult:
    """Intermediate results exposed for Phase 3 evaluation.

    Attributes:
        preprocessed: Query preprocessing output.
        raw_candidates: All candidates before dedup/reranking.
        deduped_candidates: Candidates after dedup, before reranking.
        reranked_candidates: Final candidates after reranking.
        context_block: Assembled context sent to the generator.
        metrics: Pipeline timing and count metrics.
    """

    preprocessed: PreprocessedQuery | None = None
    raw_candidates: list[RetrievedChunk] = field(default_factory=list)
    deduped_candidates: list[RetrievedChunk] = field(default_factory=list)
    reranked_candidates: list[RetrievedChunk] = field(default_factory=list)
    context_block: ContextBlock | None = None
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


def _run_preprocessing(
    query: str,
    history: list[dict],
) -> tuple[PreprocessedQuery, float]:
    """Preprocess the query synchronously (runs in a thread).

    Args:
        query: Raw student query.
        history: Recent conversation history.

    Returns:
        Tuple of (preprocessed_query, elapsed_ms).
    """
    t0 = time.perf_counter()
    preprocessed = preprocess_query(query, history)
    return preprocessed, (time.perf_counter() - t0) * 1000


def _run_search_and_dedup(
    preprocessed: PreprocessedQuery,
    student_id: str,
    filters: dict | None,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk], float]:
    """Run hybrid search and dedup synchronously (runs in a thread).

    Args:
        preprocessed: Output from query preprocessing.
        student_id: Student ID for scoping student_notes.
        filters: Optional metadata filters.

    Returns:
        Tuple of (raw_candidates, deduped_candidates, retrieval_ms).
    """
    t0 = time.perf_counter()
    qdrant_client = get_qdrant_client()
    search_queries = [preprocessed.rewritten_query] + preprocessed.expansion_queries
    all_candidates: list[RetrievedChunk] = []

    for sq in search_queries:
        candidates = hybrid_search(
            qdrant_client,
            sq,
            student_id=student_id,
            filters=filters,
            k_per_collection=settings.retrieval_k_per_collection,
        )
        all_candidates.extend(candidates)

    logger.info(
        "Retrieved {} raw candidates from {} search queries",
        len(all_candidates),
        len(search_queries),
    )

    deduped = dedup_chunks(all_candidates)
    return all_candidates, deduped, (time.perf_counter() - t0) * 1000


def _run_rerank_and_assemble(
    rewritten_query: str,
    deduped: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], ContextBlock, float, float]:
    """Rerank candidates and assemble context synchronously (runs in a thread).

    Args:
        rewritten_query: Preprocessed rewritten query for reranker scoring.
        deduped: Deduplicated candidate chunks.

    Returns:
        Tuple of (reranked, context_block, reranking_ms, context_assembly_ms).
    """
    t_r = time.perf_counter()
    if deduped:
        reranked = rerank(
            rewritten_query,
            deduped,
            max_results=settings.reranker_max_results,
            min_results=settings.reranker_min_results,
            min_score=settings.reranker_min_score,
        )
    else:
        reranked = []
    reranking_ms = (time.perf_counter() - t_r) * 1000

    t_c = time.perf_counter()
    context_block = assemble_context(reranked)
    context_assembly_ms = (time.perf_counter() - t_c) * 1000

    return reranked, context_block, reranking_ms, context_assembly_ms


async def run_pipeline(
    session_id: str,
    query: str,
    answer_mode: str,
    filters: dict | None = None,
) -> AsyncGenerator[ChatEvent, None]:
    """Execute the full query pipeline and stream the response.

    Blocking stages (preprocessing, retrieval, reranking, generation)
    are offloaded to a thread so the async event loop is not blocked.

    Args:
        session_id: Active session identifier.
        query: The student's raw query.
        answer_mode: One of "long", "short", "eli5".
        filters: Optional metadata filters dict.

    Yields:
        ChatEvent objects for SSE streaming.
    """
    # 1. Session + history
    session = get_session(session_id)
    if session is None:
        yield ChatEvent(type="error", data="Session not found.")
        return

    history = get_recent_history(session_id)
    metrics = PipelineMetrics()

    # Stage 1: Preprocess query
    yield ChatEvent(type="status", data="Thinking...")
    preprocessed, preprocessing_ms = await asyncio.to_thread(
        _run_preprocessing, query, history
    )
    metrics.preprocessing_ms = preprocessing_ms
    metrics.strategy_used = preprocessed.strategy

    # Out-of-scope short-circuit
    if preprocessed.is_out_of_scope:
        refusal = preprocessed.refusal_message or (
            "I can only help with questions related to this course."
        )
        yield ChatEvent(type="chunk", data=refusal)
        add_turn(session_id, "user", query)
        add_turn(session_id, "assistant", refusal)
        yield ChatEvent(
            type="done",
            data={"citations": [], "metrics": metrics.model_dump()},
        )
        return

    # Stage 2: Hybrid search + dedup
    yield ChatEvent(type="status", data="Searching...")
    all_candidates, deduped, retrieval_ms = await asyncio.to_thread(
        _run_search_and_dedup, preprocessed, session.student_id, filters
    )
    metrics.total_candidates = len(all_candidates)
    metrics.deduped_candidates = len(deduped)
    metrics.retrieval_ms = retrieval_ms

    # Stage 3: Rerank + context assembly
    yield ChatEvent(type="status", data="Reranking...")
    reranked, context_block, reranking_ms, context_assembly_ms = await asyncio.to_thread(
        _run_rerank_and_assemble, preprocessed.rewritten_query, deduped
    )
    metrics.reranking_ms = reranking_ms
    metrics.context_assembly_ms = context_assembly_ms
    metrics.final_candidates = len(reranked)

    # Stage 4: Generate streamed response
    system_prompt = ANSWER_MODE_PROMPTS.get(answer_mode, ANSWER_MODE_PROMPTS["long"])
    user_message = _build_user_message(
        preprocessed.rewritten_query, context_block, history
    )

    fragments: list[str] = []
    try:
        async for text in stream_gemini(system_prompt, user_message):
            fragments.append(text)
            yield ChatEvent(type="chunk", data=text)
    except Exception as e:
        logger.error("Generation failed after retries: {}", e)
        yield ChatEvent(type="error", data=str(e))
        return

    # Done event: citations + metrics
    citations = [
        Citation(
            index=c.index,
            source_filename=c.source_filename,
            page_num=c.page_num,
            section=c.section,
            chapter=c.chapter,
            module_week=c.module_week,
            collection=c.collection,
            content_category=c.content_category,
            relevance_score=c.relevance_score,
            text_preview=c.text_preview,
        )
        for c in context_block.citations
    ]

    done_data = {
        "citations": [c.model_dump() for c in citations],
        "metrics": metrics.model_dump(),
    }
    yield ChatEvent(type="done", data=json.dumps(done_data))

    # 9. Update session history
    full_response = "".join(fragments)
    citations_dicts = [
        {"index": c.index, "source_filename": c.source_filename}
        for c in context_block.citations
    ]
    add_turn(session_id, "user", query)
    add_turn(session_id, "assistant", full_response, citations=citations_dicts)
