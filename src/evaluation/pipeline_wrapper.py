"""Non-streaming pipeline wrapper for batch evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from google.genai import types
from loguru import logger

from src.api.models.query import PipelineMetrics
from src.clients import get_genai_client
from src.config import settings
from src.evaluation.datasets import EvalSample, MultiTurnEvalSample
from src.retrieval.context import ContextBlock
from src.retrieval.pipeline import (
    PipelineResult,
    _run_preprocessing,
    _run_rerank_and_assemble,
    _run_search_and_dedup,
)
from src.retrieval.generator import _build_user_message
from src.retrieval.prompts import ANSWER_MODE_PROMPTS
from src.retrieval.search import RetrievedChunk


@dataclass
class EvalPipelineResult:
    """Result of running the RAG pipeline on a single evaluation question.

    Attributes:
        question: The original question.
        answer: Full generated answer text.
        context_texts: Plain text of each retrieved chunk (for RAGAS).
        total_latency_ms: End-to-end wall time in milliseconds.
        generation_ms: Generation stage time in milliseconds.
        pipeline_result: Intermediate pipeline data (candidates, metrics).
        error: Error message if the pipeline failed for this sample.
    """

    question: str = ""
    answer: str = ""
    context_texts: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    generation_ms: float = 0.0
    pipeline_result: PipelineResult = field(default_factory=PipelineResult)
    error: str | None = None


async def generate_full(system_prompt: str, user_message: str) -> tuple[str, float]:
    """Generate a complete (non-streaming) response via Gemini.

    Args:
        system_prompt (str): System instruction for the answer mode.
        user_message (str): Formatted user message with context.

    Returns:
        tuple[str, float]: (full_response_text, generation_ms).

    Raises:
        Exception: Propagated after 2 retry attempts.
    """
    import asyncio

    client = get_genai_client(required=True)
    last_exc: Exception | None = None

    for attempt in range(2):
        try:
            t0 = time.perf_counter()
            response = await asyncio.wait_for(
                client.aio.models.generate_content(  # type: ignore[union-attr]
                    model=settings.gemini_generation_model,
                    contents=[user_message],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.3,
                    ),
                ),
                timeout=60,
            )
            gen_ms = (time.perf_counter() - t0) * 1000
            return response.text or "", gen_ms
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                await asyncio.sleep(2)

    raise last_exc  # type: ignore[misc]


async def run_single(
    question: str,
    history: list[dict] | None = None,
    student_id: str = "eval",
    filters: dict | None = None,
    answer_mode: str = "long",
) -> EvalPipelineResult:
    """Run the full RAG pipeline on a single question without streaming.

    Args:
        question (str): The student question to evaluate.
        history (list[dict] | None): Conversation history for multi-turn.
            Defaults to empty list.
        student_id (str): Student ID for retrieval scoping.
        filters (dict | None): Optional metadata filters.
        answer_mode (str): One of "long", "short", "eli5".

    Returns:
        EvalPipelineResult: Pipeline output with answer, contexts, and timing.
    """
    if history is None:
        history = []

    t_total = time.perf_counter()
    result = EvalPipelineResult(question=question)
    pr = PipelineResult()
    metrics = PipelineMetrics()

    try:
        # Stage 1: Preprocess
        preprocessed, preprocessing_ms = _run_preprocessing(question, history)
        metrics.preprocessing_ms = preprocessing_ms
        metrics.strategy_used = preprocessed.strategy
        pr.preprocessed = preprocessed

        # Short-circuit out-of-scope
        if preprocessed.is_out_of_scope:
            result.answer = preprocessed.refusal_message or (
                "I can only help with questions related to this course."
            )
            result.total_latency_ms = (time.perf_counter() - t_total) * 1000
            pr.metrics = metrics
            result.pipeline_result = pr
            return result

        # Stage 2: Search + dedup
        all_candidates, deduped, retrieval_ms = _run_search_and_dedup(
            preprocessed, student_id, filters
        )
        metrics.total_candidates = len(all_candidates)
        metrics.deduped_candidates = len(deduped)
        metrics.retrieval_ms = retrieval_ms
        pr.raw_candidates = all_candidates
        pr.deduped_candidates = deduped

        # Stage 3: Rerank + context assembly
        reranked, context_block, reranking_ms, context_assembly_ms = (
            _run_rerank_and_assemble(preprocessed.rewritten_query, deduped)
        )
        metrics.reranking_ms = reranking_ms
        metrics.context_assembly_ms = context_assembly_ms
        metrics.final_candidates = len(reranked)
        pr.reranked_candidates = reranked
        pr.context_block = context_block

        # Extract context texts for RAGAS
        result.context_texts = [c.text_preview for c in context_block.citations]

        # Stage 4: Generate (non-streaming)
        system_prompt = ANSWER_MODE_PROMPTS.get(
            answer_mode, ANSWER_MODE_PROMPTS["long"]
        )
        user_message = _build_user_message(
            preprocessed.rewritten_query, context_block, history
        )
        answer, gen_ms = await generate_full(system_prompt, user_message)
        result.answer = answer
        result.generation_ms = gen_ms

    except Exception as e:
        logger.error("Pipeline failed for question '{}': {}", question[:80], e)
        result.error = str(e)

    result.total_latency_ms = (time.perf_counter() - t_total) * 1000
    pr.metrics = metrics
    result.pipeline_result = pr
    return result


async def run_multi_turn(
    scenario: MultiTurnEvalSample,
    student_id: str = "eval",
    filters: dict | None = None,
    answer_mode: str = "long",
) -> list[EvalPipelineResult]:
    """Run a multi-turn evaluation scenario, evaluating every turn.

    Runs each turn sequentially with accumulated conversation history
    from previous turns.

    Args:
        scenario (MultiTurnEvalSample): Multi-turn scenario with 2-3 turns.
        student_id (str): Student ID for retrieval scoping.
        filters (dict | None): Optional metadata filters.
        answer_mode (str): One of "long", "short", "eli5".

    Returns:
        list[EvalPipelineResult]: One result per turn.
    """
    results: list[EvalPipelineResult] = []
    history: list[dict] = []

    for i, turn in enumerate(scenario.turns):
        logger.debug(
            "Running turn {}/{} for multi-turn scenario",
            i + 1,
            len(scenario.turns),
        )
        result = await run_single(
            question=turn.question,
            history=history,
            student_id=student_id,
            filters=filters,
            answer_mode=answer_mode,
        )
        results.append(result)

        # Accumulate history for next turn
        history.append({"role": "user", "content": turn.question})
        history.append({"role": "assistant", "content": result.answer})

    return results


async def run_batch(
    samples: list[EvalSample],
    student_id: str = "eval",
    filters: dict | None = None,
    answer_mode: str = "long",
) -> list[EvalPipelineResult]:
    """Run the pipeline on a batch of single-turn evaluation samples.

    Executes sequentially to avoid overwhelming Qdrant/Gemini.

    Args:
        samples (list[EvalSample]): Evaluation samples to process.
        student_id (str): Student ID for retrieval scoping.
        filters (dict | None): Optional metadata filters.
        answer_mode (str): One of "long", "short", "eli5".

    Returns:
        list[EvalPipelineResult]: Results in the same order as input samples.
    """
    results: list[EvalPipelineResult] = []

    for i, sample in enumerate(samples):
        if (i + 1) % 5 == 0 or i == 0:
            logger.info("Processing sample {}/{}", i + 1, len(samples))

        result = await run_single(
            question=sample.question,
            student_id=student_id,
            filters=filters,
            answer_mode=answer_mode,
        )
        results.append(result)

    logger.info("Batch complete: {}/{} succeeded", sum(1 for r in results if not r.error), len(results))
    return results


async def run_batch_multi_turn(
    scenarios: list[MultiTurnEvalSample],
    student_id: str = "eval",
    filters: dict | None = None,
    answer_mode: str = "long",
) -> list[list[EvalPipelineResult]]:
    """Run the pipeline on a batch of multi-turn evaluation scenarios.

    Args:
        scenarios (list[MultiTurnEvalSample]): Scenarios to process.
        student_id (str): Student ID for retrieval scoping.
        filters (dict | None): Optional metadata filters.
        answer_mode (str): One of "long", "short", "eli5".

    Returns:
        list[list[EvalPipelineResult]]: One inner list per scenario,
            one result per turn.
    """
    all_results: list[list[EvalPipelineResult]] = []

    for i, scenario in enumerate(scenarios):
        logger.info(
            "Processing multi-turn scenario {}/{} ({} turns)",
            i + 1,
            len(scenarios),
            len(scenario.turns),
        )
        results = await run_multi_turn(
            scenario=scenario,
            student_id=student_id,
            filters=filters,
            answer_mode=answer_mode,
        )
        all_results.append(results)

    total_turns = sum(len(r) for r in all_results)
    total_ok = sum(1 for r_list in all_results for r in r_list if not r.error)
    logger.info(
        "Multi-turn batch complete: {}/{} turns succeeded across {} scenarios",
        total_ok,
        total_turns,
        len(scenarios),
    )
    return all_results
