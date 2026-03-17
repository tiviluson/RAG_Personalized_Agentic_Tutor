"""RAGAS metric configuration and evaluation helpers."""

from __future__ import annotations

import instructor
from loguru import logger
from openai import AsyncOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import InstructorLLM
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from src.config import settings
from src.evaluation.datasets import EvalSample, MultiTurnEvalSample, TurnSample
from src.evaluation.pipeline_wrapper import EvalPipelineResult


def _patch_instructor_generate_multiple() -> None:
    """Patch PydanticPrompt.generate_multiple to support n>1 for InstructorLLM.

    RAGAS's InstructorLLM code path ignores the n parameter and always returns
    1 generation. This patch fans out n>1 requests into n separate calls.
    """
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.prompt import PydanticPrompt

    original = PydanticPrompt.generate_multiple

    async def _patched(self, llm, data, n=1, **kwargs):
        if isinstance(llm, InstructorBaseRagasLLM) and n > 1:
            results = []
            for _ in range(n):
                result = await original(self, llm, data, n=1, **kwargs)
                results.extend(result)
            return results
        return await original(self, llm, data, n=n, **kwargs)

    PydanticPrompt.generate_multiple = _patched
    logger.info("Patched PydanticPrompt.generate_multiple for InstructorLLM n>1 support")


_patch_instructor_generate_multiple()


# User-friendly metric name -> constructor mapping
METRIC_REGISTRY: dict[str, type] = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": LLMContextPrecisionWithReference,
    "context_recall": LLMContextRecall,
}


def get_evaluator_llm():
    """Create a RAGAS-compatible LLM using a local OpenAI-compatible endpoint.

    Connects to a local LLM server (e.g., LM Studio) at the URL configured
    in settings.eval_llm_base_url. Uses JSON_SCHEMA instructor mode for
    structured output compatibility with local servers.

    Returns:
        A RAGAS InstructorLLM instance backed by the local model.
    """
    client = AsyncOpenAI(
        base_url=settings.eval_llm_base_url,
        api_key="lm-studio",
    )
    patched = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
    llm = InstructorLLM(
        model=settings.eval_llm_model,
        client=patched,
        provider="openai",
    )
    llm.model_args["max_tokens"] = 8192

    logger.info("Evaluator LLM initialized: {} at {}", settings.eval_llm_model, settings.eval_llm_base_url)
    return llm


def get_evaluator_embeddings():
    """Create RAGAS-compatible embeddings using nomic-embed-text-v1.5.

    Returns:
        A RAGAS HuggingFaceEmbeddings instance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.eval_embedding_model,
        model_kwargs={"trust_remote_code": True},
    )
    logger.info("Evaluator embeddings initialized: {}", settings.eval_embedding_model)
    return embeddings


def get_metrics(subset: list[str] | None = None) -> list:
    """Create configured RAGAS metric instances.

    Args:
        subset (list[str] | None): Metric names to include. If None, all
            metrics are returned. Valid names: "faithfulness",
            "answer_relevancy", "context_precision", "context_recall".

    Returns:
        list: Configured RAGAS metric instances.

    Raises:
        ValueError: If an unknown metric name is provided.
    """
    evaluator_llm = get_evaluator_llm()

    names = subset if subset else list(METRIC_REGISTRY.keys())

    unknown = set(names) - set(METRIC_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. "
            f"Valid: {list(METRIC_REGISTRY.keys())}"
        )

    # Only load embeddings if answer_relevancy is requested
    evaluator_embeddings = None
    if "answer_relevancy" in names:
        evaluator_embeddings = get_evaluator_embeddings()

    metrics = []
    for name in names:
        cls = METRIC_REGISTRY[name]
        if cls is AnswerRelevancy:
            metrics.append(cls(llm=evaluator_llm, embeddings=evaluator_embeddings))
        else:
            metrics.append(cls(llm=evaluator_llm))

    logger.info("Configured {} metrics: {}", len(metrics), names)
    return metrics


def build_ragas_dataset(
    samples: list[EvalSample | TurnSample],
    results: list[EvalPipelineResult],
) -> EvaluationDataset:
    """Convert evaluation samples and pipeline results into a RAGAS dataset.

    Args:
        samples (list[EvalSample | TurnSample]): Original evaluation samples
            (provides ground_truth).
        results (list[EvalPipelineResult]): Pipeline results (provides
            response and retrieved_contexts).

    Returns:
        EvaluationDataset: RAGAS-compatible dataset for evaluate().

    Raises:
        ValueError: If samples and results have different lengths.
    """
    if len(samples) != len(results):
        raise ValueError(
            f"Mismatch: {len(samples)} samples vs {len(results)} results"
        )

    ragas_samples = []
    for sample, result in zip(samples, results):
        if result.error:
            logger.warning(
                "Skipping errored sample: '{}'", sample.question[:60]
            )
            continue

        ragas_samples.append(
            SingleTurnSample(
                user_input=sample.question,
                response=result.answer,
                retrieved_contexts=result.context_texts,
                reference=sample.ground_truth,
            )
        )

    logger.info(
        "Built RAGAS dataset: {} samples ({} skipped due to errors)",
        len(ragas_samples),
        len(samples) - len(ragas_samples),
    )
    return EvaluationDataset(samples=ragas_samples)


def run_ragas_evaluation(
    dataset: EvaluationDataset,
    metrics: list,
    max_workers: int = settings.eval_max_workers,
    batch_size: int | None = None,
) -> dict:
    """Run RAGAS evaluation and return scores.

    Args:
        dataset (EvaluationDataset): RAGAS dataset with samples.
        metrics (list): Configured RAGAS metric instances.
        max_workers (int): Max concurrent LLM calls. Should match the
            LM Studio max concurrency setting.
        batch_size (int | None): Number of samples per evaluation batch.
            Defaults to None (RAGAS default).

    Returns:
        dict: Evaluation results with per-sample and aggregate scores.
    """
    logger.info(
        "Running RAGAS evaluation: {} samples x {} metrics (max_workers={})",
        len(dataset.samples),
        len(metrics),
        max_workers,
    )

    run_config = RunConfig(
        max_workers=max_workers,
        timeout=300,
        max_retries=3,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        batch_size=batch_size,
        show_progress=True,
    )

    return result.to_pandas().to_dict(orient="records")
