"""Evaluation runner: CLI entry point for batch RAG evaluation.

Two-stage design:
  Stage 1 (pipeline): Run retrieval + generation, save intermediate results.
  Stage 2 (scoring):  Load intermediate results, run RAGAS + accuracy scoring.

Usage:
    # Full run (both stages):
    python -m src.evaluation.runner \
        --dataset data/evaluation/rag_course_eval.jsonl \
        --smoke-test --metrics faithfulness

    # Pipeline only (save intermediate results):
    python -m src.evaluation.runner \
        --dataset data/evaluation/rag_course_eval.jsonl \
        --pipeline-only

    # Scoring only (reuse pipeline results):
    python -m src.evaluation.runner \
        --eval-only reports/pipeline_results_2026-03-15T01-30-00.json \
        --metrics faithfulness,answer_relevancy
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from src.config import settings
from src.evaluation.datasets import (
    EvalSample,
    MultiTurnEvalSample,
    TurnSample,
    load_dataset,
    load_multi_turn_dataset,
)
from src.evaluation.metrics import (
    build_ragas_dataset,
    get_metrics,
    run_ragas_evaluation,
)
from src.evaluation.pipeline_wrapper import (
    EvalPipelineResult,
    run_batch,
    run_batch_multi_turn,
)
from src.evaluation.prompts import OUTPUT_ACCURACY_PROMPT
from src.evaluation.report import (
    EvalReport,
    build_report,
    print_summary,
    save_report,
)


# ---------------------------------------------------------------------------
# Intermediate results: save / load
# ---------------------------------------------------------------------------

def _serialize_result(
    sample: EvalSample | TurnSample,
    result: EvalPipelineResult,
) -> dict:
    """Serialize one sample + pipeline result pair to a JSON-safe dict.

    Args:
        sample: The evaluation sample (provides ground_truth).
        result: The pipeline result (provides answer, contexts, timing).

    Returns:
        dict: JSON-serializable representation.
    """
    metrics = result.pipeline_result.metrics
    entry = {
        "question": sample.question,
        "ground_truth": sample.ground_truth,
        "answer": result.answer,
        "context_texts": result.context_texts,
        "total_latency_ms": result.total_latency_ms,
        "generation_ms": result.generation_ms,
        "error": result.error,
        "pipeline_metrics": metrics.model_dump(),
    }
    if isinstance(sample, EvalSample):
        entry["difficulty"] = sample.difficulty
        entry["category"] = sample.category
    return entry


def _deserialize_result(entry: dict) -> tuple[TurnSample, EvalPipelineResult]:
    """Reconstruct a TurnSample and EvalPipelineResult from a serialized dict.

    Args:
        entry: Dict from the intermediate JSON file.

    Returns:
        tuple: (TurnSample, EvalPipelineResult) pair.
    """
    from src.api.models.query import PipelineMetrics
    from src.retrieval.pipeline import PipelineResult

    sample = TurnSample(
        question=entry["question"],
        ground_truth=entry["ground_truth"],
    )
    pr = PipelineResult()
    pr.metrics = PipelineMetrics(**entry.get("pipeline_metrics", {}))

    result = EvalPipelineResult(
        question=entry["question"],
        answer=entry.get("answer", ""),
        context_texts=entry.get("context_texts", []),
        total_latency_ms=entry.get("total_latency_ms", 0.0),
        generation_ms=entry.get("generation_ms", 0.0),
        pipeline_result=pr,
        error=entry.get("error"),
    )
    return sample, result


def save_pipeline_results(
    single_turn_samples: list[EvalSample],
    single_turn_results: list[EvalPipelineResult],
    multi_turn_scenarios: list[MultiTurnEvalSample],
    multi_turn_results: list[list[EvalPipelineResult]],
    dataset_names: list[str],
    answer_mode: str,
    output_dir: Path,
) -> Path:
    """Save pipeline results to a JSON file for later scoring.

    Args:
        single_turn_samples: Single-turn evaluation samples.
        single_turn_results: Pipeline results for single-turn samples.
        multi_turn_scenarios: Multi-turn evaluation scenarios.
        multi_turn_results: Pipeline results per scenario per turn.
        dataset_names: Paths to the evaluation dataset files.
        answer_mode: Answer mode used for generation.
        output_dir: Directory to save the results file.

    Returns:
        Path: Path to the saved intermediate results file.
    """
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_paths": dataset_names,
            "answer_mode": answer_mode,
            "num_single_turn": len(single_turn_samples),
            "num_multi_turn_scenarios": len(multi_turn_scenarios),
        },
        "single_turn": [
            _serialize_result(s, r)
            for s, r in zip(single_turn_samples, single_turn_results)
        ],
        "multi_turn": [
            {
                "scenario_index": i,
                "difficulty": scenario.difficulty,
                "turns": [
                    _serialize_result(turn, result)
                    for turn, result in zip(scenario.turns, scenario_results)
                ],
            }
            for i, (scenario, scenario_results)
            in enumerate(zip(multi_turn_scenarios, multi_turn_results))
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filepath = output_dir / f"pipeline_results_{timestamp}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Pipeline results saved to {}", filepath)
    return filepath


def load_pipeline_results(
    path: Path,
) -> tuple[
    list[TurnSample],
    list[EvalPipelineResult],
    list[dict],
    list[list[EvalPipelineResult]],
    list[str],
    int,
]:
    """Load pipeline results from an intermediate JSON file.

    Args:
        path: Path to the intermediate results file.

    Returns:
        tuple: (combined_samples, combined_results,
                multi_turn_scenario_info, multi_turn_results,
                dataset_names, num_single_turn)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    dataset_names = metadata["dataset_paths"]
    num_single_turn = metadata["num_single_turn"]

    combined_samples: list[TurnSample] = []
    combined_results: list[EvalPipelineResult] = []

    # Single-turn
    for entry in data["single_turn"]:
        sample, result = _deserialize_result(entry)
        combined_samples.append(sample)
        combined_results.append(result)

    # Multi-turn (flatten for RAGAS, keep structure for report)
    mt_scenario_info: list[dict] = []
    mt_results: list[list[EvalPipelineResult]] = []
    for scenario_data in data.get("multi_turn", []):
        scenario_results: list[EvalPipelineResult] = []
        for turn_entry in scenario_data["turns"]:
            sample, result = _deserialize_result(turn_entry)
            combined_samples.append(sample)
            combined_results.append(result)
            scenario_results.append(result)
        mt_results.append(scenario_results)
        mt_scenario_info.append({
            "scenario_index": scenario_data["scenario_index"],
            "num_turns": len(scenario_data["turns"]),
        })

    logger.info(
        "Loaded pipeline results: {} single-turn + {} multi-turn scenarios from {}",
        num_single_turn,
        len(mt_results),
        path.name,
    )
    return (
        combined_samples,
        combined_results,
        mt_scenario_info,
        mt_results,
        dataset_names,
        num_single_turn,
    )


# ---------------------------------------------------------------------------
# Output accuracy (LLM judge)
# ---------------------------------------------------------------------------

async def _run_output_accuracy(
    samples: list[EvalSample | TurnSample],
    results: list[EvalPipelineResult],
) -> list[dict]:
    """Score output accuracy using the local evaluator LLM.

    Compares each predicted answer against its ground truth reference.

    Args:
        samples: Evaluation samples with ground_truth.
        results: Pipeline results with generated answers.

    Returns:
        list[dict]: Per-sample accuracy scores with reasoning.
    """
    client = AsyncOpenAI(
        base_url=settings.eval_llm_base_url,
        api_key="lm-studio",
    )
    scores: list[dict] = []

    for sample, result in zip(samples, results):
        if result.error:
            scores.append({"output_accuracy": None, "accuracy_reasoning": None})
            continue

        prompt = OUTPUT_ACCURACY_PROMPT.format(
            ground_truth=sample.ground_truth,
            predicted_answer=result.answer,
        )
        try:
            response = await client.chat.completions.create(
                model=settings.eval_llm_model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = (response.choices[0].message.content or "").strip()
            lines = text.split("\n", 1)
            judgment = lines[0].strip().upper()
            reasoning = lines[1].strip() if len(lines) > 1 else ""

            scores.append({
                "output_accuracy": 1.0 if judgment == "CORRECT" else 0.0,
                "accuracy_reasoning": reasoning,
            })
        except Exception as e:
            logger.warning("Output accuracy failed for '{}': {}", sample.question[:60], e)
            scores.append({"output_accuracy": None, "accuracy_reasoning": str(e)})

    return scores


# ---------------------------------------------------------------------------
# Stage 1: Pipeline (retrieval + generation)
# ---------------------------------------------------------------------------

async def run_pipeline_stage(
    dataset_paths: list[Path],
    multi_turn_path: Path | None = None,
    output_dir: Path = Path("reports"),
    smoke_test: bool = False,
    answer_mode: str = "long",
) -> Path:
    """Stage 1: Run retrieval + generation pipeline and save results.

    Args:
        dataset_paths: Paths to single-turn JSONL files.
        multi_turn_path: Path to multi-turn JSONL file.
        output_dir: Directory for saving the intermediate results.
        smoke_test: If True, run on first N samples only.
        answer_mode: Answer mode for generation.

    Returns:
        Path: Path to the saved intermediate results file.
    """
    # Load datasets
    all_samples: list[EvalSample] = []
    dataset_names: list[str] = []
    for path in dataset_paths:
        samples = load_dataset(path)
        all_samples.extend(samples)
        dataset_names.append(str(path))

    multi_turn_scenarios: list[MultiTurnEvalSample] = []
    if multi_turn_path:
        multi_turn_scenarios = load_multi_turn_dataset(multi_turn_path)
        dataset_names.append(str(multi_turn_path))

    logger.info(
        "Loaded {} single-turn samples + {} multi-turn scenarios",
        len(all_samples),
        len(multi_turn_scenarios),
    )

    # Apply smoke test slicing
    if smoke_test:
        n = settings.eval_smoke_test_samples
        all_samples = all_samples[:n]
        multi_turn_scenarios = multi_turn_scenarios[:1] if multi_turn_scenarios else []
        logger.info(
            "Smoke test: sliced to {} samples + {} scenarios",
            len(all_samples),
            len(multi_turn_scenarios),
        )

    # Run pipeline on single-turn samples
    logger.info("Running pipeline on {} single-turn samples...", len(all_samples))
    st_results = await run_batch(all_samples, answer_mode=answer_mode)

    # Run pipeline on multi-turn scenarios
    mt_results: list[list[EvalPipelineResult]] = []
    if multi_turn_scenarios:
        logger.info(
            "Running pipeline on {} multi-turn scenarios...",
            len(multi_turn_scenarios),
        )
        mt_results = await run_batch_multi_turn(
            multi_turn_scenarios, answer_mode=answer_mode
        )

    # Save intermediate results
    filepath = save_pipeline_results(
        single_turn_samples=all_samples,
        single_turn_results=st_results,
        multi_turn_scenarios=multi_turn_scenarios,
        multi_turn_results=mt_results,
        dataset_names=dataset_names,
        answer_mode=answer_mode,
        output_dir=output_dir,
    )

    ok = sum(1 for r in st_results if not r.error)
    mt_ok = sum(1 for rs in mt_results for r in rs if not r.error)
    mt_total = sum(len(rs) for rs in mt_results)
    logger.info(
        "Pipeline stage complete: {}/{} single-turn + {}/{} multi-turn succeeded",
        ok, len(st_results), mt_ok, mt_total,
    )
    return filepath


# ---------------------------------------------------------------------------
# Stage 2: Scoring (RAGAS + output accuracy)
# ---------------------------------------------------------------------------

async def run_scoring_stage(
    pipeline_results_path: Path,
    output_dir: Path = Path("reports"),
    metrics_subset: list[str] | None = None,
) -> EvalReport:
    """Stage 2: Run RAGAS + accuracy scoring on pre-computed pipeline results.

    Args:
        pipeline_results_path: Path to intermediate results JSON file.
        output_dir: Directory for saving the final report.
        metrics_subset: Subset of metrics to run.

    Returns:
        EvalReport: The complete evaluation report.
    """
    # Load pipeline results
    (
        combined_samples,
        combined_results,
        mt_scenario_info,
        mt_results,
        dataset_names,
        num_single_turn,
    ) = load_pipeline_results(pipeline_results_path)

    # Build RAGAS dataset and run evaluation
    logger.info("Building RAGAS dataset...")
    ragas_dataset = build_ragas_dataset(combined_samples, combined_results)

    metric_names = metrics_subset or [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    logger.info("Configuring metrics: {}", metric_names)
    metrics = get_metrics(metric_names)

    logger.info("Running RAGAS evaluation...")
    ragas_scores = run_ragas_evaluation(ragas_dataset, metrics)

    # Run output accuracy
    logger.info("Running output accuracy scoring...")
    accuracy_scores = await _run_output_accuracy(combined_samples, combined_results)

    # Merge scores
    per_sample_scores: list[dict] = []
    for i, ragas_score in enumerate(ragas_scores):
        merged = {
            "question": combined_samples[i].question,
            **{
                k: v
                for k, v in ragas_score.items()
                if k not in ("user_input", "response", "retrieved_contexts", "reference")
            },
        }
        if i < len(accuracy_scores):
            merged.update(accuracy_scores[i])
        if combined_results[i].error:
            merged["error"] = combined_results[i].error
        per_sample_scores.append(merged)

    # Multi-turn specific scores (grouped by scenario)
    mt_report_scores = None
    if mt_scenario_info:
        mt_report_scores = []
        offset = num_single_turn
        for info in mt_scenario_info:
            scenario_scores = []
            for turn_idx in range(info["num_turns"]):
                flat_idx = offset + turn_idx
                if flat_idx < len(per_sample_scores):
                    scenario_scores.append(per_sample_scores[flat_idx])
            mt_report_scores.append({
                "scenario_index": info["scenario_index"],
                "num_turns": info["num_turns"],
                "turn_scores": scenario_scores,
            })
            offset += info["num_turns"]

    # Build and save report
    report = build_report(
        per_sample_scores=per_sample_scores,
        pipeline_results=combined_results,
        dataset_path=", ".join(dataset_names),
        metrics_used=metric_names,
        multi_turn_scores=mt_report_scores,
        multi_turn_results=mt_results,
    )

    save_report(report, output_dir)
    print_summary(report)

    return report


# ---------------------------------------------------------------------------
# Combined convenience (runs both stages)
# ---------------------------------------------------------------------------

def _cleanup_pipeline_models() -> None:
    """Release pipeline models (embedding + reranker) to free memory."""
    import gc

    import torch

    from src.ingestion.embedders.dense import unload_model as unload_dense
    from src.retrieval.reranker import unload_model as unload_reranker

    unload_dense()
    unload_reranker()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    logger.info("Pipeline models released")


async def run_evaluation(
    dataset_paths: list[Path],
    multi_turn_path: Path | None = None,
    output_dir: Path = Path("reports"),
    metrics_subset: list[str] | None = None,
    smoke_test: bool = False,
    answer_mode: str = "long",
) -> EvalReport:
    """Run both pipeline and scoring stages end-to-end.

    Args:
        dataset_paths: Paths to single-turn JSONL files.
        multi_turn_path: Path to multi-turn JSONL file.
        output_dir: Directory for saving reports.
        metrics_subset: Subset of metrics to run.
        smoke_test: If True, run on first N samples only.
        answer_mode: Answer mode for generation.

    Returns:
        EvalReport: The complete evaluation report.
    """
    pipeline_path = await run_pipeline_stage(
        dataset_paths=dataset_paths,
        multi_turn_path=multi_turn_path,
        output_dir=output_dir,
        smoke_test=smoke_test,
        answer_mode=answer_mode,
    )

    _cleanup_pipeline_models()

    return await run_scoring_stage(
        pipeline_results_path=pipeline_path,
        output_dir=output_dir,
        metrics_subset=metrics_subset,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation pipeline",
        prog="python -m src.evaluation.runner",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="Path to single-turn JSONL dataset (can be passed multiple times)",
    )
    parser.add_argument(
        "--multi-turn",
        type=Path,
        default=None,
        help="Path to multi-turn JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for saving reports (default: reports/)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric names (default: all)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Run on first {settings.eval_smoke_test_samples} samples only",
    )
    parser.add_argument(
        "--answer-mode",
        type=str,
        default="long",
        choices=["long", "short", "eli5"],
        help="Answer mode for generation (default: long)",
    )

    # Stage selection (mutually exclusive)
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Run pipeline (retrieval + generation) only; save intermediate results",
    )
    stage_group.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        metavar="RESULTS_FILE",
        help="Run scoring only on a pre-computed pipeline results file",
    )

    args = parser.parse_args()

    metrics_subset = None
    if args.metrics:
        metrics_subset = [m.strip() for m in args.metrics.split(",")]

    if args.eval_only:
        # Stage 2 only: scoring
        if not args.eval_only.exists():
            parser.error(f"Results file not found: {args.eval_only}")
        asyncio.run(
            run_scoring_stage(
                pipeline_results_path=args.eval_only,
                output_dir=args.output_dir,
                metrics_subset=metrics_subset,
            )
        )
    elif args.pipeline_only:
        # Stage 1 only: pipeline
        if not args.dataset:
            parser.error("--dataset is required for --pipeline-only")
        asyncio.run(
            run_pipeline_stage(
                dataset_paths=args.dataset,
                multi_turn_path=args.multi_turn,
                output_dir=args.output_dir,
                smoke_test=args.smoke_test,
                answer_mode=args.answer_mode,
            )
        )
    else:
        # Both stages
        if not args.dataset:
            parser.error("--dataset is required")
        asyncio.run(
            run_evaluation(
                dataset_paths=args.dataset,
                multi_turn_path=args.multi_turn,
                output_dir=args.output_dir,
                metrics_subset=metrics_subset,
                smoke_test=args.smoke_test,
                answer_mode=args.answer_mode,
            )
        )


if __name__ == "__main__":
    main()
