"""Evaluation report generation and console summary."""

from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config import settings


@dataclass
class EvalReport:
    """Complete evaluation report with metadata, scores, and latency.

    Attributes:
        metadata: Run metadata (timestamp, git commit, etc.).
        per_sample_scores: List of per-sample score dicts.
        aggregate_scores: Mean score per metric.
        latency_summary: Latency statistics (mean, median, p95).
        multi_turn_scores: Per-scenario multi-turn scores (optional).
    """

    metadata: dict = field(default_factory=dict)
    per_sample_scores: list[dict] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    latency_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    multi_turn_scores: list[dict] | None = None


def _get_git_commit() -> str:
    """Get the current git commit hash.

    Returns:
        str: Short git commit hash, or "unknown" on failure.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _compute_latency_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, median, and p95 for a list of latency values.

    Args:
        values (list[float]): Latency values in milliseconds.

    Returns:
        dict[str, float]: Stats dict with mean, median, p95 keys.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}

    sorted_vals = sorted(values)
    p95_idx = int(len(sorted_vals) * 0.95)
    return {
        "mean": statistics.mean(sorted_vals),
        "median": statistics.median(sorted_vals),
        "p95": sorted_vals[min(p95_idx, len(sorted_vals) - 1)],
    }


def build_report(
    per_sample_scores: list[dict],
    pipeline_results: list,
    dataset_path: str,
    metrics_used: list[str],
    multi_turn_scores: list[dict] | None = None,
    multi_turn_results: list[list] | None = None,
) -> EvalReport:
    """Build a complete evaluation report from scores and pipeline results.

    Args:
        per_sample_scores: RAGAS per-sample score dicts.
        pipeline_results: List of EvalPipelineResult objects.
        dataset_path: Path(s) to the evaluation dataset file(s).
        metrics_used: Names of metrics that were computed.
        multi_turn_scores: Optional per-scenario multi-turn scores.
        multi_turn_results: Optional multi-turn EvalPipelineResult lists.

    Returns:
        EvalReport: Complete report ready for saving.
    """
    # Metadata
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "dataset_path": dataset_path,
        "evaluator_model": settings.eval_llm_model,
        "metrics_used": metrics_used,
        "total_samples": len(per_sample_scores),
    }

    # Aggregate scores (mean per metric)
    aggregate_scores: dict[str, float] = {}
    for metric_name in metrics_used:
        values = [
            s.get(metric_name, 0.0)
            for s in per_sample_scores
            if s.get(metric_name) is not None
        ]
        if values:
            aggregate_scores[metric_name] = statistics.mean(values)

    # Include output_accuracy if present
    accuracy_values = [
        s.get("output_accuracy", 0.0)
        for s in per_sample_scores
        if s.get("output_accuracy") is not None
    ]
    if accuracy_values:
        aggregate_scores["output_accuracy"] = statistics.mean(accuracy_values)

    # Latency summary from pipeline results
    all_results = list(pipeline_results)
    if multi_turn_results:
        for scenario_results in multi_turn_results:
            all_results.extend(scenario_results)

    total_latencies = [r.total_latency_ms for r in all_results if not r.error]
    gen_latencies = [r.generation_ms for r in all_results if not r.error]
    preprocessing_latencies = [
        r.pipeline_result.metrics.preprocessing_ms
        for r in all_results
        if not r.error
    ]
    retrieval_latencies = [
        r.pipeline_result.metrics.retrieval_ms
        for r in all_results
        if not r.error
    ]
    reranking_latencies = [
        r.pipeline_result.metrics.reranking_ms
        for r in all_results
        if not r.error
    ]

    latency_summary = {
        "total_ms": _compute_latency_stats(total_latencies),
        "generation_ms": _compute_latency_stats(gen_latencies),
        "preprocessing_ms": _compute_latency_stats(preprocessing_latencies),
        "retrieval_ms": _compute_latency_stats(retrieval_latencies),
        "reranking_ms": _compute_latency_stats(reranking_latencies),
    }

    # Enrich per-sample scores with latency
    for i, scores in enumerate(per_sample_scores):
        if i < len(pipeline_results):
            r = pipeline_results[i]
            scores["total_latency_ms"] = r.total_latency_ms
            scores["generation_ms"] = r.generation_ms
            scores["pipeline_metrics"] = r.pipeline_result.metrics.model_dump()

    return EvalReport(
        metadata=metadata,
        per_sample_scores=per_sample_scores,
        aggregate_scores=aggregate_scores,
        latency_summary=latency_summary,
        multi_turn_scores=multi_turn_scores,
    )


def save_report(report: EvalReport, output_dir: Path) -> Path:
    """Save an evaluation report as JSON.

    Args:
        report (EvalReport): The report to save.
        output_dir (Path): Directory to save the report in.

    Returns:
        Path: Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"eval_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    logger.info("Report saved to {}", filepath)
    return filepath


def print_summary(report: EvalReport) -> None:
    """Print a console summary of the evaluation report.

    Args:
        report (EvalReport): The report to summarize.
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 60)
    print(f"  Timestamp:  {report.metadata.get('timestamp', 'N/A')}")
    print(f"  Git commit: {report.metadata.get('git_commit', 'N/A')}")
    print(f"  Dataset:    {report.metadata.get('dataset_path', 'N/A')}")
    print(f"  Evaluator:  {report.metadata.get('evaluator_model', 'N/A')}")
    print(f"  Samples:    {report.metadata.get('total_samples', 0)}")

    print("\n--- Aggregate Scores ---")
    for metric, score in report.aggregate_scores.items():
        print(f"  {metric:25s} {score:.4f}")

    print("\n--- Latency Summary ---")
    for stage, stats in report.latency_summary.items():
        if isinstance(stats, dict) and stats.get("mean", 0) > 0:
            print(
                f"  {stage:25s} "
                f"mean={stats['mean']:.0f}ms  "
                f"median={stats['median']:.0f}ms  "
                f"p95={stats['p95']:.0f}ms"
            )

    errors = sum(
        1 for s in report.per_sample_scores if s.get("error") is not None
    )
    if errors:
        print(f"\n  Errors: {errors}/{report.metadata.get('total_samples', 0)}")

    print("=" * 60 + "\n")
