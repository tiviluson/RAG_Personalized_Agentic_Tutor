"""Evaluation dataset models and JSONL loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, field_validator


class EvalSample(BaseModel):
    """A single-turn evaluation sample.

    Args:
        question: The student question to evaluate.
        ground_truth: Reference answer for scoring.
        difficulty: Question difficulty level.
        category: Question category for analysis.
        expected_sources: Optional list of expected source filenames.
    """

    question: str
    ground_truth: str
    difficulty: Literal["easy", "medium", "hard"]
    category: Literal[
        "factual_recall",
        "conceptual",
        "multi_hop",
        "cross_course",
        "out_of_scope",
    ]
    expected_sources: list[str] | None = None

    @field_validator("question", "ground_truth")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        """Validate that question and ground_truth are non-empty."""
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class TurnSample(BaseModel):
    """A single turn within a multi-turn evaluation scenario.

    Args:
        question: The student question for this turn.
        ground_truth: Reference answer for this turn (can be brief for
            intermediate turns -- 1-2 sentences capturing the key point).
    """

    question: str
    ground_truth: str

    @field_validator("question", "ground_truth")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        """Validate that question and ground_truth are non-empty."""
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class MultiTurnEvalSample(BaseModel):
    """A multi-turn evaluation scenario with 2-3 conversational turns.

    Args:
        turns: Ordered list of conversation turns (2-3 turns).
        difficulty: Scenario difficulty level.
        category: Always "multi_turn".
        expected_sources: Optional list of expected source filenames.
    """

    turns: list[TurnSample]
    difficulty: Literal["easy", "medium", "hard"]
    category: Literal["multi_turn"] = "multi_turn"
    expected_sources: list[str] | None = None

    @field_validator("turns")
    @classmethod
    def must_have_multiple_turns(cls, v: list[TurnSample]) -> list[TurnSample]:
        """Validate that the scenario has at least 2 turns."""
        if len(v) < 2:
            raise ValueError("multi-turn scenario must have at least 2 turns")
        return v


def load_dataset(path: Path) -> list[EvalSample]:
    """Load and validate a single-turn evaluation dataset from JSONL.

    Each line in the file must be a valid JSON object matching the
    EvalSample schema. Lines that fail validation are logged and skipped.

    Args:
        path (Path): Path to the JSONL file.

    Returns:
        list[EvalSample]: Validated evaluation samples.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: list[EvalSample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                samples.append(EvalSample(**data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Skipping line {} in {}: {}", line_num, path.name, e)

    logger.info("Loaded {} samples from {}", len(samples), path.name)
    return samples


def load_multi_turn_dataset(path: Path) -> list[MultiTurnEvalSample]:
    """Load and validate a multi-turn evaluation dataset from JSONL.

    Each line in the file must be a valid JSON object matching the
    MultiTurnEvalSample schema. Lines that fail validation are logged
    and skipped.

    Args:
        path (Path): Path to the JSONL file.

    Returns:
        list[MultiTurnEvalSample]: Validated multi-turn scenarios.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    scenarios: list[MultiTurnEvalSample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                scenarios.append(MultiTurnEvalSample(**data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Skipping line {} in {}: {}", line_num, path.name, e)

    total_turns = sum(len(s.turns) for s in scenarios)
    logger.info(
        "Loaded {} multi-turn scenarios ({} total turns) from {}",
        len(scenarios),
        total_turns,
        path.name,
    )
    return scenarios
