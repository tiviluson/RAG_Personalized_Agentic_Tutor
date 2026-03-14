"""Prompt templates for custom evaluation metrics."""

OUTPUT_ACCURACY_PROMPT = """\
You are an expert evaluator for a university-level AI tutoring system. \
Your task is to judge whether the predicted answer is factually correct \
and complete compared to the reference answer.

## Reference Answer
{ground_truth}

## Predicted Answer
{predicted_answer}

## Instructions
1. Compare the predicted answer against the reference answer.
2. Focus on factual correctness and completeness of key concepts.
3. Minor phrasing or formatting differences are acceptable.
4. The predicted answer may include additional correct information beyond \
the reference -- this is fine.
5. The predicted answer must not contain factually incorrect statements \
that contradict the reference.

## Output
Respond with EXACTLY one of the following on the first line:
CORRECT
INCORRECT

Then on the next line, provide a brief (1-2 sentence) explanation of your judgment.
"""
