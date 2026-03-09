"""DeepEval evaluation integration.

Wraps the DeepEval library to evaluate RAG pipeline outputs.
Requires an LLM for evaluation (configured via DeepEval settings).
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

_METRIC_MAP = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "context_precision": ContextualPrecisionMetric,
    "context_recall": ContextualRecallMetric,
}

_DEFAULT_METRICS = ["faithfulness", "answer_relevancy"]


def run_deepeval_evaluation(
    dataset: list[dict],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate RAG outputs using DeepEval metrics.

    Each item in the dataset should have:
        - "question": the user query
        - "answer": the generated answer
        - "contexts": list of retrieved context strings
        - "reference": the expected answer (for precision/recall metrics)

    Args:
        dataset: List of dicts with question, answer, contexts, and reference keys.
        metrics: List of metric names to compute. Defaults to faithfulness + answer_relevancy.

    Returns:
        Dict mapping metric names to their average scores across all items (0-1).

    Raises:
        ValueError: If an unknown metric name is provided.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    metric_classes = {}
    for name in metrics:
        cls = _METRIC_MAP.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown DeepEval metric: {name!r}. "
                f"Available: {list(_METRIC_MAP.keys())}"
            )
        metric_classes[name] = cls

    scores: dict[str, list[float]] = {name: [] for name in metrics}

    for item in dataset:
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["answer"],
            expected_output=item.get("reference", ""),
            retrieval_context=item["contexts"],
        )
        for name, cls in metric_classes.items():
            metric = cls()
            metric.measure(test_case)
            scores[name].append(metric.score)

    return {
        name: sum(vals) / len(vals) if vals else 0.0
        for name, vals in scores.items()
    }
