"""RAGAS evaluation integration.

Wraps the RAGAS library to evaluate RAG pipeline outputs on
faithfulness, answer relevancy, context precision, and context recall.
Requires an LLM for evaluation (uses Ollama by default).
"""

from __future__ import annotations

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

_METRIC_MAP = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
}

_DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def run_ragas_evaluation(
    dataset: list[dict],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate RAG outputs using RAGAS metrics.

    Each item in the dataset should have:
        - "question": the user query
        - "answer": the generated answer
        - "contexts": list of retrieved context strings
        - "reference": the ground truth answer (needed for context_recall)

    Args:
        dataset: List of dicts with question, answer, contexts, and reference keys.
        metrics: List of metric names to compute. Defaults to all four RAGAS metrics.

    Returns:
        Dict mapping metric names to their aggregate scores (0-1).

    Raises:
        ValueError: If an unknown metric name is provided.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    metric_objects = []
    for name in metrics:
        cls = _METRIC_MAP.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown RAGAS metric: {name!r}. "
                f"Available: {list(_METRIC_MAP.keys())}"
            )
        metric_objects.append(cls())

    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item.get("reference", ""),
        )
        for item in dataset
    ]
    eval_dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=eval_dataset, metrics=metric_objects)
    return {name: float(result[name]) for name in metrics if name in result}
