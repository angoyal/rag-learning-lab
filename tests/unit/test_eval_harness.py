"""Tests for the RAGAS and DeepEval evaluation wrappers.

Tests verify the interface and input validation without requiring
a running LLM (which is needed for actual metric computation).
"""

from __future__ import annotations

import pytest
from src.evaluate.deepeval_eval import _METRIC_MAP as DEEPEVAL_METRICS
from src.evaluate.deepeval_eval import run_deepeval_evaluation
from src.evaluate.ragas_eval import _METRIC_MAP as RAGAS_METRICS
from src.evaluate.ragas_eval import run_ragas_evaluation

# -- RAGAS interface tests --


@pytest.mark.unit
def test_ragas_unknown_metric():
    with pytest.raises(ValueError, match="Unknown RAGAS metric"):
        run_ragas_evaluation([], metrics=["nonexistent_metric"])


@pytest.mark.unit
def test_ragas_metric_map_has_expected_keys():
    expected = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert set(RAGAS_METRICS.keys()) == expected


# -- DeepEval interface tests --


@pytest.mark.unit
def test_deepeval_unknown_metric():
    with pytest.raises(ValueError, match="Unknown DeepEval metric"):
        run_deepeval_evaluation([], metrics=["nonexistent_metric"])


@pytest.mark.unit
def test_deepeval_metric_map_has_expected_keys():
    expected = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
    assert set(DEEPEVAL_METRICS.keys()) == expected
