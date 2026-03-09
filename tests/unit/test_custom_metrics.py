"""Tests for custom evaluation metrics."""

from __future__ import annotations

import pytest
from src.evaluate.custom_metrics import (
    compute_chunk_metrics,
    compute_cost_estimate,
    compute_latency_metrics,
    estimate_token_count,
)

# -- Latency metrics --


@pytest.mark.unit
def test_latency_metrics_basic():
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = compute_latency_metrics(latencies)
    assert result["avg"] == pytest.approx(0.3)
    assert result["p50"] == pytest.approx(0.3)
    assert result["min"] == pytest.approx(0.1)
    assert result["max"] == pytest.approx(0.5)


@pytest.mark.unit
def test_latency_metrics_single_value():
    result = compute_latency_metrics([1.5])
    assert result["avg"] == pytest.approx(1.5)
    assert result["p50"] == pytest.approx(1.5)
    assert result["p95"] == pytest.approx(1.5)
    assert result["min"] == pytest.approx(1.5)
    assert result["max"] == pytest.approx(1.5)


@pytest.mark.unit
def test_latency_metrics_empty():
    result = compute_latency_metrics([])
    assert result["avg"] == 0.0
    assert result["p50"] == 0.0


@pytest.mark.unit
def test_latency_p95_high_outlier():
    latencies = [0.1] * 95 + [10.0] * 5
    result = compute_latency_metrics(latencies)
    assert result["p95"] >= 0.1
    assert result["avg"] < result["p95"]


@pytest.mark.unit
def test_latency_metrics_unsorted_input():
    result = compute_latency_metrics([0.5, 0.1, 0.3])
    assert result["min"] == pytest.approx(0.1)
    assert result["max"] == pytest.approx(0.5)


# -- Chunk metrics --


@pytest.mark.unit
def test_chunk_metrics_basic():
    result = compute_chunk_metrics([3, 5, 4])
    assert result["avg_chunks"] == pytest.approx(4.0)
    assert result["min_chunks"] == 3
    assert result["max_chunks"] == 5


@pytest.mark.unit
def test_chunk_metrics_empty():
    result = compute_chunk_metrics([])
    assert result["avg_chunks"] == 0.0


@pytest.mark.unit
def test_chunk_metrics_with_lengths():
    counts = [2, 3]
    lengths = [[100, 200], [150, 250, 300]]
    result = compute_chunk_metrics(counts, lengths)
    assert "avg_chunk_length" in result
    assert result["avg_chunk_length"] == pytest.approx(200.0)


@pytest.mark.unit
def test_chunk_metrics_without_lengths():
    result = compute_chunk_metrics([5, 5])
    assert "avg_chunk_length" not in result


# -- Token estimation --


@pytest.mark.unit
def test_estimate_token_count():
    text = "Hello world, this is a test of token estimation."
    tokens = estimate_token_count(text)
    assert tokens >= 1
    assert tokens == len(text) // 4


@pytest.mark.unit
def test_estimate_token_count_empty():
    assert estimate_token_count("") == 1  # min 1


@pytest.mark.unit
def test_estimate_token_count_long_text():
    text = "word " * 1000
    tokens = estimate_token_count(text)
    assert tokens > 100


# -- Cost estimation --


@pytest.mark.unit
def test_cost_estimate_zero_prices():
    result = compute_cost_estimate(1000, 500)
    assert result["total_cost"] == 0.0
    assert result["prompt_cost"] == 0.0
    assert result["completion_cost"] == 0.0


@pytest.mark.unit
def test_cost_estimate_with_prices():
    result = compute_cost_estimate(
        prompt_tokens=2000,
        completion_tokens=500,
        prompt_price_per_1k=0.01,
        completion_price_per_1k=0.03,
    )
    assert result["prompt_cost"] == pytest.approx(0.02)
    assert result["completion_cost"] == pytest.approx(0.015)
    assert result["total_cost"] == pytest.approx(0.035)
