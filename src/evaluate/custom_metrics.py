"""Custom evaluation metrics that don't require an LLM judge.

Computes latency statistics, chunk retrieval metrics, and rough token counts.
"""

from __future__ import annotations

import statistics


def compute_latency_metrics(latencies: list[float]) -> dict[str, float]:
    """Compute latency statistics from a list of per-query latencies.

    Args:
        latencies: List of latency values in seconds.

    Returns:
        Dict with keys: avg, p50, p95, min, max (all in seconds).
    """
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    sorted_lats = sorted(latencies)
    p95_idx = int(len(sorted_lats) * 0.95)
    p95_idx = min(p95_idx, len(sorted_lats) - 1)
    return {
        "avg": statistics.mean(sorted_lats),
        "p50": statistics.median(sorted_lats),
        "p95": sorted_lats[p95_idx],
        "min": sorted_lats[0],
        "max": sorted_lats[-1],
    }


def compute_chunk_metrics(
    chunk_counts: list[int],
    chunk_lengths: list[list[int]] | None = None,
) -> dict[str, float]:
    """Compute retrieval chunk statistics.

    Args:
        chunk_counts: Number of chunks retrieved per query.
        chunk_lengths: Optional list of lists, where each inner list contains
            the character lengths of chunks for one query.

    Returns:
        Dict with avg_chunks, min_chunks, max_chunks, and optionally
        avg_chunk_length.
    """
    if not chunk_counts:
        return {"avg_chunks": 0.0, "min_chunks": 0, "max_chunks": 0}
    result: dict[str, float] = {
        "avg_chunks": statistics.mean(chunk_counts),
        "min_chunks": min(chunk_counts),
        "max_chunks": max(chunk_counts),
    }
    if chunk_lengths:
        all_lengths = [length for query_lengths in chunk_lengths for length in query_lengths]
        if all_lengths:
            result["avg_chunk_length"] = statistics.mean(all_lengths)
    return result


def estimate_token_count(text: str) -> int:
    """Estimate token count from text using a word-based heuristic.

    Uses the rule-of-thumb that 1 token is roughly 0.75 words (or 4 chars).

    Args:
        text: Input text string.

    Returns:
        Estimated number of tokens.
    """
    return max(1, len(text) // 4)


def compute_cost_estimate(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_price_per_1k: float = 0.0,
    completion_price_per_1k: float = 0.0,
) -> dict[str, float]:
    """Estimate cost for a query based on token counts and pricing.

    For local Ollama usage, both prices are 0. For cloud APIs (Bedrock, Vertex),
    pass the per-1K-token prices.

    Args:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        prompt_price_per_1k: Price per 1K prompt tokens.
        completion_price_per_1k: Price per 1K completion tokens.

    Returns:
        Dict with prompt_cost, completion_cost, and total_cost.
    """
    prompt_cost = (prompt_tokens / 1000) * prompt_price_per_1k
    completion_cost = (completion_tokens / 1000) * completion_price_per_1k
    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": prompt_cost + completion_cost,
    }
